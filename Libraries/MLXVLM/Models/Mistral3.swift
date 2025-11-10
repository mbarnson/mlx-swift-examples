// Copyright © 2024 Apple Inc.

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/mistral3

import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Configuration

public struct Mistral3Configuration: Codable, Sendable {
    public let textConfig: PixtralConfiguration.TextConfiguration
    public let visionConfig: PixtralConfiguration.VisionConfiguration
    public let modelType: String
    public let ignoreIndex: Int
    public let imageTokenIndex: Int
    public let visionFeatureSelectStrategy: String
    public let visionFeatureLayer: Int
    public let vocabularySize: Int
    public let spatialMergeSize: Int
    public let multimodalProjectorBias: Bool
    public let eosTokenId: [Int]?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case ignoreIndex = "ignore_index"
        case imageTokenIndex = "image_token_index"
        case visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case visionFeatureLayer = "vision_feature_layer"
        case vocabularySize = "vocab_size"
        case spatialMergeSize = "spatial_merge_size"
        case multimodalProjectorBias = "multimodal_projector_bias"
        case eosTokenId = "eos_token_id"
    }
}

// MARK: - Unfold Operation (im2col)

/// Extract sliding local blocks from a batched input tensor (MLX implementation).
///
/// This operation extracts all kernel-sized patches from the input tensor and reorganizes them into columns,
/// which is commonly used in convolution operations (hence "im2col" - image to column). Each output column
/// represents one patch flattened into a 1D vector.
///
/// The operation is equivalent to PyTorch's `nn.functional.unfold` or traditional im2col implementations.
/// It's used in Mistral3's spatial patch merging to group image patches into 2x2 blocks before projection.
///
/// ## Example
/// ```swift
/// let input = MLXArray(shape: [1, 3, 4, 4])  // 1 image, 3 channels, 4x4 spatial
/// let output = unfold(input, kernelSize: (2, 2), stride: (2, 2))
/// // output shape: [1, 12, 4]
/// // - 12 = 3 channels * 2 * 2 kernel size
/// // - 4 = number of non-overlapping 2x2 blocks in 4x4 image
/// ```
///
/// - Parameters:
///   - input: Input tensor of shape (B, C, H, W) where B=batch, C=channels, H=height, W=width
///   - kernelSize: Size of the sliding blocks (height, width)
///   - dilation: Controls spacing between kernel elements. Default (1, 1) means no gaps
///   - padding: Amount of implicit zero-padding added to input edges. Default (0, 0)
///   - stride: Step size between blocks (height, width). Default (1, 1) means maximum overlap
/// - Returns: Unfolded tensor of shape (B, C*kH*kW, L) where L is the number of extracted blocks
public func unfold(
    _ input: MLXArray,
    kernelSize: (Int, Int),
    dilation: (Int, Int) = (1, 1),
    padding: (Int, Int) = (0, 0),
    stride: (Int, Int) = (1, 1)
) -> MLXArray {
    // Input shape
    let batchSize = input.dim(0)
    let channels = input.dim(1)
    let height = input.dim(2)
    let width = input.dim(3)

    // Add padding if needed
    var paddedInput = input
    if padding.0 > 0 || padding.1 > 0 {
        let paddingSpec: [IntOrPair] = [
            IntOrPair((0, 0)),
            IntOrPair((0, 0)),
            IntOrPair((padding.0, padding.0)),
            IntOrPair((padding.1, padding.1)),
        ]
        paddedInput = padded(input, widths: paddingSpec)
    }

    // Calculate output dimensions
    let heightOut = (height + 2 * padding.0 - dilation.0 * (kernelSize.0 - 1) - 1) / stride.0 + 1
    let widthOut = (width + 2 * padding.1 - dilation.1 * (kernelSize.1 - 1) - 1) / stride.1 + 1

    // Initialize output arrays
    var blocks: [MLXArray] = []

    // Extract blocks
    var i = 0
    while i < height + 2 * padding.0 - kernelSize.0 * dilation.0 + 1 {
        var j = 0
        while j < width + 2 * padding.1 - kernelSize.1 * dilation.1 + 1 {
            // Extract the block for all channels
            var block: [MLXArray] = []

            for di in 0 ..< kernelSize.0 {
                for dj in 0 ..< kernelSize.1 {
                    let hIdx = i + di * dilation.0
                    let wIdx = j + dj * dilation.1
                    // Get the block for all channels
                    block.append(paddedInput[0..., 0..., hIdx, wIdx])
                }
            }

            // Stack the channel-blocks
            var stackedBlock = stacked(block, axis: 1)  // Shape: (B, k*k, C)
            stackedBlock = stackedBlock.transposed(0, 2, 1)  // Shape: (B, C, k*k)
            blocks.append(stackedBlock)

            j += stride.1
        }
        i += stride.0
    }

    // Stack all blocks together
    var result = stacked(blocks, axis: -1)  // Shape: (B, C, k*k, L)

    // Reshape to match PyTorch's unfold output format: (B, C*k*k, L)
    result = result.reshaped(
        batchSize,
        channels * kernelSize.0 * kernelSize.1,
        heightOut * widthOut
    )

    return result
}

// MARK: - Mistral3 Components

/// Spatial patch merger for Mistral3 vision model.
///
/// Reduces the number of vision tokens by merging adjacent patches in a spatial grid.
/// This is a key difference from Pixtral: Mistral3 uses 2x2 spatial merging to reduce
/// token count by 4x, making inference more efficient for high-resolution images.
///
/// The merger reshapes image patches from a 1D sequence back into a 2D grid, then uses
/// the unfold operation to group them into spatial blocks (default 2x2), which are then
/// projected down to the original hidden dimension.
internal class Mistral3PatchMerger: Module {
    let spatialMergeSize: Int
    let patchSize: Int

    @ModuleInfo(key: "merging_layer") var mergingLayer: Linear

    public init(_ config: Mistral3Configuration) {
        self.spatialMergeSize = config.spatialMergeSize
        self.patchSize = config.visionConfig.patchSize

        let hiddenSize = config.visionConfig.hiddenSize
        self._mergingLayer.wrappedValue = Linear(
            hiddenSize * spatialMergeSize * spatialMergeSize,
            hiddenSize,
            bias: false
        )
    }

    public func callAsFunction(_ imageFeatures: MLXArray, imageSizes: MLXArray) -> MLXArray {
        // Convert image sizes to patch grid dimensions
        var imageSizesPatch: [(Int, Int)] = []
        for i in 0 ..< imageSizes.dim(0) {
            let h = Int(imageSizes[i, 0].item(Int.self)) / patchSize
            let w = Int(imageSizes[i, 1].item(Int.self)) / patchSize
            imageSizesPatch.append((h, w))
        }

        let tokensPerImage = imageSizesPatch.map { h, w in h * w }
        let d = imageFeatures.dim(imageFeatures.ndim - 1)

        // Cast to bfloat16
        let features = imageFeatures.asType(.bfloat16)

        // Split into chunks based on tokens per image
        var splitIndices: [Int] = []
        var currentIndex = 0
        for tokens in tokensPerImage {
            currentIndex += tokens
            splitIndices.append(currentIndex)
        }
        splitIndices = Array(splitIndices.dropLast())

        let chunks = split(features, indices: splitIndices, axis: 1)

        // Process each chunk
        var permutedTensors: [MLXArray] = []
        for (imageIndex, imageTokens) in chunks.enumerated() {
            if imageTokens.dim(1) > 0 {
                let (h, w) = imageSizesPatch[imageIndex]

                // Reshape to 2D grid and transpose
                let imageGrid =
                    imageTokens
                    .reshaped(h, w, d)
                    .transposed(2, 0, 1)
                    .expandedDimensions(axis: 0)  // Add batch dim

                // Apply unfold operation
                var grid = unfold(
                    imageGrid,
                    kernelSize: (spatialMergeSize, spatialMergeSize),
                    stride: (spatialMergeSize, spatialMergeSize)
                )

                grid = grid.reshaped(d * spatialMergeSize * spatialMergeSize, -1).T
                permutedTensors.append(grid)
            }
        }

        // Concatenate all processed chunks
        var result = concatenated(permutedTensors, axis: 0)
        result = mergingLayer(result)

        return result.expandedDimensions(axis: 0)  // Add batch dim back
    }
}

internal class Mistral3MultiModalProjector: Module {
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "patch_merger") var patchMerger: Mistral3PatchMerger
    @ModuleInfo(key: "linear_1") var linear1: Linear
    let gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(_ config: Mistral3Configuration) {
        // RMS normalization
        self._norm.wrappedValue = RMSNorm(dimensions: config.visionConfig.hiddenSize)

        // Patch merger
        self._patchMerger.wrappedValue = Mistral3PatchMerger(config)

        // Determine number of feature layers
        let numFeatureLayers = 1  // Simplified - could support array of layers

        // Projection layers
        let inputDim = config.visionConfig.hiddenSize * numFeatureLayers
        let outputDim = config.textConfig.hiddenSize

        self._linear1.wrappedValue = Linear(
            inputDim,
            outputDim,
            bias: config.multimodalProjectorBias
        )
        self.gelu = GELU()
        self._linear2.wrappedValue = Linear(
            outputDim,
            outputDim,
            bias: config.multimodalProjectorBias
        )
    }

    public func callAsFunction(_ x: MLXArray, imageSizes: MLXArray) -> MLXArray {
        var features = norm(x)
        features = patchMerger(features, imageSizes: imageSizes)
        features = linear1(features)
        features = gelu(features)
        features = linear2(features)
        return features
    }
}

// MARK: - Main Mistral3 Model

public class Mistral3: Pixtral {

    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: Mistral3MultiModalProjector
    fileprivate let mistral3Config: Mistral3Configuration

    public init(_ config: Mistral3Configuration) {
        // Create Pixtral configuration from Mistral3 config
        let pixtralConfig = PixtralConfiguration(
            textConfig: config.textConfig,
            visionConfig: config.visionConfig,
            modelType: "pixtral",  // Use pixtral as base type
            ignoreIndex: config.ignoreIndex,
            imageTokenIndex: config.imageTokenIndex,
            visionFeatureSelectStrategy: config.visionFeatureSelectStrategy,
            visionFeatureLayer: config.visionFeatureLayer,
            vocabularySize: config.vocabularySize,
            multimodalProjectorBias: config.multimodalProjectorBias,
            eosTokenId: config.eosTokenId
        )

        self.mistral3Config = config
        self._multiModalProjector.wrappedValue = Mistral3MultiModalProjector(config)

        super.init(pixtralConfig)
    }

    /// Override getInputEmbeddings to use Mistral3's projector instead of Pixtral's
    override func getInputEmbeddings(
        inputIds: MLXArray? = nil,
        pixelValues: [MLXArray]? = nil,
        imageSizes: MLXArray? = nil
    ) -> MLXArray {
        guard let pixelValues = pixelValues, let imageSizes = imageSizes else {
            return getTextEmbeddings(inputIds!)
        }

        // Get text embeddings
        let inputsEmbeds = getTextEmbeddings(inputIds!)

        // Get vision embeddings
        let hiddenStates = getVisionHiddenStates(pixelValues)
        let selectedImageFeature = hiddenStates[visionFeatureLayer]

        // Use Mistral3's multimodal projector (with patch merging)
        let imageFeatures = multiModalProjector(selectedImageFeature, imageSizes: imageSizes)

        // Merge vision and text embeddings
        return Pixtral.mergeInputIdsWithImageFeatures(
            imageTokenIndex: mistral3Config.imageTokenIndex,
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds!
        )
    }

    // Note: prepare() is inherited from Pixtral
    // Pixtral's prepare() now extracts imageSizes from LMInput.ProcessedImage.frames
    // and passes them to getInputEmbeddings(), so Mistral3 works automatically
}

// MARK: - Processor

public struct Mistral3ProcessorConfiguration: Codable, Sendable {
    // Processor configuration if needed
    // Mistral3 inherits Pixtral's preprocessing but adds image size tracking
    public init() {}
}

/// Processor for Mistral3 vision-language models
///
/// Extends Pixtral processor with additional image size tracking for spatial patch merging.
/// Mistral3 uses 2x2 spatial merging to reduce token count (4x fewer tokens than Pixtral).
///
/// Key differences from Pixtral:
/// - Tracks original image dimensions (height, width) for each image
/// - Passes image_sizes to model for patch merging computation
/// - Otherwise identical message generation and chat template application
///
/// mistral-common compatibility:
/// - Uses same message protocol and chat template application as Pixtral
/// - Image size tracking is model-specific, not part of mistral-common
///
/// References:
/// - mistral-common: https://github.com/mistralai/mistral-common
/// - mlx-vlm Mistral3: https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/mistral3
public class Mistral3Processor: UserInputProcessor {
    private let config: Mistral3ProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let messageGenerator: PixtralMessageGenerator

    public init(
        _ config: Mistral3ProcessorConfiguration,
        tokenizer: any Tokenizer
    ) {
        self.config = config
        self.tokenizer = tokenizer
        // Reuse Pixtral's message generator - message format is identical
        self.messageGenerator = PixtralMessageGenerator()
    }

    /// Preprocess images and track their sizes for Mistral3 spatial merging
    ///
    /// Returns:
    /// - pixelValues: MLXArray [N, C, H, W] concatenated images
    /// - frames: [THW] with (1, height, width) for each image (time=1 for images)
    private func preprocessImagesWithSizes(
        _ images: [UserInput.Image], processing: UserInput.Processing?
    ) throws -> (pixelValues: MLXArray, frames: [THW]) {
        var processedImages: [MLXArray] = []
        var frames: [THW] = []

        for imageInput in images {
            // Load as CIImage
            var image = try imageInput.asCIImage()

            // Apply user-requested processing (resize, etc.)
            image = MediaProcessing.apply(image, processing: processing)

            // Track size BEFORE final processing
            // Mistral3 needs original dimensions for patch merging
            let extent = image.extent
            let height = Int(extent.height)
            let width = Int(extent.width)
            frames.append(THW(1, height, width))  // t=1 for images (not video)

            // Convert to sRGB tone curve space
            image = MediaProcessing.inSRGBToneCurveSpace(image)

            // Convert to MLXArray [1, C, H, W]
            let array = MediaProcessing.asMLXArray(image)
            processedImages.append(array)
        }

        // Concatenate images along batch dimension
        let pixelValues = concatenated(processedImages, axis: 0)

        return (pixelValues, frames)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // 1. Generate messages in mistral-common compatible format
        // Reuses Pixtral's message generation - format is identical
        let messages = messageGenerator.generate(from: input)

        // 2. Apply chat template using tokenizer
        // THIS IS THE KEY STEP for mistral-common compatibility
        // Identical to Pixtral - same Jinja templates, same special token handling
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )

        // 3. Handle images if present
        if !input.images.isEmpty {
            // Mistral3-specific: track image sizes for spatial merging
            let (pixelValues, frames) = try preprocessImagesWithSizes(
                input.images, processing: input.processing)

            return LMInput(
                text: .init(tokens: MLXArray(promptTokens)),
                image: .init(pixels: pixelValues, frames: frames)
            )
        }

        // Text-only input
        return LMInput(tokens: MLXArray(promptTokens))
    }
}
