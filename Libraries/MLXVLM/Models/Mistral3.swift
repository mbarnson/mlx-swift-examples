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

// MARK: - Utility Functions

/// Convert value to pair tuple
private func pair<T>(_ x: T) -> (T, T) where T: BinaryInteger {
    return (x, x)
}

private func pair<T>(_ x: [T]) -> (T, T) {
    assert(x.count == 2, "Array must have exactly 2 elements")
    return (x[0], x[1])
}

// MARK: - Unfold Operation (im2col)

/// Extract sliding local blocks from a batched input tensor (MLX implementation)
/// Equivalent to PyTorch's nn.functional.unfold or im2col operation
/// - Parameters:
///   - input: Input tensor of shape (B, C, H, W)
///   - kernelSize: Size of the sliding blocks
///   - dilation: Controls spacing between kernel elements
///   - padding: Amount of implicit padding
///   - stride: Stride between blocks
/// - Returns: Unfolded tensor of shape (B, C*kernelHeight*kernelWidth, L) where L is the number of blocks
func unfold(
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
            IntOrPair((padding.1, padding.1))
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

            for di in 0..<kernelSize.0 {
                for dj in 0..<kernelSize.1 {
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

fileprivate class Mistral3PatchMerger: Module {
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
        for i in 0..<imageSizes.dim(0) {
            let h = Int(imageSizes[i, 0].item(Int.self)) / patchSize
            let w = Int(imageSizes[i, 1].item(Int.self)) / patchSize
            imageSizesPatch.append((h, w))
        }

        let tokensPerImage = imageSizesPatch.map { h, w in h * w }
        let d = imageFeatures.dim(imageFeatures.ndim - 1)

        // Cast to bfloat16
        var features = imageFeatures.asType(.bfloat16)

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
                var imageGrid = imageTokens
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

fileprivate class Mistral3MultiModalProjector: Module {
    @ModuleInfo var norm: RMSNorm
    let patchMerger: Mistral3PatchMerger
    @ModuleInfo(key: "linear_1") var linear1: Linear
    let gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(_ config: Mistral3Configuration) {
        // RMS normalization
        self._norm.wrappedValue = RMSNorm(dimensions: config.visionConfig.hiddenSize)

        // Patch merger
        self.patchMerger = Mistral3PatchMerger(config)

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

    fileprivate let mistral3MultiModalProjector: Mistral3MultiModalProjector
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
            eosTokenId: config.eosTokenId
        )

        self.mistral3Config = config
        self.mistral3MultiModalProjector = Mistral3MultiModalProjector(config)

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
        let imageFeatures = mistral3MultiModalProjector(selectedImageFeature, imageSizes: imageSizes)

        // Merge vision and text embeddings
        return Pixtral.mergeInputIdsWithImageFeatures(
            imageTokenIndex: mistral3Config.imageTokenIndex,
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds!
        )
    }
}

// MARK: - Processor

public struct Mistral3ProcessorConfiguration: Codable, Sendable {
    // Processor configuration if needed
}

public class Mistral3Processor: UserInputProcessor {
    private let config: Mistral3ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(
        _ config: Mistral3ProcessorConfiguration,
        tokenizer: any Tokenizer
    ) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Basic processor - delegates to default behavior
        // In production, would handle Mistral3-specific image preprocessing
        fatalError("Mistral3Processor not yet implemented - use default processor")
    }
}
