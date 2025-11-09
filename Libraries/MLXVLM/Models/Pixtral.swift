// Copyright © 2024 Apple Inc.

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/pixtral

import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Configuration

public struct PixtralConfiguration: Codable, Sendable {
    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let headDim: Int?
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let rmsNormEps: Float
        public let vocabularySize: Int
        public let kvHeads: Int
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let ropeScaling: [String: StringOrNumber]?
        public let maxPositionEmbeddings: Int?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case headDim = "head_dim"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case maxPositionEmbeddings = "max_position_embeddings"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenLayers: Int
        public let hiddenSize: Int
        public let headDim: Int?
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        public let projectionDim: Int?
        public let vocabularySize: Int
        public let numChannels: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenLayers = "num_hidden_layers"
            case hiddenSize = "hidden_size"
            case headDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case projectionDim = "projection_dim"
            case vocabularySize = "vocab_size"
            case numChannels = "num_channels"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
        }
    }

    public let textConfig: TextConfiguration
    public let visionConfig: VisionConfiguration
    public let modelType: String
    public let ignoreIndex: Int
    public let imageTokenIndex: Int
    public let visionFeatureSelectStrategy: String
    public let visionFeatureLayer: Int
    public let vocabularySize: Int
    // Default to true for backward compatibility with existing Pixtral models
    public var multimodalProjectorBias: Bool? = nil
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
        case multimodalProjectorBias = "multimodal_projector_bias"
        case eosTokenId = "eos_token_id"
    }

    // Memberwise initializer for programmatic construction
    public init(
        textConfig: TextConfiguration,
        visionConfig: VisionConfiguration,
        modelType: String,
        ignoreIndex: Int,
        imageTokenIndex: Int,
        visionFeatureSelectStrategy: String,
        visionFeatureLayer: Int,
        vocabularySize: Int,
        multimodalProjectorBias: Bool = true,
        eosTokenId: [Int]? = nil
    ) {
        self.textConfig = textConfig
        self.visionConfig = visionConfig
        self.modelType = modelType
        self.ignoreIndex = ignoreIndex
        self.imageTokenIndex = imageTokenIndex
        self.visionFeatureSelectStrategy = visionFeatureSelectStrategy
        self.visionFeatureLayer = visionFeatureLayer
        self.vocabularySize = vocabularySize
        self.multimodalProjectorBias = multimodalProjectorBias
        self.eosTokenId = eosTokenId
    }

    // Computed property for bias with backward-compatible default
    public var effectiveMultimodalProjectorBias: Bool {
        multimodalProjectorBias ?? true
    }
}

// MARK: - Vision Encoder

fileprivate enum Vision {

    /// Check if the array has the expected shape for conv weights
    static func checkArrayShape(_ arr: MLXArray) -> Bool {
        let shape = arr.shape

        // Check if the shape has 4 dimensions
        guard shape.count == 4 else {
            return false
        }

        let outChannels = shape[0]
        let kH = shape[1]
        let kW = shape[2]

        // Check if outChannels is the largest, and kH and kW are the same
        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    }

    /// Generate position IDs in meshgrid format for patch embeddings
    static func positionIdsInMeshgrid(patchEmbedsList: [MLXArray], maxWidth: Int) -> MLXArray {
        var positions: [MLXArray] = []

        for patch in patchEmbedsList {
            let height = patch.dim(0)
            let width = patch.dim(1)

            let indices = MLXArray.zeros([height, width, 2], dtype: .int32)
            for h in 0..<height {
                for w in 0..<width {
                    indices[h, w, 0] = MLXArray(h)
                    indices[h, w, 1] = MLXArray(w)
                }
            }

            let hGrid = indices[0..., 0..., 0]
            let vGrid = indices[0..., 0..., 1]

            let ids = hGrid * MLXArray(maxWidth) + vGrid
            positions.append(ids.flattened())
        }

        return concatenated(positions)
    }

    /// Generate block attention mask for patch embeddings
    static func generateBlockAttentionMask(patchEmbedsList: [Int], tensor: MLXArray) -> MLXArray {
        let seqLen = tensor.dim(1)
        let dMin: Float = -1e9

        // Create a matrix filled with dMin (equivalent to mx.full in Python)
        var causalMask = MLXArray.zeros([seqLen, seqLen]) + dMin

        let blockEndIdx = MLXArray(patchEmbedsList).cumsum()
        var blockStartIdx = MLXArray([0] + Array(patchEmbedsList.dropLast()))
        blockStartIdx = blockStartIdx.cumsum()

        for i in 0..<patchEmbedsList.count {
            let start = Int(blockStartIdx[i].item(Int.self))
            let end = Int(blockEndIdx[i].item(Int.self))
            causalMask[start..<end, start..<end] = MLXArray(0.0)
        }

        let batchSize = tensor.dim(0)
        let expanded = causalMask.expandedDimensions(axes: [0, 1])
        return broadcast(expanded, to: [batchSize, 1, seqLen, seqLen])
            .asType(tensor.dtype)
    }

    /// Rotate half of the hidden dims of the input
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let lastDim = x.dim(-1)
        let x1 = x[.ellipsis, 0..<(lastDim / 2)]
        let x2 = x[.ellipsis, (lastDim / 2)...]
        return concatenated([-x2, x1], axis: -1)
    }

    /// Apply rotary position embedding to query and key tensors
    static func applyRotaryPosEmb(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray, unsequeezeDim: Int = 1) -> (MLXArray, MLXArray) {
        let cosExpanded = cos.expandedDimensions(axis: unsequeezeDim)
        let sinExpanded = sin.expandedDimensions(axis: unsequeezeDim)

        let qEmbed = (q * cosExpanded) + (rotateHalf(q) * sinExpanded)
        let kEmbed = (k * cosExpanded) + (rotateHalf(k) * sinExpanded)

        return (qEmbed, kEmbed)
    }

    fileprivate class Attention: Module {
        let embedDim: Int
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        public init(dims: Int, numHeads: Int, queryInputDims: Int? = nil, keyInputDims: Int? = nil, valueInputDims: Int? = nil, valueDims: Int? = nil, valueOutputDims: Int? = nil, bias: Bool = false) {

            precondition(dims % numHeads == 0, "dims should be divisible by num_heads")

            let queryInputDims = queryInputDims ?? dims
            let keyInputDims = keyInputDims ?? dims
            let valueInputDims = valueInputDims ?? keyInputDims
            let valueDims = valueDims ?? dims
            let valueOutputDims = valueOutputDims ?? dims

            self.embedDim = dims
            self.numHeads = numHeads
            self.headDim = embedDim / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qProj.wrappedValue = Linear(queryInputDims, embedDim, bias: bias)
            self._kProj.wrappedValue = Linear(keyInputDims, embedDim, bias: bias)
            self._vProj.wrappedValue = Linear(valueInputDims, valueDims, bias: bias)
            self._oProj.wrappedValue = Linear(valueDims, valueOutputDims, bias: bias)
        }

        public func callAsFunction(_ queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            var q = qProj(queries)
            var k = kProj(keys)
            var v = vProj(values)

            let B = q.dim(0)
            let L = q.dim(1)
            let S = k.dim(1)

            // Reshape for multi-head attention
            q = q.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            k = k.reshaped(B, S, numHeads, headDim).transposed(0, 2, 1, 3)
            v = v.reshaped(B, S, numHeads, headDim).transposed(0, 2, 1, 3)

            // Scaled dot product attention
            var attn = (q * scale).matmul(k.transposed(0, 1, 3, 2))

            if let mask = mask {
                attn = attn + mask
            }

            attn = softmax(attn, axis: -1)
            let output = attn.matmul(v)
                .transposed(0, 2, 1, 3)
                .reshaped(B, L, -1)

            return oProj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class VisionEncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

        public init(_ args: PixtralConfiguration.VisionConfiguration) {
            self._selfAttn.wrappedValue = Attention(
                dims: args.hiddenSize,
                numHeads: args.attentionHeads
            )
            self._mlp.wrappedValue = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayernorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            let r = selfAttn(inputLayernorm(x), keys: inputLayernorm(x), values: inputLayernorm(x), mask: mask)
            var h = x + r
            h = h + mlp(postAttentionLayernorm(h))
            return h
        }
    }

    fileprivate class VisionTransformer: Module {
        @ModuleInfo(key: "patch_conv") var patchConv: Conv2d
        @ModuleInfo(key: "ln_pre") var lnPre: RMSNorm

        let layers: [VisionEncoderLayer]
        let positionEmbedding: RoPE
        let patchSize: Int

        public init(_ args: PixtralConfiguration.VisionConfiguration) {
            self.patchSize = args.patchSize

            self._patchConv.wrappedValue = Conv2d(
                inputChannels: args.numChannels,
                outputChannels: args.hiddenSize,
                kernelSize: IntOrPair(args.patchSize),
                stride: IntOrPair(args.patchSize),
                padding: IntOrPair(0),
                bias: false
            )

            self._lnPre.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

            self.layers = (0..<args.hiddenLayers).map { _ in
                VisionEncoderLayer(args)
            }

            let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
            self.positionEmbedding = RoPE(dimensions: headDim, traditional: false, base: args.ropeTheta)
        }

        public func callAsFunction(_ x: [MLXArray], outputHiddenStates: Bool = false) -> [MLXArray] {
            var hiddenStates: [MLXArray] = []
            var allHiddenStates: [MLXArray] = []

            // Process each image through conv
            for pixelValues in x {
                // Get output channels from weight shape (first dimension)
                let outputChannels = patchConv.weight.dim(0)
                var patchEmbeds = patchConv(pixelValues).reshaped(-1, outputChannels)
                patchEmbeds = lnPre(patchEmbeds)
                hiddenStates.append(patchEmbeds)
            }

            // Concatenate all patches
            var h = concatenated(hiddenStates, axis: 0).expandedDimensions(axis: 0)

            // Generate attention mask for separate images
            let patchSizes = hiddenStates.map { $0.dim(0) }
            let mask = Vision.generateBlockAttentionMask(patchEmbedsList: patchSizes, tensor: h)

            if outputHiddenStates {
                allHiddenStates.append(h)
            }

            // Forward through transformer layers
            for layer in layers {
                h = layer(h, mask: mask)
                if outputHiddenStates {
                    allHiddenStates.append(h)
                }
            }

            return outputHiddenStates ? allHiddenStates : [h]
        }
    }

    fileprivate class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: VisionTransformer

        public init(_ args: PixtralConfiguration.VisionConfiguration) {
            self._visionModel.wrappedValue = VisionTransformer(args)
        }

        public func callAsFunction(_ x: [MLXArray], outputHiddenStates: Bool = false) -> [MLXArray] {
            return visionModel(x, outputHiddenStates: outputHiddenStates)
        }
    }
}

// MARK: - Language Model

fileprivate enum Language {

    fileprivate class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rope: RoPE

        public init(_ args: PixtralConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            let headDim = args.headDim ?? (dim / heads)
            self.headDim = headDim
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            let ropeScale: Float = if let ropeScaling = args.ropeScaling,
                             let type = ropeScaling["type"]?.asString(),
                             type == "linear",
                             let factor = ropeScaling["factor"]?.asFloat() {
                Float(1.0) / factor
            } else {
                Float(1.0)
            }

            self._rope.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: args.ropeTraditional,
                base: args.ropeTheta,
                scale: ropeScale
            )
        }

        public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            // Prepare for attention
            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)

            let maskConverted: MLXFast.ScaledDotProductAttentionMaskMode =
                if let mask {
                    .array(mask[.ellipsis, 0..<keys.dim(-2)])
                } else {
                    .none
                }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: maskConverted
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

        public init(_ args: PixtralConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayernorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?) -> MLXArray {
            let r = attention(inputLayernorm(x), mask: mask, cache: cache)
            var h = x + r
            h = h + mlp(postAttentionLayernorm(h))
            return h
        }
    }

    fileprivate class MistralModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [TransformerBlock]
        fileprivate let norm: RMSNorm

        public init(_ args: PixtralConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize,
                dimensions: args.hiddenSize
            )

            self.layers = (0..<args.hiddenLayers).map { _ in
                TransformerBlock(args)
            }

            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask: MLXArray? = createAttentionMask(h: h, cache: cache)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: MistralModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        var kvHeads: [Int]

        public init(_ args: PixtralConfiguration.TextConfiguration) {
            self.model = MistralModel(args)
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
            self.kvHeads = (0..<args.hiddenLayers).map { _ in args.kvHeads }
        }

        public func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil) -> LMOutput {
            let out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            return LMOutput(logits: lmHead(out))
        }
    }
}

// MARK: - Main Pixtral Model

fileprivate class LlavaMultiModalProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    let gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(_ config: PixtralConfiguration) {
        // Follow Mistral's VisionLanguageAdapter pattern: configurable bias
        let bias = config.effectiveMultimodalProjectorBias
        self._linear1.wrappedValue = Linear(
            config.visionConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: bias
        )
        self.gelu = GELU()
        self._linear2.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linear1(x)
        h = gelu(h)
        h = linear2(h)
        return h
    }
}

public class Pixtral: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") fileprivate var visionTower: Vision.VisionModel
    @ModuleInfo(key: "language_model") fileprivate var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") fileprivate var multiModalProjector: LlavaMultiModalProjector

    public let config: PixtralConfiguration
    let visionFeatureLayer: Int
    let visionFeatureSelectStrategy: String

    public var vocabularySize: Int {
        config.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: PixtralConfiguration) {
        self.config = config
        self.visionFeatureLayer = config.visionFeatureLayer
        self.visionFeatureSelectStrategy = config.visionFeatureSelectStrategy

        self._visionTower.wrappedValue = Vision.VisionModel(config.visionConfig)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfig)
        self._multiModalProjector.wrappedValue = LlavaMultiModalProjector(config)
    }

    // Helper methods for subclasses to access embeddings
    internal func getTextEmbeddings(_ inputIds: MLXArray) -> MLXArray {
        return languageModel.model.embedTokens(inputIds)
    }

    internal func getVisionHiddenStates(_ pixelValues: [MLXArray]) -> [MLXArray] {
        return visionTower(pixelValues, outputHiddenStates: true)
    }

    internal func getInputEmbeddings(
        inputIds: MLXArray? = nil,
        pixelValues: [MLXArray]? = nil,
        imageSizes: MLXArray? = nil
    ) -> MLXArray {
        guard let pixelValues = pixelValues else {
            return languageModel.model.embedTokens(inputIds!)
        }

        // Get text embeddings
        let inputsEmbeds = languageModel.model.embedTokens(inputIds!)

        // Get vision embeddings
        let hiddenStates = visionTower(pixelValues, outputHiddenStates: true)
        let selectedImageFeature = hiddenStates[visionFeatureLayer]

        // Project vision features to text space
        let imageFeatures = multiModalProjector(selectedImageFeature)

        // Merge vision and text embeddings
        return Pixtral.mergeInputIdsWithImageFeatures(
            imageTokenIndex: config.imageTokenIndex,
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds!
        )
    }

    public static func mergeInputIdsWithImageFeatures(
        imageTokenIndex: Int,
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let numImages = imageFeatures.dim(0)
        let numImagePatches = imageFeatures.dim(1)
        let embedDim = imageFeatures.dim(2)

        // Find positions of image tokens (assuming batch size 1)
        let flatInputIds = inputIds.flattened()
        var imagePositions: [Int] = []
        for i in 0..<flatInputIds.size {
            if flatInputIds[i].item(Int.self) == imageTokenIndex {
                imagePositions.append(i)
            }
        }

        var textSegments: [MLXArray] = []
        var startIdx = 0

        for position in imagePositions {
            textSegments.append(inputsEmbeds[0..., startIdx..<position, 0...])
            startIdx = position + 1
        }

        // Split image features into separate embeddings for each image
        var imageEmbeddings: [MLXArray] = []
        for i in 0..<numImages {
            imageEmbeddings.append(imageFeatures[i, 0..., 0...])
        }

        // Interleave text and image embeddings
        var finalEmbeddings: [MLXArray] = []
        for (text, image) in zip(textSegments, imageEmbeddings) {
            finalEmbeddings.append(text)
            finalEmbeddings.append(image)
        }
        finalEmbeddings.append(inputsEmbeds[0..., startIdx..., 0...])

        return concatenated(finalEmbeddings, axis: 1)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        // Get vision model dtype for type consistency
        let dtype = visionTower.visionModel.patchConv.weight.dtype

        // Extract pixel values and image sizes if present
        var pixelValues: [MLXArray]? = nil
        var imageSizes: MLXArray? = nil

        if let imageData = input.image {
            pixelValues = [imageData.pixels.asType(dtype)]

            // Extract image sizes from frames (for Mistral3 spatial merging)
            // frames contains [THW(1, height, width), ...] for each image
            if let frames = imageData.frames {
                // Convert THW to [height, width] pairs
                let sizesPairs = frames.map { [$0.h, $0.w] }.flatMap { $0 }
                imageSizes = MLXArray(sizesPairs, [frames.count, 2])
            }
        }

        // Get input embeddings (handles both text-only and multimodal cases)
        let inputEmbeddings = getInputEmbeddings(
            inputIds: input.text.tokens[0],
            pixelValues: pixelValues,
            imageSizes: imageSizes
        )

        // Forward through language model
        let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbeddings.expandedDimensions(axis: 0))

        return .logits(result)
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        inputEmbedding: MLXArray?
    ) -> LMOutput {
        var out = languageModel(inputs, cache: cache, inputEmbedding: inputEmbedding)
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        // Transform vision tower keys to match Swift implementation structure
        for (key, value) in weights {
            guard key.contains("vision_tower") else { continue }

            var newKey = key

            // Add .vision_model wrapper if missing (for older checkpoints)
            if !key.contains("vision_model") {
                if key.contains("transformer") || key.contains("patch_conv") || key.contains("ln_pre") {
                    newKey = newKey.replacingOccurrences(of: "vision_tower", with: "vision_tower.vision_model")
                }
            }

            // Remove .transformer wrapper (Python has Encoder wrapper, Swift doesn't)
            newKey = newKey.replacingOccurrences(of: ".transformer.layers.", with: ".layers.")

            // Rename layer components to match Swift naming
            newKey = newKey.replacingOccurrences(of: ".attention.", with: ".self_attn.")
            newKey = newKey.replacingOccurrences(of: ".attention_norm.", with: ".input_layernorm.")
            newKey = newKey.replacingOccurrences(of: ".feed_forward.", with: ".mlp.")
            newKey = newKey.replacingOccurrences(of: ".ffn_norm.", with: ".post_attention_layernorm.")

            if newKey != key {
                sanitized[newKey] = value
                sanitized.removeValue(forKey: key)
            }
        }

        // Note: language_model.* keys should NOT be transformed
        // The @ModuleInfo(key: "language_model") decorator expects this prefix

        return sanitized
    }
}

// MARK: - Processor

public struct PixtralProcessorConfiguration: Codable, Sendable {
    // Processor configuration if needed
    // Pixtral doesn't require special preprocessing parameters
    // Images are passed directly to vision tower which handles patching
    public init() {}
}

/// Message generator for Pixtral models following mistral-common message protocol
///
/// Made public so it can be tested and reused (e.g., by Mistral3Processor)
public class PixtralMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .chat(let messages):
            // Convert structured Chat.Message to tokenizer format
            // This matches mistral-common's UserMessage/AssistantMessage protocol
            return messages.map { message in
                var dict: [String: Any] = ["role": message.role.rawValue]

                // Handle multimodal content (text + images)
                if !message.images.isEmpty {
                    // Multi-part content: text + image placeholders
                    // Matches mistral-common's ImageURLChunk/TextChunk pattern
                    var content: [[String: Any]] = []

                    // Add text chunk (always, even if empty - mistral-common behavior)
                    content.append(["type": "text", "text": message.content])

                    // Add image placeholders
                    // The tokenizer's chat template will insert [IMG] tokens
                    for _ in message.images {
                        content.append(["type": "image"])
                    }

                    dict["content"] = content
                } else {
                    // Text-only message
                    dict["content"] = message.content
                }

                return dict
            }

        case .text(let text):
            // Simple text prompt wrapped as user message
            return [["role": "user", "content": text]]

        case .messages(let messages):
            // Already in correct format
            return messages
        }
    }
}

/// Processor for Pixtral vision-language models
///
/// Implements the input preprocessing pipeline that aligns with mistral-common's approach:
/// 1. Message generation (equivalent to UserMessage/AssistantMessage protocol)
/// 2. Chat template application via tokenizer (CRITICAL mistral-common compatibility point)
/// 3. Image preprocessing (MLX-specific but semantically equivalent)
///
/// References:
/// - mistral-common: https://github.com/mistralai/mistral-common
/// - mlx-vlm Pixtral: https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/pixtral
public class PixtralProcessor: UserInputProcessor {
    private let config: PixtralProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let messageGenerator: PixtralMessageGenerator

    public init(
        _ config: PixtralProcessorConfiguration,
        tokenizer: any Tokenizer
    ) {
        self.config = config
        self.tokenizer = tokenizer
        self.messageGenerator = PixtralMessageGenerator()
    }

    /// Preprocess images for Pixtral vision tower
    ///
    /// Pixtral's vision encoder handles most preprocessing internally via Conv2d patch embedding
    /// We just need to provide images as MLXArrays in the correct format
    private func preprocessImages(_ images: [UserInput.Image], processing: UserInput.Processing?) throws -> MLXArray {
        var processedImages: [MLXArray] = []

        for imageInput in images {
            // Load as CIImage
            var image = try imageInput.asCIImage()

            // Apply user-requested processing (resize, etc.)
            image = MediaProcessing.apply(image, processing: processing)

            // Convert to sRGB tone curve space (standard for vision models)
            image = MediaProcessing.inSRGBToneCurveSpace(image)

            // Convert to MLXArray [1, C, H, W]
            // Pixtral vision tower expects this format
            let array = MediaProcessing.asMLXArray(image)
            processedImages.append(array)
        }

        // Concatenate along batch dimension
        return concatenated(processedImages, axis: 0)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // 1. Generate messages in mistral-common compatible format
        // This is the critical compatibility point with mistral-common's message protocol
        let messages = messageGenerator.generate(from: input)

        // 2. Apply chat template using tokenizer
        // THIS IS THE KEY STEP for mistral-common compatibility
        // The tokenizer's applyChatTemplate() uses the same Jinja templates that
        // mistral-common uses, ensuring identical tokenization and special token handling
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )

        // 3. Handle images if present
        if !input.images.isEmpty {
            let pixelValues = try preprocessImages(input.images, processing: input.processing)
            return LMInput(
                text: .init(tokens: MLXArray(promptTokens)),
                image: .init(pixels: pixelValues)
            )
        }

        // Text-only input
        return LMInput(tokens: MLXArray(promptTokens))
    }
}
