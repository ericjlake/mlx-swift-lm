// Mistral-Small-4 (119B) — MLA attention + MoE MLP
// Outer config: model_type="mistral3" (VLM wrapper)
// Inner text_config: model_type="mistral4"
//
// Architecture:
//   Attention  : Multi-head Latent Attention (MLA), same structural pattern as DeepSeek V3
//   MLP        : 128 routed experts (SwitchGLU, pre-stacked) + 1 shared expert
//   Gate       : 8-bit quantized Linear; dequantized in sanitize() to plain float
//   RoPE       : Standard interleaved (rope_interleave=true → traditional:false)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Decoded from the outer VLM config.json; actual model params live under text_config.
public struct Mistral4Configuration: Codable, Sendable {
    var hiddenSize: Int
    var moeIntermediateSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var nRoutedExperts: Int
    var nSharedExperts: Int
    var numExpertsPerTok: Int
    var firstKDenseReplace: Int
    var routedScalingFactor: Float
    var normTopkProb: Bool
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeParameters: [String: StringOrNumber]?
    var vocabSize: Int
    var tieWordEmbeddings: Bool
    var maxPositionEmbeddings: Int

    var qHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case vocabSize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    // Decodes from either text_config-wrapped (VLM) or flat top-level layout.
    enum OuterKeys: String, CodingKey { case textConfig = "text_config" }
    enum TLKeys: String, CodingKey { case tieWordEmbeddings = "tie_word_embeddings" }

    public init(from decoder: Decoder) throws {
        let outerContainer = try decoder.container(keyedBy: OuterKeys.self)
        let tlContainer = try decoder.container(keyedBy: TLKeys.self)
        let c: KeyedDecodingContainer<CodingKeys>
        if outerContainer.contains(.textConfig) {
            c = try outerContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
        } else {
            c = try decoder.container(keyedBy: CodingKeys.self)
        }

        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? moeIntermediateSize
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        nRoutedExperts = try c.decode(Int.self, forKey: .nRoutedExperts)
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        ropeParameters = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        tieWordEmbeddings =
            (try? c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings))
            ?? (try? tlContainer.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings))
            ?? false
    }
}

// MARK: - MLA Attention

class Mistral4Attention: Module {
    let numHeads: Int
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    let scale: Float

    let rope: RoPELayer

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    init(_ config: Mistral4Configuration) {
        let hiddenSize = config.hiddenSize
        numHeads = config.numAttentionHeads
        qkRopeHeadDim = config.qkRopeHeadDim
        kvLoraRank = config.kvLoraRank
        vHeadDim = config.vHeadDim
        qkNopeHeadDim = config.qkNopeHeadDim
        qHeadDim = config.qHeadDim
        scale = pow(Float(qHeadDim), -0.5)

        _qAProj.wrappedValue = Linear(hiddenSize, config.qLoraRank, bias: false)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank)
        _qBProj.wrappedValue = Linear(config.qLoraRank, numHeads * qHeadDim, bias: false)
        _kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize, kvLoraRank + qkRopeHeadDim, bias: false)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank)
        _kvBProj.wrappedValue = Linear(
            kvLoraRank, numHeads * (qkNopeHeadDim + vHeadDim), bias: false)
        _oProj.wrappedValue = Linear(numHeads * vHeadDim, hiddenSize, bias: false)

        // rope_interleave: true → traditional: false (standard interleaved, not YARN)
        rope = initializeRope(
            dims: qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeParameters,
            maxPositionEmbeddings: config.maxPositionEmbeddings)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Low-rank Q decomposition
        var q = qBProj(qALayerNorm(qAProj(x)))
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        let (qNope, qPe) = (splitQ[0], splitQ[1])

        // Latent KV compression (MLA)
        let raw = kvAProjWithMqa(x)
        let splitRaw = split(raw, indices: [kvLoraRank], axis: -1)
        let compressedKv = splitRaw[0]
        var kPe = splitRaw[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let splitKv = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let (kNope, values) = (splitKv[0], splitKv[1])

        let offset = cache?.offset ?? 0
        let rotQPe = rope(qPe, offset: offset)
        let rotKPe = rope(kPe, offset: offset)
        let kPeExpanded = repeated(rotKPe, count: numHeads, axis: 1)

        let keys = concatenated([kNope, kPeExpanded], axis: -1)
        let queries = concatenated([qNope, rotQPe], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - MoE MLP

class Mistral4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenDim: Int, intermediateDim: Int) {
        _gateProj.wrappedValue = Linear(hiddenDim, intermediateDim, bias: false)
        _upProj.wrappedValue = Linear(hiddenDim, intermediateDim, bias: false)
        _downProj.wrappedValue = Linear(intermediateDim, hiddenDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Expert router — kept as a raw Module (not Linear) so the 8-bit packed gate weight
/// bypasses the global 4-bit quantization system. The weight is dequantized to float
/// in Mistral4Model.sanitize() before being loaded into this module.
class Mistral4Gate: Module {
    var weight: MLXArray  // [nExperts, hiddenSize] — dequantized BF16

    init(_ config: Mistral4Configuration) {
        weight = zeros([config.nRoutedExperts, config.hiddenSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x.matmul(weight.T)  // → [B, L, nExperts]
    }
}

class Mistral4MoE: Module, UnaryLayer {
    // Gate weight stored as raw float (dequantized in sanitize); key path: mlp.gate.weight
    var gate: Mistral4Gate
    // Pre-stacked expert weights already in switch_mlp.{gate,up,down}_proj format
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    // Single shared expert: n_shared_experts × moeIntermediateSize
    @ModuleInfo(key: "shared_experts") var sharedExperts: Mistral4MLP

    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    init(_ config: Mistral4Configuration) {
        topK = config.numExpertsPerTok
        normTopkProb = config.normTopkProb
        routedScalingFactor = config.routedScalingFactor

        gate = Mistral4Gate(config)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: silu
        )
        _sharedExperts.wrappedValue = Mistral4MLP(
            hiddenDim: config.hiddenSize,
            intermediateDim: config.moeIntermediateSize * config.nSharedExperts
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Softmax gate: n_group=1 / topk_group=1 → no group masking needed
        let logits = gate(x)                                    // [B, L, nExperts]
        let scores = softmax(logits.asType(.float32), axis: -1)

        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var selected = takeAlong(scores, inds, axis: -1)

        if normTopkProb {
            selected = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20)
        }
        selected = (selected * routedScalingFactor).asType(x.dtype)

        // switchMLP → [B, L, topK, hiddenDim]; weighted sum → [B, L, hiddenDim]
        var y = switchMLP(x, inds)
        y = (y * selected[.ellipsis, .newAxis]).sum(axis: -2)
        return y + sharedExperts(x)
    }
}

// MARK: - Decoder Layer

class Mistral4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Mistral4Attention
    @ModuleInfo(key: "mlp") var mlp: Mistral4MoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: Mistral4Configuration) {
        _selfAttn.wrappedValue = Mistral4Attention(config)
        _mlp.wrappedValue = Mistral4MoE(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Note: partitionedLayerCall() in Mistral4ModelInner already performs
        // eval + Stream.gpu.synchronize() after this function returns when
        // ExpertStreamingConfig.shared.isEnabled == true (see LayerPartitioning.swift:130-139).
        // QuantizedSwitchLinear.callAsFunction() performs its own eval+sync around
        // each expert kernel (SwitchLayers.swift:270-271).
        // No additional prefault or GPU syncs are needed here.
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Inner Model

public class Mistral4ModelInner: Module, LayerPartitionable, StreamableMoE {
    let config: Mistral4Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Mistral4DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    // LayerPartitionable (SSD streaming support)
    public var startIdx: Int
    public var endIdx: Int
    public var numLayers: Int
    public var pipelineRank: Int = 0
    public var pipelineSize: Int = 1

    // StreamableMoE
    public var gpuLayerCount: Int? = nil
    public var streamExperts: Bool = false
    public var totalLayerCount: Int { layers.count }

    init(_ config: Mistral4Configuration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        layers = (0..<config.numHiddenLayers).map { _ in Mistral4DecoderLayer(config) }
        startIdx = 0
        endIdx = config.numHiddenLayers
        numLayers = config.numHiddenLayers
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = partitionedLayerCall(
                index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts
            ) {
                layer(h, mask: mask, cache: cache?[i])
            }
        }
        return norm(h)
    }
}

// MARK: - Top-Level Model

public class Mistral4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    /// One entry per layer: each layer uses full MHA (numAttentionHeads heads).
    /// Keys store qHeadDim per head; values store vHeadDim per head.
    /// Must match numHiddenLayers or the server will create a wrong-sized cache array,
    /// causing index-out-of-bounds on cache?[i] for any layer i.
    public var kvHeads: [Int]

    let args: Mistral4Configuration
    public let model: Mistral4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Mistral4Configuration) {
        self.args = args
        kvHeads = Array(repeating: args.numAttentionHeads, count: args.numHiddenLayers)
        model = Mistral4ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead { return lmHead(out) }
        return model.embedTokens.asLinear(out)
    }

    // MARK: Weight Sanitization
    //
    // 1. Strip "language_model." VLM wrapper prefix
    // 2. Drop vision_tower.*, multi_modal_projector.*, rotary_emb.inv_freq
    // 3. Handle tied embeddings
    // 4. fp8 weight_scale_inv dequantization (if present in FP8 conversions)
    // 5. Dequantize gate router weights from 8-bit to float
    //      gate.weight [nExperts, hidden/4] U32       →  [nExperts, hidden] BF16
    //      gate.scales [nExperts, hidden/groupSize]
    //      gate.biases [nExperts, hidden/groupSize]
    //    Stored in Mistral4Gate.weight (plain MLXArray) — bypasses 4-bit quantization.
    //
    // NOTE: switch_mlp weights are pre-stacked → no expert aggregation needed.
    // NOTE: kv_b_proj is used directly in Mistral4Attention; no reshape needed.

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let langPrefix = "language_model."
        var out: [String: MLXArray] = [:]

        // Step 1+2: strip prefix, skip vision keys
        for (key, value) in weights {
            guard key.hasPrefix(langPrefix) else { continue }
            let stripped = String(key.dropFirst(langPrefix.count))
            guard !stripped.contains("rotary_emb.inv_freq") else { continue }
            guard !stripped.hasPrefix("vision_tower") else { continue }
            guard !stripped.hasPrefix("multi_modal_projector") else { continue }
            out[stripped] = value
        }

        // Step 3: tied embeddings
        if args.tieWordEmbeddings {
            out.removeValue(forKey: "lm_head.weight")
        }

        // Step 4: fp8 weight_scale_inv dequantization
        var processed: [String: MLXArray] = [:]
        for (key, value) in out {
            if key.contains("weight_scale_inv") {
                let wk = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let w = out[wk], processed[wk] == nil {
                    processed[wk] = w * value
                }
            } else if processed[key] == nil {
                processed[key] = value
            }
        }
        if !processed.isEmpty { out = processed }

        // Step 5: dequantize 8-bit gate weights to float
        // gate.weight shape [nExperts, packed], dtype U32 — 4 int8 values per uint32
        // gate.scales shape [nExperts, nGroups], dtype BF16
        // group_size = realNIn / nGroups = (packed * 4) / nGroups
        for l in 0..<args.numHiddenLayers {
            let gp = "model.layers.\(l).mlp.gate"
            if let w = out["\(gp).weight"],
               let scales = out["\(gp).scales"],
               let biases = out["\(gp).biases"]
            {
                let packedWidth = w.dim(-1)                       // 1024
                let realNIn = packedWidth * 4                     // 4096 (8-bit: 4 vals/u32)
                let groupSize = realNIn / scales.dim(-1)          // 64
                let dq = dequantized(w, scales: scales, biases: biases,
                                     groupSize: groupSize, bits: 8)
                out["\(gp).weight"] = dq
                out.removeValue(forKey: "\(gp).scales")
                out.removeValue(forKey: "\(gp).biases")
            }
        }

        return out
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
