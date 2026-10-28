# llm.c Audit 09 — cuDNN Flash Attention (`llmc/cudnn_att.h` + `cudnn_att.cpp`)

Optional fast path used by train_gpt2.cu when compiled with `-DENABLE_CUDNN` (Makefile flag `USE_CUDNN=1`).
Separate translation unit so unrelated edits don't force recompiling the whole .cu (header comment).
The fallback dense implementation lives in `llmc/attention.cuh` (doc 08 §11); the trainer picks between them at compile time.

## `cudnn_att.h` (20 lines)
Include guard; includes cuda_common.h; forward-declares:
- `create_cudnn()` / `destroy_cudnn()` — called from common_start/common_free under `#ifdef ENABLE_CUDNN`.
- `attention_forward_cudnn(floatX* out /*(B,T,NH,HS)*/, float* stats /*(B,NH,T) fp32, NULL for inference-only*/, floatX* inp /*(B,T,3,NH,HS) packed QKV*/, int B,int T,int NH,int C, cudaStream_t)`
- `attention_backward_cudnn(floatX* dqkvr, floatX* dout, floatX* qkvr, floatX* o, float* stats, B,T,NH,C, stream)`

## `cudnn_att.cpp` (296 lines) — cudnn_frontend ("fe") based
- `#define NOMINMAX`, includes unistd.h + `<cudnn_frontend.h>`.
- Precision: FP32 mode → `static_assert(false)` (**cuDNN path incompatible with FP32 builds**); FP16→fe HALF; default BFLOAT16 (`CUDNN_16BIT`).
- Statics: `cudnn_handle`, lazily-grown `cudnn_workspace` (comment: cuDNN may request up to 256 MiB; realloc only when a graph needs more than current).
- Error helpers: `cuDNNCheck(status)` and `checkCudnnFE(error_object)` both print file:line + message and exit.

### Graph caching strategy
`build_operation_graph` is "VERY SLOW", so graphs are cached:
- fwd cache key `(B,H,T,HS,is_inference_only)`; bwd key `(B,NH,T,HS)`; maps of shared_ptr<fe::graph::Graph>.

### Forward graph (`lookup_cache_or_build_graph_fwd`)
- Graph dtypes: io=CUDNN_16BIT, intermediate & compute = FLOAT.
- Q,K,V tensors dim (B,H,T,HS) but strides `{3*H*HS*T, HS, 3*H*HS, 1}` — i.e. **read directly from the interleaved packed QKV buffer (B,T,3,NH,HS)**; no permute kernel needed (unlike custom path). Pointers differ per tensor: Q=inp, K=inp+C, V=inp+2C (element offsets NH*HS... note: uses `+C` since C=NH*HS and layout has 3 blocks interleaved per token with stride 3*H*HS — see variant_pack below; strides encode the interleave).
- attn_scale: scalar fp32 tensor, `is_pass_by_value(true)` (host value pointer in variant pack).
- SDPA options: name "flash_attention", `set_is_inference(is_inference_only)`, attn_scale, **causal_mask=true**.
- Outputs: O dim (B,H,T,HS) strides {H*HS*T, HS, H*HS, 1} (i.e., contiguous head-major, matches acts.atty layout); stats (fp32, (B,H,T,1)) only when training (assert stats==nullptr || !inference_only).
- Build sequence: validate → build_operation_graph → create_execution_plans({HeurMode_t::A}) → check_support → build_plans → workspace grow/realloc → cache.

### Backward graph (`lookup_cache_or_build_graph_bwd`)
Same IO/compute dtypes. Inputs Q,K,V (packed strides from qkvr buffer), O and dO (contiguous (B,NH,T,HS)), stats fp32, pass-by-value attn_scale.
- sdpa_backward_options: causal_mask true; on cudnn_frontend ≥1.5: `set_deterministic_algorithm(true)` (needed for run-to-run determinism).
- Outputs dQ,dK,dV written back into a **packed dqkvr buffer** with same interleaved strides as inputs (so grads flow straight into the qkvw matmul backward without extra transpose).
- Same build/cache/workspace logic.

### Entry points
`attention_forward_cudnn`: computes HS=C/NH; `is_inference_only = (stats==nullptr)`; cudnnSetStream(handle, stream); fetch cached graph; variant_pack uid→ptr {Q=inp, K=inp+C, V=inp+2C, scale=&attn_scale_cpu(1/√HS), O=out (+Stats if training)}; graph->execute.
`attention_backward_cudnn`: pointers Q/K/V = qkvr + {0, NH*HS, 2*NH*HS}; O=o; dO=dout; stats; outputs dQKV = dqkvr + {0, NH*HS, 2*NH*HS}; setStream then execute.
`create_cudnn`: cudnnCreate. `destroy_cudnn`: free workspace + cudnnDestroy.

### Integration notes (from doc 07)
- Trainer allocates `acts.att` as **float stats (L,B,NH,T)** instead of floatX (L,B,NH,T,T) when ENABLE_CUDNN → massive memory saving (no T² probs).
- `acts.qkvr` (L,B,T,3C) holds the fused QKV projection output consumed here directly.
- Inference-only forward passes must still produce identical results; bit-exactness vs custom kernels documented in sampling section (T≥256 stable on cuDNN 9.2.1 / FE 1.5.2).
- The duplicated utilities comment: cuDNNCheck etc. intentionally duplicate main-file macros because this TU can't include train_gpt2.cu headers fully.
