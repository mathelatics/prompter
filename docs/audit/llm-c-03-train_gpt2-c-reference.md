# llm.c Audit 03 — `train_gpt2.c` CPU Reference Trainer (complete, line-by-line digest)

File: `rough/llm.c/train_gpt2.c`, 1182 lines. This is THE canonical readable reference of GPT-2 fwd/bwd in plain C. Conventions: B=batch, T=seq len, C=channels(768), V=vocab(50257), Vp=padded vocab(50304), L=layers(12), NH=heads(12). All tensors fp32 row-major contiguous. Includes: stdlib + llmc/utils.h, llmc/tokenizer.h, llmc/dataloader.h. Optional `<omp.h>` under `#ifdef OMP`.

## Layer kernels (each = naive loop nest; indices derived manually)

### encoder_forward(out, inp, wte, wpe, B,T,C)  [L35]
out[b,t,:] = wte[inp[b,t],:] + wpe[t,:]. Triple loop b,t,i(C). inp (B,T) int.

### encoder_backward(dwte, dwpe, dout, inp, B,T,C)  [L60]
Scatter-add: dwte[ix,:] += dout_bt; dwpe[t,:] += dout_bt. (Atomic-free because serial per (b,t); races possible across b for same token but loops are serial here.)

### layernorm_forward(out, mean, rstd, inp, weight, bias, B,T,C)  [L78]
Per row x∈R^C: m = Σx/C; v = Σ(x−m)²/C (biased variance, no Bessel correction); s = 1/sqrt(v+eps), **eps=1e-5f**; out_i = s*(x_i−m)*weight_i + bias_i; caches mean[b*T+t]=m, rstd=s.

### layernorm_backward(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B,T,C)  [L20? L120]
Two-pass classic form. First reduce: dnorm_mean = Σ_i w_i*dout_i / C ; dnorm_norm_mean = Σ_i w_i*dout_i*norm_i / C where norm_i=(x_i−m)*s. Then per i: dbias_i += dout_i; dweight_i += norm_i*dout_i; dinp_i += ((w_i*dout_i) − dnorm_mean − norm_i*dnorm_norm_mean) * s. Accumulates into dinp (+=) because multiple paths merge (residual).

### matmul_forward_naive(out, inp, weight, bias, B,T,C,OC)  [L163]
out[bt,o] = (bias?bias[o]:0) + Σ_i inp[bt,i]*weight[o,i]; weight laid (OC,C). `#pragma omp parallel for collapse(2)` over b,t. Fallback used when B*T % 8 != 0.

### matmul_forward(...)  [L184]
LOOP_UNROLL=8 tiling: outer strided loop over obt (blocks of 8 rows bt), inner over o (output channel): keeps float result[8] in registers; loads each weight scalar w once and reuses across the 8 rows → 8× fewer weight loads; relies on -Ofast to emit FMAs. Requires B*T divisible by 8 else falls back.

### matmul_backward(dinp, dweight, dbias, dout, inp, weight, B,T,C,OC)  [L231]
Split into two parallel phases (single round would parallelize badly):
1. dinp[bt,i] += Σ_o dout[bt,o]*weight[o,i] — parallel over (b,t) collapse(2).
2. dweight[o,i] += Σ_bt inp[bt,i]*dout[bt,o]; dbias[o] += Σ_bt dout[bt,o] — parallel over o only (reduction implicit since each o owns its row). dbias may be NULL.

### attention_forward(out, preatt, att, inp, B,T,C,NH)  [L271]
inp is fused QKV (B,T,3C): q at offset h*hs, k at +C, v at +2C; hs=C/NH; scale=1/sqrt(hs). For each (b,t,h) [omp collapse(3)]:
- pass1: for t2≤t: preatt[t2]=(q·k_t2)*scale; track maxval (init −10000.f "TODO something better").
- pass2: att[t2]=exp(preatt[t2]−maxval); expsum.
- pass3: att[t2]*=1/expsum for t2≤t; att[t2]=0 explicitly for t2>t (causal mask, for debug parity with PyTorch). expsum==0 → inv=0 guard.
- pass4: out[b,t,h*hs:] = Σ_{t2≤t} att[t2] * v_t2 (zero-init first).
Materializes full (NH,T,T) scores (no flash).

### attention_backward(dinp, dpreatt, datt, dout, inp, att, B,T,C,NH)  [L347]
Reverse of passes, NOT omp-parallelized (comment: could be, but careful with atomics). Per (b,t,h):
- bwd pass4: for t2≤t: datt[t2] += Σ_i v_i*dout_i ; dv_t2[i] += att[t2]*dout_i.
- bwd pass2&3 (softmax): for t3≤t: dpreatt[t3] = Σ_{t2≤t} att[t2]*( (t2==t3) − att[t3] ) * datt[t2]. (local_derivative formula)
- bwd pass1 (QK matmul): for t2≤t: dq[i] += k_i*dpreatt[t2]*scale; dk_t2[i] += q_i*dpreatt[t2]*scale.

### gelu_forward/backward  [L393/L406]
Exact OpenAI tanh approx: g(x)=0.5x(1+tanh(s*(x+0.044715x³))), s=sqrt(2/π)=GELU_SCALING_FACTOR macro (computed at compile time via sqrtf of M_PI constant expression). Backward: local_grad = 0.5(1+tanh_out) + x*0.5*sech²*s*(1+3*0.044715x²), sech²=1/cosh²; dinp[i] += local_grad*dout[i]. Wrapped in `#pragma float_control(precise,on,push)` + GCC attr `optimize("no-finite-math-only")` because -Ofast breaks tanh/cosh (issue #168); popped after.

### residual_forward(out=in1+in2) / residual_backward(dinp1+=dout; dinp2+=dout) over N=B*T*C.

### softmax_forward(probs, logits, B,T,V,Vp)  [L459]
Per (b,t) [omp collapse(2)]: max over first V only; exp & sum over V; divide over V; then force probs[V..Vp)=0 ("extra super safety"). padded dims never participate.

### crossentropy_forward(losses, probs, targets, B,T,Vp): loss_bt = −log(probs[target]).

### crossentropy_softmax_backward(dlogits, dlosses, probs, targets, B,T,V,Vp): dlogits[i] += (p_i − 1{i==ix}) * dloss, only i<V (padded grads stay 0).

## Model plumbing
### GPT2Config {max_seq_len, vocab_size, padded_vocab_size, num_layers, num_heads, channels}.
### ParameterTensors: 16 pointers exactly matching checkpoint order (wte,wpe,ln1w,ln1b,qkvw,qkvb,attprojw,attprojb,ln2w,ln2b,fcw,fcb,fcprojw,fcprojb,lnfw,lnfb). Shapes as in doc 02.
fill_in_parameter_sizes: param_sizes[i] per that list (Vp used for wte).
malloc_and_point_parameters: ONE malloc of total floats; pointer array of `float** ptrs[]` walks offsets → all params contiguous (enables flat AdamW loop & single file read). Returns base pointer.
### ActivationTensors: 23 entries (encoded, ln1, ln1_mean, ln1_rstd, qkv, atty, preatt, att, attproj, residual2, ln2, ln2_mean, ln2_rstd, fch, fch_gelu, fcproj, residual3, lnf, lnf_mean, lnf_rstd, logits, probs, losses). Sizes: per-layer ones prefixed by L or B*T*C etc.; logits/probs use Vp. Single big malloc likewise.
### GPT2 struct: config + params/param_sizes/params_memory/num_parameters + grads (same tensor struct) + grads_memory + m_memory,v_memory (Adam states) + acts + grads_acts + batch_size/seq_len cached + inputs/targets int buffers + mean_loss (−1 = none).

### gpt2_build_from_checkpoint(model, path)  [L706]
Reads 256-int header; magic must be 20240326, version must be 3 (fp32; hint to rerun python train_gpt2.py). Sets config from header[2..7], prints them. Computes sizes, counts num_parameters, malloc_and_point_parameters, then fread whole params blob directly into the contiguous buffer (file order == memory order). NULLs everything else; mean_loss=-1.

### gpt2_forward(model, inputs, targets, B, T)  [L761]
Asserts model built; validates token range [0,V) for inputs and targets (assert); lazy allocates activations on first call (records B,T; requires identical B,T later else exit); memcpy caches inputs/targets. Forward sequence (per layer l, pointers sliced by l*B*T*C...):
```
encoded = encoder_forward(inputs,wte,wpe)
for l: residual = (l==0)? encoded : residual3[l-1]
  ln1        = layernorm_forward(residual, ln1w[l], ln1b[l])
  qkv        = matmul_forward(ln1, qkvw[l], qkvb[l], C→3C)
  atty       = attention_forward(qkv)      (also fills preatt,att)
  attproj    = matmul_forward(atty, attprojw[l], attprojb[l])
  residual2  = residual_forward(residual, attproj)
  ln2        = layernorm_forward(residual2, ln2w,ln2b)
  fch        = matmul_forward(ln2, fcw,fcb, C→4C)
  fch_gelu   = gelu_forward(fch)
  fcproj     = matmul_forward(fch_gelu, fcprojw,fcprojb, 4C→C)
  residual3  = residual_forward(residual2, fcproj)
lnf = layernorm_forward(residual3[L-1], lnfw,lnfb)
logits = matmul_forward(lnf, wte /*weight tied!*/, bias=NULL, C→Vp)
probs = softmax_forward(logits)
if targets: crossentropy_forward(losses); mean_loss = avg(losses) else mean_loss=-1
```
Note lm_head reuses wte as the (Vp,C) weight matrix (weight tying baked into matmul call).

### gpt2_zero_grad: memset grads_memory and grads_acts_memory to 0 (only if allocated).

### gpt2_backward(model)  [L~880]
Requires mean_loss != −1 (forwarded with targets). Lazy-allocates grads_memory + grads_acts_memory (+ zero). Seeds chain: grads_acts.losses[:] = 1/(B*T) (mean reduction backward inline). Reverse order:
```
crossentropy_softmax_backward(logits...)
matmul_backward(dlnf, dwte, dbias=NULL!, dlogits, lnf, wte, C→Vp)   # wte grad accumulates BOTH head and embedding contributions
layernorm_backward(dresidual3[L-1], dlnfw,dlnfb, ...)
for l = L-1..0:
  residual_backward(dl_residual2, dl_fcproj ← dl_residual3)
  matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, fch_gelu, fcprojw)
  gelu_backward(dl_fch, fch, dl_fch_gelu)
  matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, ln2, fcw)
  layernorm_backward(dl_residual2, dl_ln2w,dl_ln2b, dl_ln2, residual2,...)
  residual_backward(dresidual, dl_attproj ← dl_residual2)
  matmul_backward(dl_atty, dl_attprojw,b, dl_attproj, atty, attprojw)
  attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, qkv, att)
  matmul_backward(dl_ln1, dl_qkvw,b, dl_qkv, ln1, qkvw)
  layernorm_backward(dresidual, dl_ln1w,b, dl_ln1, residual,...)
encoder_backward(dwte, dwpe, dencoded, inputs)
```
Key subtlety: dlogits→lnf matmul backward writes into grads.wte; later encoder_backward ALSO adds into grads.wte (accumulation, since forward used same matrix twice). activation grads use += everywhere due to residual fan-out.

### gpt2_update(model, lr, beta1, beta2, eps, wd, t)  [L~1010]
Vanilla AdamW over FLAT arrays (num_parameters elements): m = β1*m+(1−β1)g; v = β2*v+(1−β2)g²; m̂=m/(1−β1^t); v̂=v/(1−β2^t); param -= lr*(m̂/(√v̂+eps) + wd*param). Note eps added to denominator OUTSIDE sqrt (matches torch AdamW: √v̂ + eps). m/v lazily calloc'd. t passed as step+1.

### gpt2_free: frees 8 allocations.

## Sampler (outside TESTING guard) [L~1045]
- random_u32: xorshift64* (state ^= state>>12; ^state<<25; ^state>>27; return (state*0x2545F4914F6CDD1D)>>32). random_f32 = (u32>>8)/2^24 ∈ [0,1).
- sample_mult(probs, n, coin): linear CDF scan; returns n−1 on rounding overflow. (llmc/sampler.h also has sample_softmax working on raw logits by scaling coin instead of dividing probs.)

## main() training demo [L~1075]
Hardcoded: checkpoint "gpt2_124M.bin"; data prefers dev/data/tinyshakespeare/*.bin if exists else dev/data/tinystories/TinyStories_{train,val}.bin; B=4, T=64; train loader should_shuffle=1, val=0; val_num_batches=5; tokenizer "gpt2_tokenizer.bin"; rng_state=1337; gen_tokens buffer; genT=64.
Loop step 0..40 inclusive (41 iters): every 10 steps → val loss over 5 batches (reset loader each time); step>0 && step%20==0 → generate: seed all B*T positions with eot token, autoregressively t=1..genT−1: full gpt2_forward over entire (B,T) each token (deliberately wasteful, comment acknowledges), take probs row b=0 position t−1, sample_mult over V only, write gen_tokens[t], print via tokenizer_decode/safe_printf (fallback print ids), fflush.
Train step timed with clock_gettime(CLOCK_MONOTONIC): next_batch → forward → zero_grad → backward → update(lr=1e-4, β=(0.9,0.999), eps=1e-8, wd=0.0, t=step+1) → print loss & ms. NOTE: cu trainer uses betas (0.9,0.95); this demo uses 0.999 — historical difference.
Frees at end. `#ifndef TESTING` wraps sampler+main so test_gpt2.c can link these functions.

## Testing hooks (test_gpt2.c, 195 lines)
`#define TESTING` before include; builds model from gpt2_124M.bin, loads gpt2_124M_debug_state.bin (x,y int32; ref_logits fp32 (B,T,V); ref_loss; ref grads 16 tensors), runs gpt2_forward/backward, compares: mean_loss vs ref_loss (~1e-5 tolerance prints PASS/FAIL per tensor), logits vs ref_logits, each parameter grad vs reference with relative error checks. Same skeleton mirrored in test_gpt2_fp32.cu / test_gpt2.cu.
