// lab12 — Fused classifier: logits matmul + cross-entropy, in-place grad (llm.c kernel6 trick)
// Source map: rough/llm.c/train_gpt2_fp32.cu kernel6_time_maps_cross_entropy;
//             train_gpt2.c fused_classifier_forward/backward. docs/audit/llm-c-05 §kernel6.
// Math: loss_i = -(logits[i,t_i] - logsumexp_j logits[i,j]); d_logits = softmax - onehot.
// Trick: OVERWRITE the logits buffer with the gradient after computing loss row-wise
//        (each warp owns its row entirely -> safe to write in place).
#include "harness.h"

#define V 50257
#define C 768
#define BT 512   // batch*seq rows

// forward+grad fused: one warp per row i of logits (row length V)
__global__ void fused_ce_fwd_bwd(float* __restrict__ logits,  // in: x·Wᵀ ; out: grad (in place!)
                                 const int* __restrict__ targets,
                                 float* __restrict__ loss_out, int BT_, int V_) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= BT_) return;
    float* r = logits + (size_t)row * V_;
    int t = targets[row];

    // online max+sum over the vocab row (lab07 machinery, streaming)
    float m = -1e30f, l = 0.f;
    for (int base = 0; base < V_; base += 32) {
        int j = base + lane;
        float v = (j < V_) ? r[j] : -1e30f;
        float mc = v;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mc = fmaxf(mc, __shfl_xor_sync(0xffffffffu, mc, off));
        float mn = fmaxf(m, mc);
        float e = (j < V_) ? expf(v - mn) : 0.f;
        float sc = 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sc += __shfl_xor_sync(0xffffffffu, e, off);
        l = l * expf(m - mn) + sc;
        m = mn;
    }
    float lse = m + logf(l);
    if (lane == 0) loss_out[row] = -(r[t] - lse);

    // grad pass: p_j - onehot(t), written IN PLACE over logits row
    float inv_l = 1.f / l;
    for (int j = lane; j < V_; j += 32) {
        float p = expf(r[j] - m) * inv_l;
        r[j] = p - (j == t ? 1.f : 0.f);
    }
}

// simple logits producer: logits = X · Wᵗᵉᵀ (naive dot product, one thread per element)
__global__ void head_matmul(const float* __restrict__ x, const float* __restrict__ wte,
                            float* __restrict__ logits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // (row, vocab col)
    int row = idx / V, j = idx % V;
    if (row >= BT) return;
    const float* xr = x + (size_t)row * C;
    const float* wr = wte + (size_t)j * C;
    float acc = 0.f;
    for (int c = 0; c < C; c++) acc += xr[c] * wr[c];
    logits[idx] = acc;
}

int main() {
    print_gpu_info();
    size_t xsz = (size_t)BT*C, lsz = (size_t)BT*V;
    float *x=(float*)malloc(xsz*4), *wte=(float*)malloc((size_t)V*C*4);
    float *logits=(float*)malloc(lsz*4), *loss_h=(float*)malloc(BT*4);
    int *targets=(int*)malloc(BT*4);
    fill_rand(x,(size_t)BT*C,7); fill_rand(wte,(size_t)V*C,8);
    xs64_state=99; for(int i=0;i<BT;i++) targets[i]=(int)(xs64_next()%V);

    float *d_x,*d_w,*d_lg,*d_loss; int *d_t;
    CK(cudaMalloc(&d_x,xsz*4));CK(cudaMalloc(&d_w,(size_t)V*C*4));CK(cudaMalloc(&d_lg,lsz*4));
    CK(cudaMalloc(&d_t,BT*4));CK(cudaMalloc(&d_loss,BT*4));
    CK(cudaMemcpy(d_x,x,xsz*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w,wte,(size_t)V*C*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_t,targets,BT*4,cudaMemcpyHostToDevice));

    dim3 blk(256);
    unsigned hm_grd=(unsigned)((BT*(size_t)V+255)/256);
    // CPU oracle: recompute logits & loss & grad in double
    double *ref_loss=(double*)malloc(BT*8), *ref_grad=(double*)malloc(lsz*8);
    CK(cudaMemcpy(logits,d_lg,0,cudaMemcpyDeviceToHost)); // no-op keeps tool happy on unused var
    // compute reference logits on CPU (small enough: BT*V*C too big? 512*50257*768 ~ 2e10 flops -> use sampling check instead)
    // Strategy: verify CE math on GPU-produced logits: copy logits back BEFORE overwrite is impossible
    // (fused kernel overwrites). So run head first, snapshot logits to host, then run fused kernel on a FRESH copy.
    head_matmul<<<hm_grd,blk>>>(d_x,d_w,d_lg);
    CK(cudaMemcpy(logits,d_lg,lsz*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<BT;i++){
        double m=-1e300,s=0;
        for(int j=0;j<V;j++){double v=logits[(size_t)i*V+j];if(v>m)m=v;}
        for(int j=0;j<V;j++)s+=exp(logits[(size_t)i*V+j]-m);
        ref_loss[i]=-(logits[(size_t)i*V+targets[i]]-(m+log(s)));
        for(int j=0;j<V;j++)ref_grad[(size_t)i*V+j]=exp(logits[(size_t)i*V+j]-m)/s-(j==targets[i]?1.0:0.0);
    }
    // fresh device copy of logits, then fused kernel
    CK(cudaMemcpy(d_lg,logits,lsz*4,cudaMemcpyHostToDevice));
    float ms=bench([&]{ fused_ce_fwd_bwd<<<(unsigned)((BT+7)/8),dim3(256)>>>(d_lg,d_t,d_loss,BT,V); });
    CK(cudaMemcpy(loss_h,d_loss,BT*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(logits,d_lg,lsz*4,cudaMemcpyDeviceToHost)); // now holds GRAD
    printf("fused CE fwd+bwd: %.3f ms\n",ms);
    double sl=0; for(int i=0;i<BT;i++)sl+=ref_loss[i]; sl/=BT;
    double gl=0; for(int i=0;i<BT;i++)gl+=loss_h[i]; gl/=BT;
    CHECK("mean loss matches oracle", fabs(gl-sl)<1e-3, "cpu=%.4f gpu=%.4f",sl,gl);
    check_close("in-place grad = softmax-onehot", ref_grad, logits, lsz, 1e-4f);
    return report();
}
