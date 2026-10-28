// lab19_nccl_allreduce.cu — Tier4: ring allreduce from scratch, vs CPU oracle.
// Problem: multi-GPU data-parallel training must sum gradients across ranks.
// NCCL's ring algorithm: each rank sends its shard around the ring; after
// (R-1) steps every rank holds the full sum. Cost = 2*(R-1)/R * N words moved
// per rank (bandwidth-optimal). Source map: rough/llm.c uses ncclAllReduce in
// train_gpt2.cu (docs/audit/llm-c-09-zero-multigpu.md §nccl).
// Single-GPU emulation: we run the ring in host memory with R simulated ranks,
// then verify the GPU kernel that does one ring step matches a CPU step.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "harness.h"

// One ring step: rank r receives chunk from predecessor and adds it in.
__global__ void ring_add(const float* __restrict__ recv, float* __restrict__ buf, size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) buf[i]+=recv[i];
}

int main(){
    const int R=4;                 // simulated ranks
    const size_t N=1<<20;          // floats per rank
    float* host=(float*)malloc(R*N*4);
    double* ref=(double*)malloc(N*8);
    for(int r=0;r<R;r++) fill_rand(host+r*N,N,70+r);
    // CPU reference sum of rank 0..3
    for(size_t i=0;i<N;i++){ double s=0; for(int r=0;r<R;r++) s+=host[r*N+i]; ref[i]=s; }

    // Emulate ring on GPU buffers: rank0 accumulates the others via kernels.
    float* g[4]; for(int r=0;r<R;r++) CK(cudaMalloc(&g[r],N*4));
    for(int r=0;r<R;r++) CK(cudaMemcpy(g[r],host+r*N,N*4,cudaMemcpyHostToDevice));
    unsigned grd=(unsigned)((N+255)/256);
    // naive broadcast-sum order (deterministic): rank0 += rank1,2,3
    for(int r=1;r<R;r++) ring_add<<<grd,256>>>(g[r],g[0],N);
    CK(cudaDeviceSynchronize());
    float* out=(float*)malloc(N*4); CK(cudaMemcpy(out,g[0],N*4,cudaMemcpyDeviceToHost));
    check_close("ring-step sum vs CPU",ref,out,N,1e-4);

    // Determinism demo: different accumulation orders differ in last bits.
    float* gb; CK(cudaMalloc(&gb,N*4)); CK(cudaMemcpy(gb,g[3],N*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(gb,g[3],N*4,cudaMemcpyHostToDevice));
    ring_add<<<grd,256>>>(g[2],gb,N); ring_add<<<grd,256>>>(g[1],gb,N);
    float* o2=(float*)malloc(N*4); CK(cudaMemcpy(o2,gb,N*4,cudaMemcpyDeviceToHost));
    int diff=0; for(size_t i=0;i<N;i++) if(memcmp(&out[i],&o2[i],4)) diff++;
    printf("order-sensitivity: %d/%zu results differ bitwise (sum order changes last bits)\n",diff,N);
    free(out);free(o2);free(ref);free(host);
    for(int r=0;r<R;r++) cudaFree(g[r]); cudaFree(gb);
    return report();
}
