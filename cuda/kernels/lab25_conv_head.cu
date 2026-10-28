// lab25_conv_head.cu — whisper.cpp-style conv1d + GELU residual projection.
// Problem: whisper encoder/decoder "convolutional embedding" layers project
// C->C with kernel size 3 (or 1) then GELU; used identically in llama MLP-free
// blocks. Teaches 1-D convolution tiling along the time axis.
// Source map: whisper.cpp conv1d kernels + llm.c gelu_forward_kernel (docs/audit
// llm-c-05 §gelu). Layout: X [T,C], W [Cout,Cin,3], bias [Cout], out [T,Cout].
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

#define CIN 64
#define COUT 64
__device__ float gelu_d(float x){ // tanh approximation, same as llm.c STCF path
    float x3=x*x*x; return 0.5f*x*(1.f+tanhf(0.7978845608f*(x+0.044715f*x3)));
}
__global__ void conv1d3(const float* __restrict__ X,const float* __restrict__ W,
                        const float* __restrict__ b,float* __restrict__ O,int T){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=T) return;
    int co=blockIdx.y;
    float acc=b[co];
    for(int ci=0;ci<CIN;ci++)
      for(int k=0;k<3;k++){
        int tt=t+k-1;                      // left-pad by 1 to keep length
        if(tt>=0&&tt<T) acc+=X[tt*CIN+ci]*W[(co*CIN+ci)*3+k];
      }
    O[t*COUT+co]=gelu_d(acc);
}
int main(){
    int T=128; size_t nx=(size_t)T*CIN,nw=(size_t)COUT*CIN*3;
    float *X=(float*)malloc(nx*4),*W=(float*)malloc(nw*4),*B=(float*)malloc(COUT*4),*O=(float*)malloc((size_t)T*COUT*4);
    fill_rand(X,nx,120);fill_rand(W,nw,121);fill_rand(B,COUT,122);
    float *gX,*gW,*gB,*gO; CK(cudaMalloc(&gX,nx*4));CK(cudaMalloc(&gW,nw*4));CK(cudaMalloc(&gB,COUT*4));CK(cudaMalloc(&gO,(size_t)T*COUT*4));
    CK(cudaMemcpy(gX,X,nx*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gW,W,nw*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gB,B,COUT*4,cudaMemcpyHostToDevice));
    dim3 grd((T+31)/32,COUT); conv1d3<<<grd,32>>>(gX,gW,gB,gO,T);
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(O,gO,(size_t)T*COUT*4,cudaMemcpyDeviceToHost));
    double* ref=(double*)malloc((size_t)T*COUT*8);
    for(int t=0;t<T;t++)for(int co=0;co<COUT;co++){
        double acc=B[co];
        for(int ci=0;ci<CIN;ci++)for(int k=0;k<3;k++){int tt=t+k-1;
            if(tt>=0&&tt<T) acc+=(double)X[tt*CIN+ci]*W[(co*CIN+ci)*3+k];}
        double g=0.5*acc*(1.0+tanh(0.7978845608*(acc+0.044715*acc*acc*acc)));
        ref[t*COUT+co]=g; }
    check_close("conv1d+gelu vs CPU",ref,O,(size_t)T*COUT,2e-3f);
    free(X);free(W);free(B);free(O);free(ref);
    cudaFree(gX);cudaFree(gW);cudaFree(gB);cudaFree(gO);
    return report();
}
