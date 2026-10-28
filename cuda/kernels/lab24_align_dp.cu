// lab24_align_dp.cu — cPrompter word-align Needleman-Wunsch as GPU DP.
// Problem: align ASR words (possibly mis-recognized) to script words with
// edit-distance scoring; the O(M*N) DP is the bottleneck for long pages.
// Standard NW recurrence has loop-carried deps, so we wavefront by anti-diagonal:
// D[i][j] depends on D[i-1][j-1], D[i-1][j], D[i][j-1] -> all in diagonal k-1,k-2.
// Source map: cprompter audio/align.c semantics incl <=3-char exact-match rule
// (docs/proj/cprompter-02-audio-ui.md).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "harness.h"

#define MAXW 64
__constant__ int c_cost[MAXW*MAXW]; // substitution cost matrix (0/1/2)

// one block computes a tile of the DP; here whole matrix fits shared memory
__global__ void nw_kernel(const int* sub,const int M,const int N,int* D){
    __shared__ int ds[MAXW+1][MAXW+1];
    int tid=threadIdx.x;
    for(int i=tid;i<=M;i+=blockDim.x) for(int j=0;j<=N;j+=blockDim.x){
        if(i==0&&j==0)ds[0][0]=0; else if(i==0)ds[0][j]=2*j; else if(j==0)ds[i][0]=2*i; else ds[i][j]=0; }
    __syncthreads();
    for(int k=1;k<=M+N;k++){                       // anti-diagonal
        for(int i=(1>k-N)?1:k-N; i<=(M<k-1?M:k-1);i++){   // parallel over i within diagonal
            int j=k-i;
            if(tid==0){ // teaching version serializes diagonal members (deps only across diags)
                int del=ds[i-1][j]+2, ins=ds[i][j-1]+2, rep=ds[i-1][j-1]+sub[(i-1)*N+(j-1)];
                int m=del<ins?del:ins; if(rep<m)m=rep; ds[i][j]=m;
            }
        }
        __syncthreads();
    }
    if(tid==0) D[0]=ds[M][N];
}
int main(){
    const int M=40,N=48; // script words vs ASR words
    int* sub=(int*)malloc(M*N*4);
    unsigned long long sd=7;
    for(int i=0;i<M*N;i++){ sd=sd*6364136223846793005ull+1; int r=(sd>>33)&0xF; sub[i]=(r<10)?0:(r<13)?1:2; }
    // CPU oracle NW
    int* dc=(int*)malloc((size_t)(M+1)*(N+1)*4);
    #define DC(i,j) dc[(i)*(N+1)+(j)]
    for(int i=0;i<=M;i++)for(int j=0;j<=N;j++)DC(i,j)=(i==0&&j==0)?0:(i==0?2*j:(j==0?2*i:0));
    for(int i=1;i<=M;i++)for(int j=1;j<=N;j++){
        int del=DC(i-1,j)+2,ins=DC(i,j-1)+2,rep=DC(i-1,j-1)+sub[(i-1)*N+(j-1)];
        int m=del<ins?del:ins; if(rep<m)m=rep; DC(i,j)=m; }
    int* gsub; int* gd; CK(cudaMalloc(&gsub,M*N*4));CK(cudaMalloc(&gd,4));
    CK(cudaMemcpy(gsub,sub,M*N*4,cudaMemcpyHostToDevice));
    cudaMemcpyToSymbol(c_cost,gsub,0); // unused symbol kept for doc parity; real path uses param
    nw_kernel<<<1,32>>>(gsub,M,N,gd);
    CK(cudaDeviceSynchronize());
    int gpuD; CK(cudaMemcpy(&gpuD,gd,4,cudaMemcpyDeviceToHost));
    char msg[128]; snprintf(msg,sizeof msg,"gpu=%d cpu=%d",gpuD,DC(M,N));
    check("NW distance matches CPU",gpuD==DC(M,N),msg);
    free(sub);free(dc);cudaFree(gsub);cudaFree(gd);
    return report();
}
