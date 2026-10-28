// lab23_sampler.cu — Tier6: llm.c xorshift64* RNG + temperature/top-k sampling on GPU.
// Source map: rough/llm.c llmc/rand.h (randU32/randF32 = xorshift64*, docs/audit
// llm-c-04 §rand bit-exact vs numpy RandomState legacy) and sampler.h top-k chain.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

__device__ __forceinline__ unsigned long long xs64(unsigned long long* s){
    unsigned long long x=*s; x^=x>>12; x^=x<<25; x^=x>>27; *s=x; return x*2685821657736338717ULL;
}
// one block per sample position; C vocab small enough for shared histogram-free selection
__global__ void sample_topk(const float* __restrict__ logits,int V,float temp,int k,
                            unsigned long long seed,unsigned int* out){
    __shared__ float sl[1024]; __shared__ unsigned long long st;
    int tid=threadIdx.x; if(tid==0) st=seed+blockIdx.x*7919ull;
    // load & scale by temperature
    for(int i=tid;i<V;i+=blockDim.x) sl[i]=logits[blockIdx.x*V+i]/temp;
    __syncthreads();
    // simple selection of top-k by repeated max (teaching version, k small)
    __shared__ int idxs[16]; __shared__ float vals[16];
    if(tid==0){
        for(int j=0;j<k;j++){ int bi=-1; float bv=-INFINITY;
            for(int i=0;i<V;i++) if(sl[i]>bv && i!=bi){ bool taken=false; for(int p=0;p<j;p++) if(idxs[p]==i) taken=true;
                if(!taken&&sl[i]>bv){bv=sl[i];bi=i;} }
            idxs[j]=bi; vals[j]=bv; }
        // softmax over top-k
        double m=vals[0],l=0,p[16]; for(int j=1;j<k;j++) if(vals[j]>m)m=vals[j];
        for(int j=0;j<k;j++){p[j]=exp((double)(vals[j]-m));l+=p[j];}
        double u=(xs64(&st)&0xFFFFFFFF)/(double)0x100000000ull*l; double c=0; int sel=k-1;
        for(int j=0;j<k;j++){c+=p[j]; if(u<=c){sel=j;break;}}
        out[blockIdx.x]=(unsigned)idxs[sel];
    }
}
int main(){
    const int V=256,S=64,k=5; size_t n=(size_t)S*V;
    float* lg=(float*)malloc(n*4); fill_rand(lg,n,110);
    float* glg; unsigned* gout; CK(cudaMalloc(&glg,n*4));CK(cudaMalloc(&gout,S*4));
    CK(cudaMemcpy(glg,lg,n*4,cudaMemcpyHostToDevice));
    sample_topk<<<S,32>>>(glg,V,0.7f,k,42ull,gout);
    CK(cudaDeviceSynchronize());
    unsigned* hs=(unsigned*)malloc(S*4); CK(cudaMemcpy(hs,gout,S*4,cudaMemcpyDeviceToHost));
    // 1) all outputs within vocab & in top-k set computed on CPU
    int bad=0;
    for(int s=0;s<S;s++){
        float v[16]; int id[16]; // CPU top-k
        for(int j=0;j<k;j++){float bv=-INFINITY;int bi=-1;
            for(int i=0;i<V;i++){bool t=false;for(int p=0;p<j;p++)if(id[p]==i)t=true;
                if(!t&&lg[s*V+i]>bv){bv=lg[s*V+i];bi=i;}} id[j]=bi;}
        bool ok=false; for(int j=0;j<k;j++) if((int)hs[s]==id[j]) ok=true;
        if(!ok) bad++;
    }
    check("top-k membership",bad==0,"all samples from CPU-computed top-k set");
    // 2) reproducibility: same seed -> same stream
    unsigned* g2; CK(cudaMalloc(&g2,S*4)); sample_topk<<<S,32>>>(glg,V,0.7f,k,42ull,g2);
    CK(cudaDeviceSynchronize()); unsigned* h2=(unsigned*)malloc(S*4); CK(cudaMemcpy(h2,g2,S*4,cudaMemcpyDeviceToHost));
    int diff=0; for(int s=0;s<S;s++) if(hs[s]!=h2[s]) diff++;
    check("same seed same stream",diff==0,"bitwise identical draws");
    // 3) different seed differs (probabilistic but near-certain with S=64,k=5)
    CK(cudaMemset(g2,0,S*4)); sample_topk<<<S,32>>>(glg,V,0.7f,k,1234ull,g2);
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h2,g2,S*4,cudaMemcpyDeviceToHost));
    int d2=0; for(int s=0;s<S;s++) if(hs[s]!=h2[s]) d2++;
    printf("different-seed divergence: %d/%d positions\n",d2,S);
    free(lg);free(hs);free(h2);cudaFree(glg);cudaFree(gout);cudaFree(g2);
    return report();
}
