#include "nrldpc/ldpc.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <climits>
#include <cmath>
namespace nrldpc {
namespace {
__device__ __forceinline__ int bit_get(const uint32_t* a,int idx) {
    return (a[idx>>5]>>(idx&31))&1u;
}
__device__ __forceinline__ void bit_flip(uint32_t* a,int idx) {
    a[idx>>5]^=1u<<(idx&31);
}
// One CTA per codeword. A fully device-side, straightforward GF(2)
// Gaussian parity solver. Unlike optimized NR encoding it is intentionally
// heavyweight; use it as a correctness baseline, not a throughput claim.
__global__ void encode_gauss(const int* H,int rb,int cb,int kb,int Z,
                             const uint8_t* info,uint8_t* output,
                             uint32_t* aug,int words,int P) {
    const int job=blockIdx.x, tid=threadIdx.x;
    const int K=kb*Z,N=cb*Z;
    const uint8_t* u=info+size_t(job)*K;
    uint8_t* out=output+size_t(job)*N;
    uint32_t* mat=aug+size_t(job)*P*words;
    for(int i=tid;i<N;i+=blockDim.x) out[i]=i<K?u[i]:0;
    // Expanded parity system [Hp | syndrome(Hu)] in GF(2).
    for(int r=tid;r<P;r+=blockDim.x) {
        uint32_t* row=mat+size_t(r)*words;
        for(int w=0;w<words;++w) row[w]=0;
        int br=r/Z,t=r%Z, rhs=0;
        for(int c=0;c<cb;++c){
            int s=H[br*cb+c];if(s<0)continue;
            int v=c*Z+(t+s)%Z;
            if(c<kb) rhs^=u[v];
            else bit_flip(row,v-K);
        }
        if(rhs)bit_flip(row,P);
    }
    __syncthreads();
    __shared__ int pivotRow, singular;
    if(tid==0)singular=0;
    __syncthreads();
    for(int p=0;p<P;++p){
        if(tid==0){
            pivotRow=-1;
            for(int r=p;r<P;++r)if(bit_get(mat+size_t(r)*words,p)){pivotRow=r;break;}
            if(pivotRow<0)singular=1;
            else if(pivotRow!=p){
                uint32_t* a=mat+size_t(p)*words;
                uint32_t* b=mat+size_t(pivotRow)*words;
                for(int w=0;w<words;++w){uint32_t x=a[w];a[w]=b[w];b[w]=x;}
            }
        }
        __syncthreads();
        if(singular)break;
        const uint32_t* lead=mat+size_t(p)*words;
        for(int r=p+1+tid;r<P;r+=blockDim.x){
            uint32_t* dst=mat+size_t(r)*words;
            if(bit_get(dst,p))for(int w=p>>5;w<words;++w)dst[w]^=lead[w];
        }
        __syncthreads();
    }
    if(singular){for(int i=tid;i<N;i+=blockDim.x)out[i]=0xff;return;}
    // Back substitution sequential in thread zero; all on the GPU.
    if(tid==0){
        for(int p=P-1;p>=0;--p){
            uint32_t* a=mat+size_t(p)*words;
            int rhs=bit_get(a,P);
            for(int j=p+1;j<P;++j)rhs^=(bit_get(a,j)&out[K+j]);
            out[K+p]=uint8_t(rhs);
        }
    }
}
__global__ void check_kernel(const int* offsets,const int* cols,const int* shifts,
                             int rows,int Z,int N,const uint8_t* code,uint8_t* valid){
    const int job=blockIdx.x;
    const uint8_t* x=code+size_t(job)*N;
    __shared__ int bad;
    if(threadIdx.x==0)bad=0;
    __syncthreads();
    for(int row=threadIdx.x;row<rows*Z;row+=blockDim.x){
        int r=row/Z,t=row%Z,parity=0;
        for(int e=offsets[r];e<offsets[r+1];++e)
            parity^=(x[cols[e]*Z+(t+shifts[e])%Z]&1);
        if(parity)atomicOr(&bad,1);
    }
    __syncthreads();
    if(threadIdx.x==0)valid[job]=uint8_t(bad==0);
}
// Layered normalized min-sum. Each check-row within a QC layer owns
// disjoint variable nodes because circulant permutations are bijections.
// This enables safe in-place variable LLR updates without atomics.
__global__ void decode_layered(const int* offsets,const int* cols,const int* shifts,
                               int R,int Z,int N,int edges,
                               const float* input,uint8_t* result,
                               float* work,float* messages,int iterations,float alpha){
    int job=blockIdx.x,t=threadIdx.x;
    const float* llr=input+size_t(job)*N;
    uint8_t* out=result+size_t(job)*N;
    float* L=work+size_t(job)*N;
    float* msg=messages+size_t(job)*edges*Z;
    for(int i=t;i<N;i+=blockDim.x)L[i]=llr[i];
    for(int i=t;i<edges*Z;i+=blockDim.x)msg[i]=0.f;
    __syncthreads();
    __shared__ int bad;
    for(int it=0;it<iterations;++it){
        for(int r=0;r<R;++r){
            for(int k=t;k<Z;k+=blockDim.x){
                // Algorithm-5-inspired register-backed suffix minimum reuse.
                // Each lane owns one check node in the current QC row.
                // No CPU decoding, no atomics for variable updates.
                constexpr int MAX_DEGREE=64;
                float suffix[MAX_DEGREE];
                int begin=offsets[r], end=offsets[r+1];
                int sign=0;
                float minimum=1.e30f;
                // Backward pass: suffix[e] = min_{j >= e} |q_j|.
                for(int e=end-1;e>=begin;--e){
                    int idx=cols[e]*Z+(k+shifts[e])%Z;
                    float q=L[idx]-msg[e*Z+k];
                    minimum=fminf(minimum,fabsf(q));
                    suffix[e-begin]=minimum;
                    sign^=(q<0.f);
                }
                // Forward pass: output uses min of all other edges.
                // Circulant edges address distinct variables within each
                // check-node layer, so updated L values do not perturb q.
                float prefix=1.e30f;
                for(int e=begin;e<end;++e){
                    int idx=cols[e]*Z+(k+shifts[e])%Z;
                    float q=L[idx]-msg[e*Z+k];
                    float right=(e+1<end)?suffix[e+1-begin]:1.e30f;
                    float v=alpha*fminf(prefix,right);
                    if(sign^(q<0.f))v=-v;
                    msg[e*Z+k]=v;
                    L[idx]=q+v;
                    prefix=fminf(prefix,fabsf(q));
                }
            }
            __syncthreads();
        }
        if(t==0)bad=0;
        __syncthreads();
        for(int i=t;i<R*Z;i+=blockDim.x){
            int r=i/Z,k=i%Z,par=0;
            for(int e=offsets[r];e<offsets[r+1];++e){
                int idx=cols[e]*Z+(k+shifts[e])%Z;
                par^=(L[idx]<0.f);
            }
            if(par)atomicOr(&bad,1);
        }
        __syncthreads();
        if(!bad)break;
    }
    for(int i=t;i<N;i+=blockDim.x)out[i]=uint8_t(L[i]<0.f);
}
}
void Codec::encode(const uint8_t* in,uint8_t* out,int batch,cudaStream_t stream){
    if(batch<=0)throw std::invalid_argument("batch must be positive");
    int P=parity_bits(),words=(P+1+31)/32;
    uint32_t* matrix=nullptr;
    cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&matrix),size_t(batch)*P*words*sizeof(uint32_t),stream),"encode workspace");
    encode_gauss<<<batch,256,0,stream>>>(d_shifts_,h_.rows,h_.cols,h_.kb,h_.z,in,out,matrix,words,P);
    cuda_check(cudaGetLastError(),"encode kernel launch");
    cuda_check(cudaFreeAsync(matrix,stream),"free encoder workspace");
}
void Codec::decode(const float* in,uint8_t* out,int batch,int it,float alpha,cudaStream_t stream){
    if(batch<=0||it<1||it>100||alpha<=0.f||alpha>1.f)throw std::invalid_argument("invalid decoder arguments");
    float *L=nullptr,*msg=nullptr;
    cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&L),size_t(batch)*codeword_bits()*sizeof(float),stream),"decode LLR workspace");
    cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&msg),size_t(batch)*edges_*h_.z*sizeof(float),stream),"decode messages");
    decode_layered<<<batch,256,0,stream>>>(d_offsets_,d_cols_,d_edgeshifts_,h_.rows,h_.z,codeword_bits(),edges_,in,out,L,msg,it,alpha);
    cuda_check(cudaGetLastError(),"decode kernel launch");
    cuda_check(cudaFreeAsync(L,stream),"free decoder LLR");
    cuda_check(cudaFreeAsync(msg,stream),"free decoder messages");
}
void Codec::check(const uint8_t* bits,uint8_t* valid,int batch,cudaStream_t stream){
    if(batch<=0)throw std::invalid_argument("batch must be positive");
    check_kernel<<<batch,256,0,stream>>>(d_offsets_,d_cols_,d_edgeshifts_,h_.rows,h_.z,codeword_bits(),bits,valid);
    cuda_check(cudaGetLastError(),"syndrome kernel launch");
}
}
