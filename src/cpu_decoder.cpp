#include "nrldpc/cpu_decoder.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
# include <immintrin.h>
# define NRLDPC_X86 1
#else
# define NRLDPC_X86 0
#endif
#if NRLDPC_X86 && (defined(__GNUC__) || defined(__clang__))
# define NRLDPC_AVX512_IMPL 1
# define NRLDPC_TARGET_AVX512 __attribute__((target("avx512f")))
#else
# define NRLDPC_AVX512_IMPL 0
#endif
namespace nrldpc::cpu {
void Matrix::validate() const {
    if(rows<=0 || cols<=kb || kb<=0 || z<=0 || z>384 || rows!=cols-kb ||
       shifts.size()!=static_cast<size_t>(rows)*cols)
        throw std::invalid_argument("invalid QC matrix geometry");
    for(auto s:shifts)if(s < -1 || s >= z) throw std::invalid_argument("invalid QC shift");
}
Matrix load_matrix(const std::string& file){
    std::ifstream in(file);
    if(!in)throw std::runtime_error("cannot open matrix: "+file);
    Matrix h; if(!(in>>h.rows>>h.cols>>h.kb>>h.z)) throw std::runtime_error("bad matrix header");
    if(h.rows<=0 || h.rows>64 || h.cols<=0 || h.cols>68)throw std::runtime_error("invalid matrix dimensions");
    h.shifts.resize(static_cast<size_t>(h.rows)*h.cols);
    for(int& v:h.shifts)if(!(in>>v))throw std::runtime_error("matrix truncated");
    h.validate(); return h;
}
struct Graph {
    std::vector<int> offsets, cols, shifts;
    explicit Graph(const Matrix& h): offsets(h.rows+1) {
        for(int r=0;r<h.rows;r++){
            offsets[r]=int(cols.size());
            for(int c=0;c<h.cols;c++){
                int s=h.shifts[r*h.cols+c];
                if(s>=0){cols.push_back(c);shifts.push_back(s);}
            }
            if(int(cols.size())-offsets[r]<2)throw std::invalid_argument("check row degree < 2");
        }
        offsets[h.rows]=int(cols.size());
    }
};
static void args(const Matrix& h,int batch,int iter,float alpha){
    h.validate();
    if(batch<1 || iter<1 || iter>100 || !(alpha>0 && alpha<=1))
        throw std::invalid_argument("invalid decoder settings");
}
bool check(const Matrix& h,const uint8_t* b){
    h.validate();
    for(int r=0;r<h.rows;r++)for(int k=0;k<h.z;k++){
        int s=0;
        for(int c=0;c<h.cols;c++){
            int sh=h.shifts[r*h.cols+c];
            if(sh>=0)s^=b[c*h.z+(k+sh)%h.z]&1;
        }
        if(s)return false;
    }
    return true;
}
void decode_scalar(const Matrix& h,const float* input,uint8_t* output,int batch,int it,float alpha){
    args(h,batch,it,alpha);Graph g(h);const int n=h.codeword_bits(),z=h.z;
    std::vector<float> L(n), msg(static_cast<size_t>(g.cols.size())*z);
    for(int b=0;b<batch;b++){
        std::copy(input+static_cast<size_t>(b)*n,input+static_cast<size_t>(b+1)*n,L.begin());
        std::fill(msg.begin(),msg.end(),0.f);
        for(int iteration=0;iteration<it;iteration++){
            for(int r=0;r<h.rows;r++)for(int k=0;k<z;k++){
                float min1=std::numeric_limits<float>::infinity(),min2=min1;
                int argmin=-1,sgn=0;
                for(int e=g.offsets[r];e<g.offsets[r+1];e++){
                    int idx=g.cols[e]*z+(k+g.shifts[e])%z;
                    float q=L[idx]-msg[e*z+k],a=std::abs(q);
                    if(a<min1){min2=min1;min1=a;argmin=e;}else if(a<min2)min2=a;
                    sgn^=(q<0);
                }
                for(int e=g.offsets[r];e<g.offsets[r+1];e++){
                    int idx=g.cols[e]*z+(k+g.shifts[e])%z;
                    float q=L[idx]-msg[e*z+k];
                    float v=alpha*(e==argmin?min2:min1);
                    if(sgn^(q<0))v=-v;
                    msg[e*z+k]=v;L[idx]=q+v;
                }
            }
            for(int i=0;i<n;i++)output[static_cast<size_t>(b)*n+i]=(L[i]<0);
            if(check(h,output+static_cast<size_t>(b)*n))break;
        }
    }
}
bool avx512_available(){
#if NRLDPC_AVX512_IMPL
    __builtin_cpu_init();return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}
#if NRLDPC_AVX512_IMPL
// 16 codewords processed in lockstep, vectorized across the batch dimension.
// Data is transposed into [VN][lane] so LLR updates are contiguous AVX-512 loads.
NRLDPC_TARGET_AVX512
static void decode_avx512_impl(const Matrix& h,const Graph& g,const float* input,uint8_t* output,int batch,int it,float alpha){
    constexpr int W=16;
    const int n=h.codeword_bits(),z=h.z,edges=int(g.cols.size());
    const __m512 zero=_mm512_setzero_ps(),scale=_mm512_set1_ps(alpha);
    const __m512 inf=_mm512_set1_ps(std::numeric_limits<float>::infinity());
    std::vector<float> L(static_cast<size_t>(n)*W),msg(static_cast<size_t>(edges)*z*W);
    std::vector<float> prefix(static_cast<size_t>(h.cols+1)*W);
    std::vector<float> suffix(static_cast<size_t>(h.cols+1)*W);
    for(int start=0;start<batch;start+=W){
        const int active=std::min(W,batch-start);
        for(int i=0;i<n;i++)for(int lane=0;lane<W;lane++)
            L[static_cast<size_t>(i)*W+lane]=lane<active?input[static_cast<size_t>(start+lane)*n+i]:12.f;
        std::fill(msg.begin(),msg.end(),0.f);
        for(int iteration=0;iteration<it;iteration++){
            for(int r=0;r<h.rows;r++)for(int k=0;k<z;k++){
                const int first=g.offsets[r],last=g.offsets[r+1],deg=last-first;
                __mmask16 parity=0;
                _mm512_storeu_ps(prefix.data(),inf);
                _mm512_storeu_ps(suffix.data()+static_cast<size_t>(deg)*W,inf);
                // Prefix/suffix minimum reuse: O(degree) comparisons per check.
                for(int j=0;j<deg;j++){
                    int e=first+j,idx=g.cols[e]*z+(k+g.shifts[e])%z;
                    __m512 q=_mm512_sub_ps(_mm512_loadu_ps(L.data()+static_cast<size_t>(idx)*W),
                                             _mm512_loadu_ps(msg.data()+(static_cast<size_t>(e)*z+k)*W));
                    parity ^= _mm512_cmp_ps_mask(q,zero,_CMP_LT_OQ);
                    __m512 prev=_mm512_loadu_ps(prefix.data()+static_cast<size_t>(j)*W);
                    _mm512_storeu_ps(prefix.data()+static_cast<size_t>(j+1)*W,
                                     _mm512_min_ps(prev,_mm512_max_ps(q,_mm512_sub_ps(zero,q))));
                }
                for(int j=deg-1;j>=0;j--){
                    int e=first+j,idx=g.cols[e]*z+(k+g.shifts[e])%z;
                    __m512 q=_mm512_sub_ps(_mm512_loadu_ps(L.data()+static_cast<size_t>(idx)*W),
                                             _mm512_loadu_ps(msg.data()+(static_cast<size_t>(e)*z+k)*W));
                    __m512 next=_mm512_loadu_ps(suffix.data()+static_cast<size_t>(j+1)*W);
                    _mm512_storeu_ps(suffix.data()+static_cast<size_t>(j)*W,
                                     _mm512_min_ps(next,_mm512_max_ps(q,_mm512_sub_ps(zero,q))));
                }
                for(int j=0;j<deg;j++){
                    int e=first+j,idx=g.cols[e]*z+(k+g.shifts[e])%z;
                    float* lp=L.data()+static_cast<size_t>(idx)*W;
                    float* mp=msg.data()+(static_cast<size_t>(e)*z+k)*W;
                    __m512 q=_mm512_sub_ps(_mm512_loadu_ps(lp),_mm512_loadu_ps(mp));
                    __m512 v=_mm512_mul_ps(scale,_mm512_min_ps(
                        _mm512_loadu_ps(prefix.data()+static_cast<size_t>(j)*W),
                        _mm512_loadu_ps(suffix.data()+static_cast<size_t>(j+1)*W)));
                    v=_mm512_mask_sub_ps(v,parity ^ _mm512_cmp_ps_mask(q,zero,_CMP_LT_OQ),zero,v);
                    _mm512_storeu_ps(mp,v);_mm512_storeu_ps(lp,_mm512_add_ps(q,v));
                }
            }
            bool allgood=true;
            for(int lane=0;lane<active;lane++){
                uint8_t* dest=output+static_cast<size_t>(start+lane)*n;
                for(int i=0;i<n;i++)dest[i]=L[static_cast<size_t>(i)*W+lane]<0;
                if(!check(h,dest))allgood=false;
            }
            if(allgood)break;
        }
    }
}
#endif
void decode_avx512(const Matrix& h,const float* llr,uint8_t* bits,int batch,int it,float alpha){
    args(h,batch,it,alpha);
#if NRLDPC_AVX512_IMPL
    if(!avx512_available())throw std::runtime_error("AVX-512F unavailable on this CPU");
    Graph g(h);decode_avx512_impl(h,g,llr,bits,batch,it,alpha);
#else
    throw std::runtime_error("AVX-512 backend requires x86 GCC/Clang");
#endif
}
}
