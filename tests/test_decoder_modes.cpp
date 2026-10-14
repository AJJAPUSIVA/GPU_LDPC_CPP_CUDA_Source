#include "nrldpc/cpu_decoder.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
int main(int argc,char** argv){
    if(argc!=2)return 2;
    using namespace nrldpc::cpu;
    auto h=load_matrix(argv[1]);const int batch=3,n=h.codeword_bits();
    std::vector<float> evidence(static_cast<size_t>(batch)*n,10.f);
    std::vector<uint8_t> out(evidence.size());
    auto early=decode_scalar_detailed(h,evidence.data(),out.data(),batch,6,0.8f,true);
    for(int i=0;i<batch;i++)if(early.iterations_used[i]!=1||!early.syndrome_ok[i])
        throw std::runtime_error("early stopping iteration accounting failed");
    auto fixed=decode_scalar_detailed(h,evidence.data(),out.data(),batch,6,0.8f,false);
    for(int i=0;i<batch;i++)if(fixed.iterations_used[i]!=6||!fixed.syndrome_ok[i])
        throw std::runtime_error("fixed iteration accounting failed");
    if(avx512_available()){
        auto avx=decode_avx512_detailed(h,evidence.data(),out.data(),batch,6,0.8f,false);
        for(int i=0;i<batch;i++)if(avx.iterations_used[i]!=6||!avx.syndrome_ok[i])
            throw std::runtime_error("SIMD fixed mode failed");
    }
    std::cout<<"PASS: early=1, fixed=6 iterations\n";
}
