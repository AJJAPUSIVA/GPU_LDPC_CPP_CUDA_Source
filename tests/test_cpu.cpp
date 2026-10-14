#include "nrldpc/cpu_decoder.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc,char**argv){
    if(argc<2 || argc>3){std::cerr<<"usage: ldpc_cpu_test matrix.txt [--require-avx512]\n";return 2;}
    try{
        const bool required=argc==3 && std::string(argv[2])=="--require-avx512";
        if(argc==3 && !required)throw std::invalid_argument("unknown test option");
        if(required && !nrldpc::cpu::avx512_available())
            throw std::runtime_error("AVX-512 required but unavailable; refusing skipped test");
        auto h=nrldpc::cpu::load_matrix(argv[1]);
        constexpr int batch=19;
        const int N=h.codeword_bits();
        // The all-zero codeword satisfies every binary parity-check matrix.
        std::vector<float> llr(static_cast<size_t>(N)*batch,15.f);
        std::vector<uint8_t> scalar(static_cast<size_t>(N)*batch,255),vectorized(scalar.size(),255);
        nrldpc::cpu::decode_scalar(h,llr.data(),scalar.data(),batch,10);
        for(int b=0;b<batch;b++){
            for(int i=0;i<N;i++)if(scalar[b*N+i])throw std::runtime_error("scalar noiseless mismatch");
            if(!nrldpc::cpu::check(h,scalar.data()+b*N))throw std::runtime_error("scalar syndrome mismatch");
        }
        if(nrldpc::cpu::avx512_available()){
            nrldpc::cpu::decode_avx512(h,llr.data(),vectorized.data(),batch,10);
            if(vectorized!=scalar)throw std::runtime_error("AVX-512 mismatch");
            std::cout<<"PASS: scalar and AVX-512 19-codeword noiseless test\n";
        }else std::cout<<"PASS: scalar noiseless test; AVX-512 unavailable (not executed)\n";
        // Compare scalar and SIMD on a deterministic perturbed channel.
        for(int b=0;b<batch;b++)for(int i=0;i<N;i++){
            float v=((i*13+b*7)%17==0)?-0.7f:((i+b)%3==0?2.5f:7.f);
            llr[static_cast<size_t>(b)*N+i]=v;
        }
        nrldpc::cpu::decode_scalar(h,llr.data(),scalar.data(),batch,8);
        if(nrldpc::cpu::avx512_available()){
            nrldpc::cpu::decode_avx512(h,llr.data(),vectorized.data(),batch,8);
            if(vectorized!=scalar)throw std::runtime_error("AVX-512 noisy-case mismatch");
        }
        if(required)std::cout<<"PASS: required AVX-512 executed; noiseless and perturbed inputs exactly match scalar\n";
        // Invalid channel evidence must not silently pass as a valid codeword.
        std::vector<uint8_t> corrupt(N,0);corrupt[0]=1;
        if(nrldpc::cpu::check(h,corrupt.data()))throw std::runtime_error("syndrome missed corruption");
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
