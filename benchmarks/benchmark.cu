#include "nrldpc/ldpc.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
int main(int argc,char**argv){
    if(argc<2){std::cerr<<"usage: ldpc_bench matrix.txt [batch=32] [iterations=10]\n";return 2;}
    try{
        nrldpc::Codec codec(nrldpc::load_matrix(argv[1]));
        int batch=argc>2?std::atoi(argv[2]):32;
        int iterations=argc>3?std::atoi(argv[3]):10;
        if(batch<1)throw std::invalid_argument("batch must be positive");
        size_t K=size_t(codec.information_bits()),N=size_t(codec.codeword_bits());
        std::vector<uint8_t> input(batch*K,0);
        std::vector<float> llr(batch*N,8.f);
        uint8_t *di,*do_,*decoded;float* dl;
        nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&di),input.size()),"malloc input");
        nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&do_),batch*N),"malloc code");
        nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&decoded),batch*N),"malloc output");
        nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&dl),llr.size()*sizeof(float)),"malloc llr");
        cudaMemcpy(di,input.data(),input.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dl,llr.data(),llr.size()*sizeof(float),cudaMemcpyHostToDevice);
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
        auto time_ms=[&](bool encode){
            cudaEventRecord(a);
            if(encode)codec.encode(di,do_,batch);else codec.decode(dl,decoded,batch,iterations);
            cudaEventRecord(b);cudaEventSynchronize(b);
            float ms=0.f;cudaEventElapsedTime(&ms,a,b);return ms;
        };
        time_ms(false);
        float enc=time_ms(true),dec=time_ms(false);
        std::cout<<"GPU_CUDA,batch="<<batch<<",K="<<K<<",N="<<N
                 <<",iterations="<<iterations<<",encode_ms="<<enc
                 <<",decode_ms="<<dec<<"\n";
        std::cout<<"NOTE: times include CUDA memory allocation and kernel launches; encoder is generic slow GF(2) solver.\n";
        cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(di);cudaFree(do_);cudaFree(decoded);cudaFree(dl);
    }catch(const std::exception& ex){std::cerr<<ex.what()<<"\n";return 1;}
}
