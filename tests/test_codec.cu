#include "nrldpc/ldpc.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
int main(int argc,char**argv){
    if(argc!=2){std::cerr<<"usage: test_codec toy_matrix.txt\n";return 2;}
    try{
        auto m=nrldpc::load_matrix(argv[1]);nrldpc::Codec codec(m);
        int K=codec.information_bits(),N=codec.codeword_bits(),batch=3;
        std::vector<uint8_t> info(batch*K),bits(batch*N),valid(batch),recovered(batch*N);
        for(size_t i=0;i<info.size();++i)info[i]=uint8_t((i*17+3)%5<2);
        uint8_t *d_in,*d_bits,*d_valid,*d_dec;float* d_llr;
        nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_in),info.size()),"malloc info");
        nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bits),bits.size()),"malloc bits");
        nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_valid),valid.size()),"malloc valid");
        nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_dec),recovered.size()),"malloc recovered");
        nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_llr),bits.size()*sizeof(float)),"malloc llr");
        nrldpc::cuda_check(cudaMemcpy(d_in,info.data(),info.size(),cudaMemcpyHostToDevice),"copy info");
        codec.encode(d_in,d_bits,batch);
        codec.check(d_bits,d_valid,batch);
        nrldpc::cuda_check(cudaMemcpy(bits.data(),d_bits,bits.size(),cudaMemcpyDeviceToHost),"read bits");
        nrldpc::cuda_check(cudaMemcpy(valid.data(),d_valid,valid.size(),cudaMemcpyDeviceToHost),"read syndromes");
        for(int b=0;b<batch;++b){
            if(valid[b]!=1){std::cerr<<"parity check failed\n";return 1;}
            for(int i=0;i<K;++i)if(bits[b*N+i]!=info[b*K+i]){std::cerr<<"systematic bits mismatch\n";return 1;}
        }
        std::vector<float> llr(bits.size());
        for(size_t i=0;i<llr.size();++i)llr[i]=bits[i]?-12.f:12.f;
        nrldpc::cuda_check(cudaMemcpy(d_llr,llr.data(),llr.size()*sizeof(float),cudaMemcpyHostToDevice),"copy LLR");
        codec.decode(d_llr,d_dec,batch,10);
        nrldpc::cuda_check(cudaMemcpy(recovered.data(),d_dec,recovered.size(),cudaMemcpyDeviceToHost),"read decode");
        if(bits!=recovered){std::cerr<<"noiseless decode mismatch\n";return 1;}
        cudaFree(d_in);cudaFree(d_bits);cudaFree(d_valid);cudaFree(d_dec);cudaFree(d_llr);
        std::cout<<"PASS: GPU-only encode, parity, noiseless decode ("<<batch<<" codewords)\n";
    }catch(const std::exception& ex){std::cerr<<"ERROR: "<<ex.what()<<"\n";return 1;}
}
