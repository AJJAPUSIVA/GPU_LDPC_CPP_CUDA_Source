#include "nrldpc/cpu_decoder.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
int main(int argc,char**argv){
    if(argc<2){std::cerr<<"usage: ldpc_cpu_bench matrix.txt [batch=256] [iters=10]\n";return 2;}
    try{
        auto h=nrldpc::cpu::load_matrix(argv[1]);
        int batch=argc>2?std::atoi(argv[2]):256,iter=argc>3?std::atoi(argv[3]):10;
        if(batch<=0)throw std::invalid_argument("batch <= 0");
        size_t n=h.codeword_bits();
        std::vector<float> in(n*batch,10.f);
        std::vector<uint8_t> out(n*batch);
        bool avx=nrldpc::cpu::avx512_available();
        auto run=[&]{if(avx)nrldpc::cpu::decode_avx512(h,in.data(),out.data(),batch,iter);
                     else nrldpc::cpu::decode_scalar(h,in.data(),out.data(),batch,iter);};
        run();const int repeats=5;
        auto begin=std::chrono::steady_clock::now();
        for(int i=0;i<repeats;i++)run();
        double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        double gbit=double(repeats)*batch*h.information_bits()/1e9;
        std::cout<<"backend="<<(avx?"avx512":"scalar")<<",batch="<<batch<<",N="<<n
                 <<",iterations_max="<<iter<<",elapsed_s="<<sec
                 <<",information_Gbps="<<(gbit/sec)<<"\n";
        std::cout<<"NOTE: all-zero noiseless channel, early termination; not a BLER or 5G NR benchmark.\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
