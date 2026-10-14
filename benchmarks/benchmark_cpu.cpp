#include "nrldpc/cpu_decoder.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc,char**argv){
    if(argc<2 || argc>5){
        std::cerr<<"usage: ldpc_cpu_bench matrix.txt [batch=256] [iters=10] [backend=auto|scalar|avx512]\n";
        return 2;
    }
    try{
        auto h=nrldpc::cpu::load_matrix(argv[1]);
        const int batch=argc>2?std::stoi(argv[2]):256;
        const int iter=argc>3?std::stoi(argv[3]):10;
        const std::string requested=argc>4?argv[4]:"auto";
        if(batch<=0 || iter<=0 || iter>100)throw std::invalid_argument("invalid batch or iterations");
        if(requested!="auto" && requested!="scalar" && requested!="avx512")
            throw std::invalid_argument("backend must be auto, scalar, or avx512");
        const bool available=nrldpc::cpu::avx512_available();
        if(requested=="avx512" && !available)
            throw std::runtime_error("AVX-512 requested but unavailable; refusing scalar fallback");
        const bool use_avx=requested=="avx512" || (requested=="auto" && available);
        const size_t n=h.codeword_bits();
        std::vector<float> in(n*static_cast<size_t>(batch),10.f);
        std::vector<uint8_t> out(in.size());
        auto run=[&]{
            if(use_avx)nrldpc::cpu::decode_avx512(h,in.data(),out.data(),batch,iter);
            else nrldpc::cpu::decode_scalar(h,in.data(),out.data(),batch,iter);
        };
        run();
        constexpr int repeats=5;
        auto begin=std::chrono::steady_clock::now();
        for(int i=0;i<repeats;i++)run();
        const double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        const double gbits=double(repeats)*batch*h.information_bits()/1e9;
        std::cout<<"backend="<<(use_avx?"avx512":"scalar")
                 <<",requested="<<requested<<",avx512_available="<<available
                 <<",batch="<<batch<<",N="<<n<<",iterations_max="<<iter
                 <<",elapsed_s="<<sec<<",information_Gbps="<<(gbits/sec)<<"\n";
        std::cout<<"NOTE: all-zero noiseless channel with early termination. Shared runners are not reliable for performance claims.\n";
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
