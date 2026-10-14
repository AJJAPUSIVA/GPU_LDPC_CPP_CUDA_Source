#include "nrldpc/cpu_decoder.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
// BPSK x=+1 for bit 0, noise N(0,sigma^2), LLR=2y/sigma^2.
// The all-zero codeword is a genuine codeword for every binary linear code.
// Errors are measured across all N transmitted bits; info-bit throughput uses K.
int main(int argc,char** argv) {
    if(argc<2){std::cerr<<"usage: ldpc_awgn_bench matrix.txt [batch=32] [max_iter=10] [ebn0_db=3] [early|fixed] [scalar|avx512|auto] [seed=12345]\n";return 2;}
    try {
        using namespace nrldpc::cpu;
        auto h=load_matrix(argv[1]);
        const int batch=argc>2?std::stoi(argv[2]):32;
        const int it=argc>3?std::stoi(argv[3]):10;
        const double ebn0=argc>4?std::stod(argv[4]):3.;
        const std::string mode=argc>5?argv[5]:"early";
        const std::string selected=argc>6?argv[6]:"auto";
        const unsigned seed=argc>7?static_cast<unsigned>(std::stoul(argv[7])):12345;
        if(batch<1||batch>200000||it<1||it>100||!std::isfinite(ebn0)||
           (mode!="early"&&mode!="fixed")||
           (selected!="auto"&&selected!="avx512"&&selected!="scalar"))
            throw std::invalid_argument("invalid arguments");
        bool avx=(selected=="avx512"||(selected=="auto"&&avx512_available()));
        if(avx&&!avx512_available())throw std::runtime_error("AVX-512 is unavailable");
        const int n=h.codeword_bits(),k=h.information_bits();
        // Nominal mother-code rate, NOT effective 5G NR punctured rate.
        const double rate=double(k)/n;
        const double sigma2=1./(2.*rate*std::pow(10.,ebn0/10.));
        std::mt19937 rng(seed);
        std::normal_distribution<float> gaussian(0.f,1.f);
        std::vector<float> llr(static_cast<size_t>(batch)*n);
        std::vector<uint8_t> bits(llr.size());
        for(auto& v:llr){double y=1.+std::sqrt(sigma2)*gaussian(rng);v=float(2.*y/sigma2);}
        auto start=std::chrono::steady_clock::now();
        DecodeResult stats=avx?decode_avx512_detailed(h,llr.data(),bits.data(),batch,it,0.8f,mode=="early")
                              :decode_scalar_detailed(h,llr.data(),bits.data(),batch,it,0.8f,mode=="early");
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        uint64_t errors=std::accumulate(bits.begin(),bits.end(),uint64_t{0});
        uint64_t bad_blocks=0,failed_syndromes=0;
        for(int b=0;b<batch;b++){
            bad_blocks+=std::any_of(bits.begin()+static_cast<size_t>(b)*n,bits.begin()+static_cast<size_t>(b+1)*n,
                                    [](uint8_t x){return x!=0;});
            failed_syndromes+=!stats.syndrome_ok[b];
        }
        double iterations=std::accumulate(stats.iterations_used.begin(),stats.iterations_used.end(),0.);
        std::cout<<std::setprecision(9)
                 <<"backend="<<(avx?"avx512":"scalar")<<",matrix="<<argv[1]<<",batch="<<batch
                 <<",N="<<n<<",K_nominal="<<k<<",ebn0_db="<<ebn0<<",max_iterations="<<it
                 <<",mode="<<mode<<",seed="<<seed<<",elapsed_s="<<seconds
                 <<",information_Gbps="<<double(batch)*k/seconds/1e9
                 <<",BER_allN="<<double(errors)/(double(batch)*n)
                 <<",BLER_allN="<<double(bad_blocks)/batch
                 <<",bit_errors="<<errors<<",block_errors="<<bad_blocks
                 <<",syndrome_failures="<<failed_syndromes
                 <<",avg_iterations="<<iterations/batch
                 <<",total_iterations="<<iterations<<"\n";
        std::cout<<"NOTE: mother-code AWGN experiment, all-zero transmitted word; no 5G puncturing/rate matching/filler or CRC. "
                 <<"BER/BLER are full-codeword metrics. Timing excludes LLR generation.\n";
    }catch(const std::exception& ex){std::cerr<<"ERROR: "<<ex.what()<<'\n';return 1;}
}
