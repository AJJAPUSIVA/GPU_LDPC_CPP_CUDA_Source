#include "nrldpc/ldpc.hpp"
#include <cuda_runtime.h>
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
int main(int argc,char** argv){
 if(argc<2){std::cerr<<"usage: ldpc_awgn_cuda_bench matrix.txt [batch=32] [max_iter=10] [ebn0_db=3] [early|fixed] [seed=12345]\n";return 2;}
 try{
  nrldpc::Codec codec(nrldpc::load_matrix(argv[1]));
  int batch=argc>2?std::stoi(argv[2]):32,it=argc>3?std::stoi(argv[3]):10;
  double ebn0=argc>4?std::stod(argv[4]):3.;
  std::string mode=argc>5?argv[5]:"early";
  unsigned seed=argc>6?static_cast<unsigned>(std::stoul(argv[6])):12345;
  if(batch<1||batch>100000||it<1||it>100||!std::isfinite(ebn0)||(mode!="early"&&mode!="fixed"))throw std::invalid_argument("invalid options");
  size_t n=codec.codeword_bits(),k=codec.information_bits(),total=n*size_t(batch);
  double sigma2=1./(2.*(double(k)/n)*std::pow(10.,ebn0/10.));
  std::mt19937 rng(seed);std::normal_distribution<float> gauss(0.f,1.f);
  std::vector<float> llr(total);
  for(auto& v:llr){double y=1.+std::sqrt(sigma2)*gauss(rng);v=float(2.*y/sigma2);}
  std::vector<uint8_t> bits(total);std::vector<int> iterations(batch);
  float* d_llr=nullptr;uint8_t* d_bits=nullptr;int* d_it=nullptr;
  nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_llr),total*sizeof(float)),"malloc LLR");
  nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bits),total),"malloc decoded");
  nrldpc::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_it),batch*sizeof(int)),"malloc iterations");
  nrldpc::cuda_check(cudaMemcpy(d_llr,llr.data(),total*sizeof(float),cudaMemcpyHostToDevice),"copy LLR");
  cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
  cudaEventRecord(begin);
  codec.decode(d_llr,d_bits,batch,it,0.8f,0,mode=="early",d_it);
  cudaEventRecord(end);nrldpc::cuda_check(cudaEventSynchronize(end),"synchronize decoder");
  float ms=0;cudaEventElapsedTime(&ms,begin,end);
  nrldpc::cuda_check(cudaMemcpy(bits.data(),d_bits,total,cudaMemcpyDeviceToHost),"copy bits");
  nrldpc::cuda_check(cudaMemcpy(iterations.data(),d_it,batch*sizeof(int),cudaMemcpyDeviceToHost),"copy iterations");
  uint64_t errors=0,bad=0;
  for(int b=0;b<batch;b++){
   bool has_error=false;
   for(size_t i=0;i<n;i++){uint8_t value=bits[size_t(b)*n+i];errors+=value!=0;has_error|=value!=0;}
   bad+=has_error;
  }
  double sum_it=std::accumulate(iterations.begin(),iterations.end(),0.);
  std::cout<<std::setprecision(9)<<"backend=cuda,matrix="<<argv[1]<<",batch="<<batch
   <<",N="<<n<<",K_nominal="<<k<<",ebn0_db="<<ebn0<<",max_iterations="<<it
   <<",mode="<<mode<<",seed="<<seed<<",elapsed_s="<<ms/1000.
   <<",information_Gbps="<<double(batch)*k/(ms*1.e6)
   <<",BER_allN="<<double(errors)/total<<",BLER_allN="<<double(bad)/batch
   <<",bit_errors="<<errors<<",block_errors="<<bad<<",avg_iterations="<<sum_it/batch
   <<",total_iterations="<<sum_it<<"\n";
  std::cout<<"NOTE: full-mother-code, no 5G puncturing/rate matching/CRC. CUDA time includes allocations and launches but excludes host noise generation/transfers.\n";
  cudaEventDestroy(begin);cudaEventDestroy(end);cudaFree(d_llr);cudaFree(d_bits);cudaFree(d_it);
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
