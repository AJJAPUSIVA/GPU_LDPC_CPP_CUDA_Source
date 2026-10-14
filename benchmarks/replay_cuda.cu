#include "nrldpc/ldpc.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
static uint32_t read32(std::istream& s){unsigned char b[4];if(!s.read((char*)b,4))throw std::runtime_error("truncated fixture header");return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);}
int main(int argc,char**argv){
 if(argc!=5){std::cerr<<"usage: ldpc_replay_cuda matrix fixture.llr output.bits early|fixed\n";return 2;}
 float *d_llr=nullptr;uint8_t *d_bits=nullptr;
 try{
  nrldpc::Codec codec(nrldpc::load_matrix(argv[1]));std::ifstream in(argv[2],std::ios::binary);
  char magic[8];if(!in.read(magic,8)||std::memcmp(magic,"NRLLR001",8))throw std::runtime_error("invalid fixture magic");
  auto n=read32(in),k=read32(in),batch=read32(in);
  if(n!=static_cast<uint32_t>(codec.codeword_bits())||k!=static_cast<uint32_t>(codec.information_bits())||batch<1||batch>100000)throw std::runtime_error("fixture geometry mismatch");
  size_t total=size_t(n)*batch;std::vector<float> llr(total);
  if(!in.read((char*)llr.data(),total*sizeof(float)))throw std::runtime_error("truncated LLR data");
  if(in.peek()!=EOF)throw std::runtime_error("unexpected trailing fixture data");
  std::string mode=argv[4];if(mode!="early"&&mode!="fixed")throw std::invalid_argument("mode must be early|fixed");
  nrldpc::cuda_check(cudaMalloc((void**)&d_llr,total*sizeof(float)),"malloc LLR");
  nrldpc::cuda_check(cudaMalloc((void**)&d_bits,total),"malloc decoded bits");
  nrldpc::cuda_check(cudaMemcpy(d_llr,llr.data(),total*sizeof(float),cudaMemcpyHostToDevice),"copy LLR");
  codec.decode(d_llr,d_bits,batch,10,0.8f,0,mode=="early",nullptr);
  std::vector<uint8_t> bits(total);
  nrldpc::cuda_check(cudaMemcpy(bits.data(),d_bits,total,cudaMemcpyDeviceToHost),"copy decoded bits");
  std::ofstream out(argv[3],std::ios::binary);if(!out.write((char*)bits.data(),bits.size()))throw std::runtime_error("cannot write output");
  std::vector<uint8_t> ok(batch);uint8_t* d_ok=nullptr;
  nrldpc::cuda_check(cudaMalloc((void**)&d_ok,batch),"malloc check");
  codec.check(d_bits,d_ok,batch);nrldpc::cuda_check(cudaMemcpy(ok.data(),d_ok,batch,cudaMemcpyDeviceToHost),"copy check");cudaFree(d_ok);
  size_t bad=0;for(auto v:ok)bad+=!v;
  std::cout<<"backend=cuda,batch="<<batch<<",N="<<n<<",mode="<<mode<<",syndrome_failures="<<bad<<",output="<<argv[3]<<"\n";
  cudaFree(d_llr);cudaFree(d_bits);
 }catch(const std::exception&e){if(d_llr)cudaFree(d_llr);if(d_bits)cudaFree(d_bits);std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
