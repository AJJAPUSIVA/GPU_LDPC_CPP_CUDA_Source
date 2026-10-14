#include "nrldpc/cpu_decoder.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
static uint32_t read32(std::istream& s){unsigned char b[4];if(!s.read((char*)b,4))throw std::runtime_error("truncated fixture header");return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);}
int main(int argc,char**argv){
 if(argc!=6){std::cerr<<"usage: ldpc_replay_cpu matrix fixture.llr output.bits scalar|avx512 early|fixed\n";return 2;}
 try{
  auto h=nrldpc::cpu::load_matrix(argv[1]);std::ifstream in(argv[2],std::ios::binary);
  char magic[8];if(!in.read(magic,8)||std::memcmp(magic,"NRLLR001",8))throw std::runtime_error("invalid fixture magic");
  auto n=read32(in),k=read32(in),batch=read32(in);
  if(n!=static_cast<uint32_t>(h.codeword_bits())||k!=static_cast<uint32_t>(h.information_bits())||batch<1||batch>100000)throw std::runtime_error("fixture geometry mismatch");
  size_t total=size_t(n)*batch;std::vector<float> llr(total);
  static_assert(sizeof(float)==4);if(!in.read((char*)llr.data(),total*sizeof(float)))throw std::runtime_error("truncated LLR data");
  if(in.peek()!=EOF)throw std::runtime_error("unexpected trailing fixture data");
  std::vector<uint8_t> bits(total);
  std::string backend=argv[4],mode=argv[5];if(mode!="early"&&mode!="fixed")throw std::invalid_argument("mode must be early|fixed");
  nrldpc::cpu::DecodeResult result;
  constexpr int iterations=10;constexpr float alpha=0.8f;
  if(backend=="scalar")result=nrldpc::cpu::decode_scalar_detailed(h,llr.data(),bits.data(),batch,iterations,alpha,mode=="early");
  else if(backend=="avx512"){
   if(!nrldpc::cpu::avx512_available())throw std::runtime_error("AVX-512 unavailable");
   result=nrldpc::cpu::decode_avx512_detailed(h,llr.data(),bits.data(),batch,iterations,alpha,mode=="early");
  }else throw std::invalid_argument("backend must be scalar|avx512");
  std::ofstream out(argv[3],std::ios::binary);if(!out.write((char*)bits.data(),bits.size()))throw std::runtime_error("cannot write output");
  size_t bad=0;for(unsigned b=0;b<batch;b++)bad+=!nrldpc::cpu::check(h,bits.data()+size_t(b)*n);
  std::cout<<"backend="<<backend<<",batch="<<batch<<",N="<<n<<",mode="<<mode<<",syndrome_failures="<<bad<<",output="<<argv[3]<<"\n";
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
