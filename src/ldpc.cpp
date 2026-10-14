#include "nrldpc/ldpc.hpp"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <numeric>
namespace nrldpc {
void cuda_check(cudaError_t s,const char* where) {
    if (s!=cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(s));
}
void Matrix::validate() const {
    if(rows<=0||cols<=kb||kb<=0||z<=0||z>384||
       shifts.size()!=size_t(rows)*cols || rows!=cols-kb)
        throw std::invalid_argument("invalid QC parity-check matrix geometry");
    for(int s: shifts) if(s< -1 || s>=z) throw std::invalid_argument("shift outside [-1,Z)");
    // The GPU forward/backward kernel reserves 64 register slots per lane.
    // Reject unsupported row degrees rather than overflow local storage.
    for(int r=0;r<rows;++r){
        int degree=0;
        for(int c=0;c<cols;++c) degree+=(shifts[r*cols+c]>=0);
        if(degree<2 || degree>64)
            throw std::invalid_argument("QC row degree must be in [2,64]");
    }
}
Matrix load_matrix(const std::string& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("cannot open matrix: "+path);
    Matrix m;
    in>>m.rows>>m.cols>>m.kb>>m.z;
    for(int i=0;i<m.rows*m.cols;++i) {int s; if(!(in>>s)) throw std::runtime_error("short matrix file");m.shifts.push_back(s);}
    m.validate();return m;
}
Codec::Codec(Matrix matrix,int device):h_(std::move(matrix)) {
    h_.validate();
    int count=0;
    cuda_check(cudaGetDeviceCount(&count),"cudaGetDeviceCount");
    if(count<=device||device<0) throw std::runtime_error("NVIDIA CUDA GPU unavailable; refusing CPU fallback");
    cuda_check(cudaSetDevice(device),"cudaSetDevice");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_shifts_),h_.shifts.size()*sizeof(int)),"malloc shifts");
    cuda_check(cudaMemcpy(d_shifts_,h_.shifts.data(),h_.shifts.size()*sizeof(int),cudaMemcpyHostToDevice),"copy shifts");
    std::vector<int> offsets(h_.rows+1), cols, shifts;
    for(int r=0;r<h_.rows;++r) {
        offsets[r]=int(cols.size());
        for(int c=0;c<h_.cols;++c) if(h_.shifts[r*h_.cols+c]>=0) {
            cols.push_back(c);shifts.push_back(h_.shifts[r*h_.cols+c]);
        }
    }
    offsets[h_.rows]=int(cols.size());edges_=int(cols.size());
    auto alloc_copy=[](int*& dst,const std::vector<int>& a){
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dst),a.size()*sizeof(int)),"malloc graph");
        cuda_check(cudaMemcpy(dst,a.data(),a.size()*sizeof(int),cudaMemcpyHostToDevice),"copy graph");
    };
    alloc_copy(d_offsets_,offsets);alloc_copy(d_cols_,cols);alloc_copy(d_edgeshifts_,shifts);
}
Codec::~Codec(){cudaFree(d_shifts_);cudaFree(d_offsets_);cudaFree(d_cols_);cudaFree(d_edgeshifts_);}
}
