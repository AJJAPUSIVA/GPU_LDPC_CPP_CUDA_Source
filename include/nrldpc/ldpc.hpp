#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace nrldpc {
// Sparse QC parity check matrix. Each entry denotes a nonnegative right
// cyclic shift; -1 denotes an all-zero ZxZ circulant.
// Supply the 3GPP lifting-set-specific H matrix including ALL parity columns.
struct Matrix {
    int rows=0, cols=0, kb=0, z=0;
    std::vector<int> shifts; // rows*cols, row-major
    void validate() const;
};
Matrix load_matrix(const std::string& path);

class Codec {
public:
    // CUDA device required. No CPU LDPC implementation or fallback.
    explicit Codec(Matrix matrix, int device=0);
    ~Codec();
    Codec(const Codec&) = delete;
    Codec& operator=(const Codec&) = delete;
    int information_bits() const { return h_.kb*h_.z; }
    int codeword_bits() const { return h_.cols*h_.z; }
    int parity_bits() const { return (h_.cols-h_.kb)*h_.z; }
    // Device pointers; one byte per bit 0/1; contiguous [batch, K or N].
    // Encode is a GENERAL GF(2) parity solver on the GPU. Correctness first:
    // its cubic-cost elimination is NOT optimized for production NR block sizes.
    void encode(const uint8_t* d_info, uint8_t* d_codeword,
                int batch, cudaStream_t stream=0);
    // Input LLRs use log(P(bit=0)/P(bit=1)); output is one byte/bit.
    // All decoding arithmetic and syndrome checks run on the GPU.
    void decode(const float* d_llr, uint8_t* d_bits, int batch,
                int max_iterations, float alpha=0.8f,
                cudaStream_t stream=0);
    // Returns 1 per valid codeword, 0 otherwise (device output).
    void check(const uint8_t* d_codeword, uint8_t* d_valid,
               int batch, cudaStream_t stream=0);
private:
    Matrix h_;
    int* d_shifts_ = nullptr;
    int* d_offsets_ = nullptr;
    int* d_cols_ = nullptr;
    int* d_edgeshifts_ = nullptr;
    int edges_=0;
};
void cuda_check(cudaError_t status, const char* context);
}
