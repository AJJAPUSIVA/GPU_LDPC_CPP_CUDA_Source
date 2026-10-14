#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace nrldpc::cpu {
struct Matrix {
    int rows{}, cols{}, kb{}, z{};
    std::vector<int> shifts;
    int information_bits() const { return kb*z; }
    int codeword_bits() const { return cols*z; }
    void validate() const;
};
Matrix load_matrix(const std::string& file);
// Floats are LLR log(P(0)/P(1)); one byte per decoded bit.
// The scalar reference is CPU research code, never a CUDA fallback.
void decode_scalar(const Matrix& h, const float* llr, uint8_t* bits,
                   int batch, int max_iterations, float alpha=0.8f);
// Requires AVX-512F CPU and compatible x86 compiler; throws otherwise.
bool avx512_available();
void decode_avx512(const Matrix& h, const float* llr, uint8_t* bits,
                   int batch, int max_iterations, float alpha=0.8f);
// Returns true for zero-syndrome bit vector.
bool check(const Matrix& h, const uint8_t* bits);
}
