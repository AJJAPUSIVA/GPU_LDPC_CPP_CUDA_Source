
#include "nrldpc/ldpc.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess)
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(e));
}

static double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    int n = int(values.size());
    return n % 2 ? values[n/2] :
        (values[n/2-1] + values[n/2]) / 2.0;
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "Usage: workspace_ab MATRIX BATCH MODE REPEATS\n";
            return 2;
        }

        int batch = std::stoi(argv[2]);
        int repeats = std::stoi(argv[4]);
        std::string mode = argv[3];

        if (batch < 1 || repeats < 3 ||
            (mode != "fixed" && mode != "early"))
            throw std::invalid_argument("invalid arguments");

        bool early = mode == "early";
        nrldpc::Codec codec(nrldpc::load_matrix(argv[1]));

        size_t bits = size_t(batch) * codec.codeword_bits();
        size_t messages = size_t(batch) *
                          codec.decoder_message_floats();

        std::mt19937 rng(12345);
        std::normal_distribution<float> noise(3.0f, 2.5f);
        std::vector<float> h_llr(bits);

        for (auto& x : h_llr) x = noise(rng);

        float *d_llr=nullptr, *d_work=nullptr, *d_msg=nullptr;
        uint8_t *d_old=nullptr, *d_new=nullptr;
        int *d_old_it=nullptr, *d_new_it=nullptr;

        check(cudaMalloc((void**)&d_llr, bits*sizeof(float)), "LLR");
        check(cudaMalloc((void**)&d_work, bits*sizeof(float)), "work");
        check(cudaMalloc((void**)&d_msg, messages*sizeof(float)), "messages");
        check(cudaMalloc((void**)&d_old, bits), "original output");
        check(cudaMalloc((void**)&d_new, bits), "reused output");
        check(cudaMalloc((void**)&d_old_it, batch*sizeof(int)), "original iters");
        check(cudaMalloc((void**)&d_new_it, batch*sizeof(int)), "reused iters");

        check(cudaMemcpy(d_llr, h_llr.data(), bits*sizeof(float),
                         cudaMemcpyHostToDevice), "copy input");

        auto original = [&]() {
            codec.decode(d_llr, d_old, batch, 10, 0.8f,
                         0, early, d_old_it);
        };

        auto reused = [&]() {
            codec.decode_with_workspace(
                d_llr, d_new, batch, 10,
                d_work, d_msg, 0.8f, 0, early, d_new_it);
        };

        // First validate on identical LLRs.
        original();
        reused();
        check(cudaDeviceSynchronize(), "validation sync");

        std::vector<uint8_t> old_bits(bits), new_bits(bits);
        std::vector<int> old_it(batch), new_it(batch);

        check(cudaMemcpy(old_bits.data(), d_old, bits,
                         cudaMemcpyDeviceToHost), "original bits");
        check(cudaMemcpy(new_bits.data(), d_new, bits,
                         cudaMemcpyDeviceToHost), "reused bits");
        check(cudaMemcpy(old_it.data(), d_old_it, batch*sizeof(int),
                         cudaMemcpyDeviceToHost), "original iterations");
        check(cudaMemcpy(new_it.data(), d_new_it, batch*sizeof(int),
                         cudaMemcpyDeviceToHost), "reused iterations");

        size_t bit_diff = 0, iter_diff = 0;

        for (size_t i=0; i<bits; ++i)
            bit_diff += old_bits[i] != new_bits[i];

        for (int i=0; i<batch; ++i)
            iter_diff += old_it[i] != new_it[i];

        if (bit_diff || iter_diff) {
            std::cerr << "Correctness mismatch: bits="
                      << bit_diff << " iterations=" << iter_diff << "\n";
            return 3;
        }

        cudaEvent_t start, stop;
        check(cudaEventCreate(&start), "create start");
        check(cudaEventCreate(&stop), "create stop");

        auto time_call = [&](bool reuse) -> float {
            check(cudaEventRecord(start), "record start");
            if (reuse) reused();
            else original();
            check(cudaEventRecord(stop), "record stop");
            check(cudaEventSynchronize(stop), "sync stop");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
            return ms;
        };

        for (int i=0; i<3; ++i) {
            time_call(false);
            time_call(true);
        }

        std::vector<float> old_ms, new_ms;

        for (int i=0; i<repeats; ++i) {
            if (i % 2 == 0) {
                old_ms.push_back(time_call(false));
                new_ms.push_back(time_call(true));
            } else {
                new_ms.push_back(time_call(true));
                old_ms.push_back(time_call(false));
            }
        }

        double t_old = median(old_ms);
        double t_new = median(new_ms);
        double avg_iter = std::accumulate(
            old_it.begin(), old_it.end(), 0.0) / batch;

        std::cout << std::fixed << std::setprecision(4)
                  << "RESULT"
                  << ",batch=" << batch
                  << ",mode=" << mode
                  << ",original_ms=" << t_old
                  << ",reused_ms=" << t_new
                  << ",speedup=" << t_old/t_new
                  << ",avg_iterations=" << avg_iter
                  << ",bit_mismatches=" << bit_diff
                  << ",iteration_mismatches=" << iter_diff
                  << "\n";

        check(cudaEventDestroy(start), "destroy start");
        check(cudaEventDestroy(stop), "destroy stop");

        check(cudaFree(d_llr), "free llr");
        check(cudaFree(d_work), "free work");
        check(cudaFree(d_msg), "free messages");
        check(cudaFree(d_old), "free original bits");
        check(cudaFree(d_new), "free reused bits");
        check(cudaFree(d_old_it), "free original iters");
        check(cudaFree(d_new_it), "free reused iters");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
