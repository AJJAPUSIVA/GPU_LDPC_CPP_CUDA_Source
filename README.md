# GPU-only C++20/CUDA QC-LDPC prototype (RTX 4090)

Reference benchmark project: https://github.com/rcbarke/six-times-to-spare

**STATUS: source prototype, NOT a validated 3GPP-compliant production 5G NR codec.**
No CPU LDPC fallback exists. Encoding (GPU GF(2) elimination), decoding (GPU layered normalized min-sum) and syndrome checking are CUDA kernels. Host C++ only loads QC matrices and controls CUDA memory/launches.

## Required environment
- NVIDIA RTX 4090 (Ada, `sm_89`), CUDA toolkit with C++20 support, CMake 3.24+, Linux C++ compiler.
- No CUDA device => hard failure. This environment cannot compile or execute CUDA; tests must be executed on real GPU.

## Build and test
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/ldpc_bench matrices/toy_qc_z4.txt 32 10
```
To retrieve the original research repo on your own machine: `./tools/clone_reference.sh` (requires GitHub connectivity).

## QC matrix format
Plain whitespace-delimited: first line `rows cols kb Z`; then `rows*cols` signed integer shift values row-major. `-1` means zero block. Nonnegative values describe circulant parity-check positions `(rowZ+t, colZ+(t+shift)%Z)`. Here `cols-kb=rows`; all parity columns are included. The first `kb*Z` bits are the systematic message; the remaining bits are parity. **This representation is an expanded, fully chosen 3GPP lifting-set-specific base graph**, not the raw tables of i_ls-dependent shift candidates.

Toy file `matrices/toy_qc_z4.txt` is provided solely for smoke testing.

## API
See `include/nrldpc/ldpc.hpp` and `tests/test_codec.cu`. `Codec::encode`, `Codec::decode` and `Codec::check` take **CUDA device pointers** and support asynchronous stream launch. Bytes represent bits, *not* packed bits. LLR convention is positive for bit 0. Caller manages stream synchronization and device pointer lifetime.

## Explicit limitations / next development steps
- **Encoder is deliberately a generic, dense GF(2) GPU elimination solver**. It performs O(M^3/word_size) work and can be extremely slow at full NR BG1/BG2 sizes; it is not production performant. Singular matrices yield 0xff output sentinel (caller must parity-check output).
- No 3GPP segmentation, transport-block CRC, code-block CRC, puncturing, filler-bit handling, BG selection, RV rate matching/de-rate matching, HARQ buffer, or 5G MAC integration yet.
- Early stopping is parity-syndrome based; CRC-aided early stopping is not implemented.
- No validated end-to-end NR golden vectors are bundled; passing toy smoke tests does not imply NR compliance.
- No GPU compilation, test run, or throughput measurement has been performed in the creation environment, which lacks both `nvcc` and a GPU.
- For production, replace the encoder with GPU-parallel 5G NR four-core-block parity initialization + dual-diagonal parity recursion, then implement NR-aware rate matching, CRC and HARQ and obtain comprehensive conformance tests.

## Repo relationship
The referenced repo benchmarks Sionna's Python/TF LDPC decoder, not a reusable C++ CUDA encoder. This source project is an independent CUDA implementation informed by its benchmarking approach; it does not claim to port proprietary hardware APIs or its TensorFlow internals.
