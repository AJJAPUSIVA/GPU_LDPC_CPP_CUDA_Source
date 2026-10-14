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

Toy file `matrices/toy_qc_z4.txt` is provided solely for smoke testing; **it is NOT a 5G NR base graph**. To test BG1/BG2, supply the exact matrices specified by TS 38.212 Tables 5.3.2-2 to 5.3.2-3 with valid `Z` and lifting-index choices. Full 3GPP tables are not bundled. The loader accepts any consistent nonsingular systematic QC parity-check matrix in this format.

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
# GPU_LDPC_CPP_CUDA_Source
# GPU_LDPC_CPP_CUDA_Source

## GPU forward/backward layered decoder upgrade

The decoder check-node processing now uses a register-local backward suffix
minimum and a forward prefix minimum to produce the normalized min-sum
message for each incident edge. The implementation is inspired by Algorithm 5
of Usatyuk and Egorov (IEEE DSPA 2025), rather than a verbatim implementation.
It retains QC-layer synchronization and syndrome early termination, and it
adds a hard matrix row-degree bound of [2,64]. CUDA computation stays on the
GPU; the CPU only parses matrices and launches kernels.

Validation is required on an RTX 4090 (`sm_89`). The toy round-trip tests
now cover three alpha values and a non-default CUDA stream. The change does
not yet provide full 3GPP BG1/BG2 matrices, GPU rate matching, HARQ or CRC.
The forward/backward approach may reduce comparison work, but performance
improvements are **not yet measured** and register pressure could offset gains.
