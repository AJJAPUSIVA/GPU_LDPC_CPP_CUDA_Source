# CUDA QC-LDPC Reusable-Workspace Experiment

## Scope

NVIDIA Tesla T4, BG1/BG2, Z=32, full mother-code decoding.
Nonzero NVIDIA Sionna mother-codewords with simulated AWGN.

The experimental `decode_with_workspace()` method accepts
caller-managed device workspace and invokes the same
layered normalized min-sum decoding kernel.

## Large-AWGN A/B results

Ten timed repetitions per decoder path and configuration,
following warm-up calls.

| Graph | Eb/N0 (dB) | Batch | Mode | Original (ms) | Reused (ms) | Speedup | Avg iterations |
|---|---:|---:|---|---:|---:|---:|---:|
| BG1 | 1.4 | 256 | early | 2.6960 | 2.5210 | 1.069x | 6.793 |
| BG1 | 1.4 | 256 | fixed | 5.2772 | 4.6572 | 1.133x | 10.000 |
| BG1 | 1.4 | 1024 | early | 8.7918 | 8.4860 | 1.036x | 6.697 |
| BG1 | 1.4 | 1024 | fixed | 11.8253 | 11.4095 | 1.036x | 10.000 |
| BG1 | 2.2 | 256 | early | 1.6970 | 1.5500 | 1.095x | 4.356 |
| BG1 | 2.2 | 256 | fixed | 3.4743 | 3.2593 | 1.066x | 10.000 |
| BG1 | 2.2 | 1024 | early | 5.6454 | 5.4085 | 1.044x | 4.286 |
| BG1 | 2.2 | 1024 | fixed | 11.8778 | 11.4003 | 1.042x | 10.000 |
| BG2 | 1.4 | 256 | early | 1.4695 | 1.2907 | 1.139x | 5.680 |
| BG2 | 1.4 | 256 | fixed | 1.9154 | 1.7665 | 1.084x | 10.000 |
| BG2 | 1.4 | 1024 | early | 4.7317 | 4.5671 | 1.036x | 5.662 |
| BG2 | 1.4 | 1024 | fixed | 7.5740 | 7.3096 | 1.036x | 10.000 |
| BG2 | 2.2 | 256 | early | 1.0738 | 0.9275 | 1.158x | 4.184 |
| BG2 | 2.2 | 256 | fixed | 1.9236 | 1.7161 | 1.121x | 10.000 |
| BG2 | 2.2 | 1024 | early | 3.4166 | 3.2111 | 1.064x | 4.147 |
| BG2 | 2.2 | 1024 | fixed | 7.5016 | 7.2723 | 1.032x | 10.000 |

## Validation

- 16/16 configurations completed.
- Zero decoded-bit mismatches.
- Zero per-codeword iteration-count mismatches.
- Observed speedup range: 1.032x to 1.158x.
- Existing original decoder API retained.

## Engineering constraints

- Caller owns the reusable LLR and message buffers.
- Buffers must remain valid until GPU operations complete.
- Overlapping decodes require separate writable workspace
  or correctly ordered stream execution.
- The fixed-mode syndrome-skip change remains experimental.

## Limitations

- These are within-run median CUDA-event measurements.
- Ten repeats do not establish statistical certainty.
- The test is not complete 3GPP NR conformance.
- No verified AVX-512 or RTX 4090 measurements.
- GPU hardware counters were unavailable on Kaggle.

Paid GPU rental cost: $0.
