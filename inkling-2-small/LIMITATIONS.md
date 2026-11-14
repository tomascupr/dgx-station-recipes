# Limitations

1. **No GPU run of the serving GEMMs.** Placement, dequant, routing and the
   peer record were checked on a CPU. The CUTLASS grouped GEMMs
   (`inkling_grouped.cuh`) have not been executed; the FP64 kernels are the
   reference to diff against. `./inkling2 up` starts day-0 vLLM only. The hook
   raises until `INKLING2_REFERENCE=1`, which is a tiny-bank mode, not a way
   to serve the checkpoint.

2. **dsv41 is not touched.** `up` refuses while a `ds41` or `ds41-sidecar`
   container exists. It never removes them. Host RAM while dsv41 is up is not
   enough to serve this model beside it (the plan's measurement: about 119 GiB
   free). Inkling replaces dsv41 for the window. Restoring dsv41 is part of
   the window.

3. **Grace spill is still required** if MXFP8 stays MXFP8. Requantizing layers
   40–47 is a quality change and is not implemented.

4. **Routing is unmeasured.** The default map gives the lowest NVFP4 ids to
   the GB300. That is a capacity prior, not a hit-rate. A flat router puts
   about 96/384 of NVFP4 expert-tokens on Grace. The plan's bandwidth
   estimates say that is the slow case. They are estimates.

5. **Day-0 vLLM is SM100 code, unproven on this GB300 (SM 10.3).** The RTX
   (SM 12.0) is outside that image. If the first smoke's kernels fail, the
   fallback in the plan is the SGLang image `lmsysorg/sglang:dev-inkling-2-small`,
   which then needs its own hook. This tree does not contain that hook.

6. **PR 67325 was unmerged** (head `d664cd1e`, mergeable state dirty when this
   was written). The image tag `vllm/vllm-openai:inkling-2-small` is not pinned
   by digest.
