# Karnbir Khera

**GPU Programming · MLSys 2026 FlashInfer Competitor**

Currently competing in the MLSys 2026 NVIDIA FlashInfer Sparse Attention track on Blackwell B200, and writing weekly about each kernel that forces the framework to grow another layer on [LinkedIn](https://www.linkedin.com/in/karnbir-khera/). Building a derivation-based mental model of GPU computation — the **Two Tree Framework** — where every kernel decision (memory binding, FSM phase, sync point, indexing) traces back to either the problem geometry or a hardware constraint, rather than being memorized from existing implementations.

**Profiled architectures:** Ada Lovelace (RTX 4060) · Blackwell (B200, sm_100a)
**Next Learning Arc:** Polyhedral model, Dataflow analysis, Abstract algebra to encode the patterns from 9 weeks of kernel work into MLIR, and learn how modern compilers produce optimized kernels.

---

## Featured Projects

### [MLSys 2026 FlashInfer Sparse Attention (Track B)](https://github.com/KarnbirKhera/KarnbirKhera-MLSys2026-dsa_sparse_attention)

Submission to the NVIDIA MLSys 2026 FlashInfer AI Kernel Generation Contest on B200. A derivation-first agentic pipeline ([GitHub](https://github.com/KarnbirKhera/MLSys2026-Kernel-Agent-Framework-Templates)) takes a kernel spec and hardware target, derives the structure (geometry → algorithm class → access pattern → FSM phases → lifetime tables → indexing) before any code is written, and binds those decisions to sm_100a only at the end so every optimization is auditable back to a problem-space, hardware-space or empirically derived reason.


### [9-Week Kernel Learning Plan](https://github.com/KarnbirKhera/MLSys2026-9Week-LearningPlan)

A weekly progression of CUDA kernels building toward sparse attention: shared memory → GEMM → softmax (FP8) → dense attention → paged KV (MLA) → top-k indexer → sparse attention → optimization. Each week's kernel is the artifact that forced a new layer of the framework. For those who are learning CUDA, I would recommend this repo as every kernel is commented line by line, especially in the latter kernels. The comments contain what each line of code means, and how it contributes to the overall structure of the kernel.


### [Two Tree Framework (V1)](https://github.com/KarnbirKhera/Two-Tree-Framework)

The Two Tree Framework started as a struggle to reconstruct a tiled GEMM kernel the day after reading it, and grew into a derivation system that builds kernels up from problem geometry rather than pattern-matching from existing code. V1 builds every index from first principles. Each reduces to Coordinate × Stride + Offset, found by mapping the execution tree (Grid → Block → Thread) onto the memory tree (Global → Shared → Register). The goal is pedagogical, making the path to a correct kernel feel learnable rather than memorized.



Each kernel that followed forced a new layer.

- Softmax surfaced algorithm classification, because row-max-before-normalization is an algorithm constraint, not a shape one.
- Dense attention surfaced FSM phases, because Load → Compute → Store breaks once softmax sits in the middle.
- Paged KV surfaced access patterns, because paged and contiguous caches cross the affine / non-affine boundary at the same shape.
- Top-K surfaced a refinement to the contraction classification, splitting REDUCE (associative and commutative) from GATE (associative but not commutative, where order is preserved, like scan-class reductions).


### [CUDA Vector Addition: A 40-Page Insight](https://github.com/KarnbirKhera/CUDA-Vector-Addition-40-Page-Insight)


My first CUDA project where I implemented vector add with six implementations (naive → grid-stride → float4 vectorization → ILP=2 → ILP=4) profiled on RTX 4060 with NVIDIA Nsight Compute, comparing measured throughput against the 272 GB/s theoretical bandwidth across 10M / 100M / 200M element runs.

The deeper investigation began with an unexpected ~31% L2 hit rate on a fully streaming kernel. This led to designing isolated micro-benchmarks (read-only, write-only, coalesced vs. uncoalesced) which revealed that the L2 hit rate only surfaced during writes, not reads. I found that odd at the time so I ended up finding an arXiv paper about the **write-validate policy** on the Volta architecture. I re-created the same testing environment on the Ada architecture and found a strong correlation between the paper and my results. I then conducted the same test on the B200 Blackwell architecture and found very similar results, strongly showing that the same **write-validate policy** was present. This is covered in the following two LinkedIn posts. [Post 1](https://www.linkedin.com/posts/karnbir-khera_cuda-nvidia-hpc-activity-7429192855476404225-FlaY) and [Post 2](https://www.linkedin.com/posts/karnbir-khera_cuda-nvidia-blackwell-activity-7430642396683599872-ZERT).


---

## Now

- Wrapping up the FlashInfer competition with extension tracks: **MoE** (sparse dispatch with runtime routing the geometry can't fully predict) and **GDN** (gated delta-net for Qwen3-Next, sitting outside semiring/monoid classification)
- Starting the next arc, MLIR + Compiler Theory, picking up the polyhedral model, dataflow analysis, and abstract algebra alongside MLIR so the patterns from 9 weeks of kernel work have a home in modern compilers.

---
