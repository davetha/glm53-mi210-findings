# Open issues

Ranked by how much they matter.

## 1. Prefill is numerically unstable — a QUALITY issue, not a speed one

The server returns materially different answers to the same prompt at temperature 0.
One forward pass (`max_tokens=1`), same prompt, seven samples: the fixed token 'The'
varies by up to **3.29 nats** — a ~27x swing in probability.

**Pre-existing.** Confirmed with every change from this work disabled, on the
original 28 GiB / 8-slot configuration with 42 layers armed and 0 declined.

The signature is specific and points somewhere:

    ~13 tokens    spread 0.000000   perfectly reproducible
    longer        spread 1.0 - 3.3  top-1 token itself flips between samples

Exactly zero when short, growing with prompt length. That fits the **sparse
attention indexer**, which only engages once a sequence exceeds its budget —
unstable tie-breaking in its top-k selection would give precisely this shape.
Reduction order in the MoE sum (the first guess) does not explain the length
dependence nearly as well.

Start at `patched_sparse_attn_indexer_kpool.py`. Reproduce with
`tools/correctness/fixedtok.py`. NOT INVESTIGATED.

## 2. Final decode number is not A/B'd to standard

The shipped configuration (off45 / SLOTS=46 / scratch on) measured ~11.8 tok/s on a
SINGLE container. Per METHODOLOGY #1 that is worth ±0.9 tok/s. It is consistent
with the simulator's prediction (46 slots miss 1.30/layer-step vs 1.12 at 56, about
2.4 ms) so it is believable — but the 12.16 figure for the previous configuration
was measured over three paired restarts and this one was not.

`tools/bench/cachecfg_ab.sh` does it properly, ~45 minutes.

## 3. Host swap is fully consumed

Both the GPU box and the dev box run with all 7 GB of swap used. On the GPU box
this matters because a large pinned allocation cannot be reclaimed and the
PROTECTED containers (`q38fn-lru` at 138 GiB, `qwen35`, `litellm*`, `open-webui`)
share that memory.

This is why the 60 GiB offload corner was NOT taken: it would add 30 GiB of
unreclaimable pinned memory for a decode gain inside the noise floor.

## 4. Context is 8192 and cheap to raise

The KV cache holds 118,468 tokens — about 14x what the limit uses. Raising
`--max-model-len` costs prefill time, which is now 3x better than it was. Caveat:
issue #1 grows with prompt length, so longer contexts sit further into the unstable
regime.

## 5. Uncommitted work in the cache repo

Four files modified in `/home/dave/vllm-expert-cache` (branch
`expert-parallel-support`), nothing committed, nothing pushed, per standing rule.
Full diff preserved at `patches/vllm-expert-cache.diff`.

The **resident-layer decline is a genuine bug fix** rather than a tuning change and
is the one most worth preserving.

All four changes are behind default-ON env gates and reverse without editing code:

    EXPERT_CACHE_WIDE_SCRATCH=0
    EXPERT_CACHE_FUSED_REMAP=0
    EXPERT_CACHE_DECLINE_RESIDENT=0
    GLM53_TOPK_REGSTORE=0

## 6. Two research contradictions never resolved

  * Whether Ling-3.0-flash-VL's `vllm-ling-v3` fork is required. One pass checked
    recipes.vllm.ai and found an official recipe needing only a nightly; another
    asserted the fork. Matters because the fork would mean porting the entire patch
    set.
  * The DeepSeek vision repos declare no vision config and an architecture of
    `DeepseekV4ForCausalLM`, with the vision encoder published separately as a 412M
    ViT under timm. How that attaches in vLLM is unknown.
