# Speculative decoding on GLM-5.3-Flash / 2× MI210: MTP runs, and is not usable

**Verdict: closed. MTP produces degenerate output with this checkpoint, and even if that
were fixed, the measured acceptance rate does not clear the cost of the draft layer.**

Speculation was the most attractive remaining lever on paper, because ~33 ms of the 76 ms
decode step is *fixed* per-step cost (all-reduce, dense GEMM, elementwise, mHC) that a
verified draft token gets for free. This is why it does not work out.

## The draft exists — it is just the wrong dtype

`num_hidden_layers: 45` means layers 0–44 are the model; **layer 45 is a complete MTP draft
layer** — `eh_proj`, `enorm`/`hnorm`, and a full 288-expert MoE, 889 tensors.

(Searching tensor names for `nextn` or `mtp` finds nothing: GLM numbers the draft as the
next layer index. It is easy to conclude the weights are absent when they are not.)

The problem is that it shipped **BF16 while the rest of the model is W4A16 int4**:

```
layer 45 (MTP draft):   14.87 GB   all BF16
  -> per rank at TP=2:   7.43 GB
free VRAM per card:      ~3.3 GB
```

Per-expert that is **51.6 MB** against 12.375 MiB for the int4 layers — 4× larger. It
cannot be resident, so it is offloaded, and each draft step routes 8 of its experts:
~400 MB of PCIe per draft step, ~14 ms at the 28 GB/s the link delivers.

## Three attempts

| config | outcome |
|---|---|
| `OFFLOAD=45 SLOTS=52` (standing) | **OOM.** 63.13 GiB allocated, 88 MiB free |
| `OFFLOAD=62 SLOTS=36` | loads, then **`hipErrorStreamCaptureInvalidated`** — device-side assertion during CUDA graph capture |
| `OFFLOAD=62 SLOTS=36 --enforce-eager` | **serves** (eager skips capture) — and generates garbage |

Note what each failure was *not*: VRAM stopped being the blocker at `OFFLOAD=62`. More
offload makes it fit. It does not make it work.

## It generates degenerate text

`temperature=0`, "What is 2+2? Answer with just the number.", 400 tokens, `finish_reason:
length`:

> *"Wait, no. I mean, the answer is 4. But the user asked me to answer with... Answer: 4.
> Wait, answer = 0. As a result, the answer is 2. I apologize. I responded with, 'I = 0, 1.
> Wait, is the answer 0..."*

Looping, self-contradicting, never terminating. Correctness **0/3**.

**This is not what a bad draft looks like.** A poor draft is simply rejected and the target
model's token is used — output stays correct, only speed suffers. Wrong output means the
verification path itself is broken.

### Isolated with a control

Identical config (`OFFLOAD=62 SLOTS=36 --enforce-eager`), only `SPEC=off`:

```
finish_reason: stop | tokens: 20
content: '4'
reasoning: 'The user wants me to answer with just the number. 2+2=4'
```

Clean. So the expert cache is fine at 36 armed layers, eager mode is fine, the high offload
is fine. **MTP is the variable.**

## The acceptance rate does not clear the bar either

From vLLM's own `SpecDecoding metrics` over ten reporting windows:

```
Mean acceptance length: 1.17 - 1.77   (typically ~1.2-1.4)
Avg draft acceptance:   17.3% - 76.7% (typically ~20-45%)
```

Break-even is simple: MTP wins only if the MTP step costs **less than the acceptance
length** times the base step. At ~1.3 the MTP step must stay under ~99 ms against a 76 ms
base. The offloaded bf16 draft alone adds ~14 ms (+18%), before the draft forward pass and
verifying two tokens' worth of routing. That is break-even at best.

It would need ~1.5+ acceptance to be clearly worth having.

**Caveat, stated honestly:** degenerate output would itself depress acceptance — draft and
target disagreeing because the text is nonsense — so 1.3 may understate a working setup.
The number is not clean. But it is the only acceptance figure obtainable without first
fixing the verification bug.

## What would have to be true to revisit this

1. **Fix the verification bug.** Unknown cause. Candidates: the bf16 draft layer against an
   int4 target, the draft's experts being read through the UVA offload path, or an
   incompatibility between this MTP implementation and W4A16. Not investigated — the
   economics below made it moot.
2. **Quantise layer 45 to int4.** 14.87 GB → ~3.9 GB, **1.95 GB/rank, which fits in the
   3.3 GB free.** The draft becomes resident, gathers nothing, and the standing config keeps
   `SLOTS=52` / `OFFLOAD=45` intact. This is the only configuration where speculation could
   pay, and it does not violate an int4 floor — it brings the one inconsistent layer in line.
3. **Acceptance would still need to beat ~1.3.** Even with a free resident draft, the draft
   forward pass and 2-token verification are not free.

Steps 1 and 2 are both real work, to reach something that is currently break-even. That is
why this is closed rather than parked.

## Related: DFlash

This vLLM build supports DFlash (`DFlashModelTypes = Literal["dflash"]`). Two DFlash2
drafts are present on the box (`curvedinf-dflash2-draft`, `qwen38-dflash2`) but both are
`model_type: qwen3`, hidden 5120, vocab 248320. GLM-5.3 is `glm5_next`, hidden 4096, vocab
154880 — a draft must share the target's tokenizer and hidden geometry, so neither applies.
They belong to the Qwen deployment.
