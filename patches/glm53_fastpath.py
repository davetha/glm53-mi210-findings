"""Decode fast path for GLM-5.3-Flash on gfx90a (mounted next to model.py, imported by it).

Every replacement here is a drop-in for a torch/vLLM path that the eager kernel trace
showed to be launch-bound or badly tiled at M=1, and every one is a plain Triton kernel
with static shapes, so it captures into the decode CUDA graph unchanged:

  1. router gate GEMM: hipBLASLt picks an MT128x32x64 tile for bf16->fp32 [1,4096]x[4096,288]
     (45-58 us, called TWICE per MoE layer by model + MoERunner) -> fp32-accumulate GEMV.
  2. noaux_tc top-k with n_group=1: ~15 torch kernels incl. two sbtopk (38+27 us) -> one kernel.
  3. shared-expert SiluAndMulWithClamp: 7 elementwise kernels on ROCm (forward_native) -> one.
     (opt-in only via GLM53_FASTPATH_PARTS=gate,topk,act -- see _PARTS below)

GLM53_FASTPATH=0 disables everything. Failures to install are logged loudly, never hidden.
"""
import os

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("GLM53_FASTPATH", "1") != "0"
# bisect switch: which replacements to install (comma list of gate,topk,act)
# "act" is deliberately NOT in the default: the fused clamped-SwiGLU matches forward_native to
# 1 bf16 ulp in isolation and answers correctly in eager mode, but under cudagraph replay it
# degrades the model's answers (3/3 launches). Root cause not found; left opt-in for debugging.
# Default = gate,topk. "layout" (keep uint8 K-packed experts) and "moe" (M=1 int4 GEMV) are
# opt-in: measured 2026-09-15 they LOSE on this box -- the GEMV halves the MoE kernel time but
# the uint8 layout makes expert_cache_gather_k ~21% slower per call and the original kernel's
# uint8 branch ~3x slower for prefill; net decode 103.6 vs 89.5 ms/step at OFFLOAD_GB=28.
_PARTS = set(os.environ.get("GLM53_FASTPATH_PARTS", "gate,topk,nodoublegate").split(","))
_GATE_MAX_M = 32
_TOPK_MAX_M = 64


# ---------------------------------------------------------------- 1. gate GEMV --------
@triton.jit
def _gate_gemv_kernel(x_ptr, w_ptr, out_ptr, M, N, K,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # one program = BLOCK_N output columns for one token; fp32 accumulate over K
    pid_n = tl.program_id(0)
    m = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        x = tl.load(x_ptr + m * K + kk, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * K + kk[None, :],
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(out_ptr + m * N + offs_n, acc, mask=n_mask)


def gate_gemv_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M,K] bf16, w [N,K] bf16 -> [M,N] fp32 (fp32 accumulation, like cuBLAS out_dtype=fp32)."""
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    BLOCK_N = 8
    BLOCK_K = 1024 if K % 1024 == 0 else 512
    grid = (triton.cdiv(N, BLOCK_N), M)
    _gate_gemv_kernel[grid](x, w, out, M, N, K, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    return out


# ---------------------------------------------------------------- 2. fused top-k --------
@triton.jit
def _topk_sigmoid_bias_kernel(logits_ptr, bias_ptr, w_ptr, id_ptr, N,
                              renorm: tl.constexpr, scale: tl.constexpr,
                              TOPK: tl.constexpr, BLOCK: tl.constexpr,
                              REGSTORE: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    logits = tl.load(logits_ptr + m * N + offs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scores = tl.sigmoid(logits)                       # routing weights come from these
    # selection uses the biased scores. NaN logits (graph-capture dummy inputs feed
    # garbage through the MoE) must never win the argmax: an id >= N would send the
    # expert-cache/gather kernels out of bounds (hardware exception during capture).
    valid = mask & (scores == scores)
    biased = tl.where(valid, scores + bias, float("-inf"))
    offs_t = tl.arange(0, TOPK)
    wvec = tl.zeros((TOPK,), dtype=tl.float32)
    ivec = tl.zeros((TOPK,), dtype=tl.int32)
    for i in tl.static_range(TOPK):
        idx = tl.minimum(tl.argmax(biased, axis=0), N - 1)
        sel = offs == idx
        wt = tl.sum(tl.where(sel, scores, 0.0), axis=0)
        if REGSTORE:
            # Keep the ids in registers like the weights. Storing inside the loop is 8
            # dependent scalar round-trips; the whole kernel measured 24.9 us that way.
            ivec = tl.where(offs_t == i, idx.to(tl.int32), ivec)
        else:
            tl.store(id_ptr + m * TOPK + i, idx.to(tl.int32))
        wvec = tl.where(offs_t == i, wt, wvec)
        biased = tl.where(sel, float("-inf"), biased)
    if renorm:
        wvec = wvec / tl.sum(wvec, axis=0)
    if REGSTORE:
        tl.store(id_ptr + m * TOPK + offs_t, ivec)
    tl.store(w_ptr + m * TOPK + offs_t, wvec * scale)


def topk_sigmoid_bias(gating_output: torch.Tensor, bias: torch.Tensor, topk: int,
                      renormalize: bool, routed_scaling_factor: float):
    M, N = gating_output.shape
    weights = torch.empty((M, topk), dtype=torch.float32, device=gating_output.device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=gating_output.device)
    _topk_sigmoid_bias_kernel[(M,)](
        gating_output, bias, weights, ids, N,
        renorm=bool(renormalize), scale=float(routed_scaling_factor),
        TOPK=topk, BLOCK=triton.next_power_of_2(N), num_warps=4,
        REGSTORE=os.environ.get("GLM53_TOPK_REGSTORE", "1") != "0")
    return weights, ids


# ---------------------------------------------------------------- 3. clamped SwiGLU ------
@triton.jit
def _silu_mul_clamp_kernel(x_ptr, out_ptr, D, limit, alpha, beta, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    g = tl.load(x_ptr + row * 2 * D + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * 2 * D + D + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.minimum(g, limit)
    u = tl.minimum(tl.maximum(u, -limit), limit)
    y = g * tl.sigmoid(alpha * g) * (u + beta)
    tl.store(out_ptr + row * D + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def silu_mul_clamp(x: torch.Tensor, limit: float, alpha: float, beta: float) -> torch.Tensor:
    d = x.shape[-1] // 2
    x2 = x.reshape(-1, 2 * d)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    out = torch.empty((x2.shape[0], d), dtype=x.dtype, device=x.device)
    BLOCK = 1024
    _silu_mul_clamp_kernel[(x2.shape[0], triton.cdiv(d, BLOCK))](
        x2, out, d, float(limit), float(alpha), float(beta), BLOCK=BLOCK, num_warps=4)
    return out.reshape(*x.shape[:-1], d)



# ---------------------------------------------------------------- 4. int4 MoE GEMV (M=1) ----
# fused_moe_kernel_gptq_awq tiles a 16-row MFMA block around a single real token and runs
# ~6x off the HBM roofline at decode (121-140 us for ~33 MB of expert weight). At M=1 every
# expert block of the moe_align layout holds at most ONE valid token, so the job is a plain
# int4 GEMV: stream B [BLOCK_N, K/2] uint8 (K-packed, low nibble = even k), dequantise in
# registers, fp32 FMA against the activation row, apply the per-(row, group) scale once per
# K-chunk. Same inputs/outputs/layout as the original launch (sorted_token_ids, expert_ids,
# C[token, :] written in compute_type, routed weight applied when the caller asks).
@triton.jit
def _moe_int4_gemv_m1(a_ptr, b_ptr, c_ptr, bs_ptr, tw_ptr, sorted_ptr, expert_ids_ptr, ntpp_ptr,
                      N, K, num_valid_tokens,
                      stride_am, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
                      stride_bse, stride_bsk, stride_bsn,
                      GS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
                      compute_type: tl.constexpr):
    GPC: tl.constexpr = BLOCK_K // GS      # quant groups per K chunk (bigger chunks = more bytes in flight)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ntpp = tl.load(ntpp_ptr)
    if pid_m * BLOCK_M >= ntpp:
        return
    offs_tok = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    toks = tl.load(sorted_ptr + offs_tok)
    tmask = toks < num_valid_tokens
    tok = tl.max(tl.where(tmask, toks, -1), axis=0)   # the block's single valid token, or -1
    if tok < 0:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    e = tl.load(expert_ids_ptr + pid_m)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if e != -1:
        row = tok // top_k
        offs_k = tl.arange(0, BLOCK_K)
        offs_kh = tl.arange(0, BLOCK_K // 2)
        offs_g = tl.arange(0, GPC)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + row * stride_am + k0 + offs_k).to(tl.float32)
            bp = (b_ptr + e * stride_be + offs_n[:, None] * stride_bn
                  + (k0 // 2 + offs_kh)[None, :] * stride_bk)
            b = tl.load(bp, mask=n_mask[:, None], other=0)
            lo = (b & 0xF).to(tl.float32) - 8.0
            hi = (b >> 4).to(tl.float32) - 8.0
            prod = tl.interleave(lo, hi) * a[None, :]       # [BLOCK_N, BLOCK_K], k = 2j -> lo
            part = tl.sum(tl.reshape(prod, (BLOCK_N, GPC, GS)), axis=2)   # per-group partials
            sc = tl.load(bs_ptr + e * stride_bse + (k0 // GS + offs_g)[None, :] * stride_bsk
                         + offs_n[:, None] * stride_bsn, mask=n_mask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(part * sc, axis=1)
        if MUL_ROUTED_WEIGHT:
            acc = acc * tl.load(tw_ptr + tok)
    tl.store(c_ptr + tok * stride_cm + offs_n * stride_cn, acc.to(compute_type), mask=n_mask)



def _repack_npacked_to_kpacked(w: torch.Tensor, chunk: int = 4) -> torch.Tensor:
    """int32 [E, K, N/8] (nibble j of word w = column 8w+j) -> uint8 [E, N, K/2] (low nibble =
    even k). The int32 layout is what fused_moe_kernel_gptq_awq's interleave path consumes; the
    uint8 layout is its other supported input AND what the M=1 GEMV streams efficiently."""
    E, K, Np = w.shape
    N = Np * 8
    out = torch.empty((E, N, K // 2), dtype=torch.uint8, device=w.device)
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
    for e0 in range(0, E, chunk):
        blk = w[e0:e0 + chunk]                                                  # [c, K, N/8]
        nib = ((blk.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)             # [c, K, N/8, 8]
        nib = nib.reshape(blk.shape[0], K, N).transpose(1, 2).contiguous()     # [c, N, K]
        nib = nib.view(blk.shape[0], N, K // 2, 2)
        out[e0:e0 + chunk] = nib[..., 0] | (nib[..., 1] << 4)
        del nib
    return out


def _install_moe_repack():
    """Keep the checkpoint's K-packed uint8 expert layout instead of vLLM's ROCm-only repack
    to N-packed int32 (compressed_tensors_moe_wna16.py:596-640, done for the interleave path
    of fused_moe_kernel_gptq_awq). The repack is skipped by neutralising repack_int4_to_int32
    (imported inside process_weights_after_loading, so the module attribute is what it sees).
    The unconditional scale permute that follows it is undone in a _setup_kernel wrapper --
    i.e. BEFORE the kernel object captures the layer's tensors (undoing it after
    process_weights_after_loading returned left the kernel holding [E, K/gs, N] scales against
    uint8 [E, N, K/2] weights: pure garbage output, 2026-09-15).
    Cost: prefill's MoE kernel takes its scalar-shift uint8 branch (measure!)."""
    from vllm.model_executor.layers.quantization.utils import moe_wna16_utils as _u
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        compressed_tensors_moe_wna16 as _m,
    )
    from vllm.model_executor.utils import replace_parameter
    _u.repack_int4_to_int32 = lambda w: w          # identity: stay uint8 [E, N, K/2]
    cls = _m.CompressedTensorsWNA16MoEMethod
    orig_setup = cls._setup_kernel

    def setup_kernel(self, layer):
        w13 = getattr(layer, "w13_weight_packed", None)
        if w13 is not None and w13.dtype == torch.uint8:
            for s_attr, w_attr in (("w13_weight_scale", "w13_weight_packed"), ("w2_weight_scale", "w2_weight_packed")):
                sc, w = getattr(layer, s_attr), getattr(layer, w_attr)
                if sc.dim() == 3 and sc.shape[1] != w.shape[1]:
                    # vLLM permuted [E, N, K/gs] -> [E, K/gs, N] for the int32 path; put it back
                    replace_parameter(layer, s_attr, sc.data.permute(0, 2, 1).contiguous())
            layer.w13_weight = layer.w13_weight_packed
            layer.w2_weight = layer.w2_weight_packed
            if not getattr(cls, "_glm53_layout_logged", False):
                cls._glm53_layout_logged = True
                logger.info("moe layout: uint8 K-packed kept: w13 %s scales %s, w2 %s scales %s",
                            tuple(layer.w13_weight_packed.shape), tuple(layer.w13_weight_scale.shape),
                            tuple(layer.w2_weight_packed.shape), tuple(layer.w2_weight_scale.shape))
        return orig_setup(self, layer)

    cls._setup_kernel = setup_kernel

class _GptqAwqDispatch:
    """Stands in for the module-level `fused_moe_kernel_gptq_awq` JITFunction: the caller does
    `fused_moe_kernel_gptq_awq[grid](*args, **kw)`, so `__getitem__` returns a launcher that
    takes the GEMV path for the decode shape and forwards everything else untouched."""

    # swept on MI210 at the real 4-expert decode shape (rp/moe_gemv_tune2.py):
    # gate_up 138 -> 67 us, down 52 -> 36 us
    BLOCK_N = 32
    BLOCK_K = 512
    NUM_WARPS = 2

    def __init__(self, orig):
        self.orig = orig
        self.taken = 0
        self.passed = 0

    def __getitem__(self, grid):
        def launch(*args, **kw):
            try:
                A, B, C, Bs, _bz, tw, sorted_ids, eids, ntpp, N, K, EM, nv = args[:13]
                (stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
                 stride_bse, stride_bsk, stride_bsn) = args[13:23]
                gs = kw["group_size"]
                # M == 1 <=> at most one valid token per expert block (the down call passes
                # top_k=1 with rows = tokens*topk, so nv/top_k does not give M; C is [M, topk, N])
                ok = (kw.get("use_int4_w4a16") and not kw.get("use_int4_interleave")
                      and not kw.get("has_zp") and B.dtype == torch.uint8
                      and C.dim() == 3 and C.shape[0] == 1
                      and stride_ak == 1 and K % gs == 0 and gs % 2 == 0 and gs <= 256)
            except Exception:
                ok = False
            if not ok:
                self.passed += 1
                if self.passed <= 2:   # say why, once: silent fall-through would hide a dead fast path
                    try:
                        logger.warning("moe_int4_gemv: pass-through #%d: B.dtype=%s B.shape=%s B_scale.shape=%s nv=%s "
                                       "top_k=%s stride_ak=%s K=%s group_size=%s int4=%s interleave=%s has_zp=%s",
                                       self.passed, args[1].dtype, tuple(args[1].shape), tuple(args[3].shape), args[12],
                                       kw.get("top_k"), args[14], args[10], kw.get("group_size"), kw.get("use_int4_w4a16"),
                                       kw.get("use_int4_interleave"), kw.get("has_zp"))
                    except Exception as e:
                        logger.warning("moe_int4_gemv: pass-through (could not describe args: %s)", e)
                return self.orig[grid](*args, **kw)
            self.taken += 1
            if self.taken == 1:
                logger.info("moe_int4_gemv: first take: B.shape=%s B_scale.shape=%s nv=%s N=%s K=%s", tuple(B.shape), tuple(Bs.shape), nv, N, K)
            BM = kw["BLOCK_SIZE_M"]
            BK = self.BLOCK_K if K % self.BLOCK_K == 0 and self.BLOCK_K % gs == 0 else gs
            g = (triton.cdiv(EM, BM), triton.cdiv(N, self.BLOCK_N))
            _moe_int4_gemv_m1[g](
                A, B, C, Bs, tw, sorted_ids, eids, ntpp, N, K, nv,
                stride_am, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
                stride_bse, stride_bsk, stride_bsn,
                GS=gs, BLOCK_M=BM, BLOCK_N=self.BLOCK_N, BLOCK_K=BK,
                MUL_ROUTED_WEIGHT=kw["MUL_ROUTED_WEIGHT"], top_k=kw["top_k"],
                compute_type=kw["compute_type"], num_warps=self.NUM_WARPS)
        return launch


# ---------------------------------------------------------------- 5. vision tower -> host ----
def _install_vision_offload():
    """BROKEN (2026-09-15): the vision GEMMs (hipBLASLt / wvSplitK) fault with
    HSA_STATUS_ERROR_MEMORY_FAULT on host-mapped weights; only the expert-cache gather kernel
    can read UVA views. Kept opt-in for reference.
    Move the (decode-idle) vision tower's weights to pinned host memory as UVA views after
    loading, the same way the expert offloader does. vLLM's --cpu-offload-params only reaches
    modules that go through make_layers(), so the 2.2 GiB tower (replicated on every TP rank)
    would otherwise stay in VRAM and push 2.2 GiB of experts out to the host instead. Image
    prefill reads these weights over PCIe -- slower, but functional (vision_probe.sh)."""
    import sys
    from vllm.model_executor.model_loader import utils as _lu
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    inner = _lu.process_weights_after_loading

    def process_weights_after_loading(model, model_config, target_device):
        inner(model, model_config, target_device)
        visual = getattr(model, "visual", None)
        if visual is None:
            logger.warning("vision offload: model has no .visual -- nothing moved")
            return
        keep, moved = [], 0
        for name, prm in visual.named_parameters():
            if prm.device.type != "cuda" or getattr(prm, "_vllm_is_uva_offloaded", False):
                continue
            cpu = torch.empty_like(prm.data, device="cpu", pin_memory=True).copy_(prm.data)
            prm.data = get_accelerator_view_from_cpu_tensor(cpu)
            prm._vllm_is_uva_offloaded = True
            keep.append(cpu); moved += cpu.numel() * cpu.element_size()
        visual._glm53_cpu_keepalive = keep
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        logger.info("vision offload: %.2f GiB of %d tensors moved to pinned host memory (UVA)",
                    moved / 2**30, len(keep))

    # base_loader imported the function by name; patch every module holding this reference
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "process_weights_after_loading", None) is inner:
            setattr(mod, "process_weights_after_loading", process_weights_after_loading)
    _lu.process_weights_after_loading = process_weights_after_loading


# ---------------------------------------------------------------- 6. shared-expert overlap ---
def _install_shared_overlap():
    """Run the shared expert on the aux stream, overlapped with the routed experts' gather.
    vLLM disables this on ROCm twice over (shared_experts.py: overlap_is_beneficial needs
    DP>1/TP=1; moe_runner.py:295 marks it unsafe for unquantized activations). Forced on here;
    correctness is checked by the prompt battery, not assumed."""
    from vllm import envs
    from vllm.model_executor.layers.fused_moe.runner import shared_experts as _se
    SE, Order = _se.SharedExperts, _se.SharedExpertsOrder
    orig = SE._determine_shared_experts_order

    def determine(self, hidden_states):
        r = orig(self, hidden_states)
        if (r == Order.NO_OVERLAP and self._stream is not None
                and not self._disable_shared_experts_overlap
                and hidden_states.shape[0] <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD):
            return Order.MULTI_STREAM_OVERLAPPED
        return r

    SE._determine_shared_experts_order = determine


# ---------------------------------------------------------------- 7. whole-model torch.compile
def install_compile():
    """Opt-in (GLM53_COMPILE=1). Give Glm5NextModel @support_torch_compile so inductor fuses the
    per-layer glue (norms, residual/mHC arithmetic, casts), which this build never compiles
    ("model does not support it"). Dynamo must not trace the pieces that read the forward
    context or run data-dependent kernels, so KDA attention, MLA attention and the MoE block
    are each wrapped as an opaque vLLM custom op that looks the layer up by key, exactly like
    vLLM's own linear_attention op does."""
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.compilation.decorators import support_torch_compile
    from vllm.forward_context import get_forward_context
    from vllm.config import get_current_vllm_config
    from . import kda as _kda, attention as _attn, model as _model

    _counts = {}

    def _register(self, tag):
        # deterministic across processes: the compiled graph is AOT-cached on disk with these
        # key strings baked in (id()-based keys -> KeyError on the next launch, 2026-09-15)
        n = _counts.get(tag, 0); _counts[tag] = n + 1
        # vLLM's extract_layer_index() wants dot-separated names with exactly one integer
        # segment ("glm53::kda::0" trips its assertion on the 53)
        key = f"glmfastpath.{tag}.{n}"
        get_current_vllm_config().compilation_config.static_forward_context[key] = self
        self._glm53_key = key

    def _wrap_init(cls, tag):
        orig_init = cls.__init__
        def __init__(self, *a, **kw):
            orig_init(self, *a, **kw)
            _register(self, tag)
        cls.__init__ = __init__

    def _layer(key):
        return get_forward_context().no_compile_layers[key]

    def kda_op(hidden_states: torch.Tensor, positions: torch.Tensor, key: str) -> torch.Tensor:
        return _layer(key)._glm53_eager_forward(hidden_states, positions)

    def mla_op(hidden_states: torch.Tensor, positions: torch.Tensor, key: str) -> torch.Tensor:
        return _layer(key)._glm53_eager_forward(hidden_states, positions)

    def moe_op(hidden_states: torch.Tensor, already_sequence_parallel: bool, key: str) -> torch.Tensor:
        return _layer(key)._glm53_eager_forward(hidden_states, already_sequence_parallel)

    def attn_fake(hidden_states: torch.Tensor, positions: torch.Tensor, key: str) -> torch.Tensor:
        return torch.empty_like(hidden_states)

    def moe_fake(hidden_states: torch.Tensor, already_sequence_parallel: bool, key: str) -> torch.Tensor:
        return torch.empty_like(hidden_states)

    direct_register_custom_op(op_name="glm53_kda", op_func=kda_op, fake_impl=attn_fake)
    direct_register_custom_op(op_name="glm53_mla", op_func=mla_op, fake_impl=attn_fake)
    direct_register_custom_op(op_name="glm53_moe", op_func=moe_op, fake_impl=moe_fake)

    for cls, tag, op in ((_kda.Glm5NextLinearAttention, "kda", torch.ops.vllm.glm53_kda),
                         (_attn.Glm5NextMLAAttention, "mla", torch.ops.vllm.glm53_mla)):
        cls._glm53_eager_forward = cls.forward
        _wrap_init(cls, tag)
        def fwd(self, hidden_states, positions, _op=op):
            return _op(hidden_states, positions, self._glm53_key)
        cls.forward = fwd

    _model.Glm5NextMoE._glm53_eager_forward = _model.Glm5NextMoE.forward
    _wrap_init(_model.Glm5NextMoE, "moe")
    def moe_fwd(self, hidden_states, already_sequence_parallel=False):
        return torch.ops.vllm.glm53_moe(hidden_states, bool(already_sequence_parallel), self._glm53_key)
    _model.Glm5NextMoE.forward = moe_fwd

    support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1,
                                            "intermediate_tensors": 0, "inputs_embeds": 0})(
        _model.Glm5NextModel)
    logger.info("glm53 compile: @support_torch_compile applied to Glm5NextModel; "
                "kda/mla/moe wrapped as vllm custom ops")


# (a cache hit-rate hook lived here and was WRONG: under expert parallelism the cache is
#  fed local ids with -1 clamped to 0, so topk_ids.numel() counts ~2x the experts this rank
#  owns and inflates the hit rate. Use the repo's own EXPERT_CACHE_STATS=<interval> instead,
#  which counts (owned_ids >= 0).)


# ---------------------------------------------------------------- 9. routing trace dump ----
def _install_trace_dump():
    """GLM53_TRACE_DUMP=<path>: record the EP-local expert ids each cache refresh sees, per
    layer, and write them as JSON once GLM53_TRACE_STEPS refreshes have gone by. Offline
    replay (rp/cache_sim.py) can then score any replacement policy -- including Belady's
    clairvoyant optimum, which bounds what ANY online policy could achieve on this trace.
    Eager only (it syncs per call); diagnostic, never on by default."""
    import json as _json
    from vllm_expert_cache.cache import ExpertSlotCache
    # both TP workers run this hook; without a per-rank suffix they race on one file
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        _rank = get_tensor_model_parallel_rank()
    except Exception:
        _rank = os.getpid()
    path = os.environ["GLM53_TRACE_DUMP"].replace(".json", f".rank{_rank}.json")
    limit = int(os.environ.get("GLM53_TRACE_STEPS", "12600"))
    orig = ExpertSlotCache.refresh
    trace, keys, st = {}, {}, {"n": 0, "done": False}

    def refresh(self, topk_ids):
        if not st["done"]:
            k = keys.setdefault(id(self), f"L{len(keys):02d}")
            trace.setdefault(k, []).append([int(v) for v in topk_ids.flatten().tolist()])
            st["n"] += 1
            if st["n"] >= limit:
                st["done"] = True
                with open(path, "w") as fh:
                    _json.dump(trace, fh)
                logger.info("glm53 trace: wrote %d refreshes over %d layers to %s",
                            st["n"], len(trace), path)
        return orig(self, topk_ids)

    ExpertSlotCache.refresh = refresh


# ---------------------------------------------------------------- 10. duplicate gate ------
def _install_no_double_gate():
    """The model computes router_logits and passes them to the MoE runner; the runner then
    calls the SAME gate module on the SAME hidden states and throws the passed value away
    (moe_runner.py: `if self.gate is not None: router_logits = self.gate(hidden_states)`).
    That is 42 redundant GEMVs per step -- 629 us in the trace across 84 calls where 42
    would do. Bypass the runner's copy whenever the caller already supplied logits of the
    right shape (the shape guard keeps sequence-parallel chunking honest)."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    orig = MoERunner._forward_impl

    def _forward_impl(self, hidden_states, router_logits, *a, **kw):
        if (self.gate is not None and router_logits is not None
                and router_logits.shape[0] == hidden_states.shape[0]):
            saved, self.gate = self.gate, None
            try:
                return orig(self, hidden_states, router_logits, *a, **kw)
            finally:
                self.gate = saved
        return orig(self, hidden_states, router_logits, *a, **kw)

    MoERunner._forward_impl = _forward_impl

# ---------------------------------------------------------------- install ---------------
def install() -> None:
    if not _ENABLED:
        logger.warning("glm53 fastpath: DISABLED by GLM53_FASTPATH=0")
        return
    installed = []

    # 1. router gate: only the M<=32 bf16 weight / fp32 out case; everything else untouched
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    _orig_gate_forward = GateLinear.forward

    def _gate_forward(self, x):
        if (x.dim() == 2 and x.shape[0] <= _GATE_MAX_M and x.dtype == torch.bfloat16
                and x.is_contiguous() and self.weight.is_contiguous()
                and self.weight.dtype == torch.bfloat16 and self.out_dtype == torch.float32
                and getattr(self, "bias", None) is None):
            return gate_gemv_fp32(x, self.weight), None
        return _orig_gate_forward(self, x)

    if "gate" in _PARTS:
        GateLinear.forward = _gate_forward
        installed.append("gate_gemv")

    # 2. top-k: GroupedTopk binds the module function at __init__, so patch the module global
    from vllm.model_executor.layers.fused_moe.router import grouped_topk_router as gtr
    _orig_grouped_topk = gtr.grouped_topk

    def _grouped_topk(hidden_states, gating_output, topk, renormalize, num_expert_group=0,
                      topk_group=0, scoring_func="softmax", routed_scaling_factor=1.0,
                      e_score_correction_bias=None):
        if (num_expert_group <= 1 and scoring_func == "sigmoid"
                and e_score_correction_bias is not None
                and gating_output.dim() == 2 and gating_output.shape[0] <= _TOPK_MAX_M
                and gating_output.is_contiguous()):
            return topk_sigmoid_bias(gating_output, e_score_correction_bias, topk,
                                     renormalize, routed_scaling_factor)
        return _orig_grouped_topk(hidden_states, gating_output, topk, renormalize,
                                  num_expert_group, topk_group, scoring_func,
                                  routed_scaling_factor, e_score_correction_bias)

    if "topk" in _PARTS:
        gtr.grouped_topk = _grouped_topk
        installed.append("fused_topk")

    # 3. clamped SwiGLU: ROCm routes SiluAndMulWithClamp to forward_native (7 kernels)
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    _orig_act_native = SiluAndMulWithClamp.forward_native
    _act_check = {"n": int(os.environ.get("GLM53_FASTPATH_ACT_CHECK", "0"))}

    def _act_forward(self, x):
        out = silu_mul_clamp(x, self.swiglu_limit, self.alpha, self.beta)
        if _act_check["n"] > 0:   # diagnostic: compare against the native path on the live inputs
            _act_check["n"] -= 1
            ref = _orig_act_native(self, x)
            d = (ref.float() - out.float()).abs()
            logger.info("act check: shape %s stride %s dtype %s contig %s | maxabs %.4g mean %.3g ref_absmax %.3g "
                        "| lim %s alpha %s beta %s | stream %s",
                        tuple(x.shape), tuple(x.stride()), x.dtype, x.is_contiguous(),
                        d.max().item(), d.mean().item(), ref.float().abs().max().item(),
                        self.swiglu_limit, self.alpha, self.beta, torch.cuda.current_stream())
        return out

    if "act" in _PARTS:
        SiluAndMulWithClamp.forward_native = _act_forward
        installed.append("silu_mul_clamp")

    # 4. int4 MoE GEMV for the M=1 decode shape (see _GptqAwqDispatch)
    if "layout" in _PARTS or "moe" in _PARTS:
        _install_moe_repack()            # keep uint8 K-packed experts (the GEMV needs it)
        installed.append("uint8_layout")
    if "moe" in _PARTS:
        from vllm.model_executor.layers.fused_moe import fused_moe as _fm
        _fm.fused_moe_kernel_gptq_awq = _GptqAwqDispatch(_fm.fused_moe_kernel_gptq_awq)
        installed.append("moe_int4_gemv")

    if "vision_offload" in _PARTS:
        _install_vision_offload()
        installed.append("vision_offload")
    if "overlap" in _PARTS:
        _install_shared_overlap()
        installed.append("shared_overlap")

    if os.environ.get("GLM53_TRACE_DUMP"):
        try:
            _install_trace_dump()
            installed.append("trace_dump")
        except Exception as e:
            logger.error("trace dump NOT installed: %s", e)

    if "nodoublegate" in _PARTS:
        _install_no_double_gate()
        installed.append("no_double_gate")

    logger.info("glm53 fastpath: installed %s", ", ".join(installed))
