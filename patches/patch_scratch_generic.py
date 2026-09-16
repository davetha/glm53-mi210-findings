p = "/home/dave/vllm-expert-cache/vllm_expert_cache/backends/generic.py"
s = open(p).read()
assert "WideScratch" not in s, "already patched"

s = s.replace("from ..cache import ExpertSlotCache",
              "from ..cache import ExpertSlotCache, WideScratch", 1)

# 1. a binding context manager that takes any mapping, so scratch and slots share it
old_ctx = '''@contextlib.contextmanager
def _slot_bound(layer, cache: ExpertSlotCache):
    """Point every mirrored attribute at its slot buffer for the duration of the block."""
    saved = {name: getattr(layer, name) for name in cache.bound}
    try:
        for name, slot in cache.bound.items():
            _set_tensor(layer, name, slot)
        yield
    finally:
        for name, t in saved.items():
            _set_tensor(layer, name, t)'''
new_ctx = '''@contextlib.contextmanager
def _bound_to(layer, mapping):
    """Point every mirrored attribute at the given replacement for the duration."""
    saved = {name: getattr(layer, name) for name in mapping}
    try:
        for name, buf in mapping.items():
            _set_tensor(layer, name, buf)
        yield
    finally:
        for name, t in saved.items():
            _set_tensor(layer, name, t)


def _slot_bound(layer, cache: ExpertSlotCache):
    """Point every mirrored attribute at its slot buffer for the duration of the block."""
    return _bound_to(layer, cache.bound)'''
assert old_ctx in s
s = s.replace(old_ctx, new_ctx, 1)

# 2. build the scratch + a quant config bound to it, at arm time
old_cfg = '''    with _slot_bound(layer, cache):
        slot_cfg = method.get_fused_moe_quant_config(layer)
'''
new_cfg = '''    with _slot_bound(layer, cache):
        slot_cfg = method.get_fused_moe_quant_config(layer)

    # Wide (prefill) steps. The scratch is full-size and indexed by local expert id, so
    # unlike the slot path NOTHING here is remapped -- which is also why the leak check
    # below must not run against it: every per-expert tensor staying at [E] is correct.
    scratch = wide_cfg = None
    if settings.wide_scratch:
        try:
            scratch = WideScratch.acquire(cache, cache.table.device)
            if not scratch.bound:
                for first, group in sources.items():
                    if first not in scratch.buf:
                        continue
                    buf = scratch.buf[first]
                    as_param = torch.nn.Parameter(buf, requires_grad=False)
                    for name in group:
                        scratch.bound[name] = (
                            as_param if name in layer._parameters else buf)
            with _bound_to(layer, scratch.bound):
                wide_cfg = method.get_fused_moe_quant_config(layer)
        except Exception as e:
            # Disclosed, not silent: without this the layer still computes correctly,
            # it just reads the host copies through the GEMM the slow way.
            logger.warning(
                "expert-cache: wide-step scratch unavailable (%r); prefill will read "
                "through to host memory. Set EXPERT_CACHE_WIDE_SCRATCH=0 to silence.", e)
            scratch = wide_cfg = None
'''
assert old_cfg in s
s = s.replace(old_cfg, new_cfg, 1)

# 3. use it in the bypass branch
old_bypass = '''        if not cache.fits(topk_ids):
            # Wide / prefill steps touch more distinct experts than there are slots and
            # read through the host copies on the stock path.
            return orig_apply(layer, x, topk_weights, topk_ids,
                              shared_experts, shared_experts_input)'''
new_bypass = '''        if not cache.fits(topk_ids):
            # Wide / prefill steps touch more distinct experts than there are slots, so
            # they cannot be served from the slot table. Stage the whole local expert set
            # into VRAM with the gather kernel and compute from there: same bytes over
            # the same link, but ~24 GB/s of wide contiguous copies instead of the GEMM
            # picking at host memory at ~7 GB/s. Falls back to the stock read-through
            # path when no scratch was allocated.
            if scratch is None:
                return orig_apply(layer, x, topk_weights, topk_ids,
                                  shared_experts, shared_experts_input)
            scratch.fill(layer, cache.chunks, cache.lanes)
            w_saved = (experts_ref.quant_config, method.moe_quant_config)
            if wide_cfg is not None:
                experts_ref.quant_config = wide_cfg
                method.moe_quant_config = wide_cfg
            try:
                with _bound_to(layer, scratch.bound):
                    return orig_apply(layer, x, topk_weights, topk_ids,
                                      shared_experts, shared_experts_input)
            finally:
                experts_ref.quant_config, method.moe_quant_config = w_saved'''
assert old_bypass in s
s = s.replace(old_bypass, new_bypass, 1)

# the bypass branch runs before `experts` is bound in the body, so hoist the reference
s = s.replace("    orig_apply = method.apply\n",
              "    orig_apply = method.apply\n    experts_ref = method.moe_kernel.fused_experts\n", 1)

import ast
ast.parse(s)
open(p, "w").write(s)
print("generic.py: wide-step scratch wired into the bypass branch")
