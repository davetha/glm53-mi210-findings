p = "/home/dave/vllm-expert-cache/vllm_expert_cache/backends/generic.py"
s = open(p).read()
assert "_is_offloaded" not in s, "already patched"

helper = '''def _is_offloaded(layer, *names) -> bool:
    """True if these expert weights actually live in host memory.

    Caching a layer whose weights never left the GPU is worse than pointless: the slot
    buffers cost real VRAM (12.38 MiB per slot per layer on GLM-5.3-Flash) and every step
    still runs a manager and a gather that copy VRAM to VRAM for a lookup that cannot
    miss. Measured on a 42-layer run with a 28 GiB offload budget, 25 of 42 layers were
    fully resident and held 2.45 GiB/rank of slots they could never use.

    vLLM's UVA offloader leaves `.device` reading as the accelerator, because an offloaded
    parameter becomes a device-addressable VIEW of pinned host memory. Its own marker is
    the only reliable signal, so read that, and consult the device only for the non-UVA
    path, which really does move the tensor to CPU.
    """
    for name in names:
        t = getattr(layer, name, None)
        if t is None:
            continue
        if getattr(t, "_vllm_is_uva_offloaded", False) or t.device.type == "cpu":
            return True
    return False


'''
s = s.replace("def _weight_names(layer)", helper + "def _weight_names(layer)", 1)

old = """    num_experts = getattr(layer, w13_name).size(0)
    slots = settings.slots_for(num_experts)"""
new = """    if not _is_offloaded(layer, w13_name, w2_name):
        # Not an error, and not silent: which layers the offloader reached depends on
        # --cpu-offload-gb, so this line is how you see that the budget stopped short.
        logger.info(
            "expert-cache: not caching %s -- its experts are still GPU-resident, so a "
            "cache would only spend VRAM and copy weights to themselves. Raise "
            "--cpu-offload-gb to put this layer on the host.", type(method).__name__)
        return False

    num_experts = getattr(layer, w13_name).size(0)
    slots = settings.slots_for(num_experts)"""
assert old in s
s = s.replace(old, new, 1)

import ast
ast.parse(s)
open(p, "w").write(s)
print("resident-layer decline added")
