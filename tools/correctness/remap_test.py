"""The fused remap kernels must agree with the torch expressions they replace, exactly.

These run inside the MoE routing path: a wrong local id sends a token to the wrong
expert, and a wrong composed entry sends it to the wrong SLOT, which is silent -- the
output is plausible text computed from the wrong weights. So this checks equality on the
real shapes plus the edge cases that matter (every id not owned, every id owned, a table
with empty slots), not just a random draw.
"""
import sys
import torch

sys.path.insert(0, "/opt/vllm-expert-cache")
from vllm_expert_cache import fused_remap

torch.manual_seed(0)
E_GLOBAL, E_LOCAL, SLOTS, TOPK = 288, 144, 8, 8
dev = "cuda"
fails = 0


def check(label, emap, ids, table):
    global fails
    want_local = emap[ids.to(torch.int64)]
    got_local = fused_remap.gather_local(emap, ids, {})
    want_comp = torch.where(emap < 0,
                            torch.full_like(emap, -1),
                            table[emap.clamp(min=0).to(torch.int64)])
    got_comp = fused_remap.compose(table, emap, torch.empty_like(emap))
    torch.cuda.synchronize()
    ok_l = torch.equal(got_local, want_local)
    ok_c = torch.equal(got_comp, want_comp)
    if not (ok_l and ok_c):
        fails += 1
    print("  {:<28} local {}  composed {}".format(
        label, "ok" if ok_l else "MISMATCH", "ok" if ok_c else "MISMATCH"))
    if not ok_l:
        i = (got_local != want_local).nonzero()[0]
        print("    local[{}]: got {} want {}".format(
            i.tolist(), got_local[tuple(i)].item(), want_local[tuple(i)].item()))
    if not ok_c:
        i = int((got_comp != want_comp).nonzero()[0])
        print("    composed[{}]: got {} want {}".format(
            i, int(got_comp[i]), int(want_comp[i])))


def emap_half():
    m = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=dev)
    own = torch.randperm(E_GLOBAL, device=dev)[:E_LOCAL]
    m[own] = torch.arange(E_LOCAL, dtype=torch.int32, device=dev)
    return m


def table_partial():
    t = torch.full((E_LOCAL,), -1, dtype=torch.int32, device=dev)
    hot = torch.randperm(E_LOCAL, device=dev)[:SLOTS]
    t[hot] = torch.arange(SLOTS, dtype=torch.int32, device=dev)
    return t


ids1 = torch.randint(0, E_GLOBAL, (1, TOPK), dtype=torch.int32, device=dev)
check("decode M=1, half owned", emap_half(), ids1, table_partial())
check("prefill M=64", emap_half(),
      torch.randint(0, E_GLOBAL, (64, TOPK), dtype=torch.int32, device=dev), table_partial())
check("nothing owned",
      torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=dev), ids1, table_partial())
check("all owned, empty table",
      torch.arange(E_GLOBAL, dtype=torch.int32, device=dev)[:E_GLOBAL],
      ids1, torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=dev))
check("table fully resident", emap_half(), ids1,
      torch.arange(E_LOCAL, dtype=torch.int32, device=dev) % SLOTS)

print("FAIL" if fails else "PASS: fused remap matches the torch path on every case")
sys.exit(1 if fails else 0)
