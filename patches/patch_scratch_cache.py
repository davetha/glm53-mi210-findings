p = "/home/dave/vllm-expert-cache/vllm_expert_cache/cache.py"
s = open(p).read()
assert "class WideScratch" not in s, "already patched"

block = '''

class WideScratch:
    """Full-size VRAM copies of one layer's expert weights, for steps too wide to cache.

    A prefill chunk routes to far more distinct experts than there are slots, so
    `fits()` is false and the layer used to read through to the host copies. That is
    correct but slow in a specific, measurable way: the fused MoE GEMM tiles for HBM and
    revisits, and pointed at a device-addressable view of pinned host memory it gets
    ~6.8 GB/s, against the 21-25 GB/s expert_cache_gather_k sustains on the same PCIe
    link doing wide contiguous copies (measured 2026-09-16: 515 ms per offloaded layer
    per 2070-token prefill, 3.48 GiB moved).

    So for a wide step, copy first and compute second. The scratch holds all E local
    experts and is indexed BY LOCAL EXPERT ID, exactly as the source tensor is, which is
    what makes this safe: expert_map, global_num_experts and the routing ids all keep
    their existing meaning and nothing is remapped. It is a pure change of where the
    weights live for the duration of the call.

    ONE allocation is shared by every layer of the same geometry, because layers run in
    order on one stream: layer N's GEMM is enqueued before layer N+1's fill, so the
    refill cannot overtake a read. At 144 experts x 12.38 MiB that is 1.74 GiB once
    instead of 1.74 GiB per layer.
    """

    _shared: dict = {}

    @classmethod
    def acquire(cls, cache, device):
        """The scratch for this layer's geometry, allocating it on first request."""
        key = (cache.E, tuple((n, tuple(cache.slots[n].shape[1:]), cache.slots[n].dtype)
                              for n in cache.names))
        inst = cls._shared.get(key)
        if inst is None:
            inst = cls(cache, device)
            cls._shared[key] = inst
        return inst

    def __init__(self, cache, device):
        self.names = list(cache.names)
        self.E = int(cache.E)
        self.buf = {
            n: torch.empty((self.E,) + tuple(cache.slots[n].shape[1:]),
                           dtype=cache.slots[n].dtype, device=device)
            for n in self.names
        }
        # The gather kernel reads its work list as (expert, slot) PAIRS
        # (expert_cache.hip: e = miss[2*j], sl = miss[2*j+1]). Repeating each index
        # twice gives [0,0, 1,1, 2,2, ...] -- the identity, so expert e lands at row e.
        self.pairs = torch.arange(self.E, dtype=torch.int32,
                                  device=device).repeat_interleave(2)
        self.n = torch.full((1,), self.E, dtype=torch.int32, device=device)
        self.bound: dict[str, torch.Tensor] = {}

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buf.values())

    def fill(self, owner, chunks: int, lanes: int) -> None:
        """Copy every local expert from wherever it lives into the scratch.

        Sources are read off `owner` at call time, not captured: the offloader moved
        these tensors to host memory after load, and the caller invokes this BEFORE
        rebinding the layer, so these are the originals rather than the scratch itself.
        """
        h = lib()
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        dsts = [self.buf[n] for n in self.names]
        srcs = [getattr(owner, n).data for n in self.names]

        # Same fixed-arity split-and-pad as the slot gather: the kernel copies exactly
        # GATHER_SLOTS buffers per launch, so a short final group repeats one it already
        # carries rather than passing unreadable memory.
        for i in range(0, len(dsts), GATHER_SLOTS):
            gd = dsts[i:i + GATHER_SLOTS]
            gs = srcs[i:i + GATHER_SLOTS]
            while len(gd) < GATHER_SLOTS:
                gd.append(gd[-1])
                gs.append(gs[-1])
            args = []
            for d, s in zip(gd, gs):
                args += [ctypes.c_void_p(d.data_ptr()), ctypes.c_void_p(s.data_ptr()),
                         ctypes.c_long(_bytes_per_expert(s))]
            rc = h.expert_cache_gather(
                *args,
                ctypes.c_void_p(self.pairs.data_ptr()),
                ctypes.c_void_p(self.n.data_ptr()),
                chunks, lanes, stream,
            )
            if rc != 0:
                raise RuntimeError(f"expert_cache_gather (wide scratch) failed rc={rc}")
'''

anchor = "\nclass ExpertSlotCache"
assert anchor in s
s = s.replace(anchor, block + anchor, 1)

import ast
ast.parse(s)
open(p, "w").write(s)
print("cache.py: WideScratch added")
