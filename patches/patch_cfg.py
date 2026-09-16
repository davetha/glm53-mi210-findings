p = "/home/dave/vllm-expert-cache/vllm_expert_cache/config.py"
s = open(p).read()
assert "wide_scratch" not in s, "already patched"
old = '''    @property
    def decay(self) -> int:'''
new = '''    @property
    def wide_scratch(self) -> bool:
        """Stage all local experts into VRAM for steps too wide to serve from slots.

        Costs one full-size buffer per layer geometry (E x per-expert bytes, shared by
        every layer of that shape), and buys the difference between the GEMM reading
        host memory at ~7 GB/s and the gather kernel streaming it at ~24 GB/s.
        """
        return os.environ.get("EXPERT_CACHE_WIDE_SCRATCH", "1") != "0"

    @property
    def decay(self) -> int:'''
assert old in s
s = s.replace(old, new, 1)
import ast
ast.parse(s)
open(p, "w").write(s)
print("config.py: EXPERT_CACHE_WIDE_SCRATCH added (default on)")
