p = "/home/dave/vllm-expert-cache/vllm_expert_cache/backends/generic.py"
s = open(p).read()
assert 'EXPERT_CACHE_DECLINE_RESIDENT' not in s, "already gated"
old = "    if not _is_offloaded(layer, w13_name, w2_name):"
new = ("    if (os.environ.get(\"EXPERT_CACHE_DECLINE_RESIDENT\", \"1\") != \"0\"\n"
       "            and not _is_offloaded(layer, w13_name, w2_name)):")
assert old in s
s = s.replace(old, new, 1)
import ast
ast.parse(s)
open(p, "w").write(s)
print("decline now gated by EXPERT_CACHE_DECLINE_RESIDENT (default on)")
