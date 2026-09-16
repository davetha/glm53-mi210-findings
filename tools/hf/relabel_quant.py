"""Relabel an AutoRound checkpoint so vLLM will load it.

vLLM registers 30 quantization methods and `auto-round` is not among them, so the
checkpoint is refused on the name alone. But its own config says
`packing_format: "auto_round:auto_gptq"` -- the tensors are packed exactly the way
auto-gptq packs them, which vLLM does support. So this is a labelling problem, not a
format problem, and the fix is to present the config as gptq.

Keeps a backup, and prints the layers AutoRound left at 16 bits, because GPTQ has no
per-layer exclusion list the way compressed-tensors does: if vLLM's GPTQ path insists
every target carry a `qweight`, those unquantized layers are where it will fail, and
that is worth seeing before the error rather than after.
"""
import json
import shutil
import sys

path = sys.argv[1] + "/config.json"
shutil.copy(path, path + ".autoround.bak")
c = json.load(open(path))
q = c.get("quantization_config") or {}

if q.get("quant_method") != "auto-round":
    sys.exit("not an auto-round config (found %r); nothing to do" % q.get("quant_method"))

extra = q.get("extra_config") or {}
kept16 = sorted(k for k, v in extra.items() if v.get("bits") == 16)
print("AutoRound kept %d entries at 16 bits, e.g.:" % len(kept16))
for k in kept16[:6]:
    print("   ", k)
if len(kept16) > 6:
    print("    ... and %d more" % (len(kept16) - 6))

c["quantization_config"] = {
    "quant_method": "gptq",
    "bits": q.get("bits", 4),
    "group_size": q.get("group_size", 128),
    "desc_act": False,
    "sym": q.get("sym", True),
}
json.dump(c, open(path, "w"), indent=2)
print("\nrelabelled auto-round -> gptq (bits=%s group_size=%s sym=%s)"
      % (q.get("bits"), q.get("group_size"), q.get("sym")))
print("backup at config.json.autoround.bak")
