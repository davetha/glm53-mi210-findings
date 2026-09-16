import json, sys, urllib.request

def get(url):
    return json.load(urllib.request.urlopen(url, timeout=90))

for repo in sys.argv[1:]:
    print(repo)
    try:
        d = get(f"https://huggingface.co/api/models/{repo}?blobs=true")
        sib = d.get("siblings") or []
        wt = sum(s.get("size") or 0 for s in sib
                 if s.get("rfilename", "").endswith((".safetensors", ".bin")))
        shards = sum(1 for s in sib if s.get("rfilename", "").endswith(".safetensors"))
        print("    weights %.1f GB over %d shards" % (wt / 1e9, shards))
    except Exception as e:
        print("    listing failed: %s" % type(e).__name__)
    try:
        c = get(f"https://huggingface.co/{repo}/raw/main/config.json")
        q = c.get("quantization_config") or {}
        g = (q.get("config_groups") or {}).get("group_0", {}).get("weights", {})
        print("    quant: method=%s format=%s bits=%s type=%s group=%s" % (
            q.get("quant_method"), q.get("format"),
            g.get("num_bits"), g.get("type"), g.get("group_size")))
        t = c.get("text_config", c)
        print("    arch=%s vision=%s layers=%s experts=%s topk=%s" % (
            c.get("architectures"), bool(c.get("vision_config")),
            t.get("num_hidden_layers"), t.get("n_routed_experts") or t.get("num_experts"),
            t.get("num_experts_per_tok")))
    except Exception as e:
        print("    config unreadable: %s" % type(e).__name__)
    print()
