import json, re, sys, urllib.request

def get_json(u):
    return json.load(urllib.request.urlopen(u, timeout=90))

def get_text(u):
    return urllib.request.urlopen(u, timeout=90).read().decode("utf-8", "replace")

for repo in sys.argv[1:]:
    print("==", repo)
    try:
        d = get_json(f"https://huggingface.co/api/models/{repo}")
        cd = d.get("cardData") or {}
        print("   base_model:", cd.get("base_model"))
        print("   lastModified:", (d.get("lastModified") or "")[:10],
              "| created:", (d.get("createdAt") or "")[:10])
        print("   tags:", [t for t in (d.get("tags") or []) if "base_model" in t or "0731" in t][:6])
    except Exception as e:
        print("   api failed:", type(e).__name__)
    try:
        rd = get_text(f"https://huggingface.co/{repo}/raw/main/README.md")
        hits = sorted(set(re.findall(r"[Dd]eep[Ss]eek[-\w.]*V4[-\w.]*", rd)))
        print("   names in README:", hits[:8])
        dates = sorted(set(re.findall(r"\b(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\b", rd)))
        print("   date-like tokens:", dates[:8])
    except Exception as e:
        print("   README failed:", type(e).__name__)
    print()
