"""Does the wide-step scratch change what the model generates? It must not.

The scratch changes only WHERE expert weights live during wide (prefill) steps, so at
temperature 0 the generated tokens must be byte-identical with it on and off. A wrong
expert index would not crash -- it would emit fluent text computed from the wrong
weights -- so comparing the text is the check that actually catches it.

Two traps this avoids, both hit on the first attempt:
  * this model returns its tokens under `reasoning` until the reasoning block closes,
    so a small max_tokens yields EMPTY content and two empty strings compare equal.
    Every sample is therefore asserted non-empty before any comparison is believed.
  * `fits()` is true only while tokens x top_k <= SLOTS, i.e. under 6 tokens here, so
    even a short prompt takes the wide path -- but a multi-chunk prompt is included
    too, since that is where the scratch is refilled most often.
"""
import json
import subprocess
import sys
import urllib.request

PORT = 8145
LONG = ("Consider the following technical notes on distributed storage. " * 90).strip()
PROMPTS = [
    ("short", "What is 17 times 23? Reply with just the number."),
    ("medium", "List the first eight prime numbers, comma separated, nothing else."),
    ("long", LONG + "\n\nIn one sentence, what topic do these notes concern?"),
]
MAX_TOKENS = 1500


def ask(prompt):
    body = json.dumps({
        "model": "glm53",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS, "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    text = (m.get("reasoning_content") or m.get("reasoning") or "") + "\n<<>>\n" + (m.get("content") or "")
    return text, d["usage"]["completion_tokens"], d["choices"][0].get("finish_reason")


arm = sys.argv[1]
out = {}
for name, p in PROMPTS:
    text, tok, fin = ask(p)
    body = text.replace("\n<<>>\n", "").strip()
    if not body:
        sys.exit(f"FAIL: {name} produced NO text ({tok} tokens, finish={fin}). "
                 "Comparing empty strings would pass vacuously; refusing to.")
    out[name] = text
    print(f"  {name:<7} {len(body):>6} chars, {tok:>4} tokens, finish={fin}")
json.dump(out, open(f"/tmp/equiv_{arm}.json", "w"))
