# hostar — host-staged two-rank all-reduce for PCIe-only GPUs

For GPU pairs with **no XGMI / no NVLink**, where a peer-VRAM flag handshake does not work.
Measured on 2× MI210 (gfx90a): replaces NCCL for small decode all-reduces and saves
**~3.5 ms/step (~5%)**. See `docs/HOSTAR-INTEGRATION.md` for the full measurement and the
five traps that produced wrong answers along the way.

## Why

A GPU-to-GPU handshake through peer VRAM stalls intermittently between these cards — writes
land once a kernel ends, but a concurrently-spinning kernel never observes them. Six
store/load memory-ordering combinations all fail; disabling PCIe ACS redirect does not help.
The same handshake through **pinned host memory** runs 2000 round trips clean at 4.5 µs.
That is also what NCCL does: its SHM transport keeps head/tail counters in host memory.

## Install

```bash
./build.sh gfx90a                      # -> libhostar.so
# put hostar.py + libhostar.so on the Python path, and register the entry point:
#   <site-packages>/hostar-0.1.0.dist-info/{METADATA,entry_points.txt}
VLLM_HOSTAR=1 vllm serve ...
```

`entry_points.txt` registers `hostar:install` under `vllm.general_plugins`, which vLLM
calls from `worker_base.py` **inside each TP worker** — the workers are separate processes,
so a parent-side import does not reach them.

## Knobs

| var | default | meaning |
|---|---|---|
| `VLLM_HOSTAR` | `0` | `1` enables; anything else falls back entirely to NCCL |
| `VLLM_HOSTAR_MAX_BYTES` | `65536` | above this, NCCL handles it |
| `VLLM_HOSTAR_SPIN_SEC` | `10` | spin budget before the kernel traps |

## Scope and limits

* **2 ranks only.** The protocol is a two-slot exchange.
* bf16 and fp32.
* Saves the *protocol* cost. The remaining all-reduce time is one rank waiting for the
  other (p50 13.6 µs, p90 405 µs) — expert parallelism gives the ranks different gather
  work per layer, so they arrive at each barrier at different times. That is load
  imbalance, not communication, and no barrier change can recover it.
