"""Host-staged two-rank all-reduce for vLLM on PCIe-only GPUs (no XGMI / no NVLink).

WHY. On 2x MI210 (gfx90a) over PCIe, NCCL costs 116 us per 8 KB all-reduce -- 14.45 ms of a
70 ms decode step, and 370x off the link's bandwidth, so it is pure protocol latency. A
minimal purpose-built exchange measures 6.73 us per launch in the same two-process shape,
dispatch included. This routes small decode all-reduces through that and leaves everything
else to NCCL.

The barrier lives in POSIX shared memory because a peer-VRAM handshake does not work
between these cards: writes land once a kernel ends but a concurrently-spinning kernel
stalls intermittently (six memory-ordering combinations tested; ACS redirect disabled makes
no difference; reported upstream as NCCL #2079 and ROCm #5480). Host memory is the
coherence point both GPUs reach, and is what NCCL's own SHM transport uses.

Enable:   VLLM_HOSTAR=1
Tune:     VLLM_HOSTAR_MAX_BYTES (default 65536), VLLM_HOSTAR_SPIN_SEC (default 10)
Disable:  unset VLLM_HOSTAR  -- every path falls back to stock vLLM.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_HIP_HOST_REGISTER_MAPPED = 0x02
_DTYPE_CODE = {torch.bfloat16: 0, torch.float32: 1}

_SEARCH = (
    lambda: os.environ.get("HOSTAR_LIB"),
    lambda: str(Path(__file__).parent / "libhostar.so"),
    lambda: "/w/libhostar.so",
)


def _lib_path() -> str:
    for probe in _SEARCH:
        p = probe()
        if p and Path(p).is_file():
            return p
    raise FileNotFoundError("libhostar.so not found; set HOSTAR_LIB")


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        h = ctypes.CDLL(_lib_path())
        h.hostar_allreduce.restype = ctypes.c_int
        h.hostar_allreduce.argtypes = [ctypes.c_void_p] * 7 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_void_p
        ]
        h.hostar_reset.restype = ctypes.c_int
        h.hostar_reset.argtypes = [ctypes.c_void_p] * 4
        _LIB = h
    return _LIB


_HIP = None


def _hip():
    global _HIP
    if _HIP is None:
        _HIP = ctypes.CDLL("libamdhip64.so")
        _HIP.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        _HIP.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    return _HIP


class HostStagedAllReduce:
    """One shared arena per TP group. Layout: [slot0][slot1][flag0 128B][flag1 128B]."""

    def __init__(self, rank: int, world: int, key: str, max_bytes: int, spin_sec: float):
        assert world == 2, "host-staged all-reduce is 2-rank only"
        self.rank, self.world, self.max_bytes = rank, world, max_bytes
        self.path = f"/dev/shm/vllm_hostar_{key}"
        arena = 2 * max_bytes + 256

        # Rank 0 creates and zeroes; the barrier below orders attach after create so rank 1
        # cannot map a half-built or stale segment.
        if rank == 0:
            if os.path.exists(self.path):
                os.unlink(self.path)
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(fd, arena)
            os.write(fd, b"\0" * arena)
            os.close(fd)
        torch.distributed.barrier()

        fd = os.open(self.path, os.O_RDWR)
        self._mm = mmap.mmap(fd, arena, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        host_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))

        rc = _hip().hipHostRegister(ctypes.c_void_p(host_addr), arena,
                                    _HIP_HOST_REGISTER_MAPPED)
        if rc != 0:
            raise RuntimeError(f"hipHostRegister on shm failed: rc={rc}")
        dev = ctypes.c_void_p()
        rc = _hip().hipHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(host_addr), 0)
        if rc != 0:
            raise RuntimeError(f"hipHostGetDevicePointer failed: rc={rc}")
        if not dev.value:
            raise RuntimeError("hipHostGetDevicePointer returned NULL for the shm arena")
        base = int(dev.value)

        self.slot_me = base + rank * max_bytes
        self.slot_peer = base + (1 - rank) * max_bytes
        self.flag_me = base + 2 * max_bytes + rank * 128
        self.flag_peer = base + 2 * max_bytes + (1 - rank) * 128

        # Sequence counter and status live in DEVICE memory: under CUDA graphs the kernel
        # arguments are baked at capture, so a host-side counter would replay one value.
        self._seq = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._status = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.spin_ticks = int(spin_sec * 100e6)   # s_memrealtime is ~100 MHz on gfx90a

        _lib().hostar_reset(ctypes.c_void_p(int(self._seq.data_ptr())),
                            ctypes.c_void_p(self.flag_me),
                            ctypes.c_void_p(self.flag_peer), None)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        msg = (f"hostar: arena {self.path}, {max_bytes} B/slot, "
               f"rank {rank}/{world}, spin {spin_sec:.1f}s")
        logger.info(msg)
        print(msg, flush=True)

    _hist: dict = {}
    _n = 0

    def should_use(self, t: torch.Tensor) -> bool:
        nb = t.numel() * t.element_size()
        why = []
        if t.dtype not in _DTYPE_CODE:
            why.append(f"dtype={t.dtype}")
        if not t.is_contiguous():
            why.append("noncontig")
        if nb > self.max_bytes:
            why.append("toobig")
        if t.data_ptr() % 16:
            why.append("misaligned")
        # Histogram every decision rather than the first N: the first calls are all from
        # the startup profiling run, so a small cap never shows what DECODE passes.
        cls = HostStagedAllReduce
        key = (tuple(t.shape), nb, ",".join(why) or "USE")
        cls._hist[key] = cls._hist.get(key, 0) + 1
        cls._n += 1
        if cls._n in (50, 500, 5000, 20000):
            print(f"hostar: after {cls._n} decisions:", flush=True)
            for (sh, b, d), c in sorted(cls._hist.items(), key=lambda kv: -kv[1])[:8]:
                print(f"hostar:   {c:>7}x shape={sh} nbytes={b} -> {d}", flush=True)
        return not why

    _used = 0

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        HostStagedAllReduce._used += 1
        if HostStagedAllReduce._used in (1, 100, 1000):
            print(f"hostar: USED {HostStagedAllReduce._used} times "
                  f"(latest shape {tuple(t.shape)})", flush=True)
        out = torch.empty_like(t)
        rc = _lib().hostar_allreduce(
            ctypes.c_void_p(int(t.data_ptr())), ctypes.c_void_p(int(out.data_ptr())),
            ctypes.c_void_p(self.slot_me), ctypes.c_void_p(self.slot_peer),
            ctypes.c_void_p(self.flag_me), ctypes.c_void_p(self.flag_peer),
            ctypes.c_void_p(int(self._seq.data_ptr())),
            t.numel(), _DTYPE_CODE[t.dtype], self.spin_ticks,
            ctypes.c_void_p(int(self._status.data_ptr())),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"hostar_allreduce launch failed rc={rc}")
        return out

    def close(self):
        try:
            self._mm.close()
        finally:
            if self.rank == 0 and os.path.exists(self.path):
                os.unlink(self.path)


def install() -> bool:
    """Patch CudaCommunicator.all_reduce to try the host-staged path first."""
    if os.environ.get("VLLM_HOSTAR", "0") == "0":
        return False
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    if getattr(CudaCommunicator, "_hostar_installed", False):
        return True
    orig = CudaCommunicator.all_reduce
    max_bytes = int(os.environ.get("VLLM_HOSTAR_MAX_BYTES", 65536))
    spin_sec = float(os.environ.get("VLLM_HOSTAR_SPIN_SEC", 10))

    def all_reduce(self, input_):
        ar = getattr(self, "_hostar", None)
        if ar is None and not getattr(self, "_hostar_failed", False):
            # Only the TP group; other groups keep stock behaviour.
            if self.world_size == 2 and "tp" in (self.unique_name or ""):
                try:
                    ar = HostStagedAllReduce(
                        self.rank_in_group, self.world_size,
                        f"{self.unique_name}_{os.environ.get('VLLM_HOSTAR_KEY', 'g0')}",
                        max_bytes, spin_sec)
                    self._hostar = ar
                except Exception as e:                       # noqa: BLE001
                    import traceback
                    print(f"hostar: DISABLED ({e!r}); using stock all-reduce", flush=True)
                    traceback.print_exc()
                    logger.warning("hostar: disabled (%s); using stock all-reduce", e)
                    self._hostar_failed = True
            else:
                self._hostar_failed = True
        if ar is not None and ar.should_use(input_):
            return ar.all_reduce(input_)
        return orig(self, input_)

    CudaCommunicator.all_reduce = all_reduce
    CudaCommunicator._hostar_installed = True
    msg = f"hostar: installed (max_bytes={max_bytes}, spin={spin_sec:.1f}s)"
    logger.info(msg)
    print(msg, flush=True)
    return True
