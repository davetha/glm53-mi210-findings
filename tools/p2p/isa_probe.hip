// What does the compiler ACTUALLY emit for each spin-wait load on gfx90a?
//
// The whole custom-kernel question turns on this. A system-scope RELAXED load that
// sets glc+slc bypasses L1/L2 and reads the peer's PCIe write directly -- cheap,
// correct, and needing no cache maintenance at all. A plain cached load can be
// served from a stale L2 line indefinitely, which is both why an ACQUIRE was needed
// for correctness and the likely source of the 131 ms outlier.
//
// If RELAXED/SYSTEM already emits glc slc, the spin is already uncached and there is
// no cache trick left to find -- the ~68 us/barrier is real PCIe round-trip and a
// custom kernel cannot beat it. If it emits a plain load, the current patch is
// relying on the post-loop fence for visibility and a glc/slc spin is strictly better.
//
// Disassembly answers this; no GPU required.
#include <hip/hip_runtime.h>

extern "C" __global__ void k_relaxed_sys(unsigned int* p, unsigned int* o, unsigned int w) {
  unsigned int v;
  do { v = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); } while (v < w);
  *o = v;
}

extern "C" __global__ void k_acquire_sys(unsigned int* p, unsigned int* o, unsigned int w) {
  unsigned int v;
  do { v = __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); } while (v < w);
  *o = v;
}

extern "C" __global__ void k_relaxed_agent(unsigned int* p, unsigned int* o, unsigned int w) {
  unsigned int v;
  do { v = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); } while (v < w);
  *o = v;
}

extern "C" __global__ void k_plain_volatile(unsigned int* p, unsigned int* o, unsigned int w) {
  volatile unsigned int* q = p;
  unsigned int v;
  do { v = *q; } while (v < w);
  *o = v;
}

// The loop the current patch runs: relaxed poll, then ONE system-scope acquire fence.
// Shows what that fence costs in cache maintenance instructions.
extern "C" __global__ void k_patch_shape(unsigned int* p, unsigned int* o, unsigned int w) {
  unsigned int v;
  do { v = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); } while (v < w);
  __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
  *o = v;
}
