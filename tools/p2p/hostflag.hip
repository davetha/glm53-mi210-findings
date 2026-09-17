// Can two GPUs handshake through PINNED HOST MEMORY when they cannot handshake through
// each other's VRAM?
//
// ESTABLISHED SO FAR: peer VRAM flags do not work between these two MI210s. A peer write
// lands once the writing kernel ENDS, but a concurrently-spinning kernel on the other GPU
// stalls -- intermittently, after 11 or 390 round trips -- in all six store/load
// memory-ordering combinations (glc+slc / glc / acquire x glc+slc / release).
//
// WHY HOST MEMORY MIGHT WORK WHERE PEER VRAM DOES NOT:
//   * System memory is a common coherence point BOTH GPUs can reach, rather than one GPU
//     reaching into the other's device memory across two root complexes.
//   * ACS on this box is ReqRedir+ / CmpltRedir+ / DirectTrans-, so peer traffic is already
//     being redirected up to the root complex. Host memory is where that path terminates
//     anyway.
//   * This is what NCCL/RCCL's SHM transport does for PCIe-only peers.
//
// If this sustains round trips reliably, the push-based all-reduce becomes viable:
// FLAGS in host memory (cheap, 4 bytes), DATA still peer-to-peer VRAM->VRAM (which already
// works and reaches ~28 GB/s). That is the standard split and it needs no system changes.
#include <hip/hip_runtime.h>
#include <cstdio>

#define HC(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP error %s line %d\n", hipGetErrorString(e), __LINE__); return 1;} } while(0)

__device__ __forceinline__ unsigned int ld_sys(const unsigned int* p) {
  unsigned int v;
  asm volatile("global_load_dword %0, %1, off glc slc\n\ts_waitcnt vmcnt(0)"
               : "=v"(v) : "v"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_sys(unsigned int* p, unsigned int v) {
  asm volatile("global_store_dword %0, %1, off glc slc\n\ts_waitcnt vmcnt(0)"
               : : "v"(p), "v"(v) : "memory");
}

#define TICKS 200000000ull     // ~2 s at 100 MHz; a real round trip needs microseconds

// inbox_me / inbox_peer both point into the SAME pinned host buffer, at different offsets.
__global__ void pingpong(unsigned int* inbox_peer, unsigned int* inbox_me,
                         int iters, int initiator, unsigned int* done) {
  if (threadIdx.x || blockIdx.x) return;
  for (int i = 1; i <= iters; ++i) {
    unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
    if (initiator) {
      st_sys(inbox_peer, i);
      while (ld_sys(inbox_me) < (unsigned)i)
        if (__builtin_amdgcn_s_memrealtime() - t0 > TICKS) { *done = i; return; }
    } else {
      while (ld_sys(inbox_me) < (unsigned)i)
        if (__builtin_amdgcn_s_memrealtime() - t0 > TICKS) { *done = i; return; }
      st_sys(inbox_peer, i);
    }
  }
  *done = 0;
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int n = 0; HC(hipGetDeviceCount(&n));
  printf("devices: %d\n", n);
  if (n < 2) return 1;

  // One coherent pinned host allocation; both GPUs map it. Two cache lines apart so the
  // two inboxes never share a line (false sharing would mask a real failure).
  unsigned int* host = nullptr;
  HC(hipHostMalloc((void**)&host, 4096, hipHostMallocMapped | hipHostMallocCoherent));
  host[0] = 0; host[64] = 0;

  unsigned int* dev_view[2];
  unsigned int* done[2];
  hipStream_t st[2];
  for (int d = 0; d < 2; ++d) {
    HC(hipSetDevice(d));
    HC(hipHostGetDevicePointer((void**)&dev_view[d], host, 0));
    HC(hipMalloc(&done[d], 256));
    HC(hipMemset(done[d], 0, 256));
    HC(hipStreamCreate(&st[d]));
  }
  printf("host buffer mapped on both devices: %p / %p\n",
         (void*)dev_view[0], (void*)dev_view[1]);

  // --- visibility smoke test: does GPU0's host write reach the CPU and GPU1? ---
  printf("\n--- can a GPU write to host memory and the other GPU read it? ---\n");
  for (int d = 0; d < 2; ++d) {
    host[0] = 0;
    HC(hipSetDevice(d));
    hipLaunchKernelGGL(pingpong, dim3(1), dim3(1), 0, 0,
                       dev_view[d] + 0, dev_view[d] + 64, 0, 1, done[d]);  // iters=0: no-op
    HC(hipDeviceSynchronize());
  }
  printf("  (kernels launch cleanly on both devices)\n");

  for (int iters : {100, 2000}) {
    host[0] = 0; host[64] = 0;
    for (int d = 0; d < 2; ++d) {
      HC(hipSetDevice(d)); HC(hipMemset(done[d], 0, 256)); HC(hipDeviceSynchronize());
    }
    hipEvent_t a, b;
    HC(hipSetDevice(0)); HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
    HC(hipEventRecord(a, st[0]));
    for (int d = 0; d < 2; ++d) {
      HC(hipSetDevice(d));
      // device 0 writes host[0] and waits on host[64]; device 1 the mirror image
      unsigned int* peer = dev_view[d] + (d == 0 ? 64 : 0);
      unsigned int* mine = dev_view[d] + (d == 0 ? 0 : 64);
      hipLaunchKernelGGL(pingpong, dim3(1), dim3(1), 0, st[d],
                         peer, mine, iters, d == 0, done[d]);
    }
    HC(hipSetDevice(0)); HC(hipEventRecord(b, st[0]));
    for (int d = 0; d < 2; ++d) { HC(hipSetDevice(d)); HC(hipStreamSynchronize(st[d])); }
    float ms = 0; HC(hipSetDevice(0)); HC(hipEventElapsedTime(&ms, a, b));
    unsigned int b0 = 0, b1 = 0;
    HC(hipSetDevice(0)); HC(hipMemcpy(&b0, done[0], 4, hipMemcpyDeviceToHost));
    HC(hipSetDevice(1)); HC(hipMemcpy(&b1, done[1], 4, hipMemcpyDeviceToHost));
    if (b0 || b1)
      printf("  %5d iters: STALLED at %u  <-- host-memory flags do NOT work either\n",
             iters, b0 ? b0 : b1);
    else
      printf("  %5d iters: OK  %.2f us per round trip  (NCCL costs 116 us per 8 KB exchange)\n",
             iters, ms * 1000.0 / iters);
  }
  printf("\nIf this works, the push all-reduce is viable: flags in host memory,\n"
         "data peer-to-peer VRAM->VRAM (already measured at ~28 GB/s).\n");
  return 0;
}
