// Host-staged two-rank all-reduce for GPUs with no fabric between them.
//
// WHY THIS EXISTS. On 2x MI210 (gfx90a) over PCIe with no XGMI, a GPU-to-GPU handshake
// through PEER VRAM does not work: a peer write lands once the writing kernel ends, but a
// concurrently-spinning kernel on the other GPU stalls intermittently (measured across six
// store/load memory-ordering combinations; also reported upstream as NCCL #2079 and
// ROCm #5480). Disabling PCIe ACS redirect does not fix it.
//
// The handshake DOES work through pinned host memory -- the coherence point both GPUs
// reach -- at ~4.9 us p50. This is the same choice NCCL makes: its SHM transport keeps
// head/tail counters in host memory, and the NCCL internals paper states the LL protocol
// "forces the intermediate buffer to reside in host memory ... because polling GPU memory
// over PCIe is much slower ... and requires explicit synchronization".
//
// Measured, two processes, shm arena, one exchange per launch, dispatch included:
//     6.73 us   vs NCCL's 116 us for the same 8 KB exchange
//     => 0.61 ms vs 14.45 ms across the 90 all-reduces in a decode step
//
// PROTOCOL (per rank, per call)
//   1. write my contribution into my slot in the shared arena
//   2. write my flag = seq                (same path as the data, so PCIe orders it after)
//   3. spin until the peer's flag reaches seq
//   4. out = my input + peer's slot
//
// Data and flag MUST share a path. An earlier design put data in peer VRAM and the flag in
// host memory; those take different PCIe routes and nothing orders them, so the peer could
// observe the flag and read stale staging.
//
// SEQUENCE NUMBERS AND CUDA GRAPHS. `seq` lives in DEVICE memory and is incremented by the
// kernel, never passed as an argument: under FULL_DECODE_ONLY graphs the kernel arguments
// are baked at capture, so a host-side counter would replay the same value forever and the
// second exchange would see an already-satisfied flag. Both ranks increment once per call
// and are called the same number of times, so they stay in lockstep.
//
// WRAPAROUND: the flag is 32-bit. At 90 exchanges/step and ~12 tok/s that is ~1080/s, so
// wrap is ~46 days of continuous decoding. `reset` zeroes the counters if that ever matters.
#include <hip/hip_runtime.h>

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

#define NT 256

// gfx90a: `glc slc` bypasses L1 and L2 so the poll observes host memory directly rather
// than a stale cache line. A system-scope acquire would instead emit buffer_invl2 and
// discard the whole L2 -- ruinous here, since the all-reduce sits between layers and that
// would evict the activations and resident expert weights the next layer needs.
__device__ __forceinline__ u32 ld_sys(const u32* p) {
  u32 v;
  asm volatile("global_load_dword %0, %1, off glc slc\n\ts_waitcnt vmcnt(0)"
               : "=v"(v) : "v"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_sys(u32* p, u32 v) {
  asm volatile("global_store_dword %0, %1, off glc slc\n\ts_waitcnt vmcnt(0)"
               : : "v"(p), "v"(v) : "memory");
}

__device__ __forceinline__ float bf16_to_f32(u16 h) {
  u32 b = ((u32)h) << 16;
  float f; __builtin_memcpy(&f, &b, 4); return f;
}
// round-to-nearest-even, matching what torch/NCCL produce for bf16 output
__device__ __forceinline__ u16 f32_to_bf16(float f) {
  u32 b; __builtin_memcpy(&b, &f, 4);
  u32 lsb = (b >> 16) & 1u;
  b += 0x7fffu + lsb;
  return (u16)(b >> 16);
}

// DTYPE: 0 = bf16, 1 = fp32. bf16 is summed in fp32 and rounded once on store, which is
// what a 2-rank NCCL sum does; summing in bf16 would round twice.
template <int DTYPE>
__global__ __launch_bounds__(NT) void hostar_k(
    const void* __restrict__ in, void* __restrict__ out,
    void* __restrict__ slot_me, const void* __restrict__ slot_peer,
    u32* flag_me, const u32* flag_peer, u32* seq_ctr,
    int nelem, unsigned long long spin_ticks, u32* status) {
  const int tid = threadIdx.x;
  __shared__ u32 seq;
  __shared__ int bail;
  if (!tid) { seq = ++(*seq_ctr); bail = 0; }
  __syncthreads();

  // 1. publish. 16-byte stores; the arena and every tensor are 16-byte aligned.
  {
    const u32x4* s = (const u32x4*)in;
    u32x4* d = (u32x4*)slot_me;
    const int nv = (DTYPE == 0 ? nelem * 2 : nelem * 4) / 16;
    for (int i = tid; i < nv; i += NT) d[i] = s[i];
  }
  __syncthreads();

  // 2. flag, after the data, on the same path
  if (!tid) st_sys(flag_me, seq);

  // 3. wait for the peer
  if (!tid) {
    unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
    while (ld_sys(flag_peer) < seq)
      if (__builtin_amdgcn_s_memrealtime() - t0 > spin_ticks) { bail = 1; break; }
  }
  __syncthreads();

  // 4. reduce.
  //
  // FAIL LOUD. Under FULL_DECODE_ONLY graphs no Python runs per step, so nothing can check
  // a status flag between exchanges -- a timed-out all-reduce would silently feed stale
  // staging into the next layer as a plausible-looking activation. That is the worst
  // failure mode available, so a timeout traps instead: the process dies with a fault that
  // names this kernel. The budget is set from Python at ~10 s, orders of magnitude beyond
  // the measured 6.7 us, so this only fires when a rank has genuinely gone away.
  if (bail) {
    if (!tid && status) *status = 1u;
    __builtin_trap();
  }
  {
    if (DTYPE == 0) {
      const u16* a = (const u16*)in;
      const u16* b = (const u16*)slot_peer;
      u16* o = (u16*)out;
      for (int i = tid; i < nelem; i += NT)
        o[i] = f32_to_bf16(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
    } else {
      const float* a = (const float*)in;
      const float* b = (const float*)slot_peer;
      float* o = (float*)out;
      for (int i = tid; i < nelem; i += NT) o[i] = a[i] + b[i];
    }
  }
  if (!tid && status) *status = 0u;
}

__global__ void hostar_reset_k(u32* seq_ctr, u32* flag0, u32* flag1) {
  if (threadIdx.x == 0) { *seq_ctr = 0; st_sys(flag0, 0); st_sys(flag1, 0); }
}

extern "C" {

// dtype: 0 = bf16, 1 = fp32.  Returns hipGetLastError().
// `status` (device u32, may be null) is set to 1 if the spin timed out -- the caller must
// check it and fall back, because a timed-out all-reduce leaves `out` unwritten.
int hostar_allreduce(const void* in, void* out,
                     void* slot_me, const void* slot_peer,
                     void* flag_me, const void* flag_peer, void* seq_ctr,
                     int nelem, int dtype, unsigned long long spin_ticks,
                     void* status, void* stream) {
  dim3 g(1), b(NT);
  hipStream_t s = (hipStream_t)stream;
  if (dtype == 0)
    hipLaunchKernelGGL((hostar_k<0>), g, b, 0, s, in, out, slot_me, slot_peer,
                       (u32*)flag_me, (const u32*)flag_peer, (u32*)seq_ctr,
                       nelem, spin_ticks, (u32*)status);
  else
    hipLaunchKernelGGL((hostar_k<1>), g, b, 0, s, in, out, slot_me, slot_peer,
                       (u32*)flag_me, (const u32*)flag_peer, (u32*)seq_ctr,
                       nelem, spin_ticks, (u32*)status);
  return (int)hipGetLastError();
}

int hostar_reset(void* seq_ctr, void* flag0, void* flag1, void* stream) {
  hipLaunchKernelGGL(hostar_reset_k, dim3(1), dim3(1), 0, (hipStream_t)stream,
                     (u32*)seq_ctr, (u32*)flag0, (u32*)flag1);
  return (int)hipGetLastError();
}

}  // extern "C"
