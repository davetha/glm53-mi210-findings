# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def repack_int4_to_int32(w: torch.Tensor) -> torch.Tensor:
    """Repack [E, N, K//2] uint8 → [E, K, N//8] int32.

    Input: K-packed uint8 (2 int4 per byte, low nibble first).
    Output: N-packed int32 (8 int4 per int32, GPTQ sequential shifts
            [0,4,...,28]).
    """
    E, N, K_half = w.shape
    assert N % 8 == 0, f"N must be divisible by 8 for int4 packing, got N={N}"
    K = K_half * 2
    N8 = N // 8
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4

    # Chunked over the expert dimension. Done whole, this holds ~16x the packed
    # input live at once -- lo and hi are int32 (4x each), unpacked is int32
    # over the doubled K (8x), and .contiguous() on the permute duplicates that
    # (8x more). On a 192-expert checkpoint that peak is ~3 GiB, which is more
    # headroom than a model sized to the card has left after its weights land.
    # Chunking bounds the transient without changing the result.
    out = torch.empty((E, K, N8), dtype=torch.int32, device=w.device)
    chunk = max(1, min(E, 8))
    for i in range(0, E, chunk):
        wc = w[i : i + chunk]
        e = wc.shape[0]
        lo = (wc & 0xF).to(torch.int32)
        hi = ((wc >> 4) & 0xF).to(torch.int32)
        unpacked = torch.stack([lo, hi], dim=-1).reshape(e, N, K)
        del lo, hi
        transposed = unpacked.permute(0, 2, 1).contiguous()
        del unpacked
        out[i : i + chunk] = (
            (transposed.view(e, K, N8, 8) << shifts).sum(dim=-1, dtype=torch.int32)
        )
        del transposed
    return out


def unpack_zp_int4_to_fp16(zp: torch.Tensor) -> torch.Tensor:
    """Unpack [E, N//2, K_groups] uint8 → [E, K_groups, N] fp16."""
    E, N_half, K_groups = zp.shape
    lo = (zp & 0xF).to(torch.int32)
    hi = ((zp >> 4) & 0xF).to(torch.int32)
    unpacked = torch.stack([lo, hi], dim=2).reshape(E, N_half * 2, K_groups)
    return unpacked.permute(0, 2, 1).contiguous().to(torch.float16)
