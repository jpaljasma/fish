// src/kernels/fishhash.cu
//
// CUDA kernels for FishHash (FIP-3), mirroring the CPU port in src/fishhash.cpp:
//   - dataset build kernel: 37,748,717 items x 128 B, pure function of the
//     72 MiB light cache (built once on the host, uploaded);
//   - hash kernel: start = BLAKE3-512(header); 32 rounds x 3 x 128-B dataset
//     fetches with FNV + IMAD.WIDE mixing; FNV collapse; final = BLAKE3-256.
//
// Device Keccak-f[1600] and BLAKE3 compress are ports of the exact vendored
// reference primitives (third_party/keccak.c, third_party/blake3_portable.c).
// Correctness is enforced by tests/test_gpu_match.cpp against the CPU oracle.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#include "keccak_device.cuh"  // device rol, round_constants, keccakf1600

namespace fishhash {

// Constants (must mirror src/fishhash.h).
static constexpr uint32_t kFnvPrime = 0x01000193u;
static constexpr uint32_t kLightCacheItems = 1179641u;
static constexpr uint32_t kDatasetItemParents = 512u;
static constexpr uint32_t kDatasetItems = 37748717u;
static constexpr uint32_t kDatasetAccesses = 32u;

__device__ __forceinline__ uint32_t fnv1(uint32_t u, uint32_t v) {
  return (u * kFnvPrime) ^ v;
}

// ── FishHash "keccak512", 64-byte input, in-place on the first 8 lanes ────
// Rate 72 B, Keccak delimiter 0x01 (not SHA3's 0x06). After the 64 input
// bytes, the first and last padding bytes both land in lane 8.
__device__ __forceinline__ void keccak512_inplace(uint64_t* st) {
  for (int i = 8; i < 25; ++i) st[i] = 0;
  st[8] = 0x8000000000000001ULL;
  keccakf1600(st);
}

// ── BLAKE3 device primitives (port of third_party/blake3_portable.c) ───────
static __device__ __forceinline__ uint32_t rotr32(uint32_t w, uint32_t c) {
  return (w >> c) | (w << (32 - c));
}

static __device__ __forceinline__ void g32(uint32_t* s, int a, int b, int c,
                                           int d, uint32_t x, uint32_t y) {
  s[a] = s[a] + s[b] + x;
  s[d] = rotr32(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + y;
  s[d] = rotr32(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 7);
}

static __device__ const uint32_t kBlakeIv[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

static __device__ const uint8_t kBlakeSchedule[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}};

static __device__ __forceinline__ void blake3Round(uint32_t* s,
                                                   const uint32_t* m,
                                                   int round) {
  const uint8_t* sc = kBlakeSchedule[round];
  g32(s, 0, 4, 8, 12, m[sc[0]], m[sc[1]]);
  g32(s, 1, 5, 9, 13, m[sc[2]], m[sc[3]]);
  g32(s, 2, 6, 10, 14, m[sc[4]], m[sc[5]]);
  g32(s, 3, 7, 11, 15, m[sc[6]], m[sc[7]]);
  g32(s, 0, 5, 10, 15, m[sc[8]], m[sc[9]]);
  g32(s, 1, 6, 11, 12, m[sc[10]], m[sc[11]]);
  g32(s, 2, 7, 8, 13, m[sc[12]], m[sc[13]]);
  g32(s, 3, 4, 9, 14, m[sc[14]], m[sc[15]]);
}

// One compression: 16-word state from (cv, block, counter, block_len, flags),
// 7 rounds. root=false: cv <- state[0..7] ^ state[8..15] (in-place step).
// root=true: writes 16-word XOF output (state ^ cv, both directions).
static __device__ __forceinline__ void blake3Compress(
    uint32_t* cv, const uint32_t* block, uint32_t block_len,
    uint64_t counter, uint32_t flags, bool root, uint32_t* out16) {
  uint32_t st[16];
  for (int i = 0; i < 8; ++i) st[i] = cv[i];
  st[8] = kBlakeIv[0];
  st[9] = kBlakeIv[1];
  st[10] = kBlakeIv[2];
  st[11] = kBlakeIv[3];
  st[12] = (uint32_t)counter;
  st[13] = (uint32_t)(counter >> 32);
  st[14] = block_len;
  st[15] = flags;
  for (int r = 0; r < 7; ++r) blake3Round(st, block, r);
  if (root) {
    for (int i = 0; i < 8; ++i) {
      out16[i] = st[i] ^ st[8 + i];
      out16[8 + i] = st[8 + i] ^ cv[i];
    }
  } else {
    for (int i = 0; i < 8; ++i) cv[i] = st[i] ^ st[8 + i];
  }
}

enum {
  kChunkStart = 1u << 0,
  kChunkEnd = 1u << 1,
  kRoot = 1u << 3,
};

// Unkeyed single-chunk BLAKE3 for msg length <= 192 B (headers are 180 B,
// final data is 96 B). Produces the 16-word XOF output block 0; callers use
// the first 8 words (BLAKE3-256) or all 16 (BLAKE3-512).
static __device__ __forceinline__ void blake3SingleChunk(
    const uint8_t* msg, uint32_t len, uint32_t* out16) {
  uint32_t cv[8];
  for (int i = 0; i < 8; ++i) cv[i] = kBlakeIv[i];
  const uint32_t nblocks = len ? (len + 63) / 64 : 1;
  for (uint32_t b = 0; b < nblocks; ++b) {
    const uint32_t remaining = len - b * 64;
    const uint32_t take = remaining < 64 ? remaining : 64;
    uint32_t block[16] = {};
    for (uint32_t i = 0; i < take; ++i)
      block[i / 4] |= uint32_t(msg[b * 64 + i]) << (8 * (i % 4));
    const bool last = b + 1 == nblocks;
    const uint32_t flags = (b == 0 ? kChunkStart : 0) |
                           (last ? kChunkEnd | kRoot : 0);
    // The counter identifies the chunk, not the block within it.
    blake3Compress(cv, block, take, 0, flags, last, out16);
  }
}

// ── Dataset build: one item (2 x 64-B states) per thread ───────────────────
// lc: light cache [kLightCacheItems x 16 u32] (72 MiB, L2-resident);
// ds: output dataset [kDatasetItems x 32 u32] (4.83 GiB).
__global__ void fishHashBuildDatasetKernel(const uint32_t* __restrict__ lc,
                                           uint32_t* __restrict__ ds) {
  const uint32_t idx =
      static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= kDatasetItems) return;

  uint32_t* out = ds + idx * 32;
  for (int s = 0; s < 2; ++s) {
    const uint32_t k = idx * 2 + s;
    const uint32_t li = k % kLightCacheItems;

    uint64_t st[25];
    const uint64_t* src = reinterpret_cast<const uint64_t*>(lc + li * 16);
    for (int i = 0; i < 8; ++i) st[i] = src[i];
    st[0] ^= k;  // mix.word32s[0] ^= k (low word of lane 0)
    keccak512_inplace(st);  // init keccak512 (64-B input)

    uint32_t* m32 = reinterpret_cast<uint32_t*>(st);
    for (uint32_t round = 0; round < kDatasetItemParents; ++round) {
      const uint32_t t = fnv1(k ^ round, m32[round % 16]);
      const uint32_t parent = t % kLightCacheItems;
      const uint4* cp = reinterpret_cast<const uint4*>(lc + parent * 16);
      uint4 f0 = cp[0];
      uint4 f1 = cp[1];
      uint4 f2 = cp[2];
      uint4 f3 = cp[3];
      m32[0] = fnv1(m32[0], f0.x);
      m32[1] = fnv1(m32[1], f0.y);
      m32[2] = fnv1(m32[2], f0.z);
      m32[3] = fnv1(m32[3], f0.w);
      m32[4] = fnv1(m32[4], f1.x);
      m32[5] = fnv1(m32[5], f1.y);
      m32[6] = fnv1(m32[6], f1.z);
      m32[7] = fnv1(m32[7], f1.w);
      m32[8] = fnv1(m32[8], f2.x);
      m32[9] = fnv1(m32[9], f2.y);
      m32[10] = fnv1(m32[10], f2.z);
      m32[11] = fnv1(m32[11], f2.w);
      m32[12] = fnv1(m32[12], f3.x);
      m32[13] = fnv1(m32[13], f3.y);
      m32[14] = fnv1(m32[14], f3.z);
      m32[15] = fnv1(m32[15], f3.w);
    }
    keccak512_inplace(st);  // final keccak512
    for (int i = 0; i < 16; ++i) out[s * 16 + i] = m32[i];
  }
}

// ── Shared hash body: header -> 32-B FishHash ──────────────────────────────
// ds: dataset [kDatasetItems x 32 u32];
// out: 8 u32 words (32-B hash).
__device__ __forceinline__ void fishHashHeaderDevice(const uint8_t* header,
                                                     uint32_t headerLen,
                                                     const uint32_t* __restrict__ ds,
                                                     uint32_t* out) {
  // start value = BLAKE3-512(header)
  uint32_t seed16[16];
  blake3SingleChunk(header, headerLen, seed16);

  uint32_t mix[32];
  for (int j = 0; j < 16; ++j) {
    mix[j] = seed16[j];
    mix[16 + j] = seed16[j];
  }

  for (uint32_t i = 0; i < kDatasetAccesses; ++i) {
    const uint32_t p0 = mix[0] % kDatasetItems;
    const uint32_t p1 = mix[4] % kDatasetItems;
    const uint32_t p2 = mix[8] % kDatasetItems;
    const uint4* f0v = reinterpret_cast<const uint4*>(ds + p0 * 32);
    const uint4* f1v = reinterpret_cast<const uint4*>(ds + p1 * 32);
    const uint4* f2v = reinterpret_cast<const uint4*>(ds + p2 * 32);
    uint4* mv = reinterpret_cast<uint4*>(mix);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint4 a = f0v[j];
      uint4 b = f1v[j];
      uint4 c = f2v[j];
      const uint4 m = mv[j];
      b.x = fnv1(m.x, b.x);
      b.y = fnv1(m.y, b.y);
      b.z = fnv1(m.z, b.z);
      b.w = fnv1(m.w, b.w);
      c.x = m.x ^ c.x;
      c.y = m.y ^ c.y;
      c.z = m.z ^ c.z;
      c.w = m.w ^ c.w;
      // mix.word64s[2j]   = fetch0[2j]   * fetch1[2j]   + fetch2[2j]
      // mix.word64s[2j+1] = fetch0[2j+1] * fetch1[2j+1] + fetch2[2j+1]
      // (u64 lanes, little-endian: .x low 32 bits, .y high 32 bits)
      const uint64_t f0_0 = (static_cast<uint64_t>(a.y) << 32) | a.x;
      const uint64_t f1_0 = (static_cast<uint64_t>(b.y) << 32) | b.x;
      const uint64_t f2_0 = (static_cast<uint64_t>(c.y) << 32) | c.x;
      const uint64_t r0 = f0_0 * f1_0 + f2_0;
      const uint64_t f0_1 = (static_cast<uint64_t>(a.w) << 32) | a.z;
      const uint64_t f1_1 = (static_cast<uint64_t>(b.w) << 32) | b.z;
      const uint64_t f2_1 = (static_cast<uint64_t>(c.w) << 32) | c.z;
      const uint64_t r1 = f0_1 * f1_1 + f2_1;
      mv[j].x = static_cast<uint32_t>(r0);
      mv[j].y = static_cast<uint32_t>(r0 >> 32);
      mv[j].z = static_cast<uint32_t>(r1);
      mv[j].w = static_cast<uint32_t>(r1 >> 32);
    }
  }

  // Collapse 128 B -> 32 B: FNV chain over each 4-word group.
  uint32_t collapsed[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const uint32_t i = g * 4;
    uint32_t h = fnv1(mix[i], mix[i + 1]);
    h = fnv1(h, mix[i + 2]);
    h = fnv1(h, mix[i + 3]);
    collapsed[g] = h;
  }

  // Final: BLAKE3-256(startValue[64] || collapsed[32]).
  uint8_t finalData[96];
  for (int j = 0; j < 16; ++j) {
    finalData[j * 4 + 0] = static_cast<uint8_t>(seed16[j] & 0xFF);
    finalData[j * 4 + 1] = static_cast<uint8_t>((seed16[j] >> 8) & 0xFF);
    finalData[j * 4 + 2] = static_cast<uint8_t>((seed16[j] >> 16) & 0xFF);
    finalData[j * 4 + 3] = static_cast<uint8_t>((seed16[j] >> 24) & 0xFF);
  }
  for (int j = 0; j < 8; ++j) {
    finalData[64 + j * 4 + 0] = static_cast<uint8_t>(collapsed[j] & 0xFF);
    finalData[64 + j * 4 + 1] = static_cast<uint8_t>((collapsed[j] >> 8) & 0xFF);
    finalData[64 + j * 4 + 2] = static_cast<uint8_t>((collapsed[j] >> 16) & 0xFF);
    finalData[64 + j * 4 + 3] = static_cast<uint8_t>((collapsed[j] >> 24) & 0xFF);
  }
  uint32_t out16[16];
  blake3SingleChunk(finalData, 96, out16);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = out16[i];
}

// ── Hash kernel: one header per thread ─────────────────────────────────────
__global__ void fishHashHeaderKernel(const uint8_t* __restrict__ header,
                                     uint32_t headerLen,
                                     const uint32_t* __restrict__ ds,
                                     uint32_t* __restrict__ out) {
  fishHashHeaderDevice(header, headerLen, ds, out);
}

// ── Batch hash kernel: nonce sweep over one base header ────────────────────
struct MiningHeader {
  uint8_t bytes[180];
};

template<bool search>
__global__ void fishHashHeaderBatchKernel(MiningHeader header,
                                          uint64_t nonceStart, uint32_t count,
                                          const uint32_t* __restrict__ ds,
                                          uint32_t* __restrict__ out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count) return;
  const uint64_t nonce = nonceStart + t;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    header.bytes[179 - i] = static_cast<uint8_t>(nonce >> (8 * i));
  if constexpr (!search) {
    fishHashHeaderDevice(header.bytes, sizeof(header.bytes), ds, out + t * 8);
  } else {
    uint32_t digest[8];
    fishHashHeaderDevice(header.bytes, sizeof(header.bytes), ds, digest);
    const auto* bytes = reinterpret_cast<const uint8_t*>(digest);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const uint8_t target = header.bytes[132 + i];
      if (bytes[i] > target) return;
      if (bytes[i] < target) break;
    }
    // Exactly one lane owns the payload, including when multiple hashes qualify.
    if (atomicCAS(out, UINT32_MAX, t) == UINT32_MAX) {
#pragma unroll
      for (int i = 0; i < 8; ++i) out[i + 1] = digest[i];
    }
  }
}

}  // namespace fishhash

// ── Host-side launchers (declared in src/gpu_context.h) ────────────────────
namespace fishhash {

void launchBuildDataset(const uint32_t* d_lightCache, uint32_t* d_dataset,
                        cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks =
      (kDatasetItems + kThreads - 1) / kThreads;  // 147,456 blocks
  fishHashBuildDatasetKernel<<<blocks, kThreads, 0, stream>>>(
      d_lightCache, d_dataset);
}

void launchHashHeader(const uint8_t* d_header, uint32_t headerLen,
                      const uint32_t* d_dataset, uint32_t* d_out,
                      cudaStream_t stream) {
  fishHashHeaderKernel<<<1, 1, 0, stream>>>(d_header, headerLen, d_dataset,
                                            d_out);
}

void launchHashHeaderBatch(const uint8_t* h_baseHeader, uint64_t nonceStart,
                           uint32_t count,
                           const uint32_t* d_dataset, uint32_t* d_out,
                           cudaStream_t stream) {
  MiningHeader header;
  std::memcpy(header.bytes, h_baseHeader, sizeof(header.bytes));
  constexpr int kThreads = 256;
  const int blocks = (count + kThreads - 1) / kThreads;
  fishHashHeaderBatchKernel<false><<<blocks, kThreads, 0, stream>>>(
      header, nonceStart, count, d_dataset, d_out);
}

void launchNonceSearch(const uint8_t* base, uint64_t first, uint32_t count,
                       const uint32_t* dataset, uint32_t* hit, cudaStream_t stream) {
  MiningHeader header;
  std::memcpy(header.bytes, base, sizeof(header.bytes));
  constexpr int threads = 256;
  fishHashHeaderBatchKernel<true><<<(count + threads - 1) / threads, threads, 0, stream>>>(
      header, first, count, dataset, hit);
}

}  // namespace fishhash
