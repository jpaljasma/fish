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

#include "../gpu_context.h"
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

struct alignas(16) Seed {
  uint32_t words[16];
};
struct CollapsedMix {
  uint32_t words[8];
};
static_assert(sizeof(Seed) == 64 && sizeof(CollapsedMix) == 32);

__device__ __forceinline__ uint64_t word64(uint32_t lo, uint32_t hi) {
  return (uint64_t{hi} << 32) | lo;
}

// Eight adjacent lanes own one nonce. Each issues one 16-byte load per
// dataset item; together they read all 128 bytes. The next round's three
// addresses come from mix words 0, 4 and 8 (lane 0, 1 and 2's first word).
__device__ __forceinline__ uint32_t mixLane(const Seed& seed, const uint4* dag) {
  const unsigned lane = threadIdx.x % 8;
  const unsigned mask = 0xffu << (threadIdx.x & 24u);
  uint4 m;
  memcpy(&m, seed.words + (lane % 4) * 4, sizeof(m));
#pragma unroll 1
  for (unsigned r = 0; r < kDatasetAccesses; ++r) {
    const uint32_t p0 = __shfl_sync(mask, m.x, 0, 8) % kDatasetItems;
    const uint32_t p1 = __shfl_sync(mask, m.x, 1, 8) % kDatasetItems;
    const uint32_t p2 = __shfl_sync(mask, m.x, 2, 8) % kDatasetItems;
    const uint4 a = dag[p0 * 8 + lane];
    const uint4 b = dag[p1 * 8 + lane];
    const uint4 c = dag[p2 * 8 + lane];
    const uint64_t lo = word64(a.x, a.y) * word64(fnv1(m.x, b.x), fnv1(m.y, b.y))
                      + word64(m.x ^ c.x, m.y ^ c.y);
    const uint64_t hi = word64(a.z, a.w) * word64(fnv1(m.z, b.z), fnv1(m.w, b.w))
                      + word64(m.z ^ c.z, m.w ^ c.w);
    m = make_uint4(uint32_t(lo), uint32_t(lo >> 32), uint32_t(hi), uint32_t(hi >> 32));
  }
  return fnv1(fnv1(fnv1(m.x, m.y), m.z), m.w);
}

__device__ __forceinline__ void finishHash(const Seed& seed, const CollapsedMix& mix,
                                          uint32_t digest[16]) {
  uint32_t cv[8], block[16] = {};
  memcpy(cv, kBlakeIv, sizeof(cv));
  memcpy(block, mix.words, sizeof(mix.words));
  blake3Compress(cv, seed.words, 64, 0, kChunkStart, false, nullptr);
  blake3Compress(cv, block, 32, 0, kChunkEnd | kRoot, true, digest);
}

// Arbitrary header length (oracle API); one complete eight-lane group.
__global__ void fishHashHeaderKernel(const uint8_t* header, uint32_t len,
                                     const uint4* dag, uint32_t* out) {
  __shared__ Seed seed;
  __shared__ CollapsedMix mix;
  if (threadIdx.x == 0) blake3SingleChunk(header, len, seed.words);
  __syncwarp(0xff);
  mix.words[threadIdx.x] = mixLane(seed, dag);
  __syncwarp(0xff);
  if (threadIdx.x == 0) {
    uint32_t digest[16];
    finishHash(seed, mix, digest);
    memcpy(out, digest, 32);
  }
}

__global__ void seedBatch(MiningJob job, uint64_t first, uint32_t count, Seed* seeds) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint64_t nonce = first + i;
  uint32_t block[16] = {};
  memcpy(block, job.tail, sizeof(job.tail));
  // Nonce bytes 172..179 are big-endian; BLAKE3 reads little-endian words.
  block[11] = __byte_perm(uint32_t(nonce >> 32), 0, 0x0123);
  block[12] = __byte_perm(uint32_t(nonce), 0, 0x0123);
  blake3Compress(job.cv, block, 52, 0, kChunkEnd | kRoot, true, seeds[i].words);
}

__global__ void mixBatch(const Seed* __restrict__ seeds, const uint4* __restrict__ dag,
                         uint32_t count, CollapsedMix* __restrict__ mixes) {
  const uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
  if (i >= count) return;  // all eight lanes participate or all eight exit
  mixes[i].words[threadIdx.x % 8] = mixLane(seeds[i], dag);
}

struct MiningTarget {
  uint8_t bytes[32];
};

template<BatchOutput mode>
__global__ void finishBatch(MiningTarget target, const Seed* seeds, const CollapsedMix* mixes,
                            uint32_t count, uint32_t* out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  uint32_t digest[16];
  finishHash(seeds[i], mixes[i], digest);
  if constexpr (mode == BatchOutput::FirstHit) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(digest);
    for (unsigned j = 0; j < sizeof(target.bytes); ++j) {
      if (bytes[j] > target.bytes[j]) return;
      if (bytes[j] < target.bytes[j]) break;
    }
    if (atomicCAS(out, UINT32_MAX, i) != UINT32_MAX) return;
    ++out;  // only the winning thread owns the eight payload words
  } else {
    out += i * 8;
  }
  memcpy(out, digest, 32);
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
  fishHashHeaderKernel<<<1, 8, 0, stream>>>(d_header, headerLen,
                                            reinterpret_cast<const uint4*>(d_dataset), d_out);
}

cudaError_t launchMiningBatch(const MiningJob& job, uint64_t first, uint32_t count,
                             const uint32_t* dataset, void* work, uint32_t* out,
                             BatchOutput mode, cudaStream_t stream) {
  auto* seeds = static_cast<Seed*>(work);
  auto* mixes = reinterpret_cast<CollapsedMix*>(seeds + count);
  constexpr int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  seedBatch<<<blocks, threads, 0, stream>>>(job, first, count, seeds);
  if (auto err = cudaGetLastError(); err != cudaSuccess) return err;
  constexpr int mixThreads = 128, groups = mixThreads / 8;
  mixBatch<<<(count + groups - 1) / groups, mixThreads, 0, stream>>>(
      seeds, reinterpret_cast<const uint4*>(dataset), count, mixes);
  if (auto err = cudaGetLastError(); err != cudaSuccess) return err;
  MiningTarget target;
  std::memcpy(target.bytes, job.tail + 1, sizeof(target.bytes));  // header offset 132
  if (mode == BatchOutput::FirstHit)
    finishBatch<BatchOutput::FirstHit><<<blocks, threads, 0, stream>>>(target, seeds, mixes, count, out);
  else
    finishBatch<BatchOutput::Digests><<<blocks, threads, 0, stream>>>(target, seeds, mixes, count, out);
  return cudaGetLastError();
}

}  // namespace fishhash
