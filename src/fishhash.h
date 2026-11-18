// src/fishhash.h
//
// CPU implementation of Iron Fish's FishHash PoW (FIP-3), written from the
// spec and verified byte-exact against the community reference
// (iron-fish/fish-hash) via tests/test_fishhash.cpp + tests/vectors/vectors.json.
//
// The dataset is static (no epochs): a pure function of one fixed 256-bit
// seed, so the light cache and full dataset are built once.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fishhash {

// ── FIP-3 constants (as implemented by the reference) ─────────────────────
constexpr uint32_t kFnvPrime = 0x01000193u;
constexpr uint32_t kLightCacheRounds = 3u;
constexpr uint32_t kLightCacheItems = 1179641u;
constexpr uint32_t kDatasetItemParents = 512u;
constexpr uint32_t kDatasetItems = 37748717u;
constexpr uint32_t kDatasetAccesses = 32u;

// ── Hash containers (layout mirrors the reference) ────────────────────────
union Hash256 {
  uint64_t word64s[4];
  uint32_t word32s[8];
  uint8_t bytes[32];
};

union Hash512 {
  uint64_t word64s[8];
  uint32_t word32s[16];
  uint8_t bytes[64];
};

union Hash1024 {
  Hash512 hash512s[2];
  uint64_t word64s[16];
  uint32_t word32s[32];
  uint8_t bytes[128];
};

// FNV: (u * prime) ^ v, wordwise where noted.
uint32_t fnv1(uint32_t u, uint32_t v);
Hash512 fnv1Wordwise(const Hash512& u, const Hash512& v);

// ── Light cache (1,179,641 x 64 B) ────────────────────────────────────────
// Stage 1: cache[0] = keccak512(seed256); cache[i] = keccak512(cache[i-1]).
std::vector<Hash512> buildLightCacheChain();

// Stage 2: three in-place rounds (ascending i):
// v = cache[i].word32[0] % N; w = (N + i - 1) % N;
// cache[i] = keccak512(cache[v] ^ cache[w]).
void applyLightCacheRounds(std::vector<Hash512>& cache);

class LightCache {
 public:
  // Builds the full cache (chain + 3 rounds). ~72 MiB, seconds of CPU.
  LightCache();
  explicit LightCache(std::vector<Hash512> chain);  // moves, applies rounds

  const Hash512& item(uint32_t index) const { return cache_[index]; }
  const Hash512* data() const { return cache_.data(); }
  uint32_t size() const { return kLightCacheItems; }

 private:
  std::vector<Hash512> cache_;
};

// ── Full dataset items (37,748,717 x 128 B) ───────────────────────────────
// Pure function of (light cache, index). Two 64-B states (k = 2i, 2i+1),
// each: init from light cache, 512 rounds of FNV-driven parent lookup,
// keccak512 final.
Hash1024 datasetItem(const LightCache& lc, uint32_t index);

// ── Item sources for the hash loop ────────────────────────────────────────
class ItemSource {
 public:
  virtual ~ItemSource() = default;
  virtual Hash1024 item(uint32_t index) const = 0;
};

// Computes items on demand (for tests / small datasets). Slow per item.
class ComputedItemSource : public ItemSource {
 public:
  explicit ComputedItemSource(const LightCache& lc) : lc_(lc) {}
  Hash1024 item(uint32_t index) const override { return datasetItem(lc_, index); }

 private:
  const LightCache& lc_;
};

// Wraps a full in-memory dataset (4.5 GiB) as item source.
class ArrayItemSource : public ItemSource {
 public:
  explicit ArrayItemSource(const std::vector<Hash1024>& dataset)
      : data_(dataset.data()) {}
  Hash1024 item(uint32_t index) const override { return data_[index]; }

 private:
  const Hash1024* data_;
};

// ── The hash ──────────────────────────────────────────────────────────────
// startValue = BLAKE3-512(header); mix = startValue || startValue;
// 32 rounds of 3 x 128-B fetches (p0/p1/p2 = mix words 0/4/8 mod N);
// collapse 128 B -> 32 B via FNV chains; hash = BLAKE3-256(start || collapsed).
struct Trace {
  Hash512 startValue;
  uint32_t p[kDatasetAccesses][3];
  Hash1024 mix[kDatasetAccesses];  // mix state after each round
  Hash256 collapsed;
};

Hash256 fishHash(const ItemSource& items, const uint8_t* header, size_t headerLen);
Trace fishHashTrace(const ItemSource& items, const uint8_t* header,
                    size_t headerLen);

// PoW rule: u256 big-endian(hash) <= u256 big-endian(target).
bool meetsTarget(const uint8_t hash[32], const uint8_t target[32]);

}  // namespace fishhash
