// src/fishhash.cpp
//
// CPU implementation of FishHash (FIP-3). Logic verified byte-exact against
// the community reference (iron-fish/fish-hash cpp/FishHash.cpp) and the
// published `fish_hash` crate v0.3.0 via tests/test_fishhash.cpp.

#include "fishhash.h"

#include <cstring>

#include "blake3.h"
#include "keccak.h"

namespace fishhash {
namespace {

// Fixed dataset seed: BLAKE3-256("FishHash"), exactly as hardcoded in the
// reference. tests/test_fishhash.cpp re-derives it from the string.
const uint8_t kSeed256[32] = {
    0xeb, 0x01, 0x63, 0xae, 0xf2, 0xab, 0x1c, 0x5a,
    0x66, 0x31, 0x0c, 0x1c, 0x14, 0xd6, 0x0f, 0x42,
    0x55, 0xa9, 0xb3, 0x9b, 0x0e, 0xdf, 0x26, 0x53,
    0x98, 0x44, 0xf1, 0x17, 0xad, 0x67, 0x21, 0x19};

// One 64-B state machine during dataset item construction.
struct ItemState {
  const Hash512* cache;
  uint32_t seed;  // the state's k (2i or 2i+1)
  Hash512 mix;

  ItemState(const Hash512* cache, uint64_t index)
      : cache(cache), seed(static_cast<uint32_t>(index)) {
    mix = cache[index % kLightCacheItems];
    mix.word32s[0] ^= seed;
    keccak(mix.word64s, 512, mix.bytes, 64);
  }

  void update(uint32_t round) {
    const uint32_t t = fnv1(seed ^ round, mix.word32s[round % 16]);
    const uint32_t parent = t % kLightCacheItems;
    mix = fnv1Wordwise(mix, cache[parent]);
  }

  Hash512 final() {
    keccak(mix.word64s, 512, mix.bytes, 64);
    return mix;
  }
};

void runLoop(const ItemSource& items, const uint8_t* header, size_t headerLen,
             Trace* trace, Hash256* outHash) {
  // Start value: BLAKE3-512(header). Default hasher (32-B init) finalized
  // at 64 B - standard BLAKE3 variable-output usage, matches the reference.
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, header, headerLen);
  Hash512 seed;
  blake3_hasher_finalize(&hasher, seed.bytes, 64);
  if (trace) trace->startValue = seed;

  // mix = seed || seed (128 B).
  Hash1024 mix;
  std::memcpy(mix.bytes, seed.bytes, 64);
  std::memcpy(mix.bytes + 64, seed.bytes, 64);

  for (uint32_t i = 0; i < kDatasetAccesses; ++i) {
    const uint32_t p0 = mix.word32s[0] % kDatasetItems;
    const uint32_t p1 = mix.word32s[4] % kDatasetItems;
    const uint32_t p2 = mix.word32s[8] % kDatasetItems;
    if (trace) {
      trace->p[i][0] = p0;
      trace->p[i][1] = p1;
      trace->p[i][2] = p2;
    }

    const Hash1024 fetch0 = items.item(p0);
    Hash1024 fetch1 = items.item(p1);
    Hash1024 fetch2 = items.item(p2);

    for (int j = 0; j < 32; ++j) {
      fetch1.word32s[j] = fnv1(mix.word32s[j], fetch1.word32s[j]);
      fetch2.word32s[j] = mix.word32s[j] ^ fetch2.word32s[j];
    }
    for (int j = 0; j < 16; ++j)
      mix.word64s[j] =
          fetch0.word64s[j] * fetch1.word64s[j] + fetch2.word64s[j];

    if (trace) trace->mix[i] = mix;
  }

  // Collapse 128 B -> 32 B: FNV chain over each 4-word group.
  Hash256 collapsed;
  for (int i = 0; i < 32; i += 4) {
    const uint32_t h1 = fnv1(mix.word32s[i], mix.word32s[i + 1]);
    const uint32_t h2 = fnv1(h1, mix.word32s[i + 2]);
    const uint32_t h3 = fnv1(h2, mix.word32s[i + 3]);
    collapsed.word32s[i / 4] = h3;
  }
  if (trace) trace->collapsed = collapsed;

  // Final: BLAKE3-256(startValue[64] || collapsed[32]).
  uint8_t finalData[96];
  std::memcpy(finalData, seed.bytes, 64);
  std::memcpy(finalData + 64, collapsed.bytes, 32);
  blake3_hasher hasher2;
  blake3_hasher_init(&hasher2);
  blake3_hasher_update(&hasher2, finalData, 96);
  if (outHash) blake3_hasher_finalize(&hasher2, outHash->bytes, 32);
}
}  // namespace

uint32_t fnv1(uint32_t u, uint32_t v) { return (u * kFnvPrime) ^ v; }

Hash512 fnv1Wordwise(const Hash512& u, const Hash512& v) {
  Hash512 r;
  for (int i = 0; i < 16; ++i) r.word32s[i] = fnv1(u.word32s[i], v.word32s[i]);
  return r;
}

std::vector<Hash512> buildLightCacheChain() {
  std::vector<Hash512> cache(kLightCacheItems);
  Hash512 item;
  keccak(item.word64s, 512, kSeed256, 32);
  cache[0] = item;
  for (uint32_t i = 1; i < kLightCacheItems; ++i) {
    keccak(item.word64s, 512, item.bytes, 64);
    cache[i] = item;
  }
  return cache;
}

void applyLightCacheRounds(std::vector<Hash512>& cache) {
  const uint32_t n = kLightCacheItems;
  for (int q = 0; q < kLightCacheRounds; ++q) {
    for (uint32_t i = 0; i < n; ++i) {
      // In-place, ascending: cache[v]/cache[w] may already be updated.
      const uint32_t v = cache[i].word32s[0] % n;
      const uint32_t w = (n + (i > 0 ? i - 1 : n - 1)) % n;
      Hash512 x;
      for (int j = 0; j < 8; ++j)
        x.word64s[j] = cache[v].word64s[j] ^ cache[w].word64s[j];
      keccak(cache[i].word64s, 512, x.bytes, 64);
    }
  }
}

LightCache::LightCache() : LightCache(buildLightCacheChain()) {}

LightCache::LightCache(std::vector<Hash512> chain) : cache_(std::move(chain)) {
  applyLightCacheRounds(cache_);
}

Hash1024 datasetItem(const LightCache& lc, uint32_t index) {
  const Hash512* cache = lc.data();
  ItemState item0(cache, uint64_t(index) * 2);
  ItemState item1(cache, uint64_t(index) * 2 + 1);
  for (uint32_t j = 0; j < kDatasetItemParents; ++j) {
    item0.update(j);
    item1.update(j);
  }
  Hash1024 out;
  out.hash512s[0] = item0.final();
  out.hash512s[1] = item1.final();
  return out;
}

Hash256 fishHash(const ItemSource& items, const uint8_t* header,
                 size_t headerLen) {
  Hash256 hash;
  runLoop(items, header, headerLen, nullptr, &hash);
  return hash;
}

Trace fishHashTrace(const ItemSource& items, const uint8_t* header,
                    size_t headerLen) {
  Trace trace;
  runLoop(items, header, headerLen, &trace, nullptr);
  return trace;
}

bool meetsTarget(const uint8_t hash[32], const uint8_t target[32]) {
  // u256 big-endian: first differing byte decides.
  for (int i = 0; i < 32; ++i) {
    if (hash[i] != target[i]) return hash[i] < target[i];
  }
  return true;  // Iron Fish Target.meets includes equality.
}

}  // namespace fishhash
