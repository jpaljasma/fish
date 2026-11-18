// tools/vecgen.cpp
//
// FishHash test-vector generator (oracle harness).
//
// Implements FIP-3 "Memory Hard Mining Algorithm (FishHash)" exactly as the
// community reference (iron-fish/fish-hash, cpp/FishHash.cpp) does: an
// ethash-style memory-hard PoW using BLAKE3 + Keccak-512. The JSON output
// carries full intermediate state — light-cache samples (pre- and post-round),
// every referenced dataset item, per-round mix state, the collapsed mix and
// the final hash — so the CPU port (U2) and CUDA kernel (U3) can be verified
// stage by stage without re-deriving anything.
//
// Usage:
//   vecgen <output.json> [header_hex ...]
// With no header_hex arguments, the standard 8-vector set is generated.

#include <chrono>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// The vendored BLAKE3 C library and the Keccak-512 implementation from the
// fish-hash repo, compiled as C and linked in (see CMakeLists.txt).
#include "blake3.h"
#include "keccak.h"

namespace {

// ── FIP-3 constants (as implemented by the reference) ─────────────────────
constexpr uint32_t kFnvPrime = 0x01000193;
constexpr uint32_t kItemParents = 512;        // full_dataset_item_parents
constexpr uint32_t kDatasetAccesses = 32;     // num_dataset_accesses
constexpr uint32_t kLightCacheRounds = 3;     // light_cache_rounds
constexpr uint32_t kLightCacheItems = 1179641; // light_cache_num_items
constexpr uint32_t kDatasetItems = 37748717;   // full_dataset_num_items

// ── Hash containers (mirror the reference layout) ─────────────────────────
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

// ── FNV mix (32-bit, unsigned wraparound — well-defined) ──────────────────
inline uint32_t fnv1(uint32_t u, uint32_t v) { return (u * kFnvPrime) ^ v; }

inline Hash512 fnv1Wordwise(const Hash512& u, const Hash512& v) {
  Hash512 r;
  for (int i = 0; i < 16; ++i) r.word32s[i] = fnv1(u.word32s[i], v.word32s[i]);
  return r;
}

// ── Hex helpers ───────────────────────────────────────────────────────────
std::string hex(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  s.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    s += d[p[i] >> 4];
    s += d[p[i] & 15];
  }
  return s;
}

bool unhex(const std::string& s, uint8_t* out, size_t n) {
  if (s.size() != n * 2) return false;
  auto val = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (size_t i = 0; i < n; ++i) {
    int hi = val(s[2 * i]), lo = val(s[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = (uint8_t)((hi << 4) | lo);
  }
  return true;
}

// ── Dataset item construction ─────────────────────────────────────────────
struct ItemState {
  const Hash512* cache;
  uint32_t seed;
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

inline Hash1024 calculateDatasetItem(const Hash512* cache, uint32_t index) {
  ItemState item0(cache, uint64_t(index) * 2);
  ItemState item1(cache, uint64_t(index) * 2 + 1);
  for (uint32_t j = 0; j < kItemParents; ++j) {
    item0.update(j);
    item1.update(j);
  }
  Hash1024 out;
  out.hash512s[0] = item0.final();
  out.hash512s[1] = item1.final();
  return out;
}

// ── Light cache (mirrors the reference FishHash.cpp exactly) ──────────────
// Hardcoded 32-byte seed from the reference (= BLAKE3-256("FishHash")).
const uint8_t kSeed256[32] = {
    0xeb, 0x01, 0x63, 0xae, 0xf2, 0xab, 0x1c, 0x5a,
    0x66, 0x31, 0x0c, 0x1c, 0x14, 0xd6, 0x0f, 0x42,
    0x55, 0xa9, 0xb3, 0x9b, 0x0e, 0xdf, 0x26, 0x53,
    0x98, 0x44, 0xf1, 0x17, 0xad, 0x67, 0x21, 0x19};

inline Hash512 bitwiseXor(const Hash512& a, const Hash512& b) {
  Hash512 out;
  for (int j = 0; j < 8; ++j) out.word64s[j] = a.word64s[j] ^ b.word64s[j];
  return out;
}

// cache[0] = keccak512(seed256), then keccak512 chain, then 3 in-place
// rounds: cache[i] = keccak512(cache[v] ^ cache[(i-1+N)%N]), ascending i
// (in-place: cache[v]/cache[w] may already be updated this round).
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
  for (int q = 0; q < kLightCacheRounds; ++q) {
    for (uint32_t i = 0; i < kLightCacheItems; ++i) {
      const uint32_t v = cache[i].word32s[0] % kLightCacheItems;
      const uint32_t w = (i == 0) ? kLightCacheItems - 1 : i - 1;
      const Hash512 x = bitwiseXor(cache[v], cache[w]);
      keccak(cache[i].word64s, 512, x.bytes, 64);
    }
  }
}

std::vector<Hash512> buildLightCache() {
  std::vector<Hash512> cache = buildLightCacheChain();
  applyLightCacheRounds(cache);
  return cache;
}

// ── Lazy dataset item cache (records every referenced index) ──────────────
struct ItemCache {
  std::unordered_map<uint32_t, Hash1024> items;
  std::vector<uint32_t> order;  // indices in first-use order (for the JSON)

  const Hash1024& get(const std::vector<Hash512>& cache, uint32_t index) {
    auto it = items.find(index);
    if (it != items.end()) return it->second;
    Hash1024 v = calculateDatasetItem(cache.data(), index);
    auto res = items.emplace(index, v);
    order.push_back(index);
    return res.first->second;
  }
};

// ── FishHash with per-round tracing ───────────────────────────────────────
struct Vector {
  std::string header;
  std::string startValue;                       // 64 B
  std::vector<std::array<uint32_t, 3>> p;       // p0, p1, p2 per round
  std::vector<std::string> mixAfterRound;       // 128 B per round
  std::string collapsed;                        // 32 B
  std::string expected;                         // final 32 B hash
};

Vector fishHashTrace(const std::vector<Hash512>& cache, ItemCache& ic,
                     const uint8_t* header, size_t n) {
  Vector v;
  v.header = hex(header, n);

  // Start value = BLAKE3-512(header): default hasher (init out_len 32),
  // finalized to 64 bytes (BLAKE3 XOF).
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, header, n);
  Hash512 seed;
  blake3_hasher_finalize(&hasher, seed.bytes, 64);
  v.startValue = hex(seed.bytes, 64);

  // mix starts as {seed, seed} (128 bytes).
  const uint32_t indexLimit = kDatasetItems;
  Hash1024 mix;
  memcpy(mix.bytes, seed.bytes, 64);
  memcpy(mix.bytes + 64, seed.bytes, 64);

  for (uint32_t i = 0; i < kDatasetAccesses; ++i) {
    const uint32_t p0 = mix.word32s[0] % indexLimit;
    const uint32_t p1 = mix.word32s[4] % indexLimit;
    const uint32_t p2 = mix.word32s[8] % indexLimit;
    v.p.push_back({p0, p1, p2});

    Hash1024 fetch0 = ic.get(cache, p0);
    Hash1024 fetch1 = ic.get(cache, p1);
    Hash1024 fetch2 = ic.get(cache, p2);

    for (int j = 0; j < 32; ++j) {
      fetch1.word32s[j] = fnv1(mix.word32s[j], fetch1.word32s[j]);
      fetch2.word32s[j] = mix.word32s[j] ^ fetch2.word32s[j];
    }
    for (int j = 0; j < 16; ++j)
      mix.word64s[j] = fetch0.word64s[j] * fetch1.word64s[j] + fetch2.word64s[j];

    v.mixAfterRound.push_back(hex(mix.bytes, 128));
  }

  // Collapse the 128-byte mix to 32 bytes: FNV chain over 4-word groups.
  Hash256 mixHash;
  for (size_t i = 0; i < 32; i += 4) {
    const uint32_t h1 = fnv1(mix.word32s[i], mix.word32s[i + 1]);
    const uint32_t h2 = fnv1(h1, mix.word32s[i + 2]);
    const uint32_t h3 = fnv1(h2, mix.word32s[i + 3]);
    mixHash.word32s[i / 4] = h3;
  }
  v.collapsed = hex(mixHash.bytes, 32);

  // Final: BLAKE3-256(startValue(64) || mixHash(32)).
  uint8_t finalData[96];
  memcpy(finalData, seed.bytes, 64);
  memcpy(finalData + 64, mixHash.bytes, 32);
  blake3_hasher hasher2;
  blake3_hasher_init(&hasher2);
  blake3_hasher_update(&hasher2, finalData, 96);
  Hash256 result;
  blake3_hasher_finalize(&hasher2, result.bytes, 32);
  v.expected = hex(result.bytes, 32);
  return v;
}

// ── Default vector set ────────────────────────────────────────────────────
std::vector<std::string> defaultHeaders() {
  std::vector<std::string> out;
  auto pad = [](std::vector<uint8_t> v, size_t n) {
    if (v.size() < n) v.resize(n, 0);
    return std::string(hex(v.data(), n));
  };

  std::vector<uint8_t> a(180, 0x00);
  out.push_back(pad(a, 180));  // v0: all zeros

  std::vector<uint8_t> b(180, 0xFF);
  out.push_back(pad(b, 180));  // v1: all 0xFF

  std::vector<uint8_t> c(180);
  for (size_t i = 0; i < 180; ++i) c[i] = uint8_t((i * 7 + 3) & 0xFF);
  out.push_back(pad(c, 180));  // v2: linear pattern

  std::vector<uint8_t> d(180);
  for (size_t i = 0; i < 180; ++i) d[i] = uint8_t(179 - i);
  uint8_t nonce[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
  memcpy(&d[172], nonce, 8);
  out.push_back(pad(d, 180));  // v3: reverse pattern + nonce @172

  std::vector<uint8_t> e(180, 0x42);
  uint8_t seq[8] = {45, 48, 0, 0, 0, 0, 0, 0};  // sequence 12345 LE
  memcpy(&e[0], seq, 8);
  uint8_t nonce2[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
  memcpy(&e[172], nonce2, 8);
  out.push_back(pad(e, 180));  // v4: 0x42 fill + sequence + nonce @172

  std::vector<uint8_t> f(179);
  for (size_t i = 0; i < 179; ++i) f[i] = uint8_t(i & 0xFF);
  out.push_back(hex(f.data(), 179));  // v5: 179 B (size-1 edge)

  std::vector<uint8_t> g(181);
  for (size_t i = 0; i < 181; ++i) g[i] = uint8_t((i * 3 + 1) & 0xFF);
  out.push_back(hex(g.data(), 181));  // v6: 181 B (size+1 edge)

  std::vector<uint8_t> h(32, 0xA5);
  out.push_back(hex(h.data(), 32));  // v7: 32 B (minimum edge)
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: vecgen <output.json> [header_hex ...]\n");
    return 2;
  }
  const std::string outPath = argv[1];

  auto t0 = std::chrono::steady_clock::now();
  std::vector<Hash512> chain = buildLightCacheChain();

  const std::vector<uint32_t> cacheIndices = {0, 1, 2, 1000, 100000, 500000,
                                              kLightCacheItems - 1};
  // Pre-round (post-chain) state, captured before the 3 mixing rounds run.
  std::vector<std::string> preHex;
  for (uint32_t i : cacheIndices) preHex.push_back(hex(chain[i].bytes, 64));

  applyLightCacheRounds(chain);
  auto& cache = chain;
  auto t1 = std::chrono::steady_clock::now();
  const double cacheMs =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
      1000.0;

  // Post-round (final) state; the cache is not mutated by hashing, so these
  // are stable for the whole run.
  std::vector<std::string> postHex;
  for (uint32_t i : cacheIndices) postHex.push_back(hex(cache[i].bytes, 64));

  std::vector<std::string> headers;
  if (argc > 2) {
    for (int i = 2; i < argc; ++i) headers.push_back(argv[i]);
  } else {
    headers = defaultHeaders();
  }

  std::vector<std::vector<uint8_t>> raw;
  for (const auto& h : headers) {
    std::vector<uint8_t> buf(h.size() / 2);
    if (!unhex(h, buf.data(), buf.size())) {
      std::fprintf(stderr, "bad header hex: %s\n", h.c_str());
      return 2;
    }
    raw.push_back(std::move(buf));
  }

  ItemCache ic;
  std::vector<Vector> vectors;
  for (size_t i = 0; i < raw.size(); ++i) {
    vectors.push_back(fishHashTrace(cache, ic, raw[i].data(), raw[i].size()));
  }
  auto t2 = std::chrono::steady_clock::now();
  const double totalMs =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
      1000.0;

  // ── Assemble JSON (single pass) ─────────────────────────────────────────
  std::ostringstream o;
  o << "{\n";
  o << "  \"algorithm\": \"fishhash\",\n";
  o << "  \"spec\": \"FIP-3\",\n";
  o << "  \"generator\": \"tools/vecgen.cpp\",\n";
  o << "  \"reference\": \"iron-fish/fish-hash (cpp/FishHash.cpp)\",\n";
  o << "  \"constants\": {\n";
  o << "    \"seed_input\": \"FishHash\",\n";
  o << "    \"fnv_prime\": " << kFnvPrime << ",\n";
  o << "    \"full_dataset_item_parents\": " << kItemParents << ",\n";
  o << "    \"num_dataset_accesses\": " << kDatasetAccesses << ",\n";
  o << "    \"light_cache_rounds\": " << kLightCacheRounds << ",\n";
  o << "    \"light_cache_num_items\": " << kLightCacheItems << ",\n";
  o << "    \"full_dataset_num_items\": " << kDatasetItems << "\n";
  o << "  },\n";

  o << "  \"light_cache\": {\n";
  o << "    \"pre_rounds\": {\n";
  for (size_t i = 0; i < cacheIndices.size(); ++i) {
    o << "      \"" << cacheIndices[i] << "\": \"" << preHex[i] << "\"";
    o << (i + 1 < cacheIndices.size() ? ",\n" : "\n");
  }
  o << "    },\n";
  o << "    \"post_rounds\": {\n";
  for (size_t i = 0; i < cacheIndices.size(); ++i) {
    o << "      \"" << cacheIndices[i] << "\": \"" << postHex[i] << "\"";
    o << (i + 1 < cacheIndices.size() ? ",\n" : "\n");
  }
  o << "    }\n";
  o << "  },\n";

  o << "  \"dataset_items\": {\n";
  for (size_t i = 0; i < ic.order.size(); ++i) {
    o << "    \"" << ic.order[i] << "\": \""
      << hex(ic.items.at(ic.order[i]).bytes, 128) << "\"";
    o << (i + 1 < ic.order.size() ? ",\n" : "\n");
  }
  o << "  },\n";

  o << "  \"vectors\": [\n";
  for (size_t vi = 0; vi < vectors.size(); ++vi) {
    const Vector& v = vectors[vi];
    o << "    {\n";
    o << "      \"header\": \"" << v.header << "\",\n";
    o << "      \"start_value\": \"" << v.startValue << "\",\n";
    o << "      \"rounds\": [\n";
    for (size_t r = 0; r < v.p.size(); ++r) {
      o << "        [" << v.p[r][0] << ", " << v.p[r][1] << ", "
        << v.p[r][2] << ", \"" << v.mixAfterRound[r] << "\"]";
      o << (r + 1 < v.p.size() ? ",\n" : "\n");
    }
    o << "      ],\n";
    o << "      \"collapsed\": \"" << v.collapsed << "\",\n";
  o << "      \"expected\": \"" << v.expected << "\"\n";
    o << "    }";
    o << (vi + 1 < vectors.size() ? ",\n" : "\n");
  }
  o << "  ]\n";
  o << "}\n";

  std::ofstream f(outPath);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", outPath.c_str());
    return 2;
  }
  f << o.str();
  f.close();

  std::fprintf(stderr,
               "fishhash vecgen: %zu vectors, %zu dataset items, "
               "light cache %.0f ms, vectors %.0f ms\n",
               vectors.size(), ic.order.size(), cacheMs, totalMs);
  return 0;
}
