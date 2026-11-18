// tests/test_gpu_match.cpp
//
// U3 oracle gate: verifies the CUDA kernels (src/kernels/fishhash.cu)
// against the CPU port and the cross-checked vectors in vectors.json.
//
//   1. dataset build/load  — 10 items from vectors.json (cross-checked
//                             against iron-fish/fish-hash crate v0.3.0)
//                             plus items 0 and N-1 vs the CPU port
//   2. hash match          — all 8 vectors: GPU == JSON expected, and
//                             GPU == CPU, byte-exact
//   3. planted solution    — one mining batch, checked nonce-by-nonce against
//                             the CPU; only index 257 meets the header target
//
// The light-cache upload is verified indirectly: dataset items are derived
// from it and are checked against JSON/CPU.
//
// Usage: test_gpu_match <vectors.json> [--gpu N] [--dataset-cache <path>]
//
// --dataset-cache: load the dataset from that file instead of building; if
// the file is absent, build on the GPU and save the file (4.83 GB raw).

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "fishhash.h"
#include "gpu_context.h"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);              \
    }                                                                        \
  } while (0)

// ── Minimal JSON parser (copied from tests/test_fishhash.cpp) ─────────────
struct Json {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  const Json& at(const std::string& k) const {
    auto it = obj.find(k);
    if (it == obj.end()) {
      std::printf("JSON missing key '%s'\n", k.c_str());
      std::exit(2);
    }
    return it->second;
  }
};

struct JsonParser {
  std::string s;  // owned: a reference to a temporary (e.g. ss.str()) would dangle
  size_t i = 0;

  explicit JsonParser(std::string src) : s(std::move(src)) {}

  [[noreturn]] void fail(const char* what) {
    const size_t from = i < 24 ? 0 : i - 24;
    std::printf("JSON parse fail at %s (pos %zu of %zu):\n%s...\n", what, i,
                s.size(), s.substr(from, 72).c_str());
    std::exit(2);
  }

  void skip() {
    while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
  }

  Json parseValue() {
    skip();
    if (i >= s.size()) return {};
    char c = s[i];
    if (c == '{') return parseObject();
    if (c == '[') return parseArray();
    if (c == '"') return Json{Json::Str, false, 0, parseString(), {}, {}};
    if (c == 't') { i += 4; return {Json::Bool, true, 0, {}, {}, {}}; }
    if (c == 'f') { i += 5; return {Json::Bool, false, 0, {}, {}, {}}; }
    if (c == 'n') { i += 4; return {}; }
    return parseNumber();
  }

  Json parseObject() {
    Json j;
    j.type = Json::Obj;
    ++i;  // {
    skip();
    if (i < s.size() && s[i] == '}') { ++i; return j; }
    while (true) {
      skip();
      std::string key = parseString();
      skip();
      if (s[i] != ':') fail("expected ':' after key");
      ++i;
      j.obj[key] = parseValue();
      skip();
      if (s[i] == ',') { ++i; continue; }
      if (s[i] == '}') { ++i; break; }
      fail("expected ',' or '}' in object");
    }
    return j;
  }

  Json parseArray() {
    Json j;
    j.type = Json::Arr;
    ++i;  // [
    skip();
    if (i < s.size() && s[i] == ']') { ++i; return j; }
    while (true) {
      j.arr.push_back(parseValue());
      skip();
      if (s[i] == ',') { ++i; continue; }
      if (s[i] == ']') { ++i; break; }
      fail("expected ',' or ']' in array");
    }
    return j;
  }

  std::string parseString() {
    if (s[i] != '"') fail("expected string");
    ++i;
    std::string out;
    while (s[i] != '"') {
      if (s[i] == '\\') {
        ++i;
        switch (s[i]) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          default: out += s[i]; break;
        }
      } else {
        out += s[i];
      }
      ++i;
    }
    ++i;  // closing quote
    return out;
  }

  Json parseNumber() {
    size_t start = i;
    while (i < s.size() &&
           (std::isdigit((unsigned char)s[i]) || s[i] == '-' || s[i] == '+' ||
            s[i] == '.' || s[i] == 'e' || s[i] == 'E'))
      ++i;
    return {Json::Num, false, std::stod(s.substr(start, i - start)), {}, {}, {}};
  }
};

// ── Hex helpers ───────────────────────────────────────────────────────────
int hexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

std::vector<uint8_t> fromHex(const std::string& hex) {
  std::vector<uint8_t> out;
  out.reserve(hex.size() / 2);
  for (size_t i = 0; i + 1 < hex.size(); i += 2) {
    int hi = hexVal(hex[i]);
    int lo = hexVal(hex[i + 1]);
    if (hi < 0 || lo < 0) {
      std::printf("bad hex at %zu in '%s'\n", i, hex.substr(0, 32).c_str());
      std::exit(2);
    }
    out.push_back(static_cast<uint8_t>((hi << 4) | lo));
  }
  return out;
}

std::string toHex(const uint8_t* p, size_t n) {
  static const char* kDigits = "0123456789abcdef";
  std::string s;
  s.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    s += kDigits[p[i] >> 4];
    s += kDigits[p[i] & 0xF];
  }
  return s;
}

std::vector<uint8_t> hexBytes(const Json& j, const char* what) {
  if (j.type != Json::Str) {
    std::printf("JSON %s: expected string\n", what);
    std::exit(2);
  }
  return fromHex(j.str);
}

void checkMiningBatch(fishhash::GpuContext& ctx, const fishhash::ItemSource& source) {
  // Offline FishHash template in the node's serializeHeaderFishHash layout.
  // The fixed target is part of the header being hashed, not chosen afterwards.
  uint8_t header[180] = {};
  for (size_t i = 0; i < 132; ++i) header[i] = static_cast<uint8_t>(i + 1);
  uint8_t* const target = header + 132;
  std::fill(target + 1, target + 32, 0xff);  // 00ff...ff (u256 BE)
  constexpr uint64_t timestamp = 1700000000000ULL;
  for (unsigned b = 0; b < 8; ++b)
    header[164 + b] = static_cast<uint8_t>(timestamp >> (8 * b));

  // @ironfish/rust-nodejs 2.10.0, FishHashContext(false).hash(header).
  // The interval crosses a low-u32 carry and the 256-thread block boundary.
  constexpr uint64_t firstNonce = 0x01234567ffffff91ULL;
  constexpr uint64_t plantedNonce = 0x0123456800000092ULL;
  constexpr size_t count = 258;
  constexpr char plantedHash[] =
      "007644292ffd743c6bbc0f62f2d39f3c1859ba4b32c7ecf63339d75b6032d4ad";
  const auto batch = ctx.hashHeaderBatch(header, sizeof header, firstNonce, count);
  CHECK(batch.size() == count * 32, "batch returned one digest per nonce");
  if (batch.size() != count * 32) return;

  size_t cpuHits = 0, gpuHits = 0;
  uint64_t cpuWinner = 0, gpuWinner = 0;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t nonce = firstNonce + i;
    // Serialize independently of the GPU's nonce patch: final eight bytes, BE.
    for (unsigned b = 0; b < 8; ++b)
      header[179 - b] = static_cast<uint8_t>(nonce >> (8 * b));
    const auto cpu = fishhash::fishHash(source, header, sizeof header);
    if (nonce == plantedNonce)
      CHECK(toHex(cpu.bytes, 32) == plantedHash, "planted hash matches Rust oracle");
    if (std::memcmp(cpu.bytes, target, 32) <= 0) {
      ++cpuHits;
      cpuWinner = nonce;
    }

    const uint8_t* gpu = batch.data() + i * 32;
    char msg[96];
    std::snprintf(msg, sizeof msg, "batch digest at index %zu matches CPU", i);
    CHECK(std::memcmp(cpu.bytes, gpu, 32) == 0, msg);
    if (fishhash::meetsTarget(gpu, target)) {
      ++gpuHits;
      gpuWinner = nonce;
    }
  }
  CHECK(cpuHits == 1 && cpuWinner == plantedNonce, "CPU finds only the planted nonce");
  CHECK(gpuHits == 1 && gpuWinner == plantedNonce, "batch finds only the planted nonce");
  uint8_t winningHash[32];
  const auto offset = ctx.findNonce(header, firstNonce, count, winningHash);
  CHECK(offset == plantedNonce - firstNonce &&
        toHex(winningHash, 32) == plantedHash, "GPU search returns the Rust-oracle winner");
  CHECK(ctx.findNonce(header, firstNonce, count - 1, winningHash) == UINT32_MAX,
        "next search clears the prior hit when the range has no winner");
}

}  // namespace

int main(int argc, char** argv) {
  std::string vectorsPath;
  int gpuIndex = 0;
  std::string datasetCache;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--gpu" && i + 1 < argc) gpuIndex = std::atoi(argv[++i]);
    else if (a == "--dataset-cache" && i + 1 < argc)
      datasetCache = argv[++i];
    else if (vectorsPath.empty()) vectorsPath = a;
  }
  if (vectorsPath.empty()) {
    std::printf("usage: test_gpu_match <vectors.json> [--gpu N] "
                "[--dataset-cache <path>]\n");
    return 2;
  }

  std::ifstream fin(vectorsPath);
  if (!fin) {
    std::printf("cannot open %s\n", vectorsPath.c_str());
    return 2;
  }
  std::stringstream ss;
  ss << fin.rdbuf();
  JsonParser parser(ss.str());
  Json root = parser.parseValue();
  const Json& vectors = root.at("vectors");

  // ── Stage 0: CPU light cache (for CPU cross-checks) ────────────────────
  std::printf("== CPU light cache ==\n");
  auto t0 = std::chrono::steady_clock::now();
  fishhash::LightCache lc;
  auto t1 = std::chrono::steady_clock::now();
  std::printf("  built in %.2fs\n", std::chrono::duration<double>(t1 - t0).count());

  // ── Stage 1: open GPU ──────────────────────────────────────────────────
  std::printf("== GPU context (device %d) ==\n", gpuIndex);
  fishhash::GpuContext ctx = fishhash::GpuContext::open(gpuIndex);
  std::printf("  using device %d\n", ctx.deviceIndex());

  // ── Stage 2: dataset (load from cache file or build) ───────────────────
  std::printf("== dataset ==\n");
  const size_t dsBytes = static_cast<size_t>(fishhash::kDatasetItems) * 128;
  bool loadedFromCache = false;
  if (!datasetCache.empty()) {
    std::error_code error;
    const bool exists = std::filesystem::exists(datasetCache, error);
    if (error) {
      std::fprintf(stderr, "cannot inspect %s: %s\n", datasetCache.c_str(),
                   error.message().c_str());
      return 2;
    }
    if (exists) {
      std::printf("  loading %zu bytes from %s\n", dsBytes, datasetCache.c_str());
      if (!ctx.loadDataset(datasetCache.c_str())) return 2;
      loadedFromCache = true;
    } else {
      std::printf("  no cache at %s; building on GPU\n", datasetCache.c_str());
    }
  }
  if (!loadedFromCache) {
    ctx.uploadLightCache(reinterpret_cast<const uint8_t*>(lc.data()));
    ctx.buildDataset();
  }

  if (!ctx.hasDataset()) {
    std::printf("dataset build failed\n");
    return 2;
  }
  t1 = std::chrono::steady_clock::now();
  std::printf("  dataset ready; elapsed since start: %.1fs\n",
              std::chrono::duration<double>(t1 - t0).count());

  // Save the built dataset so later runs can skip the build.
  if (!loadedFromCache && !datasetCache.empty()) {
    std::ofstream of(datasetCache, std::ios::binary | std::ios::trunc);
    if (!of) {
      std::printf("cannot write %s\n", datasetCache.c_str());
      return 2;
    }
    std::vector<uint8_t> chunk(128u << 20);
    for (size_t off = 0; off < dsBytes; off += chunk.size()) {
      size_t n = std::min(chunk.size(), dsBytes - off);
      ctx.readbackDatasetChunk(off, n, chunk.data());
      of.write(reinterpret_cast<char*>(chunk.data()), (std::streamsize)n);
    }
    of.close();
    if (!of) {
      std::printf("cannot save %s\n", datasetCache.c_str());
      return 2;
    }
    std::printf("  saved dataset cache to %s\n", datasetCache.c_str());
  }

  // ── Stage 3: dataset item verification ─────────────────────────────────
  std::printf("== dataset items ==\n");
  {
    uint8_t buf[128];
    const Json& items = root.at("dataset_items");
    for (const auto& [idx, hex] : items.obj) {
      uint32_t index = static_cast<uint32_t>(std::stoul(idx));
      std::vector<uint8_t> want = fromHex(hex.str);
      ctx.readbackDatasetItem(index, buf);
      char msg[96];
      std::snprintf(msg, sizeof msg, "dataset item %u matches JSON", index);
      CHECK(std::memcmp(buf, want.data(), 128) == 0, msg);
    }
    const uint32_t extra[] = {0u, fishhash::kDatasetItems - 1};
    for (uint32_t index : extra) {
      fishhash::Hash1024 cpu = fishhash::datasetItem(lc, index);
      ctx.readbackDatasetItem(index, buf);
      char msg[96];
      std::snprintf(msg, sizeof msg, "dataset item %u matches CPU port",
                    index);
      CHECK(std::memcmp(buf, cpu.bytes, 128) == 0, msg);
    }
  }

  // ── Stage 4: hash match for all vectors ────────────────────────────────
  std::printf("== hash match (GPU vs JSON vs CPU) ==\n");
  fishhash::ComputedItemSource source(lc);
  for (size_t v = 0; v < vectors.arr.size(); ++v) {
    const Json& vec = vectors.arr[v];
    std::vector<uint8_t> header = hexBytes(vec.at("header"), "header");
    std::vector<uint8_t> expected = hexBytes(vec.at("expected"), "expected");

    std::vector<uint8_t> gpu = ctx.hashHeader(header.data(), header.size());
    char msg[220];
    std::snprintf(msg, sizeof msg,
                  "vector %zu GPU %s == JSON %s", v,
                  toHex(gpu.data(), gpu.size()).substr(0, 16).c_str(),
                  toHex(expected.data(), expected.size()).substr(0, 16).c_str());
    CHECK(gpu.size() == 32 && gpu == expected, msg);

    fishhash::Hash256 cpuHash =
        fishhash::fishHash(source, header.data(), header.size());
    std::snprintf(msg, sizeof msg, "vector %zu CPU == JSON", v);
    CHECK(expected.size() == 32 && std::memcmp(cpuHash.bytes, expected.data(), 32) == 0,
          msg);
  }

  // ── Stage 5: planted solution ──────────────────────────────────────────
  std::printf("== planted solution ==\n");
  checkMiningBatch(ctx, source);

  // ── Summary ────────────────────────────────────────────────────────────
  std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
