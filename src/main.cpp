// src/main.cpp
//
// fishminer — Iron Fish (FishHash / FIP-3) GPU miner for the POC (U4).
//
// Mine FishHash-active templates from an Iron Fish node's local plain TCP
// adapter. See docs/protocol.md for the consensus serialization (180 bytes)
// and the structured template submitted back to the node.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fishhash.h"
#include "gpu_context.h"
#include "json.h"
#include "rpc_client.h"

using fishhash::Json;

namespace {

// ── Hex helpers ───────────────────────────────────────────────────────────
int hexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

std::string toHex(const uint8_t* b, size_t n) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    out += kDigits[b[i] >> 4];
    out += kDigits[b[i] & 0x0F];
  }
  return out;
}

bool decodeHex(const std::string& s, uint8_t* out, size_t size) {
  if (s.size() != size * 2) return false;
  for (size_t i = 0; i < size; ++i) {
    const int hi = hexVal(s[2 * i]), lo = hexVal(s[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = static_cast<uint8_t>((hi << 4) | lo);
  }
  return true;
}

// ── Header field encoders (endianness verified in U4) ─────────────────────
void putU64BE(uint8_t* p, uint64_t v) {
  p[0] = (uint8_t)(v >> 56);
  p[1] = (uint8_t)(v >> 48);
  p[2] = (uint8_t)(v >> 40);
  p[3] = (uint8_t)(v >> 32);
  p[4] = (uint8_t)(v >> 24);
  p[5] = (uint8_t)(v >> 16);
  p[6] = (uint8_t)(v >> 8);
  p[7] = (uint8_t)v;
}

void putU32LE(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

void putU64LE(uint8_t* p, uint64_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
  p[4] = (uint8_t)(v >> 32);
  p[5] = (uint8_t)(v >> 40);
  p[6] = (uint8_t)(v >> 48);
  p[7] = (uint8_t)(v >> 56);
}

std::string toHexU64BE(uint64_t v) {
  uint8_t b[8];
  putU64BE(b, v);
  return toHex(b, 8);
}

int die(const std::string& msg) {
  std::fprintf(stderr, "fishminer: %s\n", msg.c_str());
  return 1;
}

// ── Args ──────────────────────────────────────────────────────────────────
struct Args {
  std::string host = "127.0.0.1";
  int port = 8020;
  int gpu = 0;
  std::string datasetCache;
  uint32_t batch = 262144;
};

bool parseArgs(int argc, char** argv, Args& a, std::string& err) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) return nullptr;
      return argv[++i];
    };
    if (arg == "--rpc") {
      const char* v = next();
      if (!v) return (err = "--rpc needs a value"), false;
      const std::string s = v;
      const size_t colon = s.rfind(':');
      if (colon == std::string::npos)
        return (err = "--rpc expects host:port"), false;
      a.host = s.substr(0, colon);
      a.port = std::atoi(s.c_str() + colon + 1);
    } else if (arg == "--gpu") {
      const char* v = next();
      if (!v) return (err = "--gpu needs a value"), false;
      a.gpu = std::atoi(v);
    } else if (arg == "--dataset-cache") {
      const char* v = next();
      if (!v) return (err = "--dataset-cache needs a value"), false;
      a.datasetCache = v;
    } else if (arg == "--batch") {
      const char* v = next();
      if (!v) return (err = "--batch needs a value"), false;
      char* end = nullptr;
      const auto batch = std::strtoul(v, &end, 10);
      if (*end || batch == 0 || batch > fishhash::GpuContext::maxSearchBatch)
        return (err = "--batch expects 1..1048576"), false;
      a.batch = static_cast<uint32_t>(batch);
    } else {
      return (err = "unknown arg '" + arg + "'"), false;
    }
  }
  return true;
}

// ── Mining one template ───────────────────────────────────────────────────
struct TemplateState {
  uint8_t header[180];
  uint32_t sequence;
  const uint8_t* target() const { return header + 132; }
};

bool parseTemplate(const Json& tmpl, TemplateState& st) {
  const Json& h = tmpl.at("header");
  const Json& sequence = h.at("sequence");
  const Json& timestamp = h.at("timestamp");
  if (sequence.type != Json::Num || sequence.num < 1 || sequence.num > UINT32_MAX ||
      sequence.num != std::floor(sequence.num) || timestamp.type != Json::Num ||
      timestamp.num < 0 || timestamp.num > 9007199254740991.0 ||
      timestamp.num != std::floor(timestamp.num) || tmpl.at("transactions").type != Json::Arr)
    return false;

  // BlockHasher.serializeHeaderFishHash, not BlockHeader.serialize (Blake3).
  st.sequence = static_cast<uint32_t>(sequence.num);
  uint8_t* p = st.header;
  putU32LE(p + 32, st.sequence);
  putU64LE(p + 164, static_cast<uint64_t>(timestamp.num));
  return decodeHex(h.at("graffiti").str, p, 32) &&
         decodeHex(h.at("previousBlockHash").str, p + 36, 32) &&
         decodeHex(h.at("noteCommitment").str, p + 68, 32) &&
         decodeHex(h.at("transactionCommitment").str, p + 100, 32) &&
         decodeHex(h.at("target").str, p + 132, 32) &&
         decodeHex(h.at("randomness").str, p + 172, 8);
}

}  // namespace

int main(int argc, char** argv) try {
  Args a;
  std::string err;
  if (!parseArgs(argc, argv, a, err)) {
    std::fprintf(stderr,
                 "usage: %s [--rpc host:port] "
                 "[--gpu 0] [--dataset-cache <file>] [--batch 262144]\n",
                 argv[0]);
    std::fprintf(stderr, "%s\n", err.c_str());
    return 2;
  }

  std::printf("fishminer (FishHash / FIP-3) — node %s:%d\n", a.host.c_str(),
              a.port);

  auto rpc = fishhash::NodeRpc::connect(a.host, a.port);
  const auto consensus = rpc->request("chain/getConsensusParameters");
  const Json& activation = consensus.at("enableFishHash");
  if (activation.type != Json::Num) return die("FishHash is not enabled on this chain");
  rpc->subscribeBlockTemplateStream();

  constexpr size_t datasetBytes = size_t{fishhash::kDatasetItems} * 128;
  constexpr double datasetGiB = datasetBytes / 1073741824.0;
  fishhash::GpuContext ctx = fishhash::GpuContext::open(a.gpu);
  if (!a.datasetCache.empty()) {
    std::printf("loading dataset from %s (%.2f GiB)...\n",
                a.datasetCache.c_str(),
                datasetGiB);
    const auto t0 = std::chrono::steady_clock::now();
    if (!ctx.loadDataset(a.datasetCache.c_str())) return 1;
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("  dataset load: %.1f s\n",
                std::chrono::duration<double>(t1 - t0).count());
  } else {
    std::printf("building CPU light cache (%u items)...\n", fishhash::kLightCacheItems);
    fishhash::LightCache lc;
    std::printf("building dataset on GPU (%.2f GiB, ~90 s)...\n",
                datasetGiB);
    ctx.uploadLightCache(reinterpret_cast<const uint8_t*>(lc.data()));
    const auto t0 = std::chrono::steady_clock::now();
    ctx.buildDataset();
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("  dataset build: %.1f s\n",
                std::chrono::duration<double>(t1 - t0).count());
  }

  uint64_t totalHashes = 0;
  uint64_t reportHashes = 0;
  auto reportAt = std::chrono::steady_clock::now();
  for (;;) {
    Json tmpl = rpc->nextBlockTemplate();
    TemplateState st;
    if (!parseTemplate(tmpl, st)) return die("malformed block template");
    if (st.sequence < activation.num) return die("FishHash is not active at this sequence");
    std::printf("template seq=%u target=%s\n", st.sequence, toHex(st.target(), 32).c_str());
    std::fflush(stdout);

    uint64_t nonceBase = 0;
    while (!rpc->hasPendingTemplate()) {
      uint8_t digest[32];
      const uint32_t winner = ctx.findNonce(st.header, nonceBase, a.batch, digest);
      totalHashes += a.batch;
      reportHashes += a.batch;
      const auto now = std::chrono::steady_clock::now();
      const double secs = std::chrono::duration<double>(now - reportAt).count();
      if (secs >= 5) {
        std::printf("  %.2f MH/s  (%llu total hashes)\n",
                    reportHashes / secs / 1e6, (unsigned long long)totalHashes);
        std::fflush(stdout);
        reportAt = now;
        reportHashes = 0;
      }

      if (winner == UINT32_MAX) {
        nonceBase += a.batch;
        continue;
      }
      const std::string nonce = toHexU64BE(nonceBase + winner);
      std::printf("candidate seq=%u randomness=%s hash=%s\n", st.sequence, nonce.c_str(),
                  toHex(digest, sizeof digest).c_str());
      tmpl.at("header").at("randomness").str = nonce;
      const Json resp = rpc->request("miner/submitBlock", &tmpl);
      const Json& added = resp.at("added");
      const std::string& reason = resp.at("reason").str;
      if (added.type != Json::Bool || reason.empty() || added.b != (reason == "SUCCESS"))
        return die("inconsistent miner/submitBlock response");
      std::printf("  submit: added=%s reason=%s\n", added.b ? "true" : "false", reason.c_str());
      if (added.b) std::puts("  *** BLOCK ACCEPTED ***");
      std::fflush(stdout);
      if (!added.b && reason != "CHAIN_CHANGED" && reason != "FORK")
        return die("block rejected: " + reason);
      break;  // Wait for a template on the new head.
    }
  }
} catch (const std::out_of_range& e) {
  return die(std::string("node RPC response missing a required field: ") + e.what());
} catch (const fishhash::RpcError& e) {
  return die(std::string("node RPC: ") + e.what());
}
