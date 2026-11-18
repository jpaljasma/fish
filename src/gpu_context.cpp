// src/gpu_context.cpp
//
// Host-side CUDA context (U3). Allocates device buffers, uploads the light
// cache, builds/loads the dataset, and runs single-header hashes.
//
// Buffer sizes:
//   light cache: kLightCacheItems * 64 B  = 75,497,024 B  (72.0 MiB)
//   dataset:     kDatasetItems * 128 B    = 4,831,835,776 B (4.497 GiB)

#include "gpu_context.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>

#include "blake3.h"
#include "fishhash.h"  // kLightCacheItems, kDatasetItems

namespace fishhash {
namespace {

#define CHECK_CUDA(expr)                                                      \
  do {                                                                        \
    cudaError_t err__ = (expr);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(err__));                               \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

size_t lightCacheBytes() { return static_cast<size_t>(kLightCacheItems) * 64; }
size_t datasetBytes() { return static_cast<size_t>(kDatasetItems) * 128; }

}  // namespace

GpuContext GpuContext::open(int deviceIndex) {
  GpuContext ctx;
  int count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&count));
  if (count == 0) {
    std::fprintf(stderr, "no CUDA devices available\n");
    std::abort();
  }
  if (deviceIndex >= 0) {
    if (deviceIndex >= count) {
      std::fprintf(stderr, "GPU index %d out of range (have %d)\n",
                   deviceIndex, count);
      std::abort();
    }
    ctx.device_ = deviceIndex;
  } else {
    // Fastest available: largest VRAM, matching the reference miner.
    size_t bestMem = 0;
    for (int i = 0; i < count; ++i) {
      cudaDeviceProp prop;
      CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
      if (static_cast<size_t>(prop.totalGlobalMem) > bestMem) {
        bestMem = static_cast<size_t>(prop.totalGlobalMem);
        ctx.device_ = i;
      }
    }
  }
  CHECK_CUDA(cudaSetDevice(ctx.device_));
  CHECK_CUDA(cudaStreamCreate(&ctx.stream_));
  CHECK_CUDA(cudaMalloc(&ctx.d_header_, 192));
  CHECK_CUDA(cudaMalloc(&ctx.d_result_, 9 * sizeof(uint32_t)));
  CHECK_CUDA(cudaHostAlloc(&ctx.h_stage_, 192, cudaHostAllocDefault));
  return ctx;
}

GpuContext::~GpuContext() {
  if (d_light_) cudaFree(d_light_);
  if (d_dataset_) cudaFree(d_dataset_);
  if (d_header_) cudaFree(d_header_);
  if (d_result_) cudaFree(d_result_);
  if (d_batchOut_) cudaFree(d_batchOut_);
  if (h_stage_) cudaFreeHost(h_stage_);
  if (h_batchStage_) cudaFreeHost(h_batchStage_);
  if (stream_) cudaStreamDestroy(stream_);
}

void GpuContext::uploadLightCache(const uint8_t* hostData) {
  if (!d_light_) CHECK_CUDA(cudaMalloc(&d_light_, lightCacheBytes()));
  CHECK_CUDA(cudaMemcpy(d_light_, hostData, lightCacheBytes(),
                        cudaMemcpyHostToDevice));
}

void GpuContext::buildDataset() {
  if (!d_light_) {
    std::fprintf(stderr, "buildDataset requires an uploaded light cache\n");
    std::abort();
  }
  if (!d_dataset_) CHECK_CUDA(cudaMalloc(&d_dataset_, datasetBytes()));
  launchBuildDataset(reinterpret_cast<const uint32_t*>(d_light_),
                     reinterpret_cast<uint32_t*>(d_dataset_), stream_);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream_));
}

bool GpuContext::loadDataset(const char* path) {
  auto fail = [&](const char* reason) {
    if (d_dataset_) CHECK_CUDA(cudaFree(d_dataset_));
    d_dataset_ = nullptr;
    std::fprintf(stderr, "dataset cache %s: %s\n", path, reason);
    return false;
  };
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return fail("cannot open file");
  const size_t size = datasetBytes();
  if (in.tellg() != static_cast<std::streamoff>(size)) return fail("incorrect file size");
  if (!in.seekg(0)) return fail("cannot seek to start");
  constexpr size_t chunkBytes = 8u << 20;
  std::unique_ptr<uint8_t[]> chunk(new uint8_t[chunkBytes]);
  if (!d_dataset_) CHECK_CUDA(cudaMalloc(&d_dataset_, size));
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  for (size_t offset = 0; offset < size;) {
    const size_t n = std::min(chunkBytes, size - offset);
    if (!in.read(reinterpret_cast<char*>(chunk.get()), n)) return fail("incomplete read");
    blake3_hasher_update(&hasher, chunk.get(), n);
    CHECK_CUDA(cudaMemcpy(static_cast<uint8_t*>(d_dataset_) + offset, chunk.get(), n,
                          cudaMemcpyHostToDevice));
    offset += n;
  }
  if (in.peek() != std::ifstream::traits_type::eof() || in.bad())
    return fail("unexpected trailing data or read error");
  constexpr uint8_t expected[] = {
      0xb9, 0x5b, 0x23, 0xfe, 0x6b, 0xcf, 0x04, 0xcf, 0xc9, 0x8a, 0x1a, 0xc5, 0x6e, 0x88, 0x32, 0xbe,
      0xa4, 0x16, 0x5a, 0x3c, 0xc4, 0xc6, 0x86, 0x44, 0x59, 0x5c, 0x41, 0x40, 0xd4, 0x9e, 0xf9, 0x04};
  uint8_t digest[BLAKE3_OUT_LEN];
  static_assert(sizeof(expected) == sizeof(digest));
  blake3_hasher_finalize(&hasher, digest, sizeof(digest));
  if (std::memcmp(digest, expected, sizeof(digest))) return fail("BLAKE3 digest mismatch");
  return true;
}

void GpuContext::readbackDatasetItem(uint32_t index, uint8_t* dst128) {
  if (!d_dataset_) {
    std::fprintf(stderr, "readbackDatasetItem before dataset build/load\n");
    std::abort();
  }
  CHECK_CUDA(cudaMemcpy(dst128,
                        reinterpret_cast<const uint8_t*>(d_dataset_) +
                            static_cast<size_t>(index) * 128,
                        128, cudaMemcpyDeviceToHost));
}

void GpuContext::readbackDatasetChunk(size_t byteOffset, size_t bytes,
                                      uint8_t* dst) {
  if (!d_dataset_ || (byteOffset & 3) || (bytes & 3) ||
      byteOffset + bytes > datasetBytes()) {
    std::fprintf(stderr, "readbackDatasetChunk: bad range\n");
    std::abort();
  }
  CHECK_CUDA(cudaMemcpy(dst,
                        reinterpret_cast<const uint8_t*>(d_dataset_) +
                            byteOffset,
                        bytes, cudaMemcpyDeviceToHost));
}

std::vector<uint8_t> GpuContext::hashHeader(const uint8_t* header,
                                            size_t headerLen) {
  if (!d_dataset_ || headerLen > 192) return {};
  std::memcpy(h_stage_, header, headerLen);
  CHECK_CUDA(cudaMemcpyAsync(d_header_, h_stage_, headerLen,
                             cudaMemcpyHostToDevice, stream_));
  launchHashHeader(reinterpret_cast<const uint8_t*>(d_header_),
                   static_cast<uint32_t>(headerLen),
                   reinterpret_cast<const uint32_t*>(d_dataset_),
                   reinterpret_cast<uint32_t*>(d_result_), stream_);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpyAsync(h_stage_, d_result_, 32,
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaStreamSynchronize(stream_));
  auto* st = reinterpret_cast<uint8_t*>(h_stage_);
  return std::vector<uint8_t>(st, st + 32);
}

std::vector<uint8_t> GpuContext::hashHeaderBatch(const uint8_t* header,
                                                 size_t headerLen,
                                                 uint64_t nonceStart,
                                                 size_t count) {
  if (!d_dataset_ || headerLen != 180 || count == 0) return {};
  constexpr size_t kMaxBatch = 65536;
  if (count > kMaxBatch) count = kMaxBatch;
  if (!d_batchOut_) {
    CHECK_CUDA(cudaMalloc(&d_batchOut_, kMaxBatch * 32));
    CHECK_CUDA(cudaHostAlloc(&h_batchStage_, kMaxBatch * 32, cudaHostAllocDefault));
  }
  launchHashHeaderBatch(header, nonceStart, static_cast<uint32_t>(count),
                        reinterpret_cast<const uint32_t*>(d_dataset_),
                        reinterpret_cast<uint32_t*>(d_batchOut_), stream_);
  CHECK_CUDA(cudaGetLastError());
  const size_t bytes = count * 32;
  CHECK_CUDA(cudaMemcpyAsync(h_batchStage_, d_batchOut_, bytes,
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaStreamSynchronize(stream_));
  auto* st = reinterpret_cast<uint8_t*>(h_batchStage_);
  return std::vector<uint8_t>(st, st + bytes);
}

uint32_t GpuContext::findNonce(const uint8_t header[180], uint64_t first,
                               uint32_t count, uint8_t digest[32]) {
  if (!d_dataset_ || !count || count > maxSearchBatch) {
    std::fprintf(stderr, "invalid nonce search: dataset=%p, count=%u\n", d_dataset_, count);
    std::abort();
  }
  CHECK_CUDA(cudaMemsetAsync(d_result_, 0xff, 9 * sizeof(uint32_t), stream_));
  launchNonceSearch(header, first, count, static_cast<const uint32_t*>(d_dataset_),
                    static_cast<uint32_t*>(d_result_), stream_);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpyAsync(h_stage_, d_result_, 9 * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaStreamSynchronize(stream_));
  const auto* hit = static_cast<const uint32_t*>(h_stage_);
  if (hit[0] != UINT32_MAX) std::memcpy(digest, hit + 1, 32);
  return hit[0];
}

}  // namespace fishhash
