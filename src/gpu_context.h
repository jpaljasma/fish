// src/gpu_context.h
//
// Host-side CUDA context for the FishHash POC (U3). Owns device buffers and
// launches the kernels from src/kernels/fishhash.cu.
//
//   GpuContext ctx = GpuContext::open(0);
//   ctx.uploadLightCache(lc.data());   // 72 MiB, from CPU LightCache
//   ctx.buildDataset();                // 4.83 GiB in VRAM, minutes
//   auto hash = ctx.hashHeader(header, 180);  // 32 bytes

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fishhash {

class GpuContext {
 public:
  // Opens the CUDA device. `deviceIndex < 0` picks the fastest available
  // device (as the reference miner does). Exits on CUDA error.
  static GpuContext open(int deviceIndex = 0);
  ~GpuContext();

  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;
  // Move only: open() returns by value. The moved-from object is left in
  // a safe empty state (destructor is a no-op on it).
  GpuContext(GpuContext&& o) noexcept
      : device_(o.device_),
        d_light_(o.d_light_),
        d_dataset_(o.d_dataset_),
        d_header_(o.d_header_),
        d_result_(o.d_result_),
        d_batchOut_(o.d_batchOut_),
        h_stage_(o.h_stage_),
        h_batchStage_(o.h_batchStage_),
        stream_(o.stream_) {
    o.device_ = -1;
    o.d_light_ = o.d_dataset_ = o.d_header_ = o.d_result_ = nullptr;
    o.d_batchOut_ = nullptr;
    o.h_stage_ = o.h_batchStage_ = nullptr;
    o.stream_ = nullptr;
  }
  GpuContext& operator=(GpuContext&&) = delete;

  int deviceIndex() const { return device_; }

  // Uploads the light cache (kLightCacheItems x 64 B = 75,497,024 bytes).
  void uploadLightCache(const uint8_t* hostData);

  // Builds the full dataset (kDatasetItems x 128 B = 4,831,835,776 bytes)
  // on the GPU. Slow (minutes); call once and cache externally if needed.
  void buildDataset();

  // Streams and verifies a raw dataset cache. On failure, reports the
  // reason to stderr and leaves the context without a dataset.
  bool loadDataset(const char* path);

  bool hasDataset() const { return d_dataset_ != nullptr; }

  // Copies one 128-byte dataset item back to the host (test hook).
  void readbackDatasetItem(uint32_t index, uint8_t* dst128);

  // Copies a byte range of the dataset back to the host (cache save).
  // Offset and length must be multiples of 4 within the dataset.
  void readbackDatasetChunk(size_t byteOffset, size_t bytes, uint8_t* dst);

  // Hashes one header via the GPU kernel (headerLen <= 192).
  // Returns 32 bytes; empty vector on error.
  std::vector<uint8_t> hashHeader(const uint8_t* header, size_t headerLen);

  // Hashes the nonce range nonceStart .. nonceStart+count-1 over one base
  // FishHash header (exactly 180 bytes; nonce at offset 172, big-endian).
  // Returns count 32-byte hashes concatenated (count <= 65536,
  // larger requests are clamped); empty vector on error.
  std::vector<uint8_t> hashHeaderBatch(const uint8_t* header, size_t headerLen,
                                       uint64_t nonceStart, size_t count);

  // Returns a winning offset and fills digest, or UINT32_MAX for no hit.
  // The target is part of the 180-byte FishHash header (bytes 132..163).
  static constexpr uint32_t maxSearchBatch = 1u << 20;
  uint32_t findNonce(const uint8_t header[180], uint64_t first, uint32_t count,
                     uint8_t digest[32]);

 private:
  GpuContext() = default;
  int device_ = -1;
  void* d_light_ = nullptr;    // uint32_t [kLightCacheItems * 16]
  void* d_dataset_ = nullptr;  // uint32_t [kDatasetItems * 32]
  void* d_header_ = nullptr;   // scratch, 192 B
  void* d_result_ = nullptr;   // single hash or [winner offset, 8 hash words]
  void* d_batchOut_ = nullptr; // scratch, 65536 * 32 B
  void* h_stage_ = nullptr;    // pinned staging for a header or hit record
  void* h_batchStage_ = nullptr;  // pinned staging, 65536 * 32 B
  cudaStream_t stream_ = nullptr;
};

// Kernel launchers (defined in src/kernels/fishhash.cu).
void launchBuildDataset(const uint32_t* d_lightCache, uint32_t* d_dataset,
                        cudaStream_t stream);
void launchHashHeader(const uint8_t* d_header, uint32_t headerLen,
                      const uint32_t* d_dataset, uint32_t* d_out,
                      cudaStream_t stream);
void launchHashHeaderBatch(const uint8_t* h_baseHeader, uint64_t nonceStart,
                           uint32_t count,
                           const uint32_t* d_dataset, uint32_t* d_out,
                           cudaStream_t stream);
void launchNonceSearch(const uint8_t* base, uint64_t first, uint32_t count,
                       const uint32_t* dataset, uint32_t* hit, cudaStream_t stream);

}  // namespace fishhash
