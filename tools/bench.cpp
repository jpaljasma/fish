// End-to-end mining batch time, including the on-device target check and hit readback.
#include "gpu_context.h"

#include <array>
#include <chrono>
#include <cstdio>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: fishbench <dataset-cache>\n");
    return 2;
  }
  auto gpu = fishhash::GpuContext::open(0);
  if (!gpu.loadDataset(argv[1])) return 2;
  using Clock = std::chrono::steady_clock;
  std::array<uint8_t, 180> header{};
  constexpr double ceiling = 1008e9 / (32 * 3 * 128);  // RTX 4090, hashes/s
  std::puts("batch,hashes,elapsed_s,MH_s,pct_4090_ceiling");
  uint8_t digest[32];
  for (const uint32_t batch : {16384u, 65536u, 262144u, 1048576u}) {
    gpu.findNonce(header.data(), 0, batch, digest);
    uint64_t hashes = 0;
    const auto begin = Clock::now();
    double seconds;
    do {
      gpu.findNonce(header.data(), batch + hashes, batch, digest);
      hashes += batch;
      seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    } while (seconds < 5);
    std::printf("%u,%llu,%.3f,%.3f,%.2f\n", batch,
                static_cast<unsigned long long>(hashes), seconds,
                hashes / seconds / 1e6, 100 * hashes / seconds / ceiling);
  }
}
