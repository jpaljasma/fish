# FishHash CUDA miner

A single-GPU Iron Fish solo miner for a full node. It builds a static 4.5 GiB FishHash dataset and searches the node's block templates. It does not speak pool Stratum or create reward transactions; payouts go to the account configured on the node.

## Windows build and correctness checks

Install a CUDA Toolkit that supports sm_89 and sm_120, CMake, and Visual Studio Build Tools with Desktop development with C++ and a Windows SDK. From a Windows command prompt at the project root:

```bat
tools\build.bat
build\test_gpu_match.exe tests\vectors\vectors.json --dataset-cache build\fishhash-dataset.bin
```

CTest checks the CPU implementation. The separate GPU check compares dataset items and header hashes to reference vectors, then checks a multi-block nonce sweep against the CPU and a known Rust-oracle winner. Its optional cache needs 4.5 GiB of disk space; omit `--dataset-cache` to build only in VRAM.
The miner and benchmark verify the cache length and BLAKE3 fingerprint while loading it; a corrupt cache is rejected rather than used to search for blocks.

## Run against your node

Use a synced, FishHash-active node with a default mining account. Enable its **plain TCP RPC on loopback only** (`enableRpcTcp: true`, `enableRpcTls: false`, `rpcTcpHost: "127.0.0.1"`). Iron Fish SDK 2.12.0 defaults to TCP port 8020. Plain TCP has no authentication or encryption: do not expose that listener to a LAN or the internet.

```bat
build\fishminer.exe --rpc 127.0.0.1:8020 --gpu 0 --dataset-cache build\fishhash-dataset.bin
```

Without `--dataset-cache`, the miner generates the dataset at startup. Stop with Ctrl+C. A locally found nonce is not proof of a mined block: use the node's `miner/submitBlock` response to distinguish acceptance from a stale or rejected submission. See [protocol and byte layout](docs/protocol.md).

## RTX 4090 tuning

Mining defaults to `--batch 262144`. The GPU checks the target and returns one 36-byte result per batch, not every digest. Reduce the batch for faster template switching, or increase it up to 1,048,576. The miner does not set clocks, power limits, or a utilization cap.

To reproduce the measurements, run the following commands **separately**: the offline benchmark with the miner stopped, and the utilization monitor in another terminal while the miner is running on GPU 0. Interrupt the monitor with Ctrl+C.

```bat
build\fishbench.exe build\fishhash-dataset.bin
nvidia-smi -i 0 --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw --format=csv -l 1
```

Live, private FishHash node; telemetry sampled once per second after startup. These are device-wide counters, not per-process occupancy. The two windows used different block templates.

| Mining path | Batch | Samples | GPU busy (median / peak) | Median miner MH/s |
|---|---:|---:|---:|---:|
| CPU target check | 16,384 | 20 | 93% / 94% | 15.34 |
| GPU target check | 262,144 | 35 | 100% / 100% | 17.89 |

The latter run reached 100% in 28 of 35 samples and received 33 accepted submissions during the window. It did not sustain 100% uninterrupted.
Median board power rose from 191 W to 228 W across those windows; the higher busy time was not a demonstrated improvement in hashes per watt.

Offline GPU search, same card, five seconds per batch size; excludes node waits and dataset setup, includes hit readback:

| Nonces per batch | MH/s | % of 82.03 MH/s bandwidth ceiling |
|---:|---:|---:|
| 16,384 | 17.678 | 21.55 |
| 65,536 | 18.362 | 22.38 |
| 262,144 | 18.574 | 22.64 |
| 1,048,576 | 18.171 | 22.15 |

The 53 MH/s throughput target remains unmet. A 100% GPU-busy reading means kernels ran throughout that sample, not that the memory bus delivered peak bandwidth. The sm_120 target was compiled, not hardware-tested.
