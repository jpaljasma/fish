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

Mining defaults to `--batch 262144`. For each template the miner prepares the BLAKE3 prefix once. For each batch it computes the nonce seeds, mixes each nonce across eight CUDA lanes, and checks the target on the GPU. Only the 36-byte hit record comes back to the host. The reusable workspace takes 24 MiB at the default batch size or 96 MiB at the maximum of 1,048,576. Smaller batches shorten the time before the miner can switch templates. The miner does not change clocks or power limits.

To reproduce the measurements, run the following commands **separately**: the offline benchmark with the miner stopped, and the utilization monitor in another terminal while the miner is running on GPU 0. Interrupt the monitor with Ctrl+C.

```bat
build\fishbench.exe build\fishhash-dataset.bin
nvidia-smi -i 0 --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw --format=csv -l 1
```

Live private-node measurements. `nvidia-smi` sampled device-wide busy time once a second; the miner reported wall-clock MH/s every five seconds, including template and submission waits. The runs used different templates and difficulties.

| Mining path | Batch | Samples | GPU busy (median / peak) | Median miner MH/s |
|---|---:|---:|---:|---:|
| Previous CPU target check | 16,384 | 20 | 93% / 94% | 15.34 |
| Previous single-lane GPU target check | 262,144 | 35 | 100% / 100% | 17.89 |
| Eight lanes, fresh low-difficulty devnet | 262,144 | 121 | 10% / 80% | 2.20 |
| Eight lanes, higher-difficulty private genesis | 262,144 | 121 | 100% / 100% | 64.88 |

On the harder chain, **all 24 five-second reports exceeded 53 MH/s** (63.98–67.02). The node accepted all 14 submissions during the 121-second window. Aggregate throughput between the first and last miner reports was 64.99 MH/s over 115.04 s. GPU busy reached 100% in 109 of 121 polls; median board power was 286 W.

The harder run changed only the isolated chain's genesis target to `floor(2^256 / 65000000000)`; the SDK assigned and verified subsequent targets normally with FishHash active at height two. The fresh low-difficulty run shows the opposite workload: 588 accepted submissions during the telemetry window and 2.48 MH/s between miner reports. Frequent template replacement leaves the GPU waiting. Neither run is a mainnet measurement.

Offline GPU search on the same RTX 4090, five seconds per size. Both paths include the target check and hit readback but exclude node waits and dataset setup.

| Nonces per batch | Previous single-lane MH/s | Current eight-lane MH/s |
|---:|---:|---:|
| 16,384 | 17.678 | 54.614 |
| 65,536 | 18.362 | 65.710 |
| 262,144 | 18.574 | 66.045 |
| 1,048,576 | 18.171 | 66.323 |

The sm_120 target was compiled, not hardware-tested.

### Throughput research: share a nonce across CUDA lanes

An offline experiment on the same RTX 4090 kept the 4.5 GiB dataset, all 32 FishHash rounds, the GPU target check and the 36-byte hit readback. Each measurement used 32 batches of 262,144 new nonces. The table shows medians from seven runs per configuration, with the order rotated. No clocks or power limits were changed.

| Search variant | MH/s |
|---|---:|
| Previous single-lane kernel, timed in the same harness | 16.14 |
| Staged, one lane per nonce | 16.38 |
| Staged, two lanes per nonce | 16.11 |
| Staged, four lanes per nonce | 35.60 |
| Staged, eight lanes per nonce | 60.17 |

The staged variants first derive the seed using the [BLAKE3 state shared by a template's nonces](https://fips.ironfish.network/fips/fip-9-swap-randomness-and-graffiti), then run the dataset mix, then finish the hash and check the target. With eight lanes, each lane reads 16 adjacent bytes of the same [128-byte dataset item](https://fips.ironfish.network/fips/fip-3-memory-hard-mining-algorithm). Changing only the lane count within that pipeline accounted for most of the gain. Processing two, four or eight nonces at once per eight-lane group did not improve the median.

In a separate sustained run, the eight-lane variant searched 7,900,495,872 nonces in 120.237 s: **65.71 MH/s**. All 120 one-second intervals exceeded 53 MH/s (65.24–66.03). Concurrent `nvidia-smi` polling reported 100% GPU busy and 100% memory busy in all 120 samples, with median board power of 286.8 W. Memory busy measures time, not bytes transferred. These offline rates exclude dataset setup, template changes and node submission.

Every research variant matched the previous kernel on 4,099 full digests across a nonce carry, returned the known Rust-oracle winner at index 257, and returned no hit when that nonce was excluded. The miner uses the eight-lane variant with one nonce per group.
