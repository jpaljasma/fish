# FishHash CUDA miner

A Windows CUDA miner for solo mining Iron Fish against a full node. It uses one GPU and a 4.5 GiB FishHash dataset. The node creates the reward transaction and pays its default account. Stratum pools are not supported.

## Build on Windows

On x64 Windows, install CMake 3.20+, Visual Studio 2022 Build Tools with Desktop development with C++ and a Windows SDK, and CUDA 13.5 or a compatible toolkit that supports `sm_89` and `sm_120`. Use a compatible NVIDIA driver and leave room in VRAM for the 4.5 GiB dataset plus working buffers. The RTX 4090 (`sm_89`) was tested on hardware; `sm_120` was only compiled.

From a Windows command prompt at the repo root:

```bat
tools\build.bat
build\test_gpu_match.exe tests\vectors\vectors.json --dataset-cache build\fishhash-dataset.bin
```

The build runs the CPU tests. The GPU test creates the cache if it is missing. Allow 4.5 GiB of disk space and look for `0 failure(s)` before using the cache. If the test rejects the file, move it aside and rerun the test.

## Run against your node

Use a synced, connected node with FishHash active and an unlocked default wallet account with a spending key. Merge these settings into `config.json` in the node's active data directory and restart it:

```json
{
  "enableRpcTcp": true,
  "enableRpcTls": false,
  "rpcTcpHost": "127.0.0.1"
}
```

Plain TCP RPC has no authentication or encryption. Keep it on loopback. The default TCP port is 8020; use your configured port if it differs.

```bat
build\fishminer.exe --rpc 127.0.0.1:8020 --gpu 0 --dataset-cache build\fishhash-dataset.bin
```

Omit `--dataset-cache` to build the dataset in VRAM at startup. The miner logs `template seq=` when it gets work. Count a block only when the node returns `added=true reason=SUCCESS`. Stop with Ctrl+C. See the [header format and node RPC reference](docs/protocol.md).

## RTX 4090 tuning

The default batch is 262,144 nonces. Use a smaller `--batch` for faster template switching. The maximum is 1,048,576.

At the default batch size, the RTX 4090 measured 66.05 MH/s offline and 64.99 MH/s on a private chain set to about 650 million hashes per block. The node accepted all 14 submissions in the live measurement window. GPU busy time reached 100% in 109 of 121 samples. This was not a mainnet test.

The benchmark requires a saved cache and runs on GPU 0. Stop the miner before benchmarking. Run `nvidia-smi` separately while mining to measure GPU busy time, not hash rate or memory bandwidth.

```bat
build\fishbench.exe build\fishhash-dataset.bin
nvidia-smi -i 0 --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw --format=csv -l 1
```
