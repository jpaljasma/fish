# FishHash miner

## Build and test

From omp bash at the repo root:

```bash
export COMSPEC='C:\Windows\System32\cmd.exe' MSYS_NO_PATHCONV=1
cmd /c "tools\build.bat"
./build/test_gpu_match.exe tests/vectors/vectors.json --dataset-cache build/fishhash-dataset.bin
```

The build runs the CPU tests. The GPU test creates the 4.5 GiB dataset if the cache is missing; reuse a valid cache if available. Regenerate fixtures with `build/vecgen.exe <output.json>`; don't hand-edit them. Use Python `blake3` to isolate BLAKE3 discrepancies.

## Protocol

- Follow [the protocol reference](docs/protocol.md) and the node oracle. The FishHash nonce is the last eight bytes of the 180-byte header, unlike the legacy layout.

## Throughput

- Compare `fishbench` runs with the same full dataset and batch size. For live measurements, poll `nvidia-smi -l 1` during mining and report wall-clock MH/s and node-accepted submissions.
- For sustained tests, use an isolated chain with `miningForce` and FishHash active at height 2. Set genesis difficulty to 65 billion; activation divides it by 100. Fresh devnet spends much of its time submitting blocks and waiting for templates.

Keep probes, logs and local cache files in ignored `.scratch/` or `build/`.
