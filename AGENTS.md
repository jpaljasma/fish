# AGENTS.md — FishHash POC (Iron Fish GPU mining, RTX 4090)

## Build (from omp bash)

One-time per session:

    export COMSPEC='C:\Windows\System32\cmd.exe' MSYS_NO_PATHCONV=1

Then, from the project root:

    cmd /c "tools\build.bat"

`tools/build.bat` locates the x64 compiler with `vswhere`, uses `CUDA_PATH`, and runs
configure, build, then the CPU oracle. PATH changes stay in the child process.

## Tests

- `ctest` (in build.bat): `test_fishhash` — CPU port vs `tests/vectors/vectors.json`.
- GPU: `build\test_gpu_match.exe tests\vectors\vectors.json --dataset-cache .scratch\dataset.bin`.
- Vectors: regenerate with `build\vecgen.exe` (~1.5 s) after header changes; never hand-edit.
- Python `blake3` (pip) is byte-identical to official BLAKE3 vectors — use to isolate BLAKE3 bugs.

## Notes

- RTX 4090 = sm_89. The GPU oracle command builds and saves the 4.5 GiB dataset on first run.
- `docs/protocol.md` — FishHash and node RPC. The FishHash nonce is the **last** 8 bytes
  of the 180-byte header, not the first 8 used by the legacy BLAKE3 serializer.
- `plans/2026-11-17-1519-feat-fishhash-cuda-miner-poc-plan.md` — correctness gates before tuning.
- `.scratch/` — disposable: probes, reference repos, logs, dataset cache. Not built.
