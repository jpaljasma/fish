# Iron Fish (FishHash) Mining Protocol — Verified Notes

Status: verified against primary sources on 2026-11-17. This document is the
specification that the POC miner is implemented and tested against.

Sources (reference snapshots under `.scratch/ref/`; node links below pin revision `c5c3c22`):

- `nodejs_fish_hash.rs`, `nodejs_fish_hash_test.ts` — Iron Fish node binding
  (crates.io `fish_hash 0.3.0` — the mainnet PoW oracle).
- `FishHash.cpp` / `lib.rs` — community reference impl (iron-fish/fish-hash).
- [`BlockHasher`](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/blockHasher.ts), [`Target.meets`](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/primitives/target.ts) and [`BlockTemplateSerde`](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/serde/BlockTemplateSerde.ts) define the hashing preimage, target check and RPC field encodings.
- [`blockTemplateStream`](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/rpc/routes/miner/blockTemplateStream.ts), [`submitBlock`](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/rpc/routes/miner/submitBlock.ts) and the [`mining manager`](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/mining/manager.ts) define the subscription and submission behavior.
- `pool_*.ts` — Jivscko/ironfish-pool (stratum pool) source.

## 1. FishHash algorithm

Memory-hard PoW, Ethash-shaped, BLAKE3+Keccak-512. Mainnet since Hardfork-1
(Feb 2024). **The dataset is static** (no epochs): it is a pure function of one
fixed 256-bit seed, so it is built once and can be persisted/shared.

Constants (verified, see `tests/vectors/vectors.json`):

| Name | Value |
|---|---|
| seed | `BLAKE3-256("FishHash")` = `eb0163aef2ab1c5a66310c1c14d60f4255a9b39b0edf26539844f117ad672119` (checked: blake3 crate output equals the hardcoded constant) |
| FNV prime (32-bit) | 16777619 |
| light cache size | 1,179,641 items × 64 B |
| full dataset size | 37,748,717 items × 128 B = 4,831,835,776 B ≈ 4.5 GiB |
| dataset item rounds | 512 (Ethash epoch-448 uses 256; FishHash doubles it) |
| light cache rounds | 3 |
| header rounds (fetches) | 32 |
| fetches per round | 3 × 128 B |

Dataset construction (verified verbatim against reference `build_light_cache`
and `item_state`/`calculate_dataset_item_1024`; oracle-matched in
`tools/vecgen.cpp`). FNV: `fnv1(u, v) = (u * 16777619) ^ v` (XOR combine,
wordwise).

- **Light cache** (1,179,641 × 64 B):
  1. `cache[0] = keccak512(seed32)`; `cache[i] = keccak512(cache[i-1])` for
     i = 1..1,179,640.
  2. Three in-place rounds, ascending i:
     `v = cache[i].word32[0] % L`; `w = (i == 0) ? L-1 : i-1` (previous
     item, wrapping: i=0 → L-1);
     `cache[i] = keccak512(cache[v] XOR cache[w])`.
- **Full dataset item** (128 B = two 64-B states):
  For states k = 2i and 2i+1 (i = item index):
  1. `mix = light[k % L]; mix.word32[0] ^= k; mix = keccak512(mix)`
  2. 512 rounds, round j:
     `t = fnv1(k ^ j, mix.word32[j % 16])` (16 words per 64-B state);
     `parent = t % L`;
     `mix = wordwise_fnv1(mix, light[parent])` (16 words: `(m*p)^l`).
  3. `state = keccak512(mix)`.
  Item = `state(2i) || state(2i+1)`.

Cost: item = 2 × (1 keccak init + 512 × (1 × 64 B fetch + 16-word FNV+XOR)
+ 1 keccak final). ~4.5 GiB total; builds in ~1-2 min on CPU, faster in VRAM.

Per-hash computation over the 180-byte header (verified against both oracles;
reference: `tools/vecgen.cpp`, oracle-matched):

1. `startValue` = BLAKE3-512(header) (64 B). The BLAKE3 hasher is initialized
   with default output length 32 and finalized at 64 B — standard BLAKE3
   variable-output usage, matches the reference.
2. `mix` = 128 B = startValue concatenated with itself.
3. 32 rounds, round i (0..31):
   - `p0 = mix.word32[0] % N`, `p1 = mix.word32[4] % N`, `p2 = mix.word32[8] % N`
     (32-bit words at indices 0, 4, 8, i.e. bytes 0-3, 16-19, 32-35;
     little-endian on x86), N = 37,748,717
   - `fetch0/1/2` = dataset items at p0/p1/p2 (128 B each)
   - per 32-bit word j (0..31): `fetch1[j] = fnv1(mix[j], fetch1[j])`
     where `fnv1(u, v) = (u * 16777619) ^ v`
   - per 32-bit word j (0..31): `fetch2[j] = mix[j] ^ fetch2[j]`
   - per 64-bit word j (0..15): `mix[j] = fetch0[j] * fetch1[j] + fetch2[j]`
4. Collapse 128 B → 32 B: for each 4-word group i (0, 4, 8, ...):
   `h = fnv1(fnv1(fnv1(w[i], w[i+1]), w[i+2]), w[i+3])`; `mixHash[i/4] = h`.
5. `hash = BLAKE3-256(startValue[64] || mixHash[32])` (96 B input, 32 B out).

Per-hash dataset traffic: 32 × 3 × 128 B = **12,288 B (12 KiB)**.
Memory-bandwidth bound. RTX 4090: ~1008 GB/s → **ceiling ≈ 82 MH/s**;
realistic 60-65 % → **target ~50-53 MH/s**.

## 2. Block header (180 bytes)

Once `enableFishHash` is active at the template's sequence, the node hashes the
output of `serializeHeaderFishHash`. `BlockHeader.serialize()` has the legacy
Blake3 layout and is not the FishHash mining preimage.

| Offset | Size | Field | Encoding |
|---|---|---|---|
| 0 | 32 | `graffiti` | bytes |
| 32 | 4 | `sequence` | u32 little-endian |
| 36 | 32 | `previousBlockHash` | bytes, in template order |
| 68 | 32 | `noteCommitment` | bytes, in template order |
| 100 | 32 | `transactionCommitment` | bytes, in template order |
| 132 | 32 | `target` | u256 big-endian |
| 164 | 8 | `timestamp` | u64 little-endian (ms since Unix epoch) |
| 172 | 8 | `randomness` (mined nonce) | u64 big-endian |

Mining rule: `u256BE(FishHash(header180)) <= u256BE(target)`. The miner varies
the final eight bytes. It preserves the template's transactions, timestamp
and target; a target computed for some other timestamp is not a substitute.

## 3. Node miner RPC (the POC's target)

The SDK defaults to IPC; TCP is opt-in, and TLS defaults on. This client
speaks **plain TCP only**. Use a trusted loopback listener
(`enableRpcTcp: true`, `enableRpcTls: false`, `rpcTcpHost: "127.0.0.1"`). The configured
TCP port defaults to 8020. The [plain TCP adapter](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/rpc/adapters/tcpAdapter.ts)
does not authenticate clients. Do not expose it on a public interface or
send it a wallet or RPC secret.

Each [socket frame](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/rpc/adapters/socketAdapter/protocol.ts)
is UTF-8 JSON followed by one literal form-feed byte (`0x0c`), not a newline.
There is no top-level request ID. The miner first calls
`chain/getConsensusParameters` and checks `enableFishHash`. The unmodified
built-in devnet has this field set to `null` and cannot exercise FishHash.

### `miner/blockTemplateStream`
- Request: `{"type":"message","data":{"mid":2,"type":"miner/blockTemplateStream"}}`.
  Omit the inner `data` field entirely: the route accepts `undefined`, not
  JSON `null` or `{}`.
- Notifications have shape `{"type":"stream","data":{"id":2,"data":TEMPLATE}}`.
  The `id` identifies the original subscription. `TEMPLATE` has this shape:
```json
{
  "header": {
    "sequence": 123,
    "previousBlockHash": "<hex 32>",
    "noteCommitment": "<hex 32>",
    "transactionCommitment": "<hex 32>",
    "target": "<hex 32, big-endian>",
    "randomness": "<hex 8>",
    "timestamp": 1234567890,
    "graffiti": "<hex 32>"
  },
  "transactions": ["<hex block tx>", ...],
  "previousBlockInfo": { "target": "<hex 32>", "timestamp": 123 }
}
```
- The node sends work for the current head and again on connected blocks.
  It may send an empty template and then one with transactions for the same
  height. The miner switches to the latest notification between batches.
- The node needs a default spending account. It skips template generation
  when unsynced or disconnected unless `miningForce` is set.

### `miner/submitBlock`

Send `{"type":"message","data":{"mid":3,"type":"miner/submitBlock","data":TEMPLATE}}`.
Set `TEMPLATE.header.randomness` to the winning nonce (16 big-endian hex
digits, no `0x`), preserving the template and its reward transaction. The
node reconstructs and hashes the header and checks the block against its
chain. The miner never submits raw header bytes or a claimed hash.

The final response is `{"type":"message","data":{"id":3,"status":200,"data":{"added":true,"reason":"SUCCESS"}}}`.
Only this result means the node accepted the block on its main chain.
`FORK` means the node added a side-chain block but returns `added:false`.
Other reasons include `CHAIN_CHANGED`, `INVALID_BLOCK`, `ADD_FAILED`, and
`UNKNOWN_REQUEST`. An RPC route error is a `message` with a non-success
`status` and a nested error object; it is not a block result.

For a private FishHash node listening on `127.0.0.1:18309`, run:

```powershell
.\build\fishminer.exe --rpc 127.0.0.1:18309
```

Use a fresh isolated custom chain with FishHash enabled from height 2 for
a low-difficulty smoke. Do not edit a mainnet template's target or disable
the node's proof-of-work checks. A `candidate` log is local; look for the
node's `submit: added=true reason=SUCCESS` before counting an accepted block.

## 4. Stratum pool situation (secondary)

Only public FishHash stratum pool found: **Jivscko/ironfish-pool**.
- Transport: TCP, newline-delimited JSON, JSON-RPC 2.0 style.
- Methods (Stratum-V2 namespace on V1 transport): `mining.subscribe`,
  `mining.authorize`, `mining.submit`, `mining.set_difficulty`, `mining.notify`.
- Job carries `headerTemplate` (180 B hex) + fields; submit is
  `{jobId, nonce(16 hex), address, worker}`.
- Its builder reserves eight bytes for sequence at offset 0 and space for a
  nonce at 136; it writes the submitted nonce at 172, over the end of its
  graffiti field. That does not produce the FishHash preimage in §2.
  Its raw-header submit call also differs from the structured node RPC in §3.

Conclusion: the POC miner targets the **node RPC directly** (§3). A stratum
client remains optional tooling (`tools/probe_stratum.py` captures pool shapes
if a working pool appears).

## 5. Test vectors and oracle

- `tests/vectors/vectors.json` — 8 headers (180 B × 6, plus 179/181/32 B) with
  per-round traces (p0/p1/p2, 128 B mix after each round, collapsed, expected
  final hash), 768 dataset items, light-cache samples, constants.
  Generator: `tools/vecgen.cpp` (builds `vecgen.exe`); regenerates in ~1.5 s
  (light cache 1.46 s).
- **Validation (2026-11-17):** all 8 vectors byte-identical to the crates.io
  `fish_hash 0.3.0` output (`.scratch/crosscheck/` harness). The node's own
  published vector — `FishHash("the quick brown fox jumps over the lazy dog")`
  = `6f4429716dc009d5d3b9775a4d6a5d58bccd9f73386bf88da7d5afdf5deb50f1` —
  also reproduced exactly by `vecgen.exe`.

## 6. Performance model (RTX 4090)

- Per hash: 12 KiB dataset traffic + ~100 Keccak-512 compresses of work only
  if dataset is not cached (in full-dataset GPU mode: pure loads).
- 4090: 24 GB GDDR6X, ~1008 GB/s. Dataset (4.5 GiB) fits with ~2× headroom.
- Ceiling: 1008e9 / 12288 ≈ 82 MH/s. Target: ≥ 53 MH/s sustained (65 % of peak).
- Light cache (1,179,641 × 64 B ≈ 72 MiB) + full dataset (4.5 GiB) ≈ 4.6 GiB
  total VRAM — fits easily.
- Nonce space 2^64: 64-bit randomness is part of the 180-byte header; no
  extranonce splitting needed for solo mining.
