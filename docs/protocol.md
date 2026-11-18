# FishHash mining protocol

## Block header (180 bytes)

The node uses [`serializeHeaderFishHash`](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/blockHasher.ts) at and above the `enableFishHash` sequence. The legacy BLAKE3 format puts the nonce first; FishHash puts it last.

On [mainnet](https://fips.ironfish.network/fips/fip-10-hardfork-1), FishHash starts at block 503,338.

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

Change only `randomness`. A digest qualifies if its unsigned 256-bit big-endian value is less than or equal to the template's `target`. Keep the template timestamp and target together.

## Node RPC

Use the [README's loopback setup](../README.md#run-against-your-node). The [plain TCP adapter](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/rpc/adapters/tcpAdapter.ts) does not authenticate clients.

The [wire format](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/rpc/adapters/socketAdapter/protocol.ts) is UTF-8 JSON followed by a literal form feed (byte `0x0c`). Here is a subscription request before the delimiter:

```json
{"type":"message","data":{"mid":2,"type":"miner/blockTemplateStream"}}
```

Replies and stream updates echo `data.mid` as `data.id`.

1. Call `chain/getConsensusParameters`. A null `enableFishHash` means the chain has not enabled FishHash. The built-in devnet defaults to null.
2. Subscribe to [`miner/blockTemplateStream`](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/rpc/routes/miner/blockTemplateStream.ts). In both requests, omit the inner `data` field. The node can send a reward-only template and replace it at the same height. The miner takes the latest template between batches.
3. Send [`miner/submitBlock`](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/rpc/routes/miner/submitBlock.ts) with the template as the inner `data` field. Keep the header and ordered transactions unchanged except for `header.randomness`: 16 hex digits, big-endian, no `0x`. Send the template, not raw header bytes or a digest.

In a `stream` update, the template is in `data.data`. The [template format](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/serde/BlockTemplateSerde.ts) uses JSON numbers for `sequence` and `timestamp`, and unprefixed hex strings for the other header fields and each transaction. `previousBlockInfo` is stream metadata, not part of the hash or required for submission.

The [node's mining manager](https://github.com/iron-fish/ironfish/blob/c5c3c22bfc26dbf40a26c58ffee3181bef6cef5e/ironfish/src/mining/manager.ts) returns `added: true` only for `SUCCESS`. `FORK` is a side-chain block, `CHAIN_CHANGED` is stale work, and `ADD_FAILED` means the node could not add the block. An RPC error is separate from a block result.

## FishHash

FishHash uses a fixed 4.5 GiB dataset: 37,748,717 items of 128 bytes. [Dataset generation](../src/fishhash.cpp) uses Keccak-512 (not SHA3-512), starting from `BLAKE3-256("FishHash")`. It builds a light cache of 1,179,641 64-byte items in three mixing passes, then uses 512 parent rounds per 64-byte half of a dataset item.

1. Take 64 bytes of BLAKE3 output from the header and duplicate them into a 128-byte mix.
2. For 32 rounds, take the little-endian 32-bit words at mix byte offsets 0, 16 and 32. Reduce each word modulo the dataset length to get indices for `fetch0`, `fetch1` and `fetch2`. The [reference code](https://github.com/iron-fish/fish-hash/blob/main/cpp/FishHash.cpp) uses these offsets; the prose around Algorithm 2 in [FIP-3](https://fips.ironfish.network/fips/fip-3-memory-hard-mining-algorithm) gives a different stride.
3. In each round, update `fetch1` with `fnv1(mix, fetch1)` and `fetch2` with `mix XOR fetch2` in 32-bit words. FNV is `fnv1(u, v) = (u * 16777619) XOR v`, with 32-bit wraparound. Then replace the mix with `fetch0 * fetch1 + fetch2` in little-endian 64-bit words, with 64-bit wraparound.
4. Reduce each group of four mix words with three chained FNV calls. Hash the original 64-byte seed followed by the 32-byte reduced mix with BLAKE3-256.

Each hash requests 12 KiB of dataset data. That counts bytes requested, not measured DRAM traffic.

The [test fixtures](../tests/vectors/vectors.json) cover dataset samples and raw hash inputs, including non-header lengths. The [GPU test](../tests/test_gpu_match.cpp) compares against the CPU and checks a winning nonce computed by the node's Rust implementation.
