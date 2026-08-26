# Fixed data splits

The ID-only text files under `code/manifests/` are the source of truth. Their
order is fixed and protected by SHA-256 fingerprints in
`tests/validate_splits.py`. They contain no paths or private metadata.

## BTCV / Synapse binary 2D

| Split | Case IDs | Slices |
|---|---|---:|
| Train | 0005, 0006, 0007, 0009, 0010, 0021, 0023, 0024, 0026, 0027, 0028, 0030, 0031, 0033, 0034, 0037, 0039, 0040 | 2,211 |
| Validation | 0008, 0001 | 295 |
| Test | 0022, 0038, 0036, 0032, 0002, 0029, 0003, 0004, 0025, 0035 | 1,273 |

The 12 held-out cases may share one physical `BTCV/test` PNG pool. The loader
filters it with the two disjoint manifests, so validation and test observations
never overlap. The historical random 10% slice subset is not included or used.

Manifest fingerprints:

- train: `dfdb73f71f36503c06c569760e77ab15fa67cdce84341127e553b327e2219c71`
- val: `47ed3d1d80a76914ffa3bb75ecc33f40b9b8ae7c113d852fe32f3aff5031efc3`
- test: `fa5ebbed2ed11e94f6a7dce1c89ff9c03bf4d9cf2e893bf28da9cd0f9f193b3c`

## ACDC four-class 2D

The fixed MT-UNet/CASCADE-style split partitions `patient001` through
`patient100` at patient level:

| Split | Patients | Slices |
|---|---:|---:|
| Train | 70 | 1,304 |
| Validation | 10 | 182 |
| Test | 20 | 416 |

Validation patients are 019, 021, 029, 033, 041, 050, 061, 071, 076, and 080.
The test patients are 002, 003, 008, 009, 012, 014, 017, 024, 042, 048, 049,
053, 055, 064, 067, 079, 081, 088, 092, and 095. The remaining 70 patients are
training cases. Images, RGB inspection masks, and class-index `label_maps` are
paired one-to-one.

Manifest fingerprints:

- train: `bfc7abb8764d17f50615d15c26653e6e395205a71aa60712a4e3cbfe4c4946b8`
- val: `4d1799fbc83e1d6fcdf000f5e5c4d9b4efae840e306bae95fdb8105180a6b6aa`
- test: `7b2aac91c98e719a7f41300eccf8d622c0fc98754c7b8e6639ce5bbf563dd89b`

## ISIC2018 Task 1

The official challenge partitions are retained without re-splitting:

| Split | Images/masks |
|---|---:|
| Training | 2,594 |
| Validation | 100 |
| Testing | 1,000 |

Manifest fingerprints:

- train: `b6a11fa50b1cf0363160258b5cac96ccae382ea524e34434c3212d1fbb1e4775`
- val: `44c159239176fe8f63c1efd5f40b5cf66fcafcfd61571c684cd8467f912a366c`
- test: `6fb4df9f361085d70966ce9098133a6491b756869717af4eaef4ca687686e4f7`

## Validation evidence

The release validator checks manifest encoding, exact ID counts/order/hash,
duplicates, cross-split leakage, image-mask pairing, ACDC label-map pairing,
owner isolation, and exact sample counts. The locally available caches passed
all checks on 2026-08-22.

