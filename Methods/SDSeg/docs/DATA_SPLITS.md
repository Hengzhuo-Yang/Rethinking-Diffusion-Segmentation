# Fixed data splits

The split unit is a case/patient for BTCV and ACDC and an official image ID for
ISIC2018. No slice from one BTCV case or ACDC patient may cross partitions.

## BTCV

BTCV uses 30 labeled cases split at case level.

| Partition | Cases | Slices | IDs |
|---|---:|---:|---|
| Train | 18 | 2211 | `case0005`, `case0006`, `case0007`, `case0009`, `case0010`, `case0021`, `case0023`, `case0024`, `case0026`, `case0027`, `case0028`, `case0030`, `case0031`, `case0033`, `case0034`, `case0037`, `case0039`, `case0040` |
| Validation | 2 | 295 | `case0001` (147), `case0008` (148) |
| Test | 10 | 1273 | `case0002` (139), `case0003` (198), `case0004` (140), `case0022` (89), `case0025` (85), `case0029` (100), `case0032` (144), `case0035` (94), `case0036` (184), `case0038` (100) |

The authoritative lists are under `code/manifests/btcv/`. The preprocessor
enforces the aggregate counts when `--max-volumes=0`. The earlier random
slice-level quick subset is not part of the release.

## ACDC

ACDC uses 100 patients split at patient level.

| Partition | Patients | Slices |
|---|---:|---:|
| Train | 70 | 1304 |
| Validation | 10 | 182 |
| Test | 20 | 416 |

Train patients:

```text
patient001 patient004 patient005 patient006 patient007 patient010 patient011
patient013 patient015 patient016 patient018 patient020 patient022 patient023
patient025 patient026 patient027 patient028 patient030 patient031 patient032
patient034 patient035 patient036 patient037 patient038 patient039 patient040
patient043 patient044 patient045 patient046 patient047 patient051 patient052
patient054 patient056 patient057 patient058 patient059 patient060 patient062
patient063 patient065 patient066 patient068 patient069 patient070 patient072
patient073 patient074 patient075 patient077 patient078 patient082 patient083
patient084 patient085 patient086 patient087 patient089 patient090 patient091
patient093 patient094 patient096 patient097 patient098 patient099 patient100
```

Validation patients:

```text
patient019 patient021 patient029 patient033 patient041 patient050 patient061
patient071 patient076 patient080
```

Test patients:

```text
patient002 patient003 patient008 patient009 patient012 patient014 patient017
patient024 patient042 patient048 patient049 patient053 patient055 patient064
patient067 patient079 patient081 patient088 patient092 patient095
```

The authoritative lists are under `code/manifests/acdc/`. Slice counts refer to
the labeled ED/ES frames materialized by the release preprocessor.

## ISIC2018

The official ISIC2018 Task 1 partitions are retained without resplitting:

| Partition | Images | Manifest |
|---|---:|---|
| Training | 2594 | `code/manifests/isic2018/training.txt` |
| Validation | 100 | `code/manifests/isic2018/validation.txt` |
| Testing | 1000 | `code/manifests/isic2018/testing.txt` |

The manifest files enumerate every official image ID and are authoritative; the
thousands of IDs are not duplicated in this document.

## Integrity requirements

Before any run, split validation must establish:

1. each manifest has the expected number of unique identifiers;
2. train, validation, and test sets are pairwise disjoint;
3. the prepared data contains exactly the expected cases/patients/image IDs for
   a formal run;
4. the expected slice/image totals match;
5. image and mask files pair one-to-one;
6. no absolute path from the preparation machine is embedded in a release
   manifest.

Any failed condition blocks formal training or final testing. Limit flags may be
used only for clearly labeled smoke runs.
