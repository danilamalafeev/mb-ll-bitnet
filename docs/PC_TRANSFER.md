# PC transfer contract (inventory and proposal)

**Prepared:** 2026-09-09  
**State:** `runs/pc_transfer_v1.tar.gz` remains audit-only. Corrected packet `runs/pc_transfer_v3.tar.gz` is now assembled locally with preserved relative paths and verified checksums; it has not been transferred, installed, loaded, or executed on AI2.
**Owner:** `/root/pc_transfer`

This document freezes the proposed input packet for the remaining PC length tests. It is a transfer contract, not a scientific result. The canonical Mac artifacts and provenance files remain untouched.

## Audit of the existing v1 archive

The existing archive must remain audit-only until a corrected packet is approved. Its measured values are:

| Artifact | Regular entries | Listed uncompressed bytes | On-disk bytes | SHA-256 |
|---|---:|---:|---:|---|
| `runs/pc_transfer_v1.tar.gz` | 53 total: 52 under `project/` plus `SHA256SUMS` | 68,642,582 | 60,278,485 | `51f8a140cc1a351da06162f4e3ce44b9bc44e819ec42f90810374339506b280b` |

It is not Option A and it is not a complete endpoint-only packet for the primary CPU gate:

* The 16 checkpoint files cover only E38 seed 1/2, float/W4, A/B, plus their four E32 and four E33 parent files. E36 seed-0 A/B endpoints are absent; all six B_match files are absent.
* Checkpoints are renamed under `project/runs/endpoints/`, and manifests are renamed to files such as `runs/e37_e38_preflight_v2_manifest.json`. The current loaders resolve the original nested paths and therefore cannot consume this layout as-is without a separately reviewed adapter.
* `results/E32_PROTECTED_HASHES.json` and its 355 protected files are absent. The current E38 loader reaches `scripts.width_e32.load_manifest`, which verifies that complete protected set. E33 reference/protected files are also absent.
* The archive contains extra E25/E26 source modules, while omitting the seed-0 lineage required by the 12-endpoint test. Its `SHA256SUMS` file covers the 52 project files but does not repair the path or lineage omissions.

The safe next channel is a reviewed, manifest-driven transfer of a corrected staging directory. Until that is built and hash-verified, do not send `pc_transfer_v1.tar.gz` to AI2 or start a temporary server for it.

## Corrected v3 packet

The corrected packet is `runs/pc_transfer_v3.tar.gz`. It contains **415 project files** totaling **759,087,344 bytes** uncompressed, plus `TRANSFER_MANIFEST.json` and `SHA256SUMS` at the archive root. Archive size is **197,594,981 bytes** and its SHA-256 is `147c78b14a1bfc3f79aea2e18e36142c25b76f035f3a1b224de34c51079cc530`.

The packet preserves the original paths required by the E38 loader, includes the complete 355-entry E32 protected snapshot and all 18 A/B/B_match endpoints, and passes local verification of every entry. `results/E33_PROTECTED_HASHES.json` is retained as provenance text, but its historical E33 report files are not expanded because the E38 inference runner does not invoke the E33 verifier; running the standalone E33 verifier would require that separate closure.

Because the chat upload limit is 45 MB, the archive is also split into five files under `runs/pc_transfer_v3_parts/`: four parts of 40,000,000 bytes and a final part of 37,594,981 bytes. Upload all five parts plus `TRANSFER_PARTS_MANIFEST.json`. On AI2, concatenate them in numeric order and verify the resulting archive against the SHA-256 above before extraction.

## Target and scope

The target is the user-authorized AI2 Codex project on the connected Windows machine:

| Field | Observed or reported value |
|---|---|
| Host | `DESKTOP-2GSSQIN` (`remote-control:env_e_6a2fe27ca91c83228cb90f77e62c53a5`) |
| Project | `fb9afead-4df7-4f57-8176-586583c4e87e` |
| Target folder | `C:\Users\я\LoopedBitNet_AI2_inference` |
| OS / hardware | Windows 11, Ryzen 5800X, RTX 3070 Ti (user-reported) |
| Transfer/runtime tools | `curl` and PowerShell available (user-reported) |
| Pending | remote harmless system inventory approval; exact Python, driver, CUDA, and free-space values |

The remaining length contract calls for 12 saved A/B8000 endpoints: A and B for seeds 0, 1, and 2 at float and W4. The packet also carries the six accepted `B_match` checkpoints so that the optional B_match diagnostic can be run under the same provenance. There is no training, data generation, artifact rebuild, or overwrite of canonical results in this phase.

## Measured packet choices

Sizes below are byte counts from the local filesystem. MiB uses 2^20 and GiB uses 2^30; GB uses 10^9.

| Option | Contents | Files | Bytes | Size | Use |
|---|---|---:|---:|---:|---|
| **A — strict closure (v3, ready)** | Dependencies required by the current E36/E38 loaders, the 18 accepted endpoints, source, manifests, tests, and complete E32 protected provenance | **415** | **759,087,344** | 723.922 MiB / 0.707 GiB / 0.759 GB | `runs/pc_transfer_v3.tar.gz`; transfer after review |
| B — endpoint-only | The 18 endpoint `.pt` files only | 18 | 73,251,142 | 69.856 MiB / 0.068 GiB / 0.073 GB | Useful only after a separately reviewed loader/portability adapter exists |
| C — workspace snapshot | All local workspace files except `.venv`, caches, and build output | — | 7,401,379,814 | 6.893 GiB / 7.401 GB | Fallback proposal if a later review requires broad audit retention; not the current packet |

Option A is a conservative exact union, not a claim that every byte in the E32 protected set is needed by every endpoint. The current E38 loader calls `scripts.width_e32.load_manifest`, which verifies the complete 355-file E32 protected map; omitting any of those paths would change the accepted loader path. The E33 protected map is included for provenance and optional guard checks.

The size calculation excludes `.venv` (about 711 MB, macOS ARM), `__pycache__`, `.pytest_cache`, and `build`. It also excludes large E36/E38 predictions and reports except where a protected manifest explicitly requires a path. A top-level E38 report is about 1.3 GB and is not an inference input.

## Endpoint inventory

Every endpoint below is an accepted saved checkpoint already present locally. Hashes are SHA-256. The E36 seed-0 files are the six accepted wave endpoints; the E38 seed-1/2 files are the twelve accepted branch endpoints.

| Family | Precision | Seed | Label | Relative path | Bytes | SHA-256 |
|---|---|---:|---|---|---:|---|
| E36 | float | 0 | A | `runs/e35_e36_length_wave/science/float128_seed0/A/u40000.pt` | 4,068,619 | `1c78ee7e13b8c747171be6db4d56d5d9c1df5eaec26bcacf2aab3f7c24a18774` |
| E36 | float | 0 | B | `runs/e35_e36_length_wave/science/float128_seed0/B/u40000.pt` | 4,068,619 | `59feda5e7d011c6c2a8ddb644a4dd9c74526a7af72cc657f0c927afe373dd3e9` |
| E36 | float | 0 | B_match | `runs/e35_e36_length_wave/science/float128_seed0/B_match/u36572.pt` | 4,068,619 | `375c722b9189a2b0e6dfb802de95c834d92bbecdd1d4172959f43a087d86c7c9` |
| E36 | W4 | 0 | A | `runs/e35_e36_length_wave/science/w4128_seed0/A/u40000.pt` | 4,068,555 | `3fcb61d6cf33fe7f8bb164f8660729a86cc39db960c42e8115d3bbfd0a730440` |
| E36 | W4 | 0 | B | `runs/e35_e36_length_wave/science/w4128_seed0/B/u40000.pt` | 4,068,555 | `3ae5f6d4a60cb86bc88386c96c7a6683e306d0f6744e5072ff860dd5e63d2b45` |
| E36 | W4 | 0 | B_match | `runs/e35_e36_length_wave/science/w4128_seed0/B_match/u36572.pt` | 4,068,619 | `2e93df47ce047af28c000a239d6c765de96d3ec551d9f2161facc4aee204e64f` |
| E38 | float | 1 | A | `runs/e37_e38_science_v3/e38/branches/float128_seed1/A/u40000.pt` | 4,069,963 | `a837211d9c393f2aee36d6622176877190c9605c1fa36645559e64e23a1ecdc9` |
| E38 | float | 1 | B | `runs/e37_e38_science_v3/e38/branches/float128_seed1/B/u40000.pt` | 4,069,963 | `39da90c43ddd78b8bd30b3f52f671743a11c7e7ccaae81f397032a12f7ba5458` |
| E38 | float | 1 | B_match | `runs/e37_e38_science_v3/e38/branches/float128_seed1/B_match/u40000.pt` | 4,069,963 | `58c16ae707fe202fdb7b5f13e4d020b82be2b57e5d3e84f52b834f173d3d23d1` |
| E38 | float | 2 | A | `runs/e37_e38_science_v3/e38/branches/float128_seed2/A/u40000.pt` | 4,069,963 | `b5ac51159a9f1775cef14c13eda85756d44c23a5466d18300fd04bf781c3adfb` |
| E38 | float | 2 | B | `runs/e37_e38_science_v3/e38/branches/float128_seed2/B/u40000.pt` | 4,069,963 | `b62a06be9310acae0c1e8d4a38bfa547d48de0811430d957bd488be35b05f4a8` |
| E38 | float | 2 | B_match | `runs/e37_e38_science_v3/e38/branches/float128_seed2/B_match/u40000.pt` | 4,069,963 | `2092db1b3d2e66d67f677f3174efda2bd3c82cdc8d7a2010855e01dd7a6f1f2d` |
| E38 | W4 | 1 | A | `runs/e37_e38_science_v3/e38/branches/w4128_seed1/A/u40000.pt` | 4,069,963 | `08e0201de7b23c3938a64f46242b81a452b1c0ba91661c4b0b3a1fc105becadc` |
| E38 | W4 | 1 | B | `runs/e37_e38_science_v3/e38/branches/w4128_seed1/B/u40000.pt` | 4,069,963 | `0d4e0045cb52f0b2c32afa6f0149e895b122977a3022901122f1ef8fd7ba13b5` |
| E38 | W4 | 1 | B_match | `runs/e37_e38_science_v3/e38/branches/w4128_seed1/B_match/u40000.pt` | 4,069,963 | `c1fe4890f95ea86985665c3b7e6712ac79c0b2e364ba8216e6ff0b472f875f0d` |
| E38 | W4 | 2 | A | `runs/e37_e38_science_v3/e38/branches/w4128_seed2/A/u40000.pt` | 4,069,963 | `e17d957b62ee18e09915679425b761936177eb0bd289acf4b014fe069e81b9e8` |
| E38 | W4 | 2 | B | `runs/e37_e38_science_v3/e38/branches/w4128_seed2/B/u40000.pt` | 4,069,963 | `93945e847e7178965bf545a3e5dc61903ca8e75494202c72686a82d4de764618` |
| E38 | W4 | 2 | B_match | `runs/e37_e38_science_v3/e38/branches/w4128_seed2/B_match/u40000.pt` | 4,069,963 | `81b11b9bbf928c5a11c6f36e14229e40ce20c3e1daf16766432f895073d91458` |

Endpoint aggregate: 73,251,142 bytes. The 12 A/B files used by the primary test are the rows with labels A or B; the six B_match rows remain optional and do not alter the primary count.

## Required closure and lineage

The strict path is deliberately rooted in the existing saved loaders. It must preserve relative paths under the packet root, including Unicode-safe copying of the target folder name. The dependency graph is:

```text
source + tests + protocols
        |
E32 source/reference/protected map + six preflight initial checkpoints
        |
E32 six final checkpoints
        |
E33 source/reference + seed-0 continuation parents
        |
E36 v2 manifest + six E36 endpoints
        |
E37/E38 corrected source reference + base manifest + runtime lineage
        |
E38 four parent extensions + twelve E38 branch endpoints
```

The measured Option A union has these groups:

| Group | Required paths or count |
|---|---|
| E32 provenance | `results/E32_REFERENCE.json`, `results/E32_PROTECTED_HASHES.json`, and all **355** paths named by the protected map |
| E32 checkpoints | six `runs/e32_width_preflight/*.pt` initial files and six `runs/e32_width/*/u16000.pt` final files |
| E33 provenance/parents | `results/E33_REFERENCE.json`, `results/E33_PROTECTED_HASHES.json`, E33 preflight provenance, and `runs/e33_continuation/{float128_seed0,w4128_seed0}/u32000.pt`; historical E33 protected files are outside this E38 packet |
| E36 lineage | `runs/e35_e36_preflight_v2/manifest.json` and the six E36 endpoint files |
| E37/E38 lineage | `runs/e37_e38_preflight_v2/manifest.json`, `results/E37_E38_SCIENCE_REFERENCE_V3.json`, `runs/e37_e38_science_v3/e38/runtime_lineage.json`, four E38 parent extensions, and twelve E38 endpoint files |
| Source/runtime | The static import closure of `scripts/followup_e37_e38.py`, plus `looped_bitnet/__init__.py`, `__main__.py`, `config.py`, `engine.py` for a normal package invocation |
| Contract/tests | `pyproject.toml`, `requirements.txt`, `requirements-tested-macos.txt`, the E32/E33/E36/E37-E38 protocol documents, and the relevant test modules |

Key manifest and provenance hashes:

| File | Bytes | SHA-256 |
|---|---:|---|
| `runs/e32_width_preflight/manifest.json` | 2,679,629 | `e3e0fbc1a06ae9480db7119a0f405462b45634349ba76d0fa2cd18e49680b083` |
| `runs/e33_continuation_preflight/manifest.json` | 3,424 | `563255aebf55cb39b963c0cfccdc512e19d726fb0bb00f82cd31a2894d0265bb` |
| `runs/e35_e36_preflight_v2/manifest.json` | 233,182 | `4adf2dd1bb1036b94bcbf7a0246d9410596a737c9d20e5ff8865632fab5988aa` |
| `runs/e37_e38_preflight_v2/manifest.json` | 15,828 | `593cd1fb98ee180b13f968060a329d11936f61f514a45517b97d8227ffddb9b2` |
| `results/E32_REFERENCE.json` | — | `98881fadd4ae4960448946f005e7d1346db1036375bfff54904c9d6e0bd386dd` |
| `results/E32_PROTECTED_HASHES.json` | — | `68adcfd7bd498889230d1a47e919028ce007e0531da2a64852709c8a204a56e0` |
| `results/E33_REFERENCE.json` | — | `8cd09d3c34c56ee941a0e16e4140febc184f7b62cbd6899fed4b499c20d9c99` |
| `results/E33_PROTECTED_HASHES.json` | — | `f4a2adaafac678ce0f5fbfc59ffd86a5c416e2076b1fab8e2d4afacc7df7b7f0` |
| `results/E37_E38_SCIENCE_REFERENCE_V3.json` | — | `d02e663f717953b224f56b2081dca5269909612bd89afcd8b7e945a7df4d8dd0` |
| `runs/e37_e38_science_v3/e38/runtime_lineage.json` | 3,200 | `4398e0e7b883ffc04ed143d5a313c53713dfc1ceeba5252281b0f1f1c8f8eb62` |

Source files that determine the current loader contract include:

| Source | SHA-256 |
|---|---|
| `scripts/followup_e37_e38.py` | `fede3bb21c1a806c9e10068b1b82afdffb57cec00874f30dfa0bd666a4dfe0b5` |
| `tests/test_followup_e37_e38.py` | `3e9cee635f8d63f1c77aaa9b58f857e99dd412de5a49b40f23a5c5f167faaf4b` |
| `scripts/length_wave_e35_e36.py` | `383658faa81b82a068f17a223450453eb8c1d2e591a3add2c3d7139d9af35c0f` |
| `looped_bitnet/width_e32.py` | `7f754f63613cdf9c90b54f8eb934e114f9f0a72aab0f61639fc0231fc95f5c2` |
| `scripts/width_e32.py` | `e28b78e1879d0c863473914f50b679978adca846dc524cd34145abbeb1b6e1ce` |
| `scripts/continuation_e33.py` | `677f25bbd51165f720afc008dae657cadd9cbc8e16fee6746c23e68286b60e73` |
| `tests/test_continuation_e33.py` | `cf6461fc6e33beccb5f68a7833d8b25e69064524a83242149837557c50ae9f5d` |

The direct protocol hashes are also retained in the corresponding reference/base manifests. In particular, `results/E37_E38_SCIENCE_REFERENCE_V3.json` is required because the corrected E37/E38 source hashes point to the current runner and test; using only the older base-manifest source snapshot would fail the effective source check.

## Environment and portability gates

The local environment is macOS ARM64 with Python 3.12.7, PyTorch 2.14.0, NumPy 2.5.2, Matplotlib 3.11.1, and pytest 9.1.1. The project metadata requires Python >=3.10 and ranges `torch>=2.6,<3`, `numpy>=1.26,<3`, `matplotlib>=3.9,<4`, and pytest >=8,<10. `requirements-tested-macos.txt` records the observed Mac pins; it is not a Windows CUDA lockfile. The target must use a target-specific PyTorch build compatible with its NVIDIA driver/CUDA runtime. Do not copy `.venv`.

The exact installed versions observed through `importlib.metadata` are:

| Package | Version |
|---|---:|
| `torch` | 2.14.0 |
| `numpy` | 2.5.2 |
| `matplotlib` | 3.11.1 |
| `pytest` | 9.1.1 |
| `setuptools` | 84.0.0 |
| `contourpy` | 1.3.3 |
| `cycler` | 0.12.1 |
| `filelock` | 3.32.5 |
| `fonttools` | 4.64.0 |
| `fsspec` | 2026.7.0 |
| `Jinja2` | 3.1.6 |
| `kiwisolver` | 1.5.1 |
| `MarkupSafe` | 3.0.3 |
| `mpmath` | 1.3.0 |
| `networkx` | 3.6.1 |
| `packaging` | 26.3 |
| `Pillow` | 12.3.0 |
| `pyparsing` | 3.3.2 |
| `python-dateutil` | 2.9.0.post0 |
| `six` | 1.17.0 |
| `sympy` | 1.14.0 |
| `typing-extensions` | 4.16.0 |

The local executable used for this observation is `.venv/bin/python` (Anaconda Python 3.12.7, macOS ARM64). It has no usable `pip` module; dependency installation on AI2 must therefore use a new target virtual environment.

The static source closure for the E37/E38 runner is the following relative-path set (plus the package `__init__.py`, `__main__.py`, `config.py`, and `engine.py` convenience entry points named in the group table above):

```text
scripts/composition_e21.py
scripts/continuation_e24.py
scripts/continuation_e33.py
scripts/followup_e37_e38.py
scripts/length_transfer_e28.py
scripts/length_wave_e35_e36.py
scripts/longer_native8_e20.py
scripts/replication_e22.py
scripts/step_budget_e18.py
scripts/width_e32.py
looped_bitnet/bit_input_e17.py
looped_bitnet/continuation_e24.py
looped_bitnet/data.py
looped_bitnet/float_qat_e16.py
looped_bitnet/longer_native8_e20.py
looped_bitnet/model.py
looped_bitnet/quantization.py
looped_bitnet/register_e15.py
looped_bitnet/replication_e22.py
looped_bitnet/runtime.py
looped_bitnet/step_budget_e18.py
looped_bitnet/w4_e27.py
looped_bitnet/width_e32.py
```

Two source-level portability gates are known before any PC run:

1. `looped_bitnet/runtime.py` imports Unix `resource` unconditionally. Standard Windows Python generally has no such module. A small reviewed guard is needed before importing the package on Windows; its RSS measurement branch is diagnostics only.
2. Device selection and deterministic settings are platform-dependent (`torch.backends.mps`, CUDA settings, and CPU/CUDA kernels). Existing loaders construct and load CPU state. A future device adapter must move the complete model explicitly and preserve checkpoint/RNG validation. Windows CUDA output cannot be called bit-for-bit equivalent from file hashes alone.

All loader paths are packet-root-relative. Absolute `/Users/...` strings in historical reports are provenance text, not path inputs. The Unicode target folder requires PowerShell/OpenSSH/7-Zip commands that preserve UTF-8 names. `scripts/run_and_wake.py` contains a hardcoded ChatGPT.app path and is outside this inference closure.

## Copying, loader portability, and numerical equivalence

These are separate gates:

| Gate | Claim established |
|---|---|
| Project copy | The selected relative files arrived with the recorded count, byte total, and SHA-256 values |
| Source/loader portability | The same saved lineage can be loaded on Windows after the reviewed `resource` and device compatibility changes, without bypassing validation |
| GPU numerical equivalence | The same endpoint/input protocol gives results within a preregistered tolerance on the target CUDA device after the CPU baseline |

A successful copy does not establish loader portability, and a successful Windows load does not establish CPU-to-CUDA numerical equivalence.

## Transfer procedure proposal

No transfer command has been executed from this document. After root and the early contract reviewer accept Option A, create a staging directory containing only the exact relative-path union above, generate a path/size/SHA-256 manifest, and transfer that staging directory. Preserve relative paths and file bytes; do not reconstruct checkpoints or copy the Mac virtual environment.

The transport remains deliberately mechanism-neutral until the source and target paths are confirmed. Candidate commands are:

* A manifest-driven `rsync -a --files-from=...` or SFTP/`scp` copy if an SSH endpoint is enabled.
* A temporary HTTP server serving only the validated staging directory, followed by PowerShell `Invoke-WebRequest` or `curl` on AI2. Root must start and announce this server; it must not serve the project root or any secrets.
* SMB/LAN `robocopy <verified-source> C:\Users\я\LoopedBitNet_AI2_inference /E /COPY:DAT /DCOPY:T /R:2 /W:2 /XJ` as a proposal only, after the actual source drive/share and Unicode behavior are confirmed.

On AI2, the first post-copy check should be a PowerShell manifest verification (`Get-FileHash -Algorithm SHA256`) plus file count and byte-total comparison. Only after that check and the reviewed portability patch should the target create a fresh Windows virtual environment and install compatible target-specific dependencies. No secrets, package credentials, or machine configuration files belong in the packet.

## Execution boundary after transfer

The execution order proposed by `results/PC_LENGTH_TEST_CONTRACT.md` is:

1. Verify Option A file count, bytes, and hashes on AI2.
2. Apply the separately reviewed Windows import/device compatibility patch, keeping source changes outside the scientific artifacts and recording new source hashes.
3. Run the 12 A/B8000 endpoints on CPU first, using the existing lineage-aware loader and 256 registered L7 cases per endpoint.
4. Save PC outputs in a new PC-only results directory; never overwrite Mac canonical outputs.
5. Only if the CPU run is accepted, run the optional identical CUDA gate using the contract's tolerance and report both results separately.
6. Run B_match only if the reviewer keeps that optional diagnostic in scope.

No local or remote inference has been executed for this transfer preparation. The target hardware facts above are user-reported pending the remote inventory response. The exact closure is measured locally; its transfer and Windows loader execution remain pending review and transport availability.
