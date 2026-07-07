# omicau — Omics Audit

A reproducible, leakage-safe, platform-agnostic command-line tool that audits
multi-omic datasets: it ingests non-standardized matrices, aligns them, locks
their provenance with a cryptographic hash, tests for missingness bias and batch
effects, benchmarks classical and neural data-fusion models under group-aware
cross-validation, attributes predictive signal to individual features, and
compiles both a clinical dashboard and publication-grade documentation.

The core runs fully offline with no LLM connection and no orchestration
framework. Optional tiers (an LLM interpretation plugin, remote data hubs)
degrade gracefully when their dependencies or network are absent.

---

## What it answers

For a set of omic layers and a clinical endpoint, omicau answers three questions:

1. **Does combining the layers actually help,** or does one layer already carry
   the signal? (marginal gain from adding each modality)
2. **Is the data trustworthy,** or is the apparent signal an artifact of batch
   effects, target-linked missingness, or information leakage? (adversarial
   diagnostics + control baselines)
3. **What drives the prediction,** biologically? (leakage-safe feature
   attribution)

The four design pillars are universality (runs out of the box on laptops, Apple
Silicon, Intel, and headless HPC nodes; MPS/CUDA/CPU auto-selected), reproducibility
(seeded loops, pinned versions, an immutable data hash), usability (a dual
clinical/research dashboard), and scientific validity (leakage prohibition,
masked missing-value handling, control baselines).

---

## Installation

```bash
pip install .                 # core, fully offline
pip install ".[llm]"          # + Anthropic LLM interpretation plugin
pip install ".[data]"         # + remote data hubs (requests, google-cloud-storage, cptac)
pip install ".[all,dev]"      # everything + pytest
```

Python ≥ 3.10. Core dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`,
`torch`, `plotly`, `click`, `jinja2`, `tqdm`.

## Quickstart

```bash
omicau check-env                                   # CPU/GPU + dependency status
omicau bootstrap --dataset mock --out-dir demo     # write a synthetic dataset
omicau run --config demo/config.json --cores 8     # full audit -> demo/run/report.html
omicau verify --config demo/config.json            # recompute the provenance hash
```

Open `demo/run/report.html` for the dashboard; `demo/run/audit.json` and the
`*.csv` files are the machine-readable assets.

### Verifying a run's provenance hash

Every run prints and stores a SHA-256 hash of the aligned data. It is
deterministic, so anyone can recompute it from the same inputs and confirm no
drift:

```bash
omicau verify --config demo/config.json --expected <hash>   # exit 1 on mismatch
omicau verify --config demo/config.json --audit demo/run/audit.json  # compare stored vs recomputed
```

---

## Use your own data

The remote data hubs are optional shortcuts; **the primary workflow is to point
omicau at your own matrices.** You provide one file per omic modality, one
clinical table with the outcome, and a small config that names them.

Ingestion is deliberately forgiving, so you rarely have to reshape anything:

- **Delimiter** is auto-detected (comma, tab, semicolon, or whitespace).
- **Orientation** is auto-detected by sample-id overlap — a `samples × features`
  matrix and a `features × samples` matrix (e.g. genes-as-rows expression) both work.
- **Sample names** are fuzzily matched across files (whitespace, case, and common
  batch/aliquot suffixes are normalized).
- Dirty headers, mixed whitespace, common NA tokens (`NA`, `null`, `.`, `?`, …),
  European decimals, and `±inf` are healed automatically.
- Missing values stay masked — never imputed at ingest.

Minimal layout (any of CSV/TSV; matrices can be either orientation):

```
mystudy/
  rna.csv          # samples × genes  (or genes × samples — auto-detected)
  protein.csv      # samples × proteins
  clinical.csv     # one row per sample: sample_id, outcome[, patient_id, batch]
  config.json
```

`config.json` (JSON shown; `.toml` and `.yaml` are also accepted):

```json
{
  "run_name": "my_study",
  "output_dir": "run",
  "modalities": [
    {"name": "rna",     "path": "rna.csv",     "description": "RNA-seq log-TPM"},
    {"name": "protein", "path": "protein.csv", "description": "Proteomics"}
  ],
  "clinical": {
    "path": "clinical.csv",
    "target": "outcome",
    "sample_id": "sample_id",
    "group": "patient_id",
    "batch": "batch",
    "task": "auto"
  },
  "cv": {"n_splits": 5, "seed": 42},
  "neural": {"enabled": true, "epochs": 60},
  "llm": {"enabled": false}
}
```

```bash
omicau run --config mystudy/config.json --cores 8
```

Field notes:

- **`sample_id`** — the clinical column holding sample identifiers (omit to use the
  table's first column / index). The same ids must appear (after normalization) in
  each modality matrix; samples are intersected across all files.
- **`target`** — the clinical column to predict. `task: "auto"` infers
  classification vs regression; set `"classification"` / `"regression"` to force it.
- **`group`** (optional, recommended) — a column such as patient id, so multiple
  samples from one patient never split across train and test (leakage-safe CV).
- **`batch`** (optional) — a column such as sequencing batch or site; drives the
  batch-effect diagnostics.
- Only rows with a non-missing target are kept. Paths are resolved **relative to the
  config file**, so keep the config next to your matrices.

Start from a working template by generating the mock dataset and editing its
`config.json` + CSV headers to match your files:

```bash
omicau bootstrap --dataset mock --out-dir template   # writes a runnable config.json + example CSVs
```

Add as many modalities as you like (methylation, metabolomics, CNV, …). A single
modality is allowed — the audit simply skips the cross-modality comparisons.

### Input format for each omics layer

Every modality is a single **sample × feature numeric matrix**: identifiers live
in the header row and the index column, and the body is numbers only (blanks or
`NA`/`null`/`.` for missing). omicau does not care which normalization you used —
it standardizes inside each cross-validation fold — but the values must be numeric
and comparable within a column. The table below is guidance, not a hard schema;
the ingester auto-detects delimiter and orientation, so any of these read as-is.

| Omics layer | Matrix body (value type) | Feature id (columns) | Notes for omicau |
| --- | --- | --- | --- |
| **RNA-seq / expression** | log-normalized expression (log2 TPM/FPKM/CPM, or RSEM/`log2(x+1)`) | gene symbol / Ensembl / Entrez | Prefer a log scale; raw counts work but log-transform heavy-tailed counts first. One value per gene per sample. |
| **Proteomics** | normalized abundance / intensity (usually log2) | UniProt id or gene symbol | Missing values are common and informative (MNAR) — leave them blank/`NA`, do **not** impute. Collapse isoforms to one column per protein. |
| **DNA methylation** | beta values in `[0,1]` **or** M-values | Illumina probe id (cg…) or gene | Beta and M are both fine as numeric features; don't mix the two in one matrix. |
| **Metabolomics** | peak area / concentration (often log-transformed) | metabolite name / RefMet / HMDB id | Keep below-detection as blank/`NA`. One column per metabolite. |
| **Copy number (CNV)** | log2 copy-ratio (continuous) **or** GISTIC discrete `-2…2` | gene symbol | Either continuous or integer levels; keep one convention per matrix. |
| **miRNA** | log-normalized expression | miRBase id (hsa-miR-…) | Same shape as RNA-seq. |
| **Somatic mutation** | binary `0/1` (mutated) or a mutation count | gene symbol | Presented as numeric features; a near-constant column (almost all 0) is dropped automatically. |

Universal rules across layers:

- **One file per modality**, samples aligned by id across all files and the
  clinical table. Feature names must be unique within a modality; the same name in
  two modalities is fine (omicau namespaces them as `modality::feature`).
- **Do not pre-impute or pre-scale.** Missing entries stay masked (the neural fuser
  ignores them; classical models median-impute inside the training fold only), and
  standardization happens inside each fold — pre-scaling across all samples leaks.
- **Log-transform skewed counts** (RNA-seq/miRNA/metabolomics) before ingest;
  omicau standardizes but does not log for you.
- **Orientation and delimiter are auto-detected**, so genes-as-rows or
  samples-as-rows, CSV or TSV, all load without a flag.

---

## Workflow

```mermaid
flowchart TB
    S0["Multi-modal ingestion — auto-delimiter, orientation, fuzzy sample-name match"]
    S1["Alignment & masking — sample intersection, drop missing endpoints, NaN masks"]
    S2["Provenance SHA-256 — hash of sample index + feature footprints"]
    S3["Cost / runtime estimate — N×P_m, K folds, E epochs, device, cores"]
    S4["Nested group-aware CV — impute + scale + select fitted inside train folds only"]
    S5["Fusion benchmarks — classical concat + masked global-pooling neural network"]
    S6["Leakage-safe XAI — permutation importance on held-out folds"]
    S7["Utility & redundancy audit — marginal gain, CKA, batch/missingness/control gates"]
    S8["Dual reporting — clinical + research dashboard, JSON/CSV assets"]
    S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8
```

---

## Methodology

### 1. Flexible ingestion and alignment

Matrices lack a standard layout, so ingestion adapts per file:

- **Delimiter** is inferred with `csv.Sniffer`, falling back to header-frequency
  counting across the candidate set `{tab, comma, semicolon, pipe}` and to
  whitespace splitting.
- **Orientation** is resolved by overlap scoring. For a matrix with row labels
  `R` and column labels `C`, and reference sample ids `S`, the fraction of each
  axis that matches the reference is `overlap(L, S) = |normalize(L) ∩ S| / |L|`.
  If `overlap(C, S) > overlap(R, S)` the matrix is transposed so samples are
  rows. This automatically corrects genes-as-rows expression matrices.
- **Fuzzy sample-name normalization** strips whitespace, upper-cases, and removes
  configurable prefix/suffix regexes. The default suffix rule collapses a TCGA
  aliquot barcode to its patient stem.
- **Numeric self-repair** coerces text columns, masks common NA tokens
  (`NA`, `null`, `.`, `?`, …), heals European decimals by comparing the
  parse yield of the raw text vs a `1.234,5 → 1234.5` repair and keeping the
  interpretation that recovers more finite values, and maps ±∞ to `NaN`.
- Samples are **intersected** across all modalities and the clinical table;
  records with a missing endpoint are dropped; duplicate features are namespaced
  per modality (`modality::feature`) to block cross-modality collisions.

Missing values are kept as `NaN` (true masks), never imputed at ingest.

### 2. Cryptographic provenance

Immediately after alignment, an immutable signature is computed as the SHA-256
of a canonical JSON manifest of the sorted sample index, each modality's sorted
feature list and shape, a **content digest of each aligned matrix** (the numeric
values in canonical row/column order), a digest of the encoded target, the
target name, and the task. It is tamper-evident at the value level: changing any
single measurement — not only which samples or features enter the study —
changes the hash, locking asset provenance across the study lifetime.

### 3. Missingness-bias diagnostics

Tests whether *missingness itself* carries information (missing-not-at-random):

- **Kruskal-Wallis** of each sample's per-modality missing fraction across
  outcome classes (or **Spearman** correlation for a continuous target).
- **Chi-squared** of a binary "any-missing" indicator against the outcome.
- **Kruskal-Wallis** of the missing fraction across batches.

p-values are corrected across all tests with the **Benjamini-Hochberg** FDR:
sort `p_(1) ≤ … ≤ p_(n)`, set `p̃_(i) = min_{k≥i} min(1, p_(k)·n/k)`.

### 4. Batch-effect and confounding diagnostics

Each modality is projected onto principal components (standardized;
mean-imputed for the projection only — never for modeling). Batch structure is
quantified by (i) the **silhouette** of batch labels in PC space, (ii) one-way
**ANOVA** and **Kruskal-Wallis** of PC1 across batches, and (iii) the fraction
of PC1 variance explained by batch, `η² = SS_between / SS_total`. A **chi-squared**
test with **Cramér's V** between batch and a categorical target flags
batch/target confounding — the regime in which a batch effect masquerades as
signal.

### 5. Leakage-safe preprocessing (nested)

All preprocessing is an sklearn `Pipeline` fitted **inside each training fold
only** and applied to the held-out fold:

1. median imputation (train-fold medians),
2. zero-variance filtering,
3. standardization `z = (x − μ_train) / σ_train`,
4. optional univariate selection (`SelectKBest`, ANOVA F-test for
   classification, F-regression for regression).

No validation-fold statistic ever influences training.

### 6. Group-aware cross-validation

When a group column is present (e.g. patient id with multiple samples), splits
keep all of a group's samples on one side, prohibiting identity leakage:
`StratifiedGroupKFold` (classification) or `GroupKFold` (regression); otherwise
`StratifiedKFold` / `KFold`. The fold count is clamped so every fold is populated
and, for classification, contains both classes. Metrics are computed on pooled
out-of-fold predictions.

### 7. Classical fusion benchmarks

Early fusion concatenates the (namespaced) modality matrices. For each estimator
— L2-regularized logistic regression / ridge, random forest, or histogram
gradient boosting — omicau cross-validates:

- each modality **alone**,
- the **full fusion**,
- each **leave-one-modality-out** subset.

The **marginal gain** of modality *m* is `Δ_m = score(FUSION) − score(FUSION∖m)`,
with a paired *t*-test across folds (fold splits are identical across runs given
the shared seed, so the comparison is paired).

### 8. Masked Global Pooling Fusion network (PyTorch)

The custom neural fuser is agnostic to feature counts and to which features are
missing per sample. Each modality *m* has a learned per-feature embedding table
`E_m ∈ ℝ^{P_m × d}`. For a sample with standardized values `x` and observed mask
`o ∈ {0,1}^{P_m}`:

- token per feature: `t_j = E_m[j] · x_j`,
- **masked mean pooling** over observed features only:
  `pooled_m = (Σ_j o_j · t_j) / max(1, Σ_j o_j)` (a `max` variant is available),
- followed by `LayerNorm`.

Missing features (`o_j = 0`) contribute nothing to the pooled embedding — no
artificial variance is injected. Per-modality embeddings are concatenated and
passed to an MLP head (`Linear → ReLU → Dropout → Linear`). Loss is cross-entropy
(classification) or MSE (regression). Standardization statistics are computed
from the training fold only (masked over observed entries), keeping the neural
path leakage-safe. Early stopping uses an internal split of the training fold.
Training is wrapped in an out-of-memory self-repair loop that halves the batch
size, clears the device cache, and retries, then falls back to CPU.

### 9. Leakage-safe feature attribution (XAI)

Primary attribution is **permutation importance** computed on each held-out
validation fold with the model trained only on that fold's training data, then
averaged across folds. For a fitted model, feature *j*'s importance is the drop
in the fold metric when column *j* is permuted. The neural fuser additionally
exposes a native score from its per-feature embedding norms weighted by observed
feature variance.

### 10. Modality-utility ledger and redundancy

Representational redundancy between modalities *X* and *Y* is the linear
**centered kernel alignment**
`CKA(X, Y) = ‖YᵀX‖_F² / (‖XᵀX‖_F · ‖YᵀY‖_F) ∈ [0, 1]`
on column-standardized matrices; high CKA with a stronger-alone modality marks a
layer as redundant. Each modality receives a verdict — *predictive*, *redundant*,
*batch-confounded*, or *control-like* — from its standalone score, marginal gain
significance, redundancy, and diagnostic flags.

### 11. Control baselines

The identical pipeline is run on three corrupted inputs to prove it does not
leak: **shuffled target**, **column-shuffled features**, and **random Gaussian
noise**. A well-behaved harness scores at chance on all three; if a control rises
above chance, a leakage warning gates the whole ledger.

### 12. Metrics

Classification: AUROC, AUPRC (average precision), accuracy, balanced accuracy,
F1, Matthews correlation. Regression: R², RMSE, MAE, Spearman ρ, Pearson r. All
guard against degenerate folds (single-class, zero-variance) and return `NaN`
rather than raising. The primary metric is AUROC (classification) or R²
(regression).

### 13. Pre-flight cost estimation

Before heavy loops, wall-time is estimated from `N × P_m` per modality, the fold
count `K`, neural epochs `E`, the device (MPS/CUDA/CPU), and the core count `C`.
A single live RandomForest fit calibrates the per-fit cost on the actual machine;
it is scaled by the model/fold counts, and the neural cost is modelled from the
epoch budget and feature footprint. The estimate is deliberately conservative so
HPC allocations are safe.

---

## Reporting

- **Dashboard** (`report.html`): a single self-contained file (Plotly bundled
  inline, so it works offline after the one-time Google Fonts load). Editorial
  serif typography (EB Garamond + JetBrains Mono), a color-blind-safe Okabe-Ito
  palette (cobalt `#0072B2` = standard, vermillion `#D55E00` = warning), an
  executive tab for PIs/clinicians and a research tab for computational
  biologists, five interactive figures, and tables that are sortable,
  text-filterable, and CSV/TSV-exportable via dependency-free vanilla JavaScript.
- **Machine-readable**: `audit.json` (full state), `model_metrics.csv`,
  `modality_ledger.csv`, `missingness_tests.csv`.

---

## Data hubs

All clients run structural gates (numeric-only, non-finite healing,
constant-feature dropping, sample-extension matching) and use retry-with-jitter.
Network access is optional and isolated; the core is unaffected if it is absent.

Only hubs whose live connection was verified are shipped. Each client was probed
directly against its real endpoint (the connection column reflects that check).

| Client | Source | Modalities → target | Connection |
| --- | --- | --- | --- |
| `tcga` | cBioPortal public REST API | mRNA + copy-number + merged sample/patient clinical | **verified** (`laml_tcga`: sex from RNA+CNV, AUROC ≈ 0.90) |
| `ccle` | DepMap 24Q4 (figshare) | RNA-seq → CRISPR gene-effect dependency | **verified** (1103 lines × 19k genes; SOX10 R² ≈ 0.74) |
| `xena` | UCSC Xena hubs (no auth) | RNA-seq / methylation / CNV / protein + phenotype | **verified** (TCGA-BRCA PAM50, 1247 samples) |
| `openpbta` | Public AWS S3 (anonymous) | putative-fusion matrix + histologies | **verified** (`open-targets/v15` S3 listing + TSV headers) |
| `metabolomics_workbench` | Metabolomics Workbench REST | metabolomics + study factors | **verified** (ST000009: gender AUROC ≈ 0.88) |
| `cptac` | `cptac` PyPI package | matched proteomics + transcriptomics | needs the `cptac` package (a build toolchain; no Windows/py3.12 wheel) — verified by docs only |
| `allofus` | All of Us Researcher Workbench | WGS / proteomics / RNA-seq | Workbench-only by design; off-platform it raises a clear error (cannot be externally connected) |

`gdsc` and `linkedomics` were evaluated and **dropped**: their public download
endpoints do not respond to a scripted client (HTTP 410 and 403 respectively).

The All of Us client runs only inside the secure Researcher Workbench and reads
the managed `WORKSPACE_CDR` / `WORKSPACE_BUCKET` / `GOOGLE_PROJECT` variables;
data cannot be exported and off-platform sessions raise a clear error. No
participant-level data is ever transmitted off-platform.

`omicau` was also validated on external multi-omics datasets that are **not** hubs
— the MOGONET benchmark (Wang et al., *Nat Commun* 2021): ROSMAP (Alzheimer's,
three omics, fusion AUROC ≈ 0.80) and BRCA (PAM50 subtype, three omics, fusion
AUROC ≈ 0.95, with methylation and miRNA correctly flagged redundant with RNA).

---

## Software versions

### Frozen Python stack (development + test reference)

| Package | Version | Package | Version |
| --- | --- | --- | --- |
| Python | 3.12.10 | scikit-learn | 1.9.0 |
| numpy | 2.5.0 | torch | 2.12.1 (CPU) |
| pandas | 3.0.3 | plotly | 6.8.0 |
| scipy | 1.18.0 | click | 8.4.1 |
| jinja2 | 3.1.6 | tqdm | 4.68.3 |
| requests | 2.34.2 | pytest | 9.1.1 |

Optional tiers (pinned floors in `pyproject.toml`): `anthropic ≥ 0.39`,
`cptac ≥ 1.5` (tested against 1.5.14), `google-cloud-storage ≥ 2.10`,
`pyyaml ≥ 6.0`. The API layer targets the current Anthropic Messages API and
model ids (default `claude-sonnet-5`).

### Upstream database / atlas releases (pinned)

| Resource | Release pinned | Route |
| --- | --- | --- |
| cBioPortal | REST API v3 (public); example study `laml_tcga` | `https://www.cbioportal.org/api` |
| DepMap / CCLE | **DepMap Public 24Q4** (figshare article 27993248) | figshare `ndownloader` |
| CPTAC | via `cptac` 1.5.14 (Zenodo-hosted, on-demand) | `cptac` cancer classes |
| OpenPedCan | **release v15** (`open-targets/v15`) | public S3 `d3b-openaccess-us-east-1-prd-pbta` |
| OpenPBTA | `release-v23-20230115` | public S3 (same bucket) |
| GDSC (optional target) | release 8.4 (24Jul22) | — |
| PRISM (optional target) | Repurposing 19Q4 | figshare article 9393293 |
| All of Us | CDR v7/v8 Workbench variable conventions | in-Workbench BigQuery + GCS |

Open-access release identifiers move over time; each client docstrings its
verified-as-of date, and the endpoints are best-effort — re-verify before
production.

---

## Reproducibility log

- **Determinism**: every stochastic step is seeded (`numpy`, `random`,
  `torch.manual_seed`; per-fold seeds derive from the master seed). Cross-validation
  splits depend only on the target, groups, and seed, so leave-one-out and
  single-vs-fusion comparisons are exactly paired. Strict bit-level determinism
  (`torch.use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG`) is opt-in via
  `--deterministic` / `compute.deterministic`; it is off by default because some
  ops lack deterministic kernels (enabled with `warn_only` so it never hard-fails).
- **Provenance**: the value-level SHA-256 of the aligned matrices, sample index,
  features, and target is written to `audit.json` and re-checkable with
  `omicau verify` (exit 1 on any drift).
- **Environment capture**: `audit.json → environment` records the Python,
  platform, numpy, and torch versions of the run; `runtime_log.txt` records the
  wall-time of every step with a device tag (`hostname/device/cores`).
- **Built and tested on**: Python 3.12.10, Windows 11 (10.0.26200), x86-64
  (AMD64), CPU-only torch. The package is cross-platform (`pathlib` throughout,
  defensive `newline=""` I/O, no shell invocation) and headless-HPC ready
  (`--cores`/`--threads` honor cgroup limits; no interactive prompts).

---

## Decoupling and HPC

The `omicau` core is 100% functional with no internet access, no LLM connection,
and no multi-agent framework. The LLM interpretation layer
(`interpretation/llm_summary.py`) is an optional plugin; when absent it falls
back to a deterministic rule-based summary filling the identical JSON schema, so
the report never breaks. Remote data hubs are optional and isolated behind lazy
imports. Thread/worker counts are set explicitly via `--cores` / `--threads`;
the PyTorch device is chosen with `--device` (MPS/CUDA/CPU, `auto` by default).

---

## License

MIT.
