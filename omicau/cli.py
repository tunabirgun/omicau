"""omicau command-line interface.

Three ergonomic commands:

* ``omicau run --config <path>`` -- ingest, hash provenance, estimate wall-time,
  then run the full audit (diagnostics, fusion benchmarks, XAI, interpretation)
  and compile the dashboard + documentation, all under strict thread limits.
* ``omicau bootstrap --dataset <name> --out-dir <path>`` -- download / assemble a
  benchmark cohort into an omicau-ready dataset in one step.
* ``omicau check-env`` -- print CPU/GPU compute status, folder access, and
  optional-dependency / API readiness.

Designed for headless HPC use: no interactive prompts, ``pathlib`` throughout,
``--cores`` / ``--threads`` honor cluster cgroup limits.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import click

from omicau import __version__


def _utf8_stdout() -> None:
    """Best-effort UTF-8 console so report glyphs never crash on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Environment + resources
# --------------------------------------------------------------------------- #
def _environment() -> dict[str, Any]:
    import numpy
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
    }
    try:
        import torch
        env["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        env["torch"] = "not installed"
    return env


def _apply_thread_limits(cores: int) -> None:
    """Constrain BLAS/torch threading to honor cluster limits (best-effort)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(cores))
    try:
        import torch
        torch.set_num_threads(max(1, cores))
    except Exception:  # noqa: BLE001
        pass


def _apply_determinism(seed: int, enabled: bool) -> None:
    """Opt-in strict determinism for reproducible neural training."""
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 - determinism is best-effort
        pass


def _human_time(seconds: float) -> str:
    if seconds < 90:
        return f"~{seconds:.0f} s"
    if seconds < 5400:
        return f"~{seconds / 60:.1f} min"
    return f"~{seconds / 3600:.1f} h"


# --------------------------------------------------------------------------- #
# Pre-flight cost / runtime estimation engine
# --------------------------------------------------------------------------- #
def estimate_runtime(aligned, config, device_type: str, cores: int) -> dict[str, Any]:
    """Approximate the end-to-end wall-time from data size, folds, epochs, hw.

    Calibrates a per-fit cost with a tiny live RandomForest probe on this machine,
    then scales by the number of models and folds; the neural cost is modelled
    from the epoch budget and feature footprint. The estimate is deliberately
    conservative (an upper-ish bound) so HPC allocations are safe.
    """
    import numpy as np

    N = aligned.n_samples
    P_tot = sum(aligned.feature_counts().values())
    M = len(aligned.modality_names)
    K = min(config.cv.n_splits, N)
    n_keys = max(1, len(config.classical.models))
    n_controls = sum([config.controls.shuffle_target, config.controls.shuffle_features,
                      config.controls.random_noise]) if config.controls.enabled else 0

    # classical model count: (single M + fusion 1 + LOO M) per estimator + controls.
    classical_models = n_keys * (2 * M + 1) + n_controls
    classical_fits = classical_models * K

    # calibrate one fit on a small random problem.
    per_fit = 0.05
    try:
        from sklearn.ensemble import RandomForestClassifier
        n0, p0, tr0 = min(N, 64), min(P_tot, 120), 40
        X0 = np.random.rand(max(8, n0), max(4, p0))
        y0 = (np.random.rand(max(8, n0)) > 0.5).astype(int)
        t0 = time.perf_counter()
        RandomForestClassifier(n_estimators=tr0, n_jobs=cores).fit(X0, y0)
        dt = time.perf_counter() - t0
        per_fit = dt * (300 / tr0) * (max(N, 1) / max(n0, 1)) * (max(P_tot, 1) / max(p0, 1)) ** 0.5
    except Exception:  # noqa: BLE001 - fall back to the default constant
        pass
    classical_seconds = classical_fits * per_fit
    # permutation importance adds ~ repeats * a fold-predict on the fusion model.
    if config.xai.enabled:
        classical_seconds *= 1.0 + 0.15 * config.xai.permutation_repeats / max(1, K)

    # neural cost.
    neural_seconds = 0.0
    if config.neural.enabled:
        neural_models = 2 * M + 1
        epochs = neural_models * K * config.neural.epochs
        dev_factor = 3.0 if device_type in ("cuda", "mps") else 1.0
        c_epoch = 0.015 * (max(N, 1) / 64.0) * (max(P_tot, 1) / 500.0) * (config.neural.embed_dim / 32.0)
        neural_seconds = epochs * c_epoch / dev_factor

    overhead = 4.0 + 0.5 * M  # diagnostics PCA + reporting compile
    total = classical_seconds + neural_seconds + overhead
    return {
        "total_seconds": round(total, 1),
        "human_readable": _human_time(total),
        "breakdown": {
            "classical_seconds": round(classical_seconds, 1),
            "neural_seconds": round(neural_seconds, 1),
            "overhead_seconds": round(overhead, 1),
            "classical_fits": int(classical_fits),
            "neural_epochs": int((2 * M + 1) * K * config.neural.epochs) if config.neural.enabled else 0,
        },
        "dims": {"N": N, "P_total": P_tot, "M": M, "K": K, "device": device_type, "cores": cores},
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
_STEP_LABEL = {
    "ingest_align": "Reading and aligning data layers",
    "diagnostics_missingness": "Checking missing-value patterns",
    "diagnostics_batch": "Checking for batch effects",
    "classical_benchmarks": "Benchmarking standard models",
    "neural_benchmark": "Benchmarking the neural fusion model",
    "utility_ledger": "Scoring each layer's usefulness",
    "interpretation": "Writing the plain-language verdict",
    "report": "Building the dashboard and files",
}


def _log_runtime(log_path: Path, step: str, elapsed: float, device_tag: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp}\t{step}\t{elapsed:.2f}s\t{device_tag}\n"
    with open(log_path, "a", encoding="utf-8", newline="") as fh:
        fh.write(line)


def run_audit(config, *, cores: int, device: str, llm: bool | None,
              api_key: str | None = None,
              echo=lambda *_a, **_k: None) -> dict[str, Any]:
    """Execute the full audit and compile all report assets; return the audit dict.

    ``api_key`` is an ephemeral, UI-supplied model key: it is used only for this
    run's plain-language verdict call and is never stored, logged, or written to
    any output (config.json / audit.json / report / provenance hash)."""
    from omicau.data.alignment import load_and_align
    from omicau.diagnostics import batch_effect_diagnostics, missingness_diagnostics
    from omicau.models.classical import resolve_cores, run_classical_benchmarks
    from omicau.models.neural import resolve_device, run_neural_benchmark
    from omicau.interpretation.utility import build_utility_ledger
    from omicau.interpretation.llm_summary import build_context, summarize
    from omicau.reporting.reporter import build_report

    # resolve compute.
    if cores:
        config.compute.cores = cores
    if device and device != "auto":
        config.compute.device = device
    if llm is not None:
        config.llm.enabled = llm
    resolved_cores = resolve_cores(config)
    _apply_thread_limits(resolved_cores)
    _apply_determinism(config.seed, config.compute.deterministic)
    dev = resolve_device(config.compute.device)
    device_tag = f"{platform.node()}/{dev.type}/{resolved_cores}c"

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "runtime_log.txt"

    def timed(step, fn):
        t = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t
        _log_runtime(log_path, step, dt, device_tag)
        echo(f"  - {_STEP_LABEL.get(step, step)}: {dt:.1f}s")
        return result

    echo("Ingesting and aligning modalities...")
    aligned = timed("ingest_align", lambda: load_and_align(config))
    from omicau.data.alignment import check_grouping
    check_grouping(aligned)   # preflight: warn on missing/no-op grouping, raise on class==group
    echo(f"  provenance SHA-256 (a fingerprint of these exact inputs): {aligned.provenance_hash}")
    echo(f"  {aligned.n_samples} samples | {aligned.task} | modalities {aligned.feature_counts()}")
    if len(aligned.modality_names) == 1:
        echo(f"  single-modality run ({aligned.modality_names[0]}): fusion / redundancy / "
             "marginal-contribution analyses do not apply; running the leakage-safe honesty check "
             "(group-aware CV, shuffled-label controls, calibration, bootstrap CIs).")

    cost = estimate_runtime(aligned, config, dev.type, resolved_cores)
    echo(f"Estimated wall-time: {cost['human_readable']} "
         f"({cost['breakdown']['classical_fits']} classical fits, "
         f"{cost['breakdown']['neural_epochs']} neural epochs on {dev.type})")

    missing = timed("diagnostics_missingness", lambda: missingness_diagnostics(aligned))
    batch = timed("diagnostics_batch", lambda: batch_effect_diagnostics(aligned, seed=config.seed))
    if aligned.task == "survival":
        from omicau.models.survival import run_survival_benchmark
        classical = timed("classical_benchmarks", lambda: run_survival_benchmark(aligned, config))
        neural = {"enabled": False, "results": []}      # neural survival deferred
    else:
        classical = timed("classical_benchmarks", lambda: run_classical_benchmarks(aligned, config, batch))
        neural = timed("neural_benchmark", lambda: run_neural_benchmark(aligned, config))
    util = timed("utility_ledger",
                 lambda: build_utility_ledger(aligned, classical, neural, batch, missing))
    summary = timed("interpretation",
                    lambda: summarize(build_context(aligned, util, missing, batch), config, api_key=api_key))

    audit = _assemble_audit(aligned, classical, neural, util, summary, missing, batch,
                            config, cost, dev.type, resolved_cores)

    if config.reporting.html or config.reporting.json or config.reporting.csv:
        assets = timed("report", lambda: build_report(audit, out_dir, config))
        audit["_assets"] = {k: str(v) for k, v in assets.items()}
    return audit


def _assemble_audit(aligned, classical, neural, util, summary, missing, batch,
                    config, cost, device_type, cores) -> dict[str, Any]:
    return {
        "meta": {"run_name": config.run_name, "tool_version": __version__,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "provenance_hash": aligned.provenance_hash, "device": device_type,
                 "cores": cores, "seed": config.seed,
                 "organism": config.organism},
        "environment": _environment(),
        "dataset": {"n_samples": aligned.n_samples, "task": aligned.task,
                    "n_groups": (int(aligned.groups.nunique()) if aligned.groups is not None else None),
                    "class_names": aligned.class_names,
                    "class_balance": aligned.report.get("class_balance"),
                    "feature_counts": aligned.feature_counts(),
                    "n_dropped": aligned.report.get("dropped", {}).get("missing_target", 0),
                    "alignment_report": aligned.report,
                    "modalities": [{"name": n, "description": m.description, "n_features": m.shape[1]}
                                   for n, m in aligned.modalities.items()]},
        "cost_estimate": cost,
        "diagnostics": {"missingness": missing, "batch": batch},
        "models": {"primary_metric": classical["primary_metric"], "task": aligned.task,
                   "reference_estimator": classical["reference_estimator"],
                   "classical": [r.to_dict() for r in classical["results"]],
                   "controls": [r.to_dict() for r in classical["controls"]],
                   "neural": {"enabled": neural.get("enabled"), "device": neural.get("device"),
                              "results": [r.to_dict() for r in neural.get("results", [])]}},
        "utility": util, "summary": summary, "config": config.to_dict(),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="omicau")
def main() -> None:
    """omicau - a reproducible, leakage-safe multi-omics data-audit CLI."""
    _utf8_stdout()


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path),
              help="Path to a JSON / TOML / YAML run configuration.")
@click.option("--cores", "--threads", "cores", type=int, default=None,
              help="Max CPU worker cores (honors cluster cgroup limits). Default: auto.")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto",
              help="PyTorch device backend.")
@click.option("--llm/--no-llm", default=None,
              help="Force-enable or force-disable the optional LLM interpretation tier.")
@click.option("--deterministic", is_flag=True, default=False,
              help="Enable strict PyTorch determinism (reproducible neural training).")
@click.option("--out-dir", type=click.Path(path_type=Path), default=None,
              help="Override the output directory from the config.")
def run(config_path: Path, cores: int | None, device: str, llm: bool | None,
        deterministic: bool, out_dir: Path | None) -> None:
    """Run the full audit and compile the interactive dashboard + assets."""
    from omicau.config import OmicauConfig

    config = OmicauConfig.from_file(config_path)
    if deterministic:
        config.compute.deterministic = True
    if out_dir:
        config.output_dir = str(out_dir)
    click.secho(f"omicau v{__version__} - {config.run_name}", fg="cyan", bold=True)
    try:
        audit = run_audit(config, cores=cores, device=device, llm=llm, echo=click.echo)
    except Exception as exc:  # noqa: BLE001 - present errors clearly to end users
        raise click.ClickException(str(exc)) from exc
    assets = audit.get("_assets", {})
    click.secho("\nDone. Assets written:", fg="green", bold=True)
    for k, v in assets.items():
        click.echo(f"  {k:16s} {v}")
    if "html" in assets:
        click.secho(f"\nOpen the dashboard: {assets['html']}", fg="cyan")


_BOOTSTRAP_EPILOG = """\
\b
Per-dataset usage (all write a runnable config.json + matrices):
  mock          synthetic, fully offline (redundant/confounded/noise layers)
                  omicau bootstrap --dataset mock --out-dir demo [--task classification|regression]
  tcga          cBioPortal (no auth): mRNA + copy-number + merged clinical
                  omicau bootstrap --dataset tcga --out-dir d --study laml_tcga --target SEX
  ccle          DepMap 24Q4 (figshare): expression -> CRISPR gene-effect (regression)
                  omicau bootstrap --dataset ccle --out-dir d --target SOX10   # any gene
  xena          UCSC Xena hub: multi-omics + phenotype
                  omicau bootstrap --dataset xena --out-dir d --preset brca --target PAM50Call_RNAseq
  openpbta      public S3 (v15): putative-fusion matrix + histologies
                  omicau bootstrap --dataset openpbta --out-dir d --target broad_histology
  metabolomics  Metabolomics Workbench REST: metabolites + study factors
                  omicau bootstrap --dataset metabolomics --out-dir d --study ST000009 --target gender
  cptac         requires the `cptac` package: matched proteomics + transcriptomics
                  omicau bootstrap --dataset cptac --out-dir d --cancer Ucec
  expression_atlas  EMBL-EBI Expression Atlas: cross-organism RNA-seq (log2 CPM) + a factor target
                  omicau bootstrap --dataset expression_atlas --out-dir d --study E-GEOD-100100 --target "RNA interference"
                  [--normalization log2cpm|tmm|median_of_ratios]  (default log2cpm; tmm/mor are whole-matrix)

Omit --target to let the client pick a sensible default. Remote clients need the
'data' extra (pip install ".[data]" from a checkout, or omicau[data] once
published); the mock is fully offline.
"""


@main.command(epilog=_BOOTSTRAP_EPILOG)
@click.option("--dataset", required=True,
              type=click.Choice(["mock", "tcga", "ccle", "cptac", "openpbta", "xena",
                                 "metabolomics", "expression_atlas"]),
              help="Benchmark cohort to assemble (see examples below).")
@click.option("--out-dir", required=True, type=click.Path(path_type=Path),
              help="Directory to write the dataset + config into.")
@click.option("--study", default=None,
              help="TCGA study id (e.g. laml_tcga), Metabolomics Workbench id (e.g. ST000009), "
                   "or Expression Atlas accession (e.g. E-GEOD-100100).")
@click.option("--target", default=None,
              help="Target column / gene (dataset-specific; omit for the client default).")
@click.option("--cancer", default=None, help="CPTAC cohort abbreviation (default: Ucec).")
@click.option("--preset", default=None, help="Xena preset cohort (default: brca).")
@click.option("--task", default="classification",
              help="Mock dataset task: classification, regression, or survival.")
@click.option("--seed", default=42, type=int, help="Seed for the mock dataset.")
@click.option("--normalization",
              type=click.Choice(["log2cpm", "tmm", "median_of_ratios"]),
              default="log2cpm", show_default=True,
              help="Expression Atlas only: cross-sample count normalization. log2cpm is "
                   "within-sample (leakage-clean default); tmm / median_of_ratios are "
                   "whole-matrix (see the dataset's config.json note).")
def bootstrap(dataset: str, out_dir: Path, study: str | None, target: str | None,
              cancer: str | None, preset: str | None, task: str, seed: int,
              normalization: str) -> None:
    """Download / assemble a benchmark cohort in one step.

    Produces an omicau-ready dataset (modality CSVs + clinical.csv + config.json)
    so that `omicau run --config <out-dir>/config.json` works immediately.
    """
    click.secho(f"Bootstrapping '{dataset}' into {out_dir}...", fg="cyan")
    try:
        from omicau.data.bootstrap import assemble
        cfg = assemble(dataset, out_dir, study=study, target=target, cancer=cancer,
                       preset=preset, task=task, seed=seed, normalization=normalization)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    click.secho(f"Ready. Run:\n  omicau run --config {cfg}", fg="green", bold=True)


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="Config to re-ingest and recompute the provenance hash from.")
@click.option("--audit", "audit_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="A prior audit.json whose stored hash to read/compare.")
@click.option("--expected", default=None, help="Expected SHA-256 to check against (exit 1 on mismatch).")
def verify(config_path: Path | None, audit_path: Path | None, expected: str | None) -> None:
    """Recompute (or read) the run provenance SHA-256 and optionally check it.

    The hash is a deterministic SHA-256 of the aligned sample index and each
    modality's sorted feature footprint, so re-running this on the same inputs
    reproduces it exactly - any drift in samples or features changes the hash.
    """
    recomputed = stored = None
    if audit_path:
        stored = json.loads(audit_path.read_text(encoding="utf-8")).get("meta", {}).get("provenance_hash")
        click.echo(f"  stored (audit.json):   {stored}")
    if config_path:
        from omicau.config import OmicauConfig
        from omicau.data.alignment import load_and_align
        try:
            aligned = load_and_align(OmicauConfig.from_file(config_path))
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(str(exc)) from exc
        recomputed = aligned.provenance_hash
        click.echo(f"  recomputed (from data): {recomputed}")
    if not (recomputed or stored):
        raise click.ClickException("Provide --config and/or --audit to obtain a hash.")

    if recomputed and stored and recomputed != stored:
        raise click.ClickException("MISMATCH: recomputed hash differs from the stored audit hash "
                                   "- the aligned data has drifted.")
    target = expected.strip() if expected else None
    got = recomputed or stored
    if target:
        if got == target:
            click.secho(f"MATCH: {got}", fg="green", bold=True)
        else:
            raise click.ClickException(f"MISMATCH: expected {target}\n            got      {got}")
    else:
        click.secho(f"provenance SHA-256: {got}", fg="cyan", bold=True)


@main.command()
@click.option("--port", type=int, default=None, help="Port to bind (default: an auto-selected free port).")
@click.option("--no-browser", is_flag=True, default=False, help="Do not open a browser automatically.")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind address. Keep it localhost - the UI is single-user and local-first.")
def ui(port: int | None, no_browser: bool, host: str) -> None:
    """Launch the optional local web UI (no-code, localhost only).

    Opens a browser-based wizard to upload data, map columns, and run the audit
    without touching a config file. Requires `pip install omicau[ui]`. Data never
    leaves the machine; the server binds localhost with a one-time token.
    """
    try:
        from omicau.ui.server import launch
        launch(host=host, port=port, open_browser=not no_browser, echo=click.echo)
    except ImportError as exc:  # missing [ui] extra -> clean message, not a traceback
        raise click.ClickException(str(exc)) from exc


@main.command(name="check-env")
def check_env() -> None:
    """Print compute status, folder access, and dependency / API readiness."""
    click.secho(f"omicau v{__version__} environment check", fg="cyan", bold=True)
    env = _environment()
    for k, v in env.items():
        click.echo(f"  {k:12s} {v}")

    # compute backend
    try:
        from omicau.models.neural import resolve_device
        import torch
        dev = resolve_device("auto")
        click.echo(f"  {'device':12s} {dev.type} "
                   f"(cuda={torch.cuda.is_available()}, "
                   f"mps={bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())})")
    except Exception as exc:  # noqa: BLE001
        click.secho(f"  torch device probe failed: {exc}", fg="yellow")
    click.echo(f"  {'cpu_cores':12s} {os.cpu_count()}")

    # folder access
    ok = os.access(Path.cwd(), os.W_OK)
    click.echo(f"  {'cwd_write':12s} {'yes' if ok else 'NO'} ({Path.cwd()})")

    # optional dependencies (names mirror the pyproject extras)
    click.secho("Optional tiers:", bold=True)
    for name, mod in [("requests ([data])", "requests"),
                      ("google-cloud-storage ([data])", "google.cloud.storage"),
                      ("anthropic ([llm])", "anthropic"), ("openai ([llm])", "openai"),
                      ("fastapi ([ui])", "fastapi"), ("cptac ([cptac])", "cptac"),
                      ("pyyaml ([yaml])", "yaml")]:
        try:
            __import__(mod)
            status, color = "available", "green"
        except Exception:  # noqa: BLE001
            status, color = "absent (optional)", "yellow"
        click.secho(f"  {name:22s} {status}", fg=color)

    # API / platform readiness
    key = os.environ.get("ANTHROPIC_API_KEY")
    click.echo(f"  {'ANTHROPIC_API_KEY':22s} {'set' if key else 'not set'}")
    try:
        from omicau.data.allofus import workbench_status
        st = workbench_status()
        click.echo(f"  {'All of Us Workbench':22s} {'inside' if st['in_workbench'] else 'off-platform'}")
    except Exception:  # noqa: BLE001
        pass
    click.secho("\nCore pipeline runs fully offline; absent optional tiers degrade gracefully.",
                fg="cyan")


if __name__ == "__main__":  # pragma: no cover
    main()
