#!/usr/bin/env python3
"""Generate a self-contained interactive HTML report from one or more benchmark runs."""

import argparse
import json
import math
from pathlib import Path
import re

BENCHES = Path(__file__).resolve().parent

FAMILIES = {
    "Rounding": ["trunc", "floor", "ceil", "round", "roundeven", "lround", "llround"],
    "Remainders": ["fmod", "remainder", "remquo"],
    "Roots and norms": ["sqrt", "rsqrt", "cbrt", "hypot"],
    "Arithmetic": ["fma"],
    "Exponentials": ["exp", "exp2", "expm1"],
    "Powers": ["pow", "powr", "pown", "rootn", "compoundn"],
    "Logarithms": ["log", "log2", "log10", "log1p"],
    "Trigonometric": ["sin", "cos", "sincos", "tan", "asin", "acos", "atan", "atan2"],
    "Hyperbolic": ["sinh", "cosh", "tanh", "asinh", "acosh", "atanh"],
    "Error functions": ["erf", "erfc"],
    "Gamma functions": ["lgamma", "tgamma"],
}
OPERATIONS = [operation for operations in FAMILIES.values() for operation in operations]
KINDS = ["f32", "f64"]


def read_measurements(build, source):
    rows = []
    measurements = build["measurements"]
    if not measurements:
        raise ValueError(f"{source}: no measurements")
    if len({event["name"] for event in measurements}) != len(measurements):
        raise ValueError(f"{source}: duplicate measurements")
    pairs = {}
    for event in measurements:
        stem, variant = event["name"].split("::")
        operation, width = stem.rsplit("_", 1)
        kind, lanes = width.split("x")
        lanes = int(lanes)
        if kind not in KINDS or lanes <= 0:
            raise ValueError(f"{source}: invalid vector width {width}")
        pairs.setdefault((operation, kind, lanes), {})[variant] = event
    for (operation, kind, lanes), pair in sorted(pairs.items()):
        if set(pair) != {"scalar", "vector"}:
            raise ValueError(f"{source}: missing scalar/vector pair for {operation}_{kind}x{lanes}")
        scalar, vector = pair["scalar"], pair["vector"]
        if any(not math.isfinite(event["median"]) or event["median"] <= 0
               for event in (scalar, vector)):
            raise ValueError(f"{source}: medians must be finite and positive")
        if any(not math.isfinite(event["deviation"]) or event["deviation"] < 0
               for event in (scalar, vector)):
            raise ValueError(f"{source}: deviations must be finite and nonnegative")
        count = 32 * lanes
        rows.append({
            "operation": operation, "kind": kind, "lanes": lanes,
            "elements_per_iteration": count,
            "scalar_ns_per_element": scalar["median"] / count,
            "vector_ns_per_element": vector["median"] / count,
            "speedup": scalar["median"] / vector["median"],
        })
    return rows


def make_report(sources):
    runs, series = [], []
    present = set()
    operation_family = {operation: family for family, operations in FAMILIES.items()
                        for operation in operations}
    for index, source in enumerate(sources):
        metadata = json.loads(source.read_text())
        builds = metadata.pop("builds")
        if not builds:
            raise ValueError(f"{source}: no builds")
        run_id = f"run-{index}"
        source_name = source.parent.name if source.name == "results.json" else source.stem
        runs.append({"id": run_id, "source": source_name, "metadata": metadata})
        for profile, build in enumerate(builds):
            measurements = read_measurements(build, source)
            for row in measurements:
                row["family"] = operation_family.get(row["operation"], "Other")
                present.add(row["operation"])
            series.append({
                "id": f"{run_id}:{profile}", "run": run_id,
                "label": build["label"], "rustflags": build["rustflags"], "rows": measurements,
            })
    # Keep separate machines/runs separate, even when their OS and profile match.
    labels = [item["label"] for item in series]
    for item in series:
        if labels.count(item["label"]) > 1:
            run = next(run for run in runs if run["id"] == item["run"])
            item["label"] += f" · {run['source']} (run {runs.index(run) + 1})"
    operations = [operation for operation in OPERATIONS if operation in present]
    operations += sorted(present - set(OPERATIONS))
    families = [family for family in [*FAMILIES, "Other"]
                if any(operation_family.get(operation, "Other") == family for operation in present)]
    return {"version": 1, "runs": runs, "series": series, "operations": operations,
            "families": families}


def write_report(report, output):
    template = (BENCHES / "report.html").read_text()
    # Script data must never close its HTML element, even with untrusted metadata.
    data = json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    data = re.sub(r"[<>&]", lambda match: f"\\u{ord(match[0]):04x}", data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__BENCHMARK_DATA__", data), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="results JSON files")
    parser.add_argument("--output", type=Path, help="output HTML file (default: first run's index.html)")
    args = parser.parse_args()
    try:
        report = make_report(args.results)
        output = args.output or args.results[0].parent / "index.html"
        write_report(report, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
