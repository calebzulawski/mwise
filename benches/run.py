#!/usr/bin/env python3
"""Run the configured benchmark profiles and save one results.json file."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess

from cpuinfo import get_cpu_info

BENCHES = Path(__file__).resolve().parent
ROOT = BENCHES.parent


def profile_config():
    return json.loads((BENCHES / "profiles.json").read_text())


def name_variables(metadata):
    host = metadata["host"]
    system = platform.system()
    architecture = host.split("-", 1)[0].lower()
    return {
        "OS": "macOS" if system == "Darwin" else system,
        "ARCH": {"aarch64": "arm64", "amd64": "x86_64"}.get(architecture, architecture),
    }


def machine_description():
    cpu = get_cpu_info().get("brand_raw") or platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        return cpu, f"macOS {platform.mac_ver()[0]}"
    system = platform.platform()
    if platform.system() == "Linux":
        system = platform.freedesktop_os_release()["PRETTY_NAME"]
    return cpu, system


def make_metadata():
    cpu, system = machine_description()
    version = subprocess.check_output(["rustc", "+nightly", "--version"], text=True).strip()
    host = subprocess.check_output(["rustc", "+nightly", "--print", "host-tuple"], text=True).strip()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = ""
    return {"cpu": cpu, "os": system, "kernel": platform.release(), "rustc": version,
            "host": host, "architecture": platform.machine(), "commit": commit,
            "date_utc": datetime.now(timezone.utc).isoformat(), "builds": []}


def capability_tags():
    """Query host capabilities using the target's default CPU features."""
    env = {**os.environ, "CARGO_BUILD_TARGET": "host-tuple",
           "CARGO_ENCODED_RUSTFLAGS": ""}
    return set(subprocess.check_output(
        ["cargo", "+nightly", "-Zscript", "--quiet", str(BENCHES / "support/capabilities.rs")],
        env=env, text=True).split())


def load_profiles(target):
    architecture = "wasm32" if target.startswith("wasm") else (
        "x86_64" if platform.machine().lower() in ("x86_64", "amd64") else "other")
    profiles = profile_config()[architecture]
    tags = capability_tags() if any(profile.get("requires") for profile in profiles) else set()
    return [profile for profile in profiles if set(profile.get("requires", [])) <= tags]


def run_profile(flags, cpu, target, label):
    print(f"Running {label}: {' '.join(flags)}", flush=True)
    env = {**os.environ, "RUSTFLAGS": " ".join(flags)}
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    command = ["cargo", "+nightly", "bench", "--locked", "--bench", "math", "--target", target]
    if target == "wasm32-wasip1":
        env["CARGO_TARGET_WASM32_WASIP1_RUNNER"] = "wasmtime run -W relaxed-simd=y --"
    subprocess.run(command + ["--no-run"], cwd=ROOT, env=env, check=True)
    result = subprocess.run(command + ["--", "--test-threads=1", "--format=json", "-Zunstable-options"],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True, check=True,
                            preexec_fn=(lambda: os.sched_setaffinity(0, {cpu})) if cpu is not None else None)
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if not events or events[-1].get("event") != "ok":
        raise ValueError(f"{label}: incomplete or failed results")
    measurements = [event for event in events if event.get("type") == "bench"]
    if not measurements:
        raise ValueError(f"{label}: no measurements")
    return {"rustflags": " ".join(flags), "label": label, "measurements": measurements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host-tuple", "wasm32-wasip1"], default="host-tuple")
    parser.add_argument("--output", type=Path, help="output directory for results.json")
    args = parser.parse_args()
    directory = "benchmarks-wasm" if args.target == "wasm32-wasip1" else "benchmarks"
    output = (args.output or ROOT / "target" / directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] += os.pathsep + str(Path.home() / ".cargo/bin")
    profiles = load_profiles(args.target)
    cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    metadata = make_metadata()
    metadata["target"] = args.target
    if args.target == "host-tuple":
        metadata["target"] = metadata["host"]
    else:
        metadata["runtime"] = subprocess.check_output(["wasmtime", "--version"], text=True).strip()
    variables = name_variables(metadata)
    results = output / "results.json"
    for profile in profiles:
        metadata["builds"].append(run_profile(profile["rustflags"], cpu, args.target,
                                             label=profile["name"].format(**variables)))
        results.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(results)


if __name__ == "__main__":
    main()
