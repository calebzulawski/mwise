# Benchmarks

From the repository root:

```sh
python3 -m venv target/bench-venv
source target/bench-venv/bin/activate
python3 -m pip install -r benches/requirements.txt
python3 benches/run.py
python3 benches/report.py target/benchmarks/results.json
```

Results are saved to `target/benchmarks/results.json`. Open `target/benchmarks/index.html`
directly in your browser to view the interactive plots.

To combine results from multiple runs:

```sh
python3 benches/report.py target/benchmarks/results.json target/benchmarks-wasm/results.json \
  --output target/benchmark-report/index.html
```

## WASM

Install [Wasmtime](https://docs.wasmtime.dev/cli-install.html) and put `wasmtime`
on `PATH`, then run the same benchmark harness through WASI:

```sh
rustup target add --toolchain nightly wasm32-wasip1
python3 benches/run.py --target wasm32-wasip1
python3 benches/report.py target/benchmarks-wasm/results.json
```

The `pown` and `compoundn` benchmarks use mixed exponents from -128 to 128.
The scalar comparison uses `powi`; all benchmark exponents fit its `i32`
parameter. The scalar compound comparison rounds `1+x` before exponentiation,
while `compoundn` retains small increments.
