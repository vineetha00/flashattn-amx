"""Run gpu_bench.py on an NVIDIA A100 via modal.com (one command, no SSH/SLURM).

    pip install modal
    modal setup          # one-time account link
    modal run modal_bench.py

Needs gpu_bench.py in the same directory. The A100 is Ampere (sm80), so
PyTorch's genuine FLASH_ATTENTION backend runs. Change GPU below to "H100",
"A10G", or "L4" if you like — all are Ampere+ and support real flash.
The run takes ~2 minutes of GPU time.
"""
from pathlib import Path

import modal

GPU = "A100"

app = modal.App("flashattn-amx-gpu-bench")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch")
    .add_local_file(Path(__file__).parent / "gpu_bench.py", "/root/gpu_bench.py")
)


@app.function(gpu=GPU, image=image, timeout=900)
def run_bench() -> str:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "/root/gpu_bench.py"],
        capture_output=True, text=True, check=True,
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.stdout


@app.local_entrypoint()
def main():
    out = run_bench.remote()
    Path("gpu_bench_results.txt").write_text(out)
    print("\nsaved to gpu_bench_results.txt")
