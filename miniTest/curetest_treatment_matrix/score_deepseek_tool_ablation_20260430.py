from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORER = PROJECT_ROOT / "test" / "score_curetest_treatment_with_qwen.py"
MATRIX_DIR = PROJECT_ROOT / "miniTest" / "curetest_treatment_matrix"
RUN_STAMP = "20260430_1114"
SCORE_STAMP = "20260430_1150"

VARIANTS = [
    "tool_no_case_access",
    "tool_no_sofa",
    "tool_no_guidelines",
]
JUDGES = [
    "gemini_flash",
    "gpt_mini",
]


def run_one(variant: str, judge: str) -> None:
    result_dir = MATRIX_DIR / f"treatment_deepseek_{variant}_{RUN_STAMP}"
    results_csv = result_dir / "results.csv"
    output_dir = result_dir / f"general_treatment_scores_{judge}_{SCORE_STAMP}"
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results.csv: {results_csv}")

    print(
        f"=== START scoring {variant} with {judge} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(SCORER),
            "--results-csv",
            str(results_csv),
            "--output-dir",
            str(output_dir),
            "--judges",
            judge,
            "--resume",
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    print(
        f"=== DONE scoring {variant} with {judge} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===",
        flush=True,
    )


def main() -> int:
    print(f"=== DeepSeek tool ablation scoring started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    for variant in VARIANTS:
        for judge in JUDGES:
            run_one(variant, judge)
    print(f"=== DeepSeek tool ablation scoring finished at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
