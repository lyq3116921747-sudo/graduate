from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import test.run_curetest_treatment_matrix as matrix


RUN_STAMP = "20260430_1114"
VARIANTS = [
    ("tool_no_case_access", ("get_case_digest", "read_case_sections", "search_case_text")),
    ("tool_no_sofa", ("run_sofa_assessment",)),
    ("tool_no_guidelines", ("search_local_guidelines",)),
]


def main() -> int:
    print(
        f"=== DeepSeek treatment tool ablation started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===",
        flush=True,
    )
    for name, disabled_tools in VARIANTS:
        output_dir = matrix.DEFAULT_OUTPUT_ROOT / f"treatment_deepseek_{name}_{RUN_STAMP}"
        print(
            f"=== START {name} at {time.strftime('%Y-%m-%d %H:%M:%S')}; "
            f"disabled={disabled_tools}; output={output_dir} ===",
            flush=True,
        )
        try:
            matrix.TOOL_ABLATION_GROUPED_VARIANTS = ((name, disabled_tools),)
            summary = matrix.run_curetest_treatment_matrix(
                curetest_dir=matrix.DEFAULT_CURETEST_DIR,
                output_dir=output_dir,
                model_presets=["deepseek"],
                architecture=matrix.ARCHITECTURE_FULL_STACK_BASE,
                medical_model_presets=[],
                mode="clinical_assist",
                limit=None,
                resume=True,
                max_new_tokens=None,
                enable_generated_code=False,
                enable_treatment_round_reminders=False,
                tool_ablation_mode=matrix.TOOL_ABLATION_GROUPED,
            )
        except Exception:
            print(f"=== FAILED {name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
            traceback.print_exc()
            raise
        print(f"=== DONE {name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        print(json.dumps({name: summary}, ensure_ascii=False, indent=2), flush=True)

    print(
        f"=== DeepSeek treatment tool ablation finished at {time.strftime('%Y-%m-%d %H:%M:%S')} ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
