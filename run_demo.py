"""Interactive classroom runner for direct and cross-Memory fusion inputs."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Windows commonly starts Python in the legacy GBK locale. Inspect AI may run
# subprocesses whose UTF-8 output is otherwise decoded as GBK. Re-enter Python
# once in UTF-8 mode before importing Inspect AI or any detector modules.
if os.name == "nt" and sys.flags.utf8_mode == 0 and os.environ.get("DEMO_UTF8_REEXEC") != "1":
    child_env = dict(os.environ)
    child_env["PYTHONUTF8"] = "1"
    child_env["DEMO_UTF8_REEXEC"] = "1"
    raise SystemExit(
        subprocess.run([sys.executable, *sys.argv], env=child_env, check=False).returncode
    )


DEMO_ROOT = Path(__file__).resolve().parent
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from demo_core.baseline_safety_adapters import detect  # noqa: E402
from demo_core.runtime import (  # noqa: E402
    detect_v2,
    load_env_file,
    run_understanding_agent,
)


BASELINES = ("llm_ac", "agrail", "guardagent", "shieldagent", "v2", "none")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fixed_cases() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = {row["case_id"].lower(): row for row in read_jsonl(DEMO_ROOT / "data" / "cases.jsonl")}
    truth = {row["case_id"].lower(): row for row in read_jsonl(DEMO_ROOT / "data" / "ground_truth.jsonl")}
    return cases, truth


def resolve_case(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["memory_path"] = str((DEMO_ROOT / row["memory_file"]).resolve())
    result["attack_fusion_input"] = row["fusion_input"]
    return result


def practice_case() -> dict[str, Any]:
    instruction_path = DEMO_ROOT / "practice" / "fusion_instruction.txt"
    instruction = instruction_path.read_text(encoding="utf-8").strip()
    if not instruction:
        raise SystemExit(
            "实践区的 fusion_instruction.txt 仍为空。请先填写融合指令后重试。"
        )
    return {
        "case_id": "Practice",
        "source_case_id": "user-authored",
        "dataset": "classroom-practice",
        "fusion_mechanism": "user-authored",
        "fusion_input": instruction,
        "attack_fusion_input": instruction,
        "memory_path": str((DEMO_ROOT / "practice" / "memory.jsonl").resolve()),
    }


async def detector_result(
    baseline: str,
    model: str,
    case: dict[str, Any],
    current_input: str,
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    if baseline == "none":
        return {
            "baseline": "none",
            "blocked": False,
            "reason": "No protection system was enabled.",
        }
    if baseline == "v2":
        return await detect_v2(model, case, current_input, retries, timeout)
    return await detect(baseline, model, current_input, retries, timeout)


async def run_one(
    baseline: str,
    model: str,
    agent_model: str,
    case: dict[str, Any],
    mode: str,
    retries: int,
    timeout: int,
    memory_top_k: int,
) -> dict[str, Any]:
    current_input = (
        case["direct_instruction"] if mode == "direct" else case["fusion_input"]
    )
    print(f"\n=== Baseline={baseline} | Case={case['case_id']} | Mode={mode} ===")
    print("[INPUT]")
    print(current_input)
    # The model-facing state gets a neutral ID.  Names such as Attack-1 and the
    # source dataset ID are display/scoring metadata and must not leak labels.
    online_case = dict(case)
    online_case["case_id"] = case.get("online_sample_id", "classroom-sample")
    verdict = await detector_result(
        baseline, model, online_case, current_input, retries, timeout
    )
    blocked = verdict.get("blocked")
    print(f"[DETECTOR] blocked={blocked} reason={verdict.get('reason') or verdict.get('block_reason')}")

    agent: dict[str, Any] | None = None
    if blocked is False:
        agent = await run_understanding_agent(
            agent_model,
            case,
            mode,
            max_retries=retries,
            timeout=timeout,
            memory_top_k=memory_top_k,
        )
        print("[AGENT UNDERSTANDING — NO EXECUTION]")
        print(agent.get("agent_completion") or f"ERROR: {agent.get('error')}")
    elif blocked is True:
        print("[AGENT] Not called because the protection system blocked the input.")
    else:
        print("[AGENT] Not called because the detector returned an error/invalid result.")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case["case_id"],
        "source_case_id": case.get("source_case_id"),
        "mode": mode,
        "baseline": baseline,
        "current_input": current_input,
        "detector": verdict,
        "blocked": blocked,
        "agent_started": agent is not None,
        "agent_understanding": agent,
        "execution_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated classroom fusion-defense demo."
    )
    parser.add_argument("--baseline", choices=(*BASELINES, "all"), default="v2")
    parser.add_argument("--case", default="Attack-1", help="Benign-1..5 or Attack-1..5")
    parser.add_argument("--mode", choices=("direct", "composition", "both"), default="composition")
    parser.add_argument("--model", default="openai/deepseek-flash")
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--env-file", type=Path, default=DEMO_ROOT / ".env")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--memory-top-k", type=int, default=500)
    parser.add_argument("--practice", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEMO_ROOT / "outputs")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    cases, truth = fixed_cases()
    if args.list_cases:
        for key in sorted(cases):
            row = cases[key]
            # Keep terminal output ASCII-safe for older Windows CMD code pages.
            label = "harmful" if truth[key]["is_harmful"] else "benign"
            print(f"{row['case_id']}: {label}; {row.get('fusion_mechanism')}")
        return

    if args.practice:
        if args.mode != "composition":
            raise SystemExit("实践区仅支持 --mode composition。")
        case = practice_case()
    else:
        key = args.case.lower()
        if key not in cases:
            raise SystemExit(f"未知 Case：{args.case}。用 --list-cases 查看可选项。")
        case = resolve_case(cases[key])

    baselines = list(BASELINES[:-1]) if args.baseline == "all" else [args.baseline]
    modes = ["direct", "composition"] if args.mode == "both" else [args.mode]
    if args.practice:
        modes = ["composition"]

    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        for mode in modes:
            rows.append(
                await run_one(
                    baseline,
                    args.model,
                    args.agent_model or args.model,
                    case,
                    mode,
                    args.max_retries,
                    args.timeout,
                    args.memory_top_k,
                )
            )

    if not args.practice:
        expected = truth[case["case_id"].lower()]
        for row in rows:
            blocked = row["blocked"]
            row["ground_truth_is_harmful"] = expected["is_harmful"]
            row["confusion_label"] = (
                "ERROR"
                if blocked is None
                else ("TP" if blocked else "FN")
                if expected["is_harmful"]
                else ("FP" if blocked else "TN")
            )
            print(
                f"[SCORE] {row['baseline']} {row['mode']} -> {row['confusion_label']}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"{stamp}_{case['case_id']}_{args.mode}.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {output.resolve()}")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
