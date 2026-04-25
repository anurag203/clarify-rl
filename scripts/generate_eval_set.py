"""Generate frozen held-out eval scenarios (seeds 10000-10099)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.scenarios import DIFFICULTIES, generate


def main() -> None:
    out_dir = ROOT / "scenarios"
    out_dir.mkdir(exist_ok=True)

    scenarios = []
    for seed in range(10000, 10100):
        for diff in DIFFICULTIES:
            sc = generate(seed=seed, task_id=diff)
            sc["seed"] = seed
            scenarios.append(sc)

    out_path = out_dir / "eval_held_out.json"
    with open(out_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"Generated {len(scenarios)} scenarios → {out_path}")
    families = {}
    for sc in scenarios:
        families[sc["family"]] = families.get(sc["family"], 0) + 1
    print(f"Family distribution: {families}")


if __name__ == "__main__":
    main()
