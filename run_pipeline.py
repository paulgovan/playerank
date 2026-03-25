#!/usr/bin/env python3
"""
Run the full PlayeRank pipeline in sequence:
  1. Compute feature weights (learning phase)
  2. Compute player roles (learning phase)
  3. Compute playerank scores and rankings (rating/ranking phase)
"""
import subprocess
import sys

steps = [
    ("Feature weights", [sys.executable, "-m", "playerank.utils.compute_features_weight"]),
    ("Player roles",    [sys.executable, "-m", "playerank.utils.compute_roles"]),
    ("PlayerRank",      [sys.executable, "-m", "playerank.utils.compute_playerank"]),
]

for name, cmd in steps:
    print(f"\n{'='*60}\nStep: {name}\n{'='*60}")
    result = subprocess.run(cmd, cwd=__file__.replace("run_pipeline.py", ""))
    if result.returncode != 0:
        print(f"\nPipeline failed at step: {name}")
        sys.exit(result.returncode)

print("\nPipeline complete.")
