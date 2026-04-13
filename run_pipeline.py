#!/usr/bin/env python3
"""
Run the full PlayeRank pipeline in sequence:
  1. Compute feature weights             (learning phase — quality + chain features)
  2. Compute lack-of-performance weights (learning phase — quality features)
  3. Compute harmful chain weights       (learning phase — chain features, loss label)
  4. Compute player roles                (learning phase)
  5. Compute playerank scores            (rating/ranking phase)
  6. Compute lack-of-performance + chain scores (rating/ranking phase → dashboard_data.csv)
"""
import subprocess
import sys

steps = [
    ("Feature weights",              [sys.executable, "-m", "playerank.utils.compute_features_weight"]),
    ("Lack-of-performance weights",  [sys.executable, "-m", "playerank.utils.compute_lack_of_performance_weights"]),
    ("Harmful chain weights",        [sys.executable, "-m", "playerank.utils.compute_harmful_chain_weights"]),
    ("Player roles",                 [sys.executable, "-m", "playerank.utils.compute_roles"]),
    ("PlayerRank",                   [sys.executable, "-m", "playerank.utils.compute_playerank"]),
    ("Lack-of-performance scores",   [sys.executable, "-m", "playerank.utils.compute_lack_of_performance"]),
]

for name, cmd in steps:
    print(f"\n{'='*60}\nStep: {name}\n{'='*60}")
    result = subprocess.run(cmd, cwd=__file__.replace("run_pipeline.py", ""))
    if result.returncode != 0:
        print(f"\nPipeline failed at step: {name}")
        sys.exit(result.returncode)

print("\nPipeline complete.")
