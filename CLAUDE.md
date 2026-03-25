# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

```bash
cd playerank
pip install -e .
```

## Data Setup

Download the public Wyscout soccer dataset (figshare) into `data/matches/` and `data/events/`:

```bash
python data_download.py
```

## Running the Pipeline

Run all phases at once from the repo root:

```bash
python run_pipeline.py
```

Or run individual phases as modules (works from any directory):

```bash
python -m playerank.utils.compute_features_weight  # Phase 1: feature weights → playerank/conf/features_weights.json
python -m playerank.utils.compute_roles            # Phase 2: player roles    → playerank/conf/role_matrix.json
python -m playerank.utils.compute_playerank        # Phase 3+4: scores/ranks  → data/playerank.json
```

## Architecture

PlayeRank is a 4-component soccer player rating pipeline:

```
Soccer-logs (Wyscout events) → Learning → Rating → Ranking
```

### Data Format
Input events are Wyscout-format JSON with fields: `id`, `type`, `position`, `timestamp`. Event taxonomy is defined in `playerank/features/wyscoutEventsDefinition.py`.

### Feature System (`playerank/features/`)
- `abstract.py`: Base classes `Feature` (computes per-match feature vectors) and `Aggregation` (merges features across matches)
- Features produce `match → entity → feature → value` dictionaries
- `qualityFeatures.py`: Accurate/failed passes, shots, duels etc. — excludes goalkeepers and penalty periods
- `playerankFeatures.py`: Applies learned weights to quality features to produce per-match scores
- `plainAggregation.py` / `relativeAggregation.py`: Merge and aggregate feature collections into DataFrames
- `centerOfPerformanceFeature.py`: Average player pitch positions (used for role assignment)
- `roleFeatures.py`: Assigns players to roles based on positional clusters

### Model Layer (`playerank/models/`)
- `Weighter.py`: Trains a `LinearSVC` on team-level features to learn which features predict match outcome; outputs feature weights. Label types: `'w-dl'`, `'wd-l'`, `'w-d-l'`
- `Rater.py`: Combines playerank score with goal-scoring using: `rating = score * (1 - alpha_goal) + goals * alpha_goal`, then MinMax-normalizes
- `Clusterer.py`: K-means clustering (8–9 clusters) on positional features to define player roles

### Configuration
- `playerank/conf/features_weights.json`: Pre-computed feature weights (30+ features). Negative weights are intentional — they reflect inverse relationships in LinearSVC coefficients.

### Key Design Pattern
The utility scripts in `playerank/utils/` orchestrate full pipeline phases by composing `Feature` and `Aggregation` subclasses, passing their outputs through model classes, and writing JSON results to `data/`.
