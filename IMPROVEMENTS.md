# Model Improvement Ideas

This document captures potential improvements to the PlayeRank extended framework for future development.

## Feature Engineering

- **Game context features**: Encode match state (score at time of action, minute of the match, home/away) as additional input features. A pass made while 1-0 up in the 85th minute carries different value than the same pass at 0-0 in the 10th minute.
- **Opponent strength adjustment**: Weight player scores by the quality of the opponent. Scoring against a top team should be valued more than scoring against the weakest side.
- **Sequence/chain context**: Model multi-event chains (e.g. pass → dribble → shot) rather than isolated events. The contribution of each action in a chain leading to a goal is currently invisible to the model.

  **Implementation plan:**

  1. **Chain detection** — Sort events by `(matchPeriod, eventSec)` within a match and group consecutive same-team events into possession chains. A chain breaks on:
     - `INTERRUPTION` (eventId 5): ball out of play or referee whistle.
     - `FOUL` (eventId 2) or `OFFSIDE` (eventId 6).
     - Match-period change.
     - Possession change (different `teamId`), confirmed by loss-of-possession tags on the previous event (`NOT_ACCURATE_TAG 1802`, `DANGEROUS_BALL_LOST_TAG`, `INTERCEPTION_TAG 1401`). Bare `teamId` changes without a loss tag (e.g. paired duel events where the attacker keeps the ball) should be treated conservatively as breaks unless a look-ahead confirms the same team retains possession.
     - Optional: large time gap (>5 s) between consecutive events, to avoid merging truly separate phases.

  2. **Chain outcome classification** — Classify each chain by its terminal event:
     - `"goal"` — last event is `SHOT` (eventId 10) with `GOAL_TAG (101)`.
     - `"shot"` — last event is `SHOT` without `GOAL_TAG`.
     - `None` — chain ends without a shot; no credit assigned.

  3. **New feature class `chainFeatures.py`** — Extends `Feature` (from `abstract.py`). Produces three features per player per match:
     - `chain-shot-participant`: count of chains ending in a shot the player touched the ball in.
     - `chain-goal-participant`: count of chains ending in a goal the player touched the ball in.
     - `chain-final-action`: count of times the player made the last non-shot action before a shot (key-pass analog).

  4. **Pipeline integration** — Add chain features alongside quality features in both training and scoring:
     - `compute_features_weight.py`: include chain features in `aggregation.set_features([quality, goals, chains])` so the `Weighter` learns weights for them.
     - `compute_playerank.py`: include player-level chain features in `playerankFeat.set_features()` so they contribute to `playerankScore`.
     - No changes needed to `Weighter.py` or `Rater.py` — chain features are simply additional columns.

  5. **README updates** (`README.md`) — Document the new capability alongside the existing lack-of-performance section:
     - Add a **Sequence/chain context** section explaining what possession chains are, how chain detection works, and which new features are produced (`chain-shot-participant`, `chain-goal-participant`, `chain-final-action`).
     - Update the pipeline step list to note that chain features are computed automatically as part of `compute_features_weight.py` and `compute_playerank.py` — no new standalone script required.
     - Add a brief note on the chain-breaking rules (interruptions, fouls, possession-loss tags) so users understand what counts as a chain boundary.

  6. **Dashboard updates** (`dashboard.py`) — Add a new **⛓️ Chain Analytics** tab (between the existing Role Breakdown and Player Profile tabs) with three sections:
     - **Chain leaderboard**: table ranking players by `chain-goal-participant` and `chain-final-action` counts per match, filterable by role cluster. Useful for identifying players who consistently appear in dangerous build-up sequences.
     - **Chain contribution scatter**: x-axis = `chain-shot-participant` per match, y-axis = `chain-goal-participant` per match, sized by matches played, coloured by role cluster. Highlights players who reach the shot phase frequently vs. those who convert chains into goals.
     - **Player chain timeline**: in the existing **👤 Player Profile** tab, extend the match-by-match section to include a second chart showing `chain-shot-participant` and `chain-goal-participant` counts per match as an area chart, giving a visual sense of how involved a player is in dangerous sequences over a season.
     - Data requirements: `dashboard_data.csv` must include the three chain feature columns; `compute_playerank.py` should write them alongside the existing score columns when chain features are enabled.

  7. **Data volume** — The full Wyscout dataset provides ~3.25 M events across 1,941 matches and 7 competitions, yielding an estimated 300,000–400,000 possession chains (~43,000 ending in a shot, ~14%). This is a sufficient training signal for LinearSVC to learn chain feature weights. Note that World Cup and Euro data (~115 matches combined) reflects international rather than club football; monitor whether learned weights differ significantly if training on competitions separately.

  8. **Chain pitch visualisation** — Add a **⛓️ Chain Map** sub-section inside the Chain Analytics tab. 99.98% of events have both a start and end position (`positions[0]` / `positions[1]` in Wyscout 0–100 coords), so chain paths can be drawn directly onto the existing `draw_soccer_field()` Plotly figure using the same coordinate transform already used for role clusters (`x-axis = Wyscout y`, `y-axis = Wyscout x`). Four views to implement:
     - **Single chain replay**: select a match → select a chain → draw event-by-event `go.Scatter` arrows coloured by event type (pass = blue, dribble = yellow, shot = red, goal = green).
     - **Goal chain origin heatmap**: overlay all `chain-goal` event start positions as a `go.Densitymapbox` or binned `go.Heatmap` on the pitch — shows which zones dangerous sequences originate from.
     - **Player contribution map**: for a selected player, plot positions where they appeared in shot/goal chains — reveals their spatial role in attacking build-up.
     - **Chain length vs outcome**: scatter plot of chain length (event count) vs outcome (shot / goal / none) — answers whether short counters or long build-up sequences are more likely to result in goals.
- **Minutes-played normalisation**: Divide per-match scores by actual minutes played rather than treating a 90-minute appearance the same as a 10-minute substitute appearance.

## Model Architecture

- **Role-specific models**: Train separate SVM models for each role cluster rather than a single global model. Defensive actions are undervalued by a model trained on all positions simultaneously.
- **Non-linear model**: Replace LinearSVC with a gradient-boosted tree or neural network to capture interaction effects between features (e.g. a high shot rate matters more when combined with high positioning quality).
- **Uncertainty quantification**: Report a confidence interval around each player score, not just a point estimate. Players with few appearances should have wider intervals to prevent small-sample flukes from dominating leaderboards.

## Label Quality

- **xG differential instead of match result**: Replace binary win/loss labels with the match-level expected goals differential. This provides a continuous, less noisy training signal and decouples luck (post shots, goalkeeper errors) from genuine team performance.
- **Weighted match outcomes**: Down-weight heavily lopsided results (e.g. 5-0 wins) when training, since they may reflect opponent weakness more than individual quality.

## Aggregation and Ranking

- **Time-decay weighting**: Apply an exponential decay so that recent matches contribute more to the current rating than matches from early in the season. Useful for tracking form.
- **Plus-minus component**: Add a simple on/off plus-minus term (team goal difference when the player is on the field vs. off) to complement the action-based playerankScore.

## Waste Model

- **Independent feature sets**: Currently `wasteScore` is the playerankScore re-weighted with loss-associated weights, so the two models share the same feature space. Training the waste model on a separate feature set (e.g. turnovers, fouls conceded, poor positional choices) would make the two scores more independent and informative.
- **Asymmetric weighting**: Some action types matter more for waste than for performance (e.g. defensive errors). Allow the waste model to up-weight defensive event types relative to the performance model.
- **Harmful chain score (`chainWasteScore`)**: Mirror the `wasteScore` pattern at the chain level to quantify which possession sequences are most detrimental to match outcomes. This is a natural extension of the chain context feature and uses the same `Weighter(label_type='l-wd')` mechanism.

  **Implementation plan:**

  1. **Two types of harmful chain** — Extend `chainFeatures.py` to detect and credit two categories of damaging chain at the player level:
     - *Turnover chains (Type 1)*: Your team's chain ends in a dangerous possession loss, and the **immediately following opponent chain** ends in a shot or goal. Link consecutive chains by checking `chain[i].team ≠ chain[i+1].team` and `chain[i+1]` outcome is shot/goal. New features:
       - `chain-turnover-precedes-shot`: player appeared in a chain that directly preceded an opponent shot.
       - `chain-turnover-precedes-goal`: same, but the opponent scored.
       - `chain-turnover-final-actor`: player made the last action before the turnover that led to a shot/goal.
     - *Conceded chains (Type 2)*: The opposing team completes a chain that ends in a shot or goal against you. Attribute these to the defensive players who appeared in duels or defensive actions *within that opponent chain*. New features:
       - `chain-conceded-shot`: player made a defensive action in a chain the opponent used to reach a shot.
       - `chain-conceded-goal`: same, but the opponent scored.

  2. **Training** — Add harmful chain features (at team level) to `relativeAggregation` and train a new `Weighter(label_type='l-wd')`, identical to `compute_lack_of_performance_weights.py` but using chain features as input. Output: `playerank/conf/harmful_chain_weights.json`.

  3. **Scoring** — Apply `harmful_chain_weights.json` to player-level harmful chain features via `playerankFeatures.createFeature()`, producing `chainWasteScore` per player per match. Rename the feature doc (`doc['feature'] = 'chainWasteScore'`), mirroring how `compute_lack_of_performance.py` renames `wasteScore`.

  4. **Net chain score** — Combine in the final aggregation step:
     ```
     chainNetScore = chainScore − chainWasteScore
     ```
     A player with high `chainScore` but high `chainWasteScore` participates in dangerous build-up but also regularly causes or appears in harmful sequences — a risk profile not visible in the existing action-count scores.

  5. **Independence from existing scores** — `chainNetScore` is built entirely from chain-level features, making it genuinely complementary to `netScore` (which is built from isolated event counts). The two can diverge meaningfully: a player may rank highly on `netScore` (good individual actions) but poorly on `chainNetScore` (frequently involved in turnover chains), or vice versa.

  6. **Dashboard updates** — Extend the Chain Analytics tab:
     - Add `chainWasteScore` and `chainNetScore` to the chain leaderboard alongside `chainScore`.
     - Add a **Harmful chain map** pitch view: overlay positions where `chain-turnover-precedes-goal` events occurred, coloured by pitch zone — identifies which areas of the field dangerous turnovers most commonly originate from.
     - Add a **chain scatter** analogous to the existing Performance vs Waste scatter: x-axis = `chainScore`, y-axis = `chainWasteScore`, coloured by `chainNetScore`, to identify players who are net positive vs net harmful in sequence play.
