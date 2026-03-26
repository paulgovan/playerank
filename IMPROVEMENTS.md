# Model Improvement Ideas

This document captures potential improvements to the PlayeRank extended framework for future development.

## Feature Engineering

- **Game context features**: Encode match state (score at time of action, minute of the match, home/away) as additional input features. A pass made while 1-0 up in the 85th minute carries different value than the same pass at 0-0 in the 10th minute.
- **Opponent strength adjustment**: Weight player scores by the quality of the opponent. Scoring against a top team should be valued more than scoring against the weakest side.
- **Sequence/chain context**: Model multi-event chains (e.g. pass → dribble → shot) rather than isolated events. The contribution of each action in a chain leading to a goal is currently invisible to the model.
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
