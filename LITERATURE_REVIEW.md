# Literature Review: Soccer Player Performance Analytics

## State of the Art

This document surveys the academic and applied literature on data-driven soccer player evaluation, situating PlayeRank within the broader field and documenting the state of the art through 2025.

---

## 1. PlayeRank

**Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P., Pedreschi, D., & Giannotti, F. (2019). "PlayeRank: Data-driven Performance Evaluation and Player Ranking in Soccer via a Machine Learning Approach." *ACM Transactions on Intelligent Systems and Technology*, 10(5), Article 59.**

PlayeRank addresses a fundamental open problem: how to evaluate a soccer player's performance in a principled, multi-dimensional, and *role-aware* manner. Prior approaches relied on subjective scout judgments, simplistic aggregate statistics (goals, assists), or proprietary black-box commercial systems.

**Methodology.** The framework has three phases:

1. *Learning phase.* Feature weights are learned by framing performance as a team-level classification problem — given aggregated player features for a team in a match, can a Linear SVM predict the match result? The resulting SVM coefficient vector encodes which features (passes, shots, duels, etc.) are most predictive of winning. In parallel, an unsupervised clustering step discovers fine-grained player roles beyond the five nominal positions, assigning players to roles if at least 40% of their matches map to each.

2. *Rating phase.* A player's per-match performance score is the dot product of the learned weight vector and the player's feature values — interpretable and efficient.

3. *Ranking phase.* Per-match scores are aggregated across a season into role-aware rankings, supporting queries such as "top 10 wide forwards in Serie A."

**Data.** Trained on Wyscout event logs from 18 competitions across four seasons. Validated against professional scout evaluations, where it showed significantly higher correlation than WhoScored, EA Sports FIFA ratings, and other competitor systems.

**Limitations.** Restricted to on-ball event data — pressing, off-ball runs, and spatial occupation are not captured. Feature weights learned at team level may confound individual contributions with team quality. The open dataset companion (Pappalardo et al., *Scientific Data* 2019) covers 7 competitions, one season each (~3M events), and is what this codebase uses.

---

## 2. Expected Goals (xG)

Expected Goals quantifies the quality of a shot as the probability it results in a goal, given contextual features at the moment of the shot.

**Origins.** The concept and abbreviation "xG" were introduced publicly by Sam Green at Opta in 2012 (*OptaPro Blog*, April 2012), applied to the 2011–12 EPL season. Core features were distance, angle, shot type (header, foot), and assist type (cross, through ball, set piece, counter-attack).

**Key academic work.** Lucey et al. (2015, *MIT Sloan*) showed that contextual strategic features — game phase, defender proximity within the shot angle, speed of play — improve xG beyond location alone. Eggels et al. (2016, *ECML/PKDD Workshop*) benchmarked Logistic Regression, Decision Tree, Random Forest, and AdaBoost on 128,667 shots, establishing large-scale baselines. Brechot & Flepp (2020, *Journal of Sports Economics*) demonstrated that xG-based evaluation filters outcome noise better than raw goals, especially over short windows.

**Post-shot xG (PSxG).** Developed independently by StatsBomb and Opta, PSxG conditions on the shot being on target and adds shot placement within the goal frame and trajectory. It enables goalkeeper evaluation: Goals Prevented = PSxG − Goals Conceded.

**Current state.** xG is a standard broadcast metric (Sky Sports, BBC, NBC). Every major data provider has its own proprietary model. Davis & Robberechts (2024, *arXiv*) show that current xG models systematically misattribute finishing quality due to selection biases and model misspecification, calling for causal treatment. Modern approaches incorporate preceding possession context, pressure data (StatsBomb 360), and transformer architectures.

---

## 3. VAEP — Valuing Actions by Estimating Probabilities

**Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). "Actions Speak Louder than Goals: Valuing Player Actions in Soccer." *KDD 2019*, pp. 1851–1861.**

VAEP provides a unified, data-driven value for every on-ball action — not just shots. The insight is that every action shifts the game from one state to another. VAEP models two probabilities for each game state: P(score within next 10 actions) and P(concede within next 10 actions), learned via gradient-boosted trees on SPADL-encoded features (see Section 4). The value of an action = (change in scoring probability) − (change in conceding probability). Player season ratings sum action values across all touches.

Van Roy et al. (2020, *AAAI Workshop on AI in Team Sports*) compared VAEP against Expected Threat (xT) and found VAEP more reliable for measuring defenders and midfielders, whose contributions to goal-scoring are indirect and poorly captured by xT.

VAEP is implemented in the open-source `socceraction` Python package (ML-KULeuven/socceraction on GitHub).

---

## 4. SPADL and the Socceraction Framework

**Decroos, T., & Davis, J. (2020). Socceraction.** A Python library providing SPADL (Soccer Player Action Description Language): a unified representation of on-ball actions that standardises event stream data from different vendors (Wyscout, StatsBomb, Opta) into a common vocabulary of 12 action types with (x, y) start/end coordinates, body part, and outcome. SPADL has become a de facto standard in academic soccer analytics, enabling direct replication across data providers.

**Atomic SPADL** (Decroos, Robberechts & Davis, 2020) decomposes compound actions (completed pass) into atomic sub-actions (ball release + ball reception), enabling finer credit assignment and better handling of direction and receiver information.

---

## 5. Pitch Control and Spatial Analytics

**Wide Open Spaces.** Fernandez & Bornn (2018, *MIT Sloan*) modelled player influence as a bivariate Gaussian distribution shaped by velocity and direction, yielding a continuous pitch control surface indicating which team controls each location at each moment.

**Beyond Expected Goals.** Spearman (2018, *MIT Sloan*) introduced Off-Ball Scoring Opportunities (OBSO) using a physics-based pitch control model that accounts for ball travel time under aerodynamic drag and player locomotion time. For any pitch location at any moment, OBSO computes the probability that the attacking team controls the ball *and* scores from it — enabling the first principled evaluation of off-ball attacker positioning.

**Expected Possession Value (EPV).** Fernandez, Bornn & Cervone (2021, *Machine Learning*, 110:3021–3062) built a full tracking-data EPV framework that updates a team's probability of scoring next at every frame of the match. The framework is decomposed into a pass value surface (deep neural network), ball-drive value, shot probability, and shot outcome. It enables counterfactual analysis: "what was the value of the pass not taken?"

**SoccerMap** (Fernandez & Bornn, *ECML PKDD 2020*) is a fully convolutional network producing full-pitch probability surfaces for pass success, pass selection, and pass expected value from spatiotemporal tracking data.

**Pressing metrics.** StatsBomb introduced PPDA (Passes Per Defensive Action) around 2017 — (opponent passes allowed) / (defensive actions in opponent's defensive 60% of pitch) — now a standard measure of pressing intensity across the industry.

---

## 6. Plus-Minus Models

Adapted from basketball, regularised Adjusted Plus-Minus (APM) models treat each match segment (or possession) as an observation unit and estimate individual player effects via penalised regression.

Macdonald (2011) first applied APM to soccer. Schultze & Wellbrock (2018, *Journal of Sports Analytics*) developed a weighted plus-minus metric. Pantuso & Hvattum (2024/2025, *IMA Journal of Management Mathematics*) introduced possession-based regularised APM with four penalisation schemes exploiting position-group and strength-group priors, addressing the collinearity problems that plague shorter seasons compared to basketball. The key advantage over action-value models is that APM captures off-ball contributions; the key limitation is that it requires many observations to isolate individual effects from team quality.

---

## 7. Player Embeddings and Representation Learning

**Pass2Vec and action embeddings.** Applying word2vec principles to soccer, Magdaci (2021) tokenised actions into spatial grid cells and action types, learning action embeddings that enable unsupervised player similarity and style clustering without manual feature engineering.

**Graph-based embeddings.** ACM SAC (2022) constructed a player-similarity graph based on performance statistics, applied GNN-based node embeddings, and used k-NN on the learned representations for player replacement recommendation.

---

## 8. Graph Neural Networks

**TacticAI** (Wang et al., 2024, *Nature Communications*, 15:1906 — Google DeepMind × Liverpool FC) uses geometric deep learning on a graph representation of corner-kick situations (players as nodes, spatial interactions as edges) to (a) predict which receiver gets the first touch and whether a shot results, and (b) generate alternative tactical setups. Trained on 7,176 Liverpool corner kicks; expert raters preferred TacticAI suggestions 90% of the time over setups actually used.

**GoalNet** (2025, *arXiv* 2503.09737) builds event-centric graphs per match, using graph attention networks (GATGoalNet) and a transformer-GNN hybrid (TransGoalNet) to distribute xT credit among all involved players, not just ball recipients. Designed to surface "hidden orchestrators" — defensive midfielders and ball-playing defenders poorly captured by traditional metrics.

**Pass Receiver Prediction** (2023) uses temporal GNNs to model dynamic player relationships evolving within a possession sequence, predicting pass recipient and outcome.

---

## 9. Foundation and Language Models

**Foundation Model for Soccer** (Baron, Hocevar & Salehe, 2024, *arXiv* 2407.14558). A transformer treating each match as a sequence of SPADL-style event tokens. Pre-trained on multiple WSL seasons, it supports diverse downstream tasks — action valuation, player evaluation, outcome prediction — via fine-tuning.

**Large Events Model (LEM)** (Mendes-Neves et al., 2024, *Machine Learning*). Directly analogous to LLMs: soccer events are tokens, a match is a sentence. Outputs event-level probabilities and enables simulation from any game state. Introduces xP+, a player's contribution to team points earned estimated via model simulation. Scaled to multiple leagues and competitions in a follow-up (2025, Springer LNCS).

**LLMs applied to soccer** (2024–2025). Multiple papers have applied general-purpose LLMs: automatic commentary generation from event streams (Knowledge-Based Systems, 2024), next-event prediction treating event sequences as text (arXiv 2402.06820), and natural-language interfaces to event databases (ECAI 2024). A multimodal RAG approach for commentary appeared at WACV 2025.

---

## 10. Deep Reinforcement Learning for Decision Optimisation

**Beyond Action Valuation** (Rahimian et al., 2022, *MIT Sloan*). Rather than valuing observed actions, this DRL framework outputs an optimal ball-destination surface from tracking data — "what should the player have done?" Extended formally in Rahimian et al. (2024, *International Journal of Sports Science & Coaching*).

**On- and Off-Ball Valuation via Multi-Agent DRL** (*IEEE Transactions on Games*, 2023). The first unified framework that simultaneously values actions for ball-possessing and off-ball players using a simulated environment based on Google Research Football.

---

## 11. Player Valuation and Transfer Markets

**CIES Football Observatory** uses multiple linear regression trained on 2,400+ actual transfers (2011–2018). Features include player age, position, contract length, league level, nationality, and recent performance. Estimates transfer values for 30,000+ players across 74 leagues.

**Machine learning for transfer fees.** Sæbø & Hvattum (2019, *MLSA/ECML*) combined Wyscout performance data with Transfermarkt values in regression models. Multiple papers (2022–2025) have used gradient-boosted models and deep neural networks on 20–30 performance metrics, finding that position, league level, contract length, and age dominate. SHAP-based explanation of XGBoost valuation models (arXiv 2311.04599, 2023) identifies position-specific drivers for practitioners.

---

## 12. Creativity and Debiasing

**un-xPass** (Robberechts, Euvrard & Davis, 2023, *KDD 2023*). Defines the Creative Decision Rating (CDR): a pass is creative if it is simultaneously *unusual* (low xPass probability given the game state distribution) and *valuable* (high VAEP outcome). Built on StatsBomb 360 data for the EPL; identifies Kevin De Bruyne, Mesut Özil, and Toni Kroos as consistently creative.

**Debiased ML for player evaluation** (ACM ICoMS 2023). Applies Double Machine Learning (DML) to disentangle individual player contributions from team-quality confounding — a principled answer to the problem acknowledged by PlayeRank (that team-level training may conflate player quality with team strength).

---

## 13. Tracking Data and Key Datasets

| Dataset | Type | Coverage | Access |
|---|---|---|---|
| Wyscout Public (Pappalardo et al., 2019) | Event stream | 7 competitions, ~3M events | Open (Figshare) |
| StatsBomb Open Data | Event + 360 freeze frames | 40+ competitions (Messi career, World Cups, WSL) | Open (GitHub) |
| StatsBomb 360 (commercial) | Event + player locations | 40+ leagues | Commercial |
| Opta / Stats Perform | Event stream | 60+ sports, 70 countries | Commercial |
| SkillCorner Open Data | Broadcast tracking | 10 sample matches | Open (GitHub) |
| SkillCorner (commercial) | Broadcast tracking | 40+ leagues | Commercial |
| TRACAB | Optical tracking (25 Hz) | Bundesliga, EPL, LaLiga, UCL | Commercial |
| Second Spectrum | Optical tracking | EPL (2020–), MLS | Commercial |

**TRACAB** (ChyronHego) is the dominant optical tracking system in professional football, covering most top-tier grounds. **SkillCorner** extracts tracking data from broadcast footage using computer vision and temporal-graph networks — democratising tracking data because broadcast coverage is near-universal and the system requires no pitch-side hardware.

---

## 14. Methodological Landscape

| Approach | Strength | Limitation | Key Work |
|---|---|---|---|
| Aggregate statistics | Simple, interpretable | Context-free; position-biased | Goals, assists, pass % |
| xG family | Shot quality, validated | Values only ~5% of actions | Green 2012; Lucey 2015 |
| VAEP / xT | Values all on-ball actions | Event data only; no off-ball | Decroos et al. 2019 |
| PlayeRank | Role-aware; multi-dimensional | Team-level learning; no off-ball | Pappalardo et al. 2019 |
| Pitch control | Spatial; off-ball | Requires tracking data | Fernandez 2018; Spearman 2018 |
| EPV | Rich context; counterfactuals | Requires full tracking data | Fernandez et al. 2021 |
| Plus-Minus / APM | Captures off-ball via team proxy | Collinearity; long data needed | Pantuso & Hvattum 2024 |
| GNN-based | Models player interactions | Requires large labelled data | TacticAI 2024; GoalNet 2025 |
| Foundation / LEM | Transfer learning; simulation | Data hungry; interpretability | Baron 2024; Mendes-Neves 2024 |
| DRL | Optimal counterfactual policy | Simulation gap; partial observability | Rahimian et al. 2022 |

---

## 15. Open Problems and Future Directions

**Off-ball evaluation.** Even the most sophisticated models largely evaluate the player in possession. Pressing, off-ball runs, positioning, and space-creation are captured only with tracking data, which is expensive and not universally available. SkillCorner's broadcast-tracking approach is the most promising path to broader access.

**Causal inference.** Current action-value models are predictive, not causal. Davis et al. (*Machine Learning*, 2024) and arXiv 2505.11841 (2025) identify correct causal framing — using double ML, instrumental variables, or synthetic controls — as the frontier for questions like "does this player cause his team to win?"

**Unifying event and tracking data.** Most academic work uses one data type. Combining rich event labels with partial tracking data (SkillCorner broadcast tracking providing approximate positions) is an emerging direction.

**Transfer across competitions.** Models trained on top European leagues do not necessarily generalise to lower divisions or other confederations. Foundation models offer a solution via pre-training on large multi-competition corpora and fine-tuning on small target datasets.

**Women's football.** Almost all published models are trained exclusively on men's data. StatsBomb and others have begun releasing women's competition datasets (particularly WSL and World Cup), but dedicated models and systematic evaluation remain scarce.

**Evaluation methodology.** Davis et al. (2024, *Machine Learning*) identifies underappreciated challenges in how player rating models are evaluated: correct data partitioning (by player, match, or competition), handling temporal dependencies, and measuring downstream utility (does the rating help clubs win transfers or predict future performance?) rather than internal consistency.

**Explainability for practitioners.** The gap between model accuracy and practitioner trust is significant. SHAP-based explanations (un-xPass, transfer valuation models) and LLM-based justification generation are emerging bridges, but integrating them rigorously into pipelines is ongoing work.

**Fairness and bias.** Player rating systems trained on historical data reflect historical biases in scouting, playing time allocation, and transfer markets. Auditing these systems and developing debiasing methods is an open research problem.

---

## References

- Pappalardo et al. (2019). PlayeRank. *ACM TIST*, 10(5). https://dl.acm.org/doi/10.1145/3343172
- Pappalardo et al. (2019). A public dataset of event-stream data. *Scientific Data*. https://www.nature.com/articles/s41597-019-0247-7
- Green, S. (2012). Assessing the performance of Premier League goalscorers. *OptaPro Blog*.
- Lucey, P. et al. (2015). Quality vs Quantity. *MIT Sloan Sports Analytics Conference*.
- Eggels, H. et al. (2016). Explaining Soccer Match Outcomes. *ECML/PKDD MLSA Workshop*.
- Brechot, M. & Flepp, R. (2020). Dealing with Randomness in Match Outcomes. *Journal of Sports Economics*, 21(4).
- Decroos, T. et al. (2019). Actions Speak Louder than Goals (VAEP). *KDD 2019*.
- Decroos, T. & Davis, J. (2020). Socceraction. https://github.com/ML-KULeuven/socceraction
- Van Roy, M. et al. (2020). Valuing On-the-Ball Actions: A Critical Comparison of xT and VAEP. *AAAI Workshop on AI in Team Sports*.
- Fernandez, J. & Bornn, L. (2018). Wide Open Spaces. *MIT Sloan Sports Analytics Conference*.
- Spearman, W. (2018). Beyond Expected Goals. *MIT Sloan Sports Analytics Conference*.
- Fernandez, J., Bornn, L., & Cervone, D. (2021). A Framework for the Fine-Grained Evaluation of Soccer Possessions. *Machine Learning*, 110(11).
- Fernandez, J. & Bornn, L. (2020). SoccerMap. *ECML PKDD 2020*.
- Macdonald, B. (2011). Adjusted Plus-Minus for soccer.
- Pantuso, G. & Hvattum, L.M. (2024). Plus-Minus Models for Soccer Using Possession Sequences. *IMA Journal of Management Mathematics*.
- Wang, Z. et al. (2024). TacticAI. *Nature Communications*, 15:1906. https://www.nature.com/articles/s41467-024-45965-x
- GoalNet (2025). arXiv:2503.09737.
- Baron, S. et al. (2024). A Foundation Model for Soccer. arXiv:2407.14558.
- Mendes-Neves, T. et al. (2024). Towards a Foundation Large Events Model for Soccer. *Machine Learning*. https://link.springer.com/article/10.1007/s10994-024-06606-y
- Robberechts, P. et al. (2023). un-xPass: Measuring Soccer Player's Creativity. *KDD 2023*.
- Rahimian, P. et al. (2022). Beyond Action Valuation. *MIT Sloan Sports Analytics Conference*.
- Davis, J. et al. (2024). Methodology and Evaluation Challenges in Sports Analytics. *Machine Learning*. https://link.springer.com/article/10.1007/s10994-024-06585-0
- Sæbø, O.I. & Hvattum, L.M. (2019). Modelling the Financial Contribution of Soccer Players. *MLSA/ECML*.
- Tureen, T. & Olthof, S. (2022). Estimated Player Impact (EPI). *StatsBomb Conference*.
- Van Roy, M. et al. (2023). A Markov Framework for Learning and Reasoning About Strategies in Soccer. *JAIR*, 77.
