#!/usr/bin/env python3
"""
PlayeRank Analytics Dashboard

Run with:
    streamlit run dashboard.py

Requires the pipeline to have been run first:
    python run_pipeline.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
_DATA = _ROOT / "data"
# results/ is tracked by git and holds the committed sample dataset;
# data/ holds locally-generated pipeline output (gitignored).
_DASHBOARD_CSV = (
    _ROOT / "data" / "dashboard_data.csv"
    if (_ROOT / "data" / "dashboard_data.csv").exists()
    else _ROOT / "results" / "dashboard_data.csv"
)
_PLAYERS_JSON = _DATA / "players.json"
_ROLE_MATRIX_JSON = _ROOT / "playerank" / "conf" / "role_matrix.json"
_CONF = _ROOT / "playerank" / "conf"
_PERF_WEIGHTS_JSON = _CONF / "features_weights.json"
_WASTE_WEIGHTS_JSON = _CONF / "lack_of_performance_weights.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PlayeRank Dashboard",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not _DASHBOARD_CSV.exists():
        return None, None
    df = pd.read_csv(_DASHBOARD_CSV)

    # Load player names for display
    names = {}
    if _PLAYERS_JSON.exists():
        players = json.loads(_PLAYERS_JSON.read_text())
        for p in players:
            names[p["wyId"]] = p.get("shortName") or f"{p.get('firstName','')} {p.get('lastName','')}".strip()

    df["playerName"] = df["entity"].map(names).fillna(df["entity"].astype(str))
    return df, names


df, names = load_data()


@st.cache_data
def load_role_cluster_centers():
    """Compute cluster centroids from role_matrix.json (Wyscout 0–100 coords)."""
    if not _ROLE_MATRIX_JSON.exists():
        return None
    matrix = json.loads(_ROLE_MATRIX_JSON.read_text())
    from collections import defaultdict
    points = defaultdict(list)
    for x_str, y_dict in matrix.items():
        for y_str, label in y_dict.items():
            # label may be "3" or "3-5" (multi-cluster); use first
            primary = int(str(label).split("-")[0])
            points[primary].append((int(x_str), int(y_str)))
    centroids = {}
    for cluster_id, pts in points.items():
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        centroids[cluster_id] = (cx, cy)
    return centroids


@st.cache_data
def load_feature_weights():
    """Load performance and waste weights; return a merged DataFrame."""
    if not _PERF_WEIGHTS_JSON.exists() or not _WASTE_WEIGHTS_JSON.exists():
        return None
    perf = json.loads(_PERF_WEIGHTS_JSON.read_text())
    waste = json.loads(_WASTE_WEIGHTS_JSON.read_text())
    all_features = sorted(set(perf) | set(waste))
    rows = []
    for feat in all_features:
        parts = feat.split("-")
        category = parts[0].strip()
        subcategory = parts[1].strip() if len(parts) > 1 else ""
        outcome = parts[2].strip() if len(parts) > 2 else ""
        rows.append({
            "feature": feat,
            "category": category,
            "subcategory": subcategory,
            "outcome": outcome,
            "perfWeight": perf.get(feat, 0.0),
            "wasteWeight": waste.get(feat, 0.0),
        })
    return pd.DataFrame(rows)


def draw_soccer_field(cluster_centers):
    """Return a Plotly figure of a soccer field with cluster centroids.

    Display orientation: x-axis = Wyscout y (field width, 0–100),
    y-axis = Wyscout x (field depth, 0=defensive end, 100=attacking end).
    """
    # helper to add a white line/rect
    def line(x0, y0, x1, y1, w=1.5):
        return dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color="white", width=w))

    def rect(x0, y0, x1, y1, w=1.5, fill="rgba(0,0,0,0)"):
        return dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color="white", width=w), fillcolor=fill)

    def circle(cx, cy, r, w=1.5):
        return dict(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                    line=dict(color="white", width=w), fillcolor="rgba(0,0,0,0)")

    shapes = [
        # pitch outline
        rect(0, 0, 100, 100, w=2),
        # halfway line
        line(0, 50, 100, 50),
        # centre circle (radius ~9 in 0-100 space)
        circle(50, 50, 9),
        # penalty areas
        rect(21, 0, 79, 17),        # defensive end
        rect(21, 83, 79, 100),      # attacking end
        # goal areas
        rect(37, 0, 63, 6),
        rect(37, 94, 63, 100),
        # goals (outside field boundary)
        rect(43, -3, 57, 0, fill="#3a6e4c"),
        rect(43, 100, 57, 103, fill="#3a6e4c"),
        # penalty arcs (approximate semicircles as circles)
        circle(50, 11, 9),
        circle(50, 89, 9),
    ]

    fig = go.Figure()
    fig.update_layout(shapes=shapes)

    # pitch background
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
                  fillcolor="#4a7c59", line=dict(width=0), layer="below")

    # penalty spots & centre spot
    fig.add_trace(go.Scatter(
        x=[50, 50, 50], y=[50, 11, 89],
        mode="markers", marker=dict(color="white", size=4),
        showlegend=False, hoverinfo="skip",
    ))

    # cluster positions
    if cluster_centers:
        palette = px.colors.qualitative.Bold
        ids = sorted(cluster_centers)
        xs = [cluster_centers[cid][1] for cid in ids]   # Wyscout y → display x
        ys = [cluster_centers[cid][0] for cid in ids]   # Wyscout x → display y
        colors = [palette[int(cid) % len(palette)] for cid in ids]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(color=colors, size=20,
                        line=dict(color="white", width=1.5)),
            text=[str(cid) for cid in ids],
            textposition="middle center",
            textfont=dict(color="white", size=9, family="Arial Black"),
            customdata=[[f"({cluster_centers[cid][0]:.0f}, {cluster_centers[cid][1]:.0f})"]
                        for cid in ids],
            hovertemplate="Cluster %{text}<br>Avg position (x,y): %{customdata[0]}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-4, 104], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-6, 106], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1,
                   fixedrange=True),
        height=280,
    )
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚽ PlayeRank Analytics Dashboard")
st.markdown(
    "Explore **performance**, **waste**, and **net** scores across players and roles. "
    "Run `python run_pipeline.py` to generate the underlying data."
)

if df is None:
    st.error(
        "No dashboard data found. Run the full pipeline to generate real data:\n\n"
        "```\npython run_pipeline.py\n```\n\n"
        "Or place a `dashboard_data.csv` file in `results/` for a sample view."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Score column detection
# ---------------------------------------------------------------------------
SCORE_COLS = [c for c in ["playerankScore", "wasteScore", "netScore"] if c in df.columns]
SCORE_LABELS = {
    "playerankScore": "Performance Score",
    "wasteScore": "Waste Score",
    "netScore": "Net Score",
}

# Chain feature columns
CHAIN_PART_COLS = [c for c in [
    "chain-shot-participant", "chain-goal-participant", "chain-final-action",
] if c in df.columns]
CHAIN_PART_LABELS = {
    "chain-shot-participant": "Shot Chain Participations",
    "chain-goal-participant": "Goal Chain Participations",
    "chain-final-action":     "Final Actions Before Shot",
}
CHAIN_HARM_COLS = [c for c in [
    "chain-turnover-precedes-shot", "chain-turnover-precedes-goal",
    "chain-turnover-final-actor", "chain-conceded-shot", "chain-conceded-goal",
] if c in df.columns]
CHAIN_HARM_LABELS = {
    "chain-turnover-precedes-shot":  "Turnovers → Opp. Shot",
    "chain-turnover-precedes-goal":  "Turnovers → Opp. Goal",
    "chain-turnover-final-actor":    "Final Actor in Turnover Chain",
    "chain-conceded-shot":           "Defensive Actions in Conceded Shot Chain",
    "chain-conceded-goal":           "Defensive Actions in Conceded Goal Chain",
}
CHAIN_SCORE_COLS = [c for c in ["chainScore", "chainWasteScore", "chainNetScore"] if c in df.columns]
CHAIN_SCORE_LABELS = {
    "chainScore":      "Chain Performance Score",
    "chainWasteScore": "Chain Waste Score",
    "chainNetScore":   "Chain Net Score",
}
HAS_CHAIN_DATA = bool(CHAIN_PART_COLS or CHAIN_SCORE_COLS)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

roles = sorted(df["roleCluster"].dropna().unique()) if "roleCluster" in df.columns else []
selected_roles = st.sidebar.multiselect(
    "Role cluster", options=roles, default=roles,
    help="Filter by player role cluster"
)

min_matches = st.sidebar.slider(
    "Minimum matches played", min_value=1,
    max_value=int(df.groupby("entity")["match"].nunique().max()),
    value=5,
)

# ── Sidebar: Role cluster field reference ────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("Role Cluster Map")
st.sidebar.caption(
    "Approximate pitch positions of each role cluster "
    "(centre of performance, Wyscout coords). "
    "Attack direction: bottom → top."
)
_cluster_centers = load_role_cluster_centers()
if _cluster_centers:
    st.sidebar.plotly_chart(
        draw_soccer_field(_cluster_centers),
        use_container_width=True,
        config={"displayModeBar": False},
    )
else:
    st.sidebar.info(
        "Run the full pipeline to generate role cluster positions:\n\n"
        "```\npython run_pipeline.py\n```"
    )

# Apply filters
match_counts = df.groupby("entity")["match"].nunique()
qualified = match_counts[match_counts >= min_matches].index
fdf = df[df["entity"].isin(qualified)]
if selected_roles and "roleCluster" in fdf.columns:
    fdf = fdf[fdf["roleCluster"].isin(selected_roles)]

# Aggregate to player level (mean per match)
player_df = (
    fdf.groupby(["entity", "playerName"])[SCORE_COLS]
    .mean()
    .reset_index()
    .round(4)
)
if "roleCluster" in fdf.columns:
    role_map = fdf.groupby("entity")["roleCluster"].agg(lambda x: x.mode()[0])
    player_df["roleCluster"] = player_df["entity"].map(role_map)
player_df["matchesPlayed"] = player_df["entity"].map(match_counts)

# Percentile ranks (0–100) for each score column
for col in SCORE_COLS:
    player_df[f"{col}_pct"] = player_df[col].rank(pct=True).mul(100).round(1)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Players", len(player_df))
col2.metric("Avg Performance", f"{player_df['playerankScore'].mean():.4f}" if "playerankScore" in player_df.columns else "—")
col3.metric("Avg Waste", f"{player_df['wasteScore'].mean():.4f}" if "wasteScore" in player_df.columns else "—")
col4.metric("Avg Net Score", f"{player_df['netScore'].mean():.4f}" if "netScore" in player_df.columns else "—")
col5.metric("Avg Chain Net", f"{player_df['chainNetScore'].mean():.4f}" if "chainNetScore" in player_df.columns else "—")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Leaderboard", "📊 Score Analysis", "🎭 Role Breakdown",
    "⛓️ Chain Analytics", "👤 Player Profile", "⚙️ Feature Weights",
])

# ── Tab 1: Leaderboard ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Player Leaderboard")

    score_choice = st.radio(
        "Rank by", options=SCORE_COLS,
        format_func=lambda c: SCORE_LABELS.get(c, c),
        horizontal=True,
    )
    top_n = st.slider("Show top/bottom N", min_value=5, max_value=50, value=20)

    top = player_df.nlargest(top_n, score_choice)
    bottom = player_df.nsmallest(top_n, score_choice)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Top {top_n} — {SCORE_LABELS.get(score_choice, score_choice)}**")
        st.dataframe(
            top[["playerName", "roleCluster", "matchesPlayed"] + SCORE_COLS]
            .rename(columns={"playerName": "Player", "roleCluster": "Role",
                              "matchesPlayed": "Matches"}),
            use_container_width=True, hide_index=True,
        )
    with c2:
        st.markdown(f"**Bottom {top_n} — {SCORE_LABELS.get(score_choice, score_choice)}**")
        st.dataframe(
            bottom[["playerName", "roleCluster", "matchesPlayed"] + SCORE_COLS]
            .rename(columns={"playerName": "Player", "roleCluster": "Role",
                              "matchesPlayed": "Matches"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.markdown("**Full player table**")
    pct_col = f"{score_choice}_pct"
    display_cols = ["playerName", "roleCluster", "matchesPlayed"] + SCORE_COLS + (
        [pct_col] if pct_col in player_df.columns else []
    )
    full_table = (
        player_df[display_cols]
        .sort_values(score_choice, ascending=False)
        .rename(columns={"playerName": "Player", "roleCluster": "Role",
                          "matchesPlayed": "Matches", pct_col: "Percentile"})
    )
    st.dataframe(full_table, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download table as CSV",
        data=full_table.to_csv(index=False),
        file_name="playerank_leaderboard.csv",
        mime="text/csv",
    )

# ── Tab 2: Score Analysis ───────────────────────────────────────────────────
with tab2:
    st.subheader("Score Analysis")

    if "playerankScore" in player_df.columns and "wasteScore" in player_df.columns:
        st.markdown("#### Performance vs Waste")
        st.markdown(
            "Players in the **bottom-right** have high performance and low waste (ideal). "
            "Players in the **top-left** have low performance and high waste."
        )
        fig_scatter = px.scatter(
            player_df,
            x="playerankScore", y="wasteScore",
            color="netScore" if "netScore" in player_df.columns else None,
            size="matchesPlayed",
            hover_name="playerName",
            hover_data={"roleCluster": True, "matchesPlayed": True,
                        "playerankScore": ":.4f", "wasteScore": ":.4f",
                        "netScore": ":.4f" if "netScore" in player_df.columns else False},
            color_continuous_scale="RdYlGn",
            labels={
                "playerankScore": "Performance Score",
                "wasteScore": "Waste Score",
                "netScore": "Net Score",
                "matchesPlayed": "Matches Played",
            },
            title="Performance Score vs Waste Score (size = matches played, colour = net score)",
        )
        # Quadrant reference lines at median values
        x_mid = player_df["playerankScore"].median()
        y_mid = player_df["wasteScore"].median()
        fig_scatter.add_vline(x=x_mid, line_dash="dot", line_color="grey", opacity=0.4)
        fig_scatter.add_hline(y=y_mid, line_dash="dot", line_color="grey", opacity=0.4)
        x_range = player_df["playerankScore"].agg(["min", "max"])
        y_range = player_df["wasteScore"].agg(["min", "max"])
        label_style = dict(xref="x", yref="y", showarrow=False,
                           font=dict(size=11, color="rgba(200,200,200,0.6)"))
        fig_scatter.add_annotation(x=x_range["min"], y=y_range["min"],
            text="Low performance, low waste", xanchor="left", yanchor="bottom", **label_style)
        fig_scatter.add_annotation(x=x_range["min"], y=y_range["max"],
            text="Low performance, high waste", xanchor="left", yanchor="top", **label_style)
        fig_scatter.add_annotation(x=x_range["max"], y=y_range["min"],
            text="High performance, low waste (ideal)", xanchor="right", yanchor="bottom", **label_style)
        fig_scatter.add_annotation(x=x_range["max"], y=y_range["max"],
            text="High performance, high waste", xanchor="right", yanchor="top", **label_style)
        fig_scatter.update_layout(height=550)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### Score Distributions")
    dist_score = st.selectbox(
        "Score to plot", SCORE_COLS,
        format_func=lambda c: SCORE_LABELS.get(c, c),
    )
    fig_hist = px.histogram(
        player_df, x=dist_score, nbins=50,
        color="roleCluster" if "roleCluster" in player_df.columns else None,
        labels={dist_score: SCORE_LABELS.get(dist_score, dist_score)},
        title=f"Distribution of {SCORE_LABELS.get(dist_score, dist_score)}",
        barmode="overlay", opacity=0.7,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Tab 3: Role Breakdown ───────────────────────────────────────────────────
with tab3:
    st.subheader("Scores by Role Cluster")

    if "roleCluster" not in player_df.columns:
        st.info("Role cluster data not available.")
    else:
        fig_box = go.Figure()
        colours = px.colors.qualitative.Set2
        for i, score_col in enumerate(SCORE_COLS):
            fig_box.add_trace(go.Box(
                x=player_df["roleCluster"].astype(str),
                y=player_df[score_col],
                name=SCORE_LABELS.get(score_col, score_col),
                marker_color=colours[i % len(colours)],
            ))
        fig_box.update_layout(
            boxmode="group",
            xaxis_title="Role Cluster",
            yaxis_title="Score",
            title="Score Distribution by Role Cluster",
            height=500,
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("#### Mean scores per role")
        role_summary = (
            player_df.groupby("roleCluster")[SCORE_COLS]
            .mean().round(4).reset_index()
            .rename(columns={"roleCluster": "Role"})
        )
        st.dataframe(role_summary, use_container_width=True, hide_index=True)

# ── Tab 4: Chain Analytics ──────────────────────────────────────────────────
with tab4:
    st.subheader("Chain Analytics")
    st.markdown(
        "Evaluate players based on their participation in **possession chains** — "
        "multi-event sequences leading to shots, goals, or dangerous turnovers. "
        "Run the full pipeline to populate chain columns in `dashboard_data.csv`."
    )

    if not HAS_CHAIN_DATA:
        st.info(
            "No chain data found in `dashboard_data.csv`. "
            "Chain features will be available after the chain context model is integrated "
            "into `compute_playerank.py` and the pipeline is re-run:\n\n"
            "```\npython run_pipeline.py\n```"
        )
    else:
        # ── Section 1: Chain score summary ───────────────────────────────────
        if CHAIN_SCORE_COLS:
            st.markdown("### Chain Score Summary")
            cs_cols = st.columns(len(CHAIN_SCORE_COLS))
            for col, sc in zip(cs_cols, CHAIN_SCORE_COLS):
                col.metric(
                    CHAIN_SCORE_LABELS.get(sc, sc),
                    f"{player_df[sc].mean():.4f}" if sc in player_df.columns else "—",
                )

            if "chainScore" in player_df.columns and "chainWasteScore" in player_df.columns:
                st.markdown("#### Chain Performance vs Chain Waste")
                st.markdown(
                    "Players in the **bottom-right** generate dangerous chains while rarely "
                    "appearing in harmful turnover sequences (ideal). "
                    "Players in the **top-left** are a net negative in sequential play."
                )
                fig_cs = px.scatter(
                    player_df,
                    x="chainScore", y="chainWasteScore",
                    color="chainNetScore" if "chainNetScore" in player_df.columns else None,
                    size="matchesPlayed",
                    hover_name="playerName",
                    hover_data={
                        "roleCluster": True, "matchesPlayed": True,
                        "chainScore": ":.4f", "chainWasteScore": ":.4f",
                        "chainNetScore": ":.4f" if "chainNetScore" in player_df.columns else False,
                    },
                    color_continuous_scale="RdYlGn",
                    labels={
                        "chainScore": "Chain Performance Score",
                        "chainWasteScore": "Chain Waste Score",
                        "chainNetScore": "Chain Net Score",
                        "matchesPlayed": "Matches Played",
                    },
                    title="Chain Score vs Chain Waste (size = matches played, colour = chain net score)",
                )
                cx_mid = player_df["chainScore"].median()
                cy_mid = player_df["chainWasteScore"].median()
                fig_cs.add_vline(x=cx_mid, line_dash="dot", line_color="grey", opacity=0.4)
                fig_cs.add_hline(y=cy_mid, line_dash="dot", line_color="grey", opacity=0.4)
                fig_cs.update_layout(height=500)
                st.plotly_chart(fig_cs, use_container_width=True)

        st.divider()

        # ── Section 2: Chain participation leaderboard ────────────────────────
        if CHAIN_PART_COLS:
            st.markdown("### Chain Participation")
            st.markdown(
                "Average chain participations **per match** for each player. "
                "`Goal Chain Participations` counts chains ending in a goal the player "
                "touched the ball in. `Final Actions Before Shot` is the key-pass analog."
            )

            sort_col = "chain-goal-participant" if "chain-goal-participant" in player_df.columns \
                else CHAIN_PART_COLS[0]
            top_n_chain = st.slider("Show top N players", 5, 50, 20, key="chain_top_n")
            chain_part_df = (
                player_df[["playerName", "roleCluster", "matchesPlayed"] + CHAIN_PART_COLS]
                .nlargest(top_n_chain, sort_col)
                .rename(columns={**{"playerName": "Player", "roleCluster": "Role",
                                    "matchesPlayed": "Matches"}, **CHAIN_PART_LABELS})
            )
            st.dataframe(chain_part_df, use_container_width=True, hide_index=True)

            # Scatter: shot-chain vs goal-chain participation
            if "chain-shot-participant" in player_df.columns and "chain-goal-participant" in player_df.columns:
                st.markdown("#### Shot Chain vs Goal Chain Participation")
                st.markdown(
                    "Players to the **right** reach the shot phase often. "
                    "Players **higher up** appear in chains that actually result in goals — "
                    "a marker of clinical build-up involvement."
                )
                fig_part = px.scatter(
                    player_df,
                    x="chain-shot-participant", y="chain-goal-participant",
                    color="roleCluster" if "roleCluster" in player_df.columns else None,
                    size="matchesPlayed",
                    hover_name="playerName",
                    hover_data={
                        "roleCluster": True, "matchesPlayed": True,
                        "chain-shot-participant": ":.2f",
                        "chain-goal-participant": ":.2f",
                        "chain-final-action": ":.2f" if "chain-final-action" in player_df.columns else False,
                    },
                    labels={
                        "chain-shot-participant": "Shot Chain Participations / match",
                        "chain-goal-participant": "Goal Chain Participations / match",
                        "roleCluster": "Role",
                    },
                    title="Shot chain vs goal chain participation rate (size = matches played)",
                )
                fig_part.update_layout(height=480)
                st.plotly_chart(fig_part, use_container_width=True)

        st.divider()

        # ── Section 3: Harmful chain leaderboard ─────────────────────────────
        if CHAIN_HARM_COLS:
            st.markdown("### Harmful Chain Involvement")
            st.markdown(
                "Players ranked by average involvement in chains that **directly preceded "
                "a conceded goal** (`Turnovers → Opp. Goal`) or in chains where the "
                "opponent scored (`Defensive Actions in Conceded Goal Chain`). "
                "Lower values are better."
            )

            harm_sort = "chain-turnover-precedes-goal" if "chain-turnover-precedes-goal" in player_df.columns \
                else CHAIN_HARM_COLS[0]
            top_n_harm = st.slider("Show top N players", 5, 50, 20, key="harm_top_n")
            harm_df = (
                player_df[["playerName", "roleCluster", "matchesPlayed"] + CHAIN_HARM_COLS]
                .nlargest(top_n_harm, harm_sort)
                .rename(columns={**{"playerName": "Player", "roleCluster": "Role",
                                    "matchesPlayed": "Matches"}, **CHAIN_HARM_LABELS})
            )
            st.dataframe(harm_df, use_container_width=True, hide_index=True)

            # Scatter: turnover-precedes-shot vs conceded-goal
            if "chain-turnover-precedes-shot" in player_df.columns and \
               "chain-conceded-goal" in player_df.columns:
                st.markdown("#### Turnover Chains vs Conceded Goal Chains")
                st.markdown(
                    "Players in the **top-right** are involved in both types of harmful chain — "
                    "high turnover rate *and* frequently present when opponents score. "
                    "Players near the **origin** are rarely involved in damaging sequences."
                )
                fig_harm = px.scatter(
                    player_df,
                    x="chain-turnover-precedes-shot", y="chain-conceded-goal",
                    color="roleCluster" if "roleCluster" in player_df.columns else None,
                    size="matchesPlayed",
                    hover_name="playerName",
                    hover_data={
                        "roleCluster": True, "matchesPlayed": True,
                        "chain-turnover-precedes-shot": ":.2f",
                        "chain-turnover-precedes-goal": ":.2f" if "chain-turnover-precedes-goal" in player_df.columns else False,
                        "chain-conceded-goal": ":.2f",
                    },
                    labels={
                        "chain-turnover-precedes-shot": "Turnover Chains → Opp. Shot / match",
                        "chain-conceded-goal": "Defensive Actions in Conceded Goal Chain / match",
                        "roleCluster": "Role",
                    },
                    title="Turnover chain rate vs conceded goal chain involvement",
                )
                fig_harm.update_layout(height=480)
                st.plotly_chart(fig_harm, use_container_width=True)

        st.divider()

        # ── Section 4: Chain pitch origin map ────────────────────────────────
        st.markdown("### Chain Origin Map")
        _CHAIN_MAP_JSON = _DATA / "chain_map.json"
        if _CHAIN_MAP_JSON.exists():
            @st.cache_data
            def load_chain_map():
                return json.loads(_CHAIN_MAP_JSON.read_text())

            chain_map = load_chain_map()
            outcome_filter = st.selectbox(
                "Show chains ending in", ["goal", "shot"],
                key="chain_map_outcome",
            )
            filtered_chains = [c for c in chain_map if c.get("outcome") == outcome_filter]
            if filtered_chains:
                fig_map = draw_soccer_field(None)
                xs = [c["start_y"] for c in filtered_chains]   # Wyscout y → display x
                ys = [c["start_x"] for c in filtered_chains]   # Wyscout x → display y
                fig_map.add_trace(go.Histogram2dContour(
                    x=xs, y=ys,
                    colorscale="YlOrRd",
                    reversescale=False,
                    showscale=True,
                    opacity=0.7,
                    name=f"{outcome_filter.title()} chain origins",
                    hoverinfo="skip",
                ))
                fig_map.update_layout(
                    title=f"Origin positions of chains ending in a {outcome_filter}",
                    height=420,
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info(f"No chains with outcome '{outcome_filter}' found in chain_map.json.")
        else:
            st.info(
                "Chain origin map requires `data/chain_map.json` — a precomputed file of "
                "chain start positions generated by the pipeline. Re-run the pipeline once "
                "chain features are integrated to generate this file."
            )

# ── Tab 5: Player Profile ───────────────────────────────────────────────────
with tab5:
    st.subheader("Player Profile")

    player_options = (
        player_df.sort_values("netScore" if "netScore" in player_df.columns else "playerankScore",
                              ascending=False)
        [["playerName", "entity"]].drop_duplicates()
    )
    player_display = dict(zip(player_options["entity"], player_options["playerName"]))

    selected_entity = st.selectbox(
        "Select player",
        options=player_options["entity"].tolist(),
        format_func=lambda e: player_display.get(e, str(e)),
    )

    player_matches = (
        fdf[fdf["entity"] == selected_entity]
        .sort_values("timestamp" if "timestamp" in fdf.columns else "match")
        .reset_index(drop=True)
    )

    if player_matches.empty:
        st.warning("No match data available for this player under current filters.")
    else:
        name = player_display.get(selected_entity, str(selected_entity))
        role = player_matches["roleCluster"].iloc[0] if "roleCluster" in player_matches.columns else "—"
        n_matches = player_matches["match"].nunique()

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Player", name)
        mc2.metric("Role cluster", str(role))
        mc3.metric("Matches", n_matches)
        if "minutesPlayed" in player_matches.columns:
            mc4.metric("Avg minutes", f"{player_matches['minutesPlayed'].mean():.0f}")

        # Match-by-match timeline
        st.markdown("#### Match-by-match scores")
        available_scores = [c for c in SCORE_COLS if c in player_matches.columns]
        if available_scores:
            timeline_df = player_matches[["match"] + available_scores].copy()
            timeline_df["match"] = timeline_df["match"].astype(str)

            roll_window = st.slider(
                "Rolling average window (matches)", min_value=1,
                max_value=max(3, len(timeline_df) // 2), value=min(5, len(timeline_df)),
                help="Set to 1 to show raw match-by-match values",
            )

            fig_line = go.Figure()
            line_colours = {"playerankScore": "#2ecc71", "wasteScore": "#e74c3c", "netScore": "#3498db"}
            xs = list(range(len(timeline_df)))
            for score_col in available_scores:
                colour = line_colours.get(score_col)
                raw_y = timeline_df[score_col].tolist()
                # Raw values as faint markers
                fig_line.add_trace(go.Scatter(
                    x=xs, y=raw_y,
                    mode="markers",
                    name=f"{SCORE_LABELS.get(score_col, score_col)} (raw)",
                    marker=dict(color=colour, opacity=0.35, size=6),
                    showlegend=False,
                    hovertemplate=f"Match %{{x}}<br>{SCORE_LABELS.get(score_col, score_col)}: %{{y:.4f}}<extra></extra>",
                ))
                # Rolling average as solid line
                rolled = timeline_df[score_col].rolling(roll_window, min_periods=1).mean()
                fig_line.add_trace(go.Scatter(
                    x=xs, y=rolled.tolist(),
                    mode="lines",
                    name=SCORE_LABELS.get(score_col, score_col),
                    line=dict(color=colour, width=2),
                    hovertemplate=f"Match %{{x}}<br>{SCORE_LABELS.get(score_col, score_col)} ({roll_window}-match avg): %{{y:.4f}}<extra></extra>",
                ))
            fig_line.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig_line.update_layout(
                xaxis_title="Match (chronological)",
                yaxis_title="Score",
                title=f"{name} — score timeline",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # Summary bar chart
        st.markdown("#### Average scores")
        avg_scores = {SCORE_LABELS.get(c, c): player_matches[c].mean() for c in available_scores}
        fig_bar = go.Figure(go.Bar(
            x=list(avg_scores.keys()),
            y=list(avg_scores.values()),
            marker_color=["#2ecc71", "#e74c3c", "#3498db"][:len(avg_scores)],
        ))
        fig_bar.update_layout(
            yaxis_title="Average score",
            title=f"{name} — average scores across {n_matches} matches",
            height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Chain participation timeline
        player_chain_cols = [c for c in CHAIN_PART_COLS if c in player_matches.columns]
        player_harm_cols  = [c for c in CHAIN_HARM_COLS  if c in player_matches.columns]
        if player_chain_cols or player_harm_cols:
            st.markdown("#### Chain participation per match")
            fig_chain_timeline = go.Figure()
            chain_colours = {
                "chain-shot-participant":        "#3498db",
                "chain-goal-participant":        "#2ecc71",
                "chain-final-action":            "#f39c12",
                "chain-turnover-precedes-shot":  "#e67e22",
                "chain-turnover-precedes-goal":  "#e74c3c",
                "chain-turnover-final-actor":    "#c0392b",
                "chain-conceded-shot":           "#9b59b6",
                "chain-conceded-goal":           "#6c3483",
            }
            xs_chain = list(range(len(player_matches)))
            all_chain_cols = player_chain_cols + player_harm_cols
            for ccol in all_chain_cols:
                label = {**CHAIN_PART_LABELS, **CHAIN_HARM_LABELS}.get(ccol, ccol)
                colour = chain_colours.get(ccol, "#95a5a6")
                is_harm = ccol in player_harm_cols
                fig_chain_timeline.add_trace(go.Scatter(
                    x=xs_chain,
                    y=player_matches[ccol].tolist(),
                    mode="lines+markers",
                    name=label,
                    line=dict(color=colour, width=2,
                              dash="dot" if is_harm else "solid"),
                    marker=dict(color=colour, size=5),
                    fill="tozeroy",
                    fillcolor=colour.replace(")", ", 0.08)").replace("rgb", "rgba")
                    if colour.startswith("rgb") else colour + "14",
                    hovertemplate=f"Match %{{x}}<br>{label}: %{{y:.1f}}<extra></extra>",
                ))
            fig_chain_timeline.update_layout(
                xaxis_title="Match (chronological)",
                yaxis_title="Chain participations",
                title=f"{name} — chain participation timeline "
                      "(solid = positive, dashed = harmful)",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_chain_timeline, use_container_width=True)

        # Raw match data
        with st.expander("Raw match data"):
            st.dataframe(player_matches, use_container_width=True, hide_index=True)

# ── Tab 6: Feature Weights ───────────────────────────────────────────────────
with tab6:
    st.subheader("Feature Weights")
    st.markdown(
        "How much each event type and sub-type drives the **Performance** "
        "(win-associated) and **Waste** (loss-associated) scores. "
        "Weights are LinearSVC coefficients normalised so |weights| sum to 1."
    )

    fw = load_feature_weights()

    if fw is None:
        st.info(
            "Weight files not found. Run the pipeline to generate them:\n\n"
            "```\npython run_pipeline.py\n```"
        )
    else:
        # ── Section 1: Top features ──────────────────────────────────────────
        st.markdown("### Top features by weight")
        top_n = st.slider("Number of features to show", 5, len(fw), 15, key="fw_topn")

        col_p, col_w = st.columns(2)

        # Performance
        with col_p:
            st.markdown("#### Performance score")
            st.caption("Positive = win-associated · Negative = loss-associated")
            top_perf = fw.nlargest(top_n // 2, "perfWeight").assign(group="positive").pipe(
                lambda d: pd.concat([d, fw.nsmallest(top_n - top_n // 2, "perfWeight").assign(group="negative")])
            ).sort_values("perfWeight")
            fig_p = px.bar(
                top_perf, x="perfWeight", y="feature", orientation="h",
                color="group",
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"},
                labels={"perfWeight": "Weight", "feature": ""},
                height=max(350, top_n * 22),
            )
            fig_p.update_layout(showlegend=False, margin=dict(l=0, r=10, t=10, b=30))
            fig_p.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
            st.plotly_chart(fig_p, use_container_width=True)

        # Waste
        with col_w:
            st.markdown("#### Waste score")
            st.caption("Positive = loss-associated · Negative = not-loss-associated")
            top_waste = fw.nlargest(top_n // 2, "wasteWeight").assign(group="positive").pipe(
                lambda d: pd.concat([d, fw.nsmallest(top_n - top_n // 2, "wasteWeight").assign(group="negative")])
            ).sort_values("wasteWeight")
            fig_w = px.bar(
                top_waste, x="wasteWeight", y="feature", orientation="h",
                color="group",
                color_discrete_map={"positive": "#e74c3c", "negative": "#2ecc71"},
                labels={"wasteWeight": "Weight", "feature": ""},
                height=max(350, top_n * 22),
            )
            fig_w.update_layout(showlegend=False, margin=dict(l=0, r=10, t=10, b=30))
            fig_w.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
            st.plotly_chart(fig_w, use_container_width=True)

        # ── Section 2: By event category ─────────────────────────────────────
        st.markdown("### By event category")
        st.caption(
            "Sum of positive weights per category — how much each event type "
            "contributes to winning (performance) vs losing (waste)."
        )
        cat_df = (
            fw.groupby("category")[["perfWeight", "wasteWeight"]]
            .apply(lambda g: pd.Series({
                "Performance (win-assoc.)": g.loc[g["perfWeight"] > 0, "perfWeight"].sum(),
                "Waste (loss-assoc.)": g.loc[g["wasteWeight"] > 0, "wasteWeight"].sum(),
            }))
            .reset_index()
            .melt(id_vars="category", var_name="model", value_name="totalWeight")
        )
        fig_cat = px.bar(
            cat_df, x="category", y="totalWeight", color="model", barmode="group",
            color_discrete_map={
                "Performance (win-assoc.)": "#2ecc71",
                "Waste (loss-assoc.)": "#e74c3c",
            },
            labels={"totalWeight": "Sum of positive weights", "category": "Event category", "model": ""},
            height=380,
        )
        fig_cat.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_cat, use_container_width=True)

        # ── Section 3: Performance vs Waste comparison scatter ────────────────
        st.markdown("### Performance vs Waste weight comparison")
        st.caption(
            "Each dot is one feature. Features in the **top-left** are strongly "
            "loss-associated (high waste) but not win-associated. "
            "Features in the **bottom-right** drive performance but not waste."
        )
        fig_cmp = px.scatter(
            fw, x="perfWeight", y="wasteWeight",
            color="category", hover_name="feature",
            hover_data={"perfWeight": ":.4f", "wasteWeight": ":.4f", "category": False},
            labels={"perfWeight": "Performance weight", "wasteWeight": "Waste weight"},
            height=450,
        )
        fig_cmp.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.4)
        fig_cmp.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.4)
        fig_cmp.update_layout(legend_title_text="Event category")
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Raw weights table ─────────────────────────────────────────────────
        with st.expander("Full weights table"):
            st.dataframe(
                fw[["feature", "category", "subcategory", "outcome", "perfWeight", "wasteWeight"]]
                .sort_values("perfWeight", ascending=False)
                .reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
