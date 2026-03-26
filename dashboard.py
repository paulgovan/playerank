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
col1, col2, col3, col4 = st.columns(4)
col1.metric("Players", len(player_df))
col2.metric("Avg Performance", f"{player_df['playerankScore'].mean():.4f}" if "playerankScore" in player_df else "—")
col3.metric("Avg Waste", f"{player_df['wasteScore'].mean():.4f}" if "wasteScore" in player_df else "—")
col4.metric("Avg Net Score", f"{player_df['netScore'].mean():.4f}" if "netScore" in player_df else "—")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Leaderboard", "📊 Score Analysis", "🎭 Role Breakdown", "👤 Player Profile"])

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

# ── Tab 4: Player Profile ───────────────────────────────────────────────────
with tab4:
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

        # Raw match data
        with st.expander("Raw match data"):
            st.dataframe(player_matches, use_container_width=True, hide_index=True)
