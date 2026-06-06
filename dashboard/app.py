import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from scipy import stats
from ingestion.loader import get_connection

app = Dash(__name__)

BG_OBSIDIAN = "#0B0F14"
BG_CARD = "#0D1117"
BORDER = "#1E2530"
AMBER = "#FFB000"
TEXT_PRIMARY = "#F0F0F0"
TEXT_SECONDARY = "#888888"
AXIS_TEXT = "#666666"
ZERO_LINE = "#333333"

PLOT_LAYOUT = dict(
    plot_bgcolor=BG_OBSIDIAN,
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", color=TEXT_PRIMARY, size=12),
    hoverlabel=dict(
        bgcolor=BG_CARD,
        bordercolor=AMBER,
        font=dict(family="JetBrains Mono", color=TEXT_PRIMARY, size=11),
    ),
    legend=dict(
        orientation="h",
        y=-0.18,
        font=dict(family="JetBrains Mono", size=10, color=TEXT_SECONDARY),
        bgcolor="rgba(0,0,0,0)",
    ),
)

AXIS_DEFAULTS = dict(
    gridcolor="rgba(30, 37, 48, 0.4)",
    gridwidth=0.5,
    tickfont=dict(family="JetBrains Mono", size=10, color=AXIS_TEXT),
    linecolor=BORDER,
    zerolinecolor=ZERO_LINE,
    zerolinewidth=1,
)

def axis(title_text: str, **extra) -> dict:
    return dict(
        title=dict(
            text=title_text,
            font=dict(family="IBM Plex Sans", size=11, color=TEXT_SECONDARY),
        ),
        **AXIS_DEFAULTS,
        **extra,
    )
TS_OPACITIES = [1.0, 0.7, 0.5, 0.35, 0.25]


def load_data() -> pd.DataFrame:
    conn = get_connection()
    query = """
        SELECT
            c.ticker,
            f.filed_at,
            lf.hedging_score,
            lf.uncertainty_score,
            lf.specificity_score,
            lf.fog_index,
            lf.finbert_positive,
            lf.finbert_negative,
            lf.finbert_neutral,
            ps.return_30d,
            ps.return_60d,
            ps.return_90d,
            t.word_count
        FROM linguistic_features lf
        JOIN transcripts t ON t.id = lf.transcript_id
        JOIN filings f ON f.id = t.filing_id
        JOIN companies c ON c.id = f.company_id
        JOIN price_snapshots ps
            ON ps.company_id = c.id
            AND ps.filing_date = f.filed_at
        WHERE ps.return_30d IS NOT NULL
        AND lf.hedging_score < 30
        AND lf.fog_index < 35
        AND t.word_count > 200
        ORDER BY c.ticker, f.filed_at
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    df["return_30d_pct"] = (df["return_30d"] * 100).round(2)
    df["return_60d_pct"] = (df["return_60d"] * 100).round(2)
    df["return_90d_pct"] = (df["return_90d"] * 100).round(2)
    return df


def compute_findings(data: pd.DataFrame) -> list[dict]:
    horizons = {
        "return_30d_pct": "30D",
        "return_60d_pct": "60D",
        "return_90d_pct": "90D",
    }
    rows = []
    for return_col, horizon in horizons.items():
        for feature, label in FEATURES.items():
            clean = data[[feature, return_col]].dropna()
            if len(clean) <= 3:
                continue
            corr, pvalue = stats.spearmanr(clean[feature], clean[return_col])
            if pvalue < 0.05:
                rows.append({
                    "feature": label,
                    "horizon": horizon,
                    "r": corr,
                    "p": pvalue,
                    "n": len(clean),
                })
    return rows


def build_ticker_tape(data: pd.DataFrame) -> list[str]:
    latest = data.sort_values("filed_at").groupby("ticker").last()
    return [
        f"{ticker}  {latest.loc[ticker, 'hedging_score']:.2f}"
        for ticker in sorted(latest.index)
    ]


def build_findings_table(findings: list[dict]) -> html.Div:
    if not findings:
        return html.Div(
            "No significant correlations found (p < 0.05).",
            className="findings-empty",
        )
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Feature"),
            html.Th("Horizon"),
            html.Th("Spearman r"),
            html.Th("P-Value"),
            html.Th("N"),
        ])),
        html.Tbody([
            html.Tr([
                html.Td(row["feature"]),
                html.Td(row["horizon"]),
                html.Td(f"{row['r']:.4f}"),
                html.Td(f"{row['p']:.4f}"),
                html.Td(str(row["n"])),
            ])
            for row in findings
        ]),
    ], className="findings-table")


def build_metric_card(label: str, value: str, value_class: str = "") -> html.Div:
    classes = "metric-value"
    if value_class:
        classes += f" {value_class}"
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className=classes),
    ], className="metric-card")


df = load_data()

FEATURES = {
    "hedging_score": "Hedging Score",
    "uncertainty_score": "Uncertainty Score",
    "specificity_score": "Specificity Score",
    "fog_index": "Gunning Fog Index",
    "finbert_positive": "FinBERT Positive",
    "finbert_negative": "FinBERT Negative",
}

TICKERS = sorted(df["ticker"].unique())
FINDINGS = compute_findings(df)
TAPE_ITEMS = build_ticker_tape(df)

tape_track = []
for item in TAPE_ITEMS * 2:
    tape_track.append(html.Span(item, className="ticker-item"))
    tape_track.append(html.Span("·", className="ticker-separator"))

app.layout = html.Div([
    html.Div(
        html.Div(tape_track, className="ticker-tape-track"),
        className="ticker-tape-wrap",
    ),

    html.Div([
        html.Div([
            html.H1("Earnings Sentiment Drift", className="main-title"),
            html.P(
                "Linguistic feature analysis of SEC 8-K filings vs subsequent stock returns",
                className="subtitle",
            ),
        ], className="title-block"),

        html.Div("<Methodology />", className="section-label"),
        html.Div([
            html.Div([
                html.Label("Company", className="control-label"),
                dcc.Checklist(
                    id="ticker-filter",
                    options=[{"label": f"  {t}", "value": t} for t in TICKERS],
                    value=TICKERS,
                    inline=True,
                ),
            ], className="control-group control-group--wide"),

            html.Div([
                html.Div([
                    html.Label("Linguistic Feature", className="control-label"),
                    dcc.Dropdown(
                        id="feature-select",
                        options=[{"label": v, "value": k} for k, v in FEATURES.items()],
                        value="hedging_score",
                        clearable=False,
                        className="feature-dropdown",
                    ),
                ], className="control-group control-group--dropdown"),

                html.Div([
                    html.Label("Return Horizon", className="control-label"),
                    dcc.RadioItems(
                        id="return-horizon",
                        options=[
                            {"label": " 30 day", "value": "return_30d_pct"},
                            {"label": " 60 day", "value": "return_60d_pct"},
                            {"label": " 90 day", "value": "return_90d_pct"},
                        ],
                        value="return_30d_pct",
                        inline=True,
                    ),
                ], className="control-group control-group--horizon"),
            ], className="controls-row"),
        ], className="controls-panel"),

        html.Div(id="metric-cards", className="metric-cards-row"),

        html.Hr(className="section-divider"),

        html.Div("<Results />", className="section-label"),
        html.Div([
            html.Div([
                html.Div(className="scatter-watermark"),
                dcc.Graph(id="scatter-plot", style={"height": "400px"}),
            ], className="chart-panel scatter-panel"),

            html.Div([
                dcc.Graph(id="time-series", style={"height": "400px"}),
            ], className="chart-panel"),
        ], className="charts-row"),

        html.Hr(className="section-divider"),

        html.Div("<Findings />", className="section-label"),
        html.Div(build_findings_table(FINDINGS), className="findings-panel"),

    ], className="dashboard-main"),
], className="dashboard-root")


@app.callback(
    Output("scatter-plot", "figure"),
    Output("time-series", "figure"),
    Output("metric-cards", "children"),
    Input("ticker-filter", "value"),
    Input("feature-select", "value"),
    Input("return-horizon", "value"),
)
def update_charts(tickers, feature, return_col):
    filtered = df[df["ticker"].isin(tickers)].copy()

    if filtered.empty:
        empty = go.Figure()
        empty.update_layout(**PLOT_LAYOUT)
        empty_cards = [
            build_metric_card("Spearman r", "—"),
            build_metric_card("P-Value", "—"),
            build_metric_card("Win Rate", "—"),
            build_metric_card("N", "0"),
        ]
        return empty, empty, empty_cards

    feature_label = FEATURES[feature]
    return_label = return_col.replace("_pct", "").replace("return_", "") \
                             .replace("d", " day return %")

    scatter = go.Figure()
    for ticker in tickers:
        t_df = filtered[filtered["ticker"] == ticker]
        scatter.add_trace(go.Scatter(
            x=t_df[feature],
            y=t_df[return_col],
            mode="markers",
            name=ticker,
            marker=dict(
                color=AMBER,
                size=8,
                opacity=0.85,
                line=dict(width=1, color="white"),
            ),
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                f"Date: %{{customdata}}<br>"
                f"{feature_label}: %{{x:.2f}}<br>"
                f"Return: %{{y:.1f}}%<extra></extra>"
            ),
            customdata=t_df["filed_at"].dt.strftime("%Y-%m-%d"),
        ))

    clean = filtered[[feature, return_col]].dropna()
    if len(clean) > 3:
        m, b, r, p, _ = stats.linregress(clean[feature], clean[return_col])
        x_range = [clean[feature].min(), clean[feature].max()]
        y_range = [m * x + b for x in x_range]
        scatter.add_trace(go.Scatter(
            x=x_range, y=y_range,
            mode="lines",
            name=f"Trend (r={r:.2f})",
            line=dict(color="rgba(255, 176, 0, 0.5)", width=1, dash="dash"),
            showlegend=True,
        ))

    scatter.update_layout(
        title=dict(
            text=f"{feature_label} vs {return_label.title()}",
            font=dict(family="IBM Plex Sans", size=13, color=TEXT_PRIMARY),
        ),
        xaxis=axis(feature_label),
        yaxis=axis("Return (%)", tickformat=".1f", ticksuffix="%", zeroline=True),
        margin=dict(l=56, r=24, t=52, b=64),
        hovermode="closest",
        **PLOT_LAYOUT,
    )

    time_fig = go.Figure()
    for i, ticker in enumerate(tickers):
        opacity = TS_OPACITIES[min(i, len(TS_OPACITIES) - 1)]
        t_df = filtered[filtered["ticker"] == ticker].sort_values("filed_at")
        time_fig.add_trace(go.Scatter(
            x=t_df["filed_at"],
            y=t_df[feature],
            mode="lines+markers",
            name=ticker,
            line=dict(color=f"rgba(255, 176, 0, {opacity})", width=1.5),
            marker=dict(size=5, color=f"rgba(255, 176, 0, {opacity})"),
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                f"Date: %{{x|%Y-%m-%d}}<br>"
                f"{feature_label}: %{{y:.2f}}<extra></extra>"
            ),
        ))

    time_fig.update_layout(
        title=dict(
            text=f"{feature_label} Over Time",
            font=dict(family="IBM Plex Sans", size=13, color=TEXT_PRIMARY),
        ),
        xaxis=dict(title="Filing Date", **AXIS_DEFAULTS),
        yaxis=dict(title=feature_label, **AXIS_DEFAULTS),
        margin=dict(l=56, r=24, t=52, b=64),
        hovermode="x unified",
        **PLOT_LAYOUT,
    )

    corr, pvalue = stats.spearmanr(
        clean[feature], clean[return_col]
    ) if len(clean) > 3 else (0, 1)

    median = clean[feature].median() if len(clean) > 0 else 0
    correct = (
        ((clean[feature] > median) & (clean[return_col] > 0))
        | ((clean[feature] <= median) & (clean[return_col] <= 0))
    ).sum()
    win_rate = (correct / len(clean) * 100) if len(clean) > 0 else 0

    pvalue_class = "metric-value--amber" if pvalue < 0.05 else "metric-value--red"

    metric_cards = [
        build_metric_card("Spearman r", f"{corr:.4f}"),
        build_metric_card("P-Value", f"{pvalue:.4f}", pvalue_class),
        build_metric_card("Win Rate", f"{win_rate:.1f}%"),
        build_metric_card("N", str(len(clean))),
    ]

    return scatter, time_fig, metric_cards


if __name__ == "__main__":
    app.run(debug=True)