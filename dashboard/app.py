import os
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

CHART_COLORS = [
    "#FFB000",  # amber
    "#FF6B6B",  # coral red
    "#4ECDC4",  # teal
    "#A78BFA",  # purple
    "#34D399",  # emerald
    "#F59E0B",  # yellow amber
    "#60A5FA",  # blue
    "#F472B6",  # pink
    "#FB923C",  # orange
    "#A3E635",  # lime
    "#38BDF8",  # sky blue
    "#E879F9",  # fuchsia
    "#2DD4BF",  # cyan
    "#FCA5A5",  # light red
    "#C4B5FD",  # light purple
    "#6EE7B7",  # light emerald
    "#FDE68A",  # light amber
    "#93C5FD",  # light blue
]

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


def load_data() -> pd.DataFrame:
    conn = get_connection()
    query = """
        SELECT
            c.ticker,
            c.sector,
            f.filed_at,
            lf.hedging_score,
            lf.uncertainty_score,
            lf.specificity_score,
            lf.fog_index,
            lf.finbert_positive,
            lf.finbert_negative,
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

    raw_features = [
        "hedging_score", "uncertainty_score", "specificity_score",
        "fog_index", "finbert_positive", "finbert_negative"
    ]
    for feature in raw_features:
        df[f"{feature}_z"] = df.groupby("ticker")[feature].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    return df


FEATURES = {
    "hedging_score": "Hedging Score",
    "uncertainty_score": "Uncertainty Score",
    "specificity_score": "Specificity Score",
    "fog_index": "Gunning Fog Index",
    "finbert_positive": "FinBERT Positive",
    "finbert_negative": "FinBERT Negative",
    "hedging_score_z": "Hedging Score (z)",
    "uncertainty_score_z": "Uncertainty Score (z)",
    "specificity_score_z": "Specificity Score (z)",
    "fog_index_z": "Fog Index (z)",
    "finbert_positive_z": "FinBERT Positive (z)",
    "finbert_negative_z": "FinBERT Negative (z)",
}


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
        f"{ticker}  hedging={latest.loc[ticker, 'hedging_score']:.2f}"
        for ticker in sorted(latest.index)
    ]


def build_findings_table(findings: list[dict]) -> html.Div:
    if not findings:
        return html.Div(
            "No significant correlations found (p < 0.05).",
            style={"color": TEXT_SECONDARY, "fontSize": "13px"},
        )
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Feature", style={"textAlign": "left"}),
            html.Th("Horizon", style={"textAlign": "left"}),
            html.Th("Spearman r", style={"textAlign": "right"}),
            html.Th("P-Value", style={"textAlign": "right"}),
            html.Th("N", style={"textAlign": "right"}),
        ], style={
            "background": AMBER,
            "color": BG_OBSIDIAN,
            "fontFamily": "JetBrains Mono",
            "fontSize": "11px",
        })),
        html.Tbody([
            html.Tr([
                html.Td(row["feature"],
                        style={"fontFamily": "IBM Plex Sans",
                               "fontSize": "12px", "padding": "6px 10px"}),
                html.Td(row["horizon"],
                        style={"fontFamily": "JetBrains Mono",
                               "fontSize": "12px", "padding": "6px 10px"}),
                html.Td(f"{row['r']:.4f}",
                        style={"fontFamily": "JetBrains Mono", "fontSize": "12px",
                               "textAlign": "right", "padding": "6px 10px",
                               "color": AMBER}),
                html.Td(f"{row['p']:.4f}",
                        style={"fontFamily": "JetBrains Mono", "fontSize": "12px",
                               "textAlign": "right", "padding": "6px 10px"}),
                html.Td(str(row["n"]),
                        style={"fontFamily": "JetBrains Mono", "fontSize": "12px",
                               "textAlign": "right", "padding": "6px 10px"}),
            ], style={
                "borderBottom": f"1px solid {BORDER}",
                "color": TEXT_PRIMARY,
            })
            for row in findings
        ]),
    ], style={
        "width": "100%",
        "borderCollapse": "collapse",
        "fontSize": "12px",
    })


def build_metric_card(label: str, value: str, color: str = AMBER) -> html.Div:
    return html.Div([
        html.Div(label, style={
            "fontFamily": "IBM Plex Sans",
            "fontSize": "10px",
            "color": TEXT_SECONDARY,
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "marginBottom": "8px",
        }),
        html.Div(value, style={
            "fontFamily": "JetBrains Mono",
            "fontSize": "26px",
            "color": color,
            "fontWeight": "400",
        }),
    ], style={
        "background": BG_CARD,
        "border": f"1px solid {BORDER}",
        "borderBottom": f"2px solid {AMBER}",
        "borderRadius": "8px",
        "padding": "14px 18px",
        "flex": "1",
        "minWidth": "120px",
    })


df = load_data()
TICKERS = sorted(df["ticker"].unique())
FINDINGS = compute_findings(df)
TAPE_ITEMS = build_ticker_tape(df)

tape_track = []
for item in TAPE_ITEMS * 3:
    tape_track.append(html.Span(item, style={
        "fontFamily": "JetBrains Mono",
        "fontSize": "11px",
        "color": AMBER,
        "padding": "0 16px",
        "whiteSpace": "nowrap",
    }))
    tape_track.append(html.Span("·", style={
        "color": TEXT_SECONDARY,
        "fontSize": "11px",
    }))

app.layout = html.Div([

    # Ticker tape
    html.Div(
        html.Div(tape_track, style={
            "display": "flex",
            "alignItems": "center",
            "animation": "scroll 40s linear infinite",
            "width": "max-content",
        }),
        style={
            "overflow": "hidden",
            "background": BG_CARD,
            "borderBottom": f"1px solid {BORDER}",
            "height": "32px",
            "display": "flex",
            "alignItems": "center",
            "marginBottom": "32px",
        }
    ),

    html.Div([

        # Title block
        html.Div([
            html.H1("EARNINGS SENTIMENT DRIFT", style={
                "fontFamily": "IBM Plex Sans, sans-serif",
                "fontSize": "28px",
                "fontWeight": "300",
                "letterSpacing": "0.2em",
                "color": TEXT_PRIMARY,
                "margin": "0 0 8px 0",
                "textTransform": "uppercase",
            }),
            html.P(
                "Linguistic feature analysis of SEC 8-K filings "
                "vs subsequent stock returns · "
                f"{len(df)} observations · "
                f"{df['ticker'].nunique()} companies · "
                f"{df['sector'].nunique()} sectors",
                style={
                    "fontFamily": "JetBrains Mono",
                    "fontSize": "11px",
                    "color": TEXT_SECONDARY,
                    "margin": 0,
                    "letterSpacing": "0.04em",
                }
            ),
        ], style={"marginBottom": "36px"}),

        # Amber divider
        html.Hr(style={
            "border": "none",
            "borderTop": f"1px solid rgba(255,176,0,0.15)",
            "boxShadow": "0 0 8px rgba(255,176,0,0.1)",
            "margin": "0 0 28px 0",
        }),

        # Methodology section
        html.Div("< Methodology />", style={
            "fontFamily": "JetBrains Mono",
            "fontSize": "11px",
            "color": AMBER,
            "letterSpacing": "0.12em",
            "marginBottom": "16px",
            "textTransform": "uppercase",
        }),

        # Controls
        html.Div([
            html.Div([
                html.Label("Sector", style={
                    "fontFamily": "IBM Plex Sans",
                    "fontSize": "11px",
                    "fontWeight": "500",
                    "color": TEXT_SECONDARY,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.06em",
                    "marginBottom": "8px",
                    "display": "block",
                }),
                dcc.Checklist(
                    id="sector-filter",
                    options=[{"label": f"  {s}", "value": s}
                             for s in sorted(df["sector"].dropna().unique())],
                    value=sorted(df["sector"].dropna().unique()),
                    inline=True,
                    labelStyle={"color": AMBER, "fontFamily": "IBM Plex Sans",
                                "fontSize": "12px", "marginRight": "12px"},
                ),
            ], style={"marginBottom": "16px"}),

            html.Div([
                html.Label("Company", style={
                    "fontFamily": "IBM Plex Sans",
                    "fontSize": "11px",
                    "fontWeight": "500",
                    "color": TEXT_SECONDARY,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.06em",
                    "marginBottom": "8px",
                    "display": "block",
                }),
                dcc.Checklist(
                    id="ticker-filter",
                    options=[{"label": f"  {t}", "value": t} for t in TICKERS],
                    value=TICKERS,
                    inline=True,
                    labelStyle={"color": AMBER, "fontFamily": "JetBrains Mono",
                                "fontSize": "12px", "marginRight": "10px"},
                ),
            ], style={"marginBottom": "16px"}),

            html.Div([
                html.Div([
                    html.Label("Linguistic Feature", style={
                        "fontFamily": "IBM Plex Sans",
                        "fontSize": "11px",
                        "color": TEXT_SECONDARY,
                        "textTransform": "uppercase",
                        "letterSpacing": "0.06em",
                        "marginBottom": "8px",
                        "display": "block",
                    }),
                    dcc.Dropdown(
                        id="feature-select",
                        options=[{"label": v, "value": k}
                                 for k, v in FEATURES.items()],
                        value="specificity_score",
                        clearable=False,
                        style={
                            "fontSize": "12px",
                            "background": BG_CARD,
                            "color": TEXT_PRIMARY,
                            "border": f"1px solid {BORDER}",
                            "fontFamily": "JetBrains Mono",
                        },
                    ),
                ], style={"width": "280px"}),

                html.Div([
                    html.Label("Return Horizon", style={
                        "fontFamily": "IBM Plex Sans",
                        "fontSize": "11px",
                        "color": TEXT_SECONDARY,
                        "textTransform": "uppercase",
                        "letterSpacing": "0.06em",
                        "marginBottom": "8px",
                        "display": "block",
                    }),
                    dcc.RadioItems(
                        id="return-horizon",
                        options=[
                            {"label": "  30 day", "value": "return_30d_pct"},
                            {"label": "  60 day", "value": "return_60d_pct"},
                            {"label": "  90 day", "value": "return_90d_pct"},
                        ],
                        value="return_30d_pct",
                        inline=True,
                        labelstyle={"fontSize": "12px", "color": AMBER,
                               "fontFamily": "JetBrains Mono"},
                    ),
                ], style={"width": "280px"}),
            ], style={"display": "flex", "gap": "32px",
                      "alignItems": "flex-end"}),

        ], style={
            "background": BG_CARD,
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
            "padding": "20px 24px",
            "marginBottom": "20px",
        }),

        # Metric cards
        html.Div(id="stats-bar", style={
            "display": "flex",
            "gap": "12px",
            "marginBottom": "28px",
            "flexWrap": "wrap",
        }),

        # Amber divider
        html.Hr(style={
            "border": "none",
            "borderTop": f"1px solid rgba(255,176,0,0.15)",
            "boxShadow": "0 0 8px rgba(255,176,0,0.1)",
            "margin": "0 0 20px 0",
        }),

        # Results section
        html.Div("< Results />", style={
            "fontFamily": "JetBrains Mono",
            "fontSize": "11px",
            "color": AMBER,
            "letterSpacing": "0.12em",
            "marginBottom": "16px",
            "textTransform": "uppercase",
        }),

        # Charts
        html.Div([
            html.Div([
                dcc.Graph(id="scatter-plot", style={"height": "380px"}),
            ], style={
                "flex": "1",
                "background": BG_CARD,
                "border": f"1px solid {BORDER}",
                "borderRadius": "10px",
                "padding": "8px",
            }),
            html.Div([
                dcc.Graph(id="time-series", style={"height": "380px"}),
            ], style={
                "flex": "1",
                "background": BG_CARD,
                "border": f"1px solid {BORDER}",
                "borderRadius": "10px",
                "padding": "8px",
            }),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "28px"}),

        # Amber divider
        html.Hr(style={
            "border": "none",
            "borderTop": f"1px solid rgba(255,176,0,0.15)",
            "boxShadow": "0 0 8px rgba(255,176,0,0.1)",
            "margin": "0 0 20px 0",
        }),

        # Findings section
        html.Div("< Findings />", style={
            "fontFamily": "JetBrains Mono",
            "fontSize": "11px",
            "color": AMBER,
            "letterSpacing": "0.12em",
            "marginBottom": "16px",
            "textTransform": "uppercase",
        }),

        html.Div([
            build_findings_table(FINDINGS),
        ], style={
            "background": BG_CARD,
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
            "padding": "16px 20px",
            "marginBottom": "40px",
        }),

    ], style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "0 24px 40px 24px",
    }),

], style={
    "background": BG_OBSIDIAN,
    "minHeight": "100vh",
    "color": TEXT_PRIMARY,
    "fontFamily": "IBM Plex Sans, sans-serif",
})


@app.callback(
    Output("scatter-plot", "figure"),
    Output("time-series", "figure"),
    Output("stats-bar", "children"),
    Input("sector-filter", "value"),
    Input("ticker-filter", "value"),
    Input("feature-select", "value"),
    Input("return-horizon", "value"),
)
def update_charts(sectors, tickers, feature, return_col):
    filtered = df[
        df["ticker"].isin(tickers) &
        df["sector"].isin(sectors)
    ].copy()

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
    return_label = (
        return_col.replace("_pct", "")
                  .replace("return_", "")
                  .replace("d", " day return %")
    )

    # Scatter plot
    scatter = go.Figure()
    visible_tickers = [t for t in tickers if t in filtered["ticker"].values]
    for idx, ticker in enumerate(visible_tickers):
        t_df = filtered[filtered["ticker"] == ticker]
        color = CHART_COLORS[idx % len(CHART_COLORS)]
        scatter.add_trace(go.Scatter(
            x=t_df[feature],
            y=t_df[return_col],
            mode="markers",
            name=ticker,
            marker=dict(
                color=color,
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
            line=dict(color="rgba(255,176,0,0.5)", width=1, dash="dash"),
            showlegend=True,
        ))

    scatter.update_layout(
        title=dict(
            text=f"{feature_label} vs {return_label.title()}",
            font=dict(family="IBM Plex Sans", size=13, color=TEXT_PRIMARY),
        ),
        xaxis=axis(feature_label),
        yaxis=axis("Return (%)", tickformat=".1f", ticksuffix="%",
                   zeroline=True),
        margin=dict(l=56, r=24, t=52, b=64),
        hovermode="closest",
        **PLOT_LAYOUT,
    )

    # Time series
    time_fig = go.Figure()
    for idx, ticker in enumerate([t for t in tickers
                                   if t in filtered["ticker"].values]):
        color = CHART_COLORS[idx % len(CHART_COLORS)]
        t_df = filtered[filtered["ticker"] == ticker].sort_values("filed_at")
        time_fig.add_trace(go.Scatter(
            x=t_df["filed_at"],
            y=t_df[feature],
            mode="lines+markers",
            name=ticker,
            line=dict(color=color, width=1.5),
            marker=dict(size=5, color=color),
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

    # Metric cards
    corr, pvalue = stats.spearmanr(
        clean[feature], clean[return_col]
    ) if len(clean) > 3 else (0, 1)

    median = clean[feature].median() if len(clean) > 0 else 0
    correct = (
        ((clean[feature] > median) & (clean[return_col] > 0))
        | ((clean[feature] <= median) & (clean[return_col] <= 0))
    ).sum()
    win_rate = (correct / len(clean) * 100) if len(clean) > 0 else 0

    p_color = AMBER if pvalue < 0.05 else "#FF4444"

    metric_cards = [
        build_metric_card("Spearman r", f"{corr:.4f}"),
        build_metric_card("P-Value", f"{pvalue:.4f}", p_color),
        build_metric_card("Win Rate", f"{win_rate:.1f}%"),
        build_metric_card("N", str(len(clean))),
    ]

    return scatter, time_fig, metric_cards


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)