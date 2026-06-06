import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from scipy import stats
from ingestion.loader import get_connection

app = Dash(__name__)

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
COLORS = {
    "AAPL": "#007AFF",
    "GOOGL": "#34A853",
    "META": "#1877F2",
    "MSFT": "#737373",
    "NVDA": "#76B900",
}

app.layout = html.Div([
    html.Div([
        html.H1("Earnings Call Sentiment Drift Tracker",
                style={"fontSize": "22px", "fontWeight": "600",
                       "margin": "0 0 4px 0"}),
        html.P("Linguistic feature analysis of SEC 8-K filings vs subsequent stock returns",
               style={"color": "#666", "fontSize": "13px", "margin": 0}),
    ], style={"marginBottom": "24px"}),

    # Controls row
    html.Div([
        html.Div([
            html.Label("Company", style={"fontSize": "12px",
                                          "fontWeight": "500",
                                          "marginBottom": "6px",
                                          "display": "block"}),
            dcc.Checklist(
                id="ticker-filter",
                options=[{"label": f"  {t}", "value": t} for t in TICKERS],
                value=TICKERS,
                inline=True,
                style={"fontSize": "13px", "gap": "12px"},
            ),
        ], style={"marginBottom": "16px"}),

        html.Div([
            html.Div([
                html.Label("Linguistic Feature",
                           style={"fontSize": "12px", "fontWeight": "500",
                                  "marginBottom": "6px", "display": "block"}),
                dcc.Dropdown(
                    id="feature-select",
                    options=[{"label": v, "value": k}
                             for k, v in FEATURES.items()],
                    value="hedging_score",
                    clearable=False,
                    style={"fontSize": "13px"},
                ),
            ], style={"width": "220px"}),

            html.Div([
                html.Label("Return Horizon",
                           style={"fontSize": "12px", "fontWeight": "500",
                                  "marginBottom": "6px", "display": "block"}),
                dcc.RadioItems(
                    id="return-horizon",
                    options=[
                        {"label": " 30 day", "value": "return_30d_pct"},
                        {"label": " 60 day", "value": "return_60d_pct"},
                        {"label": " 90 day", "value": "return_90d_pct"},
                    ],
                    value="return_30d_pct",
                    inline=True,
                    style={"fontSize": "13px"},
                ),
            ], style={"width": "260px"}),
        ], style={"display": "flex", "gap": "32px", "alignItems": "flex-end"}),

    ], style={
        "background": "#f8f8f8",
        "borderRadius": "10px",
        "padding": "16px 20px",
        "marginBottom": "20px",
        "border": "1px solid #eee",
    }),

    # Charts row
    html.Div([
        html.Div([
            dcc.Graph(id="scatter-plot", style={"height": "380px"}),
        ], style={"flex": "1"}),

        html.Div([
            dcc.Graph(id="time-series", style={"height": "380px"}),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

    # Stats bar
    html.Div(id="stats-bar", style={
        "background": "#f8f8f8",
        "borderRadius": "10px",
        "padding": "14px 20px",
        "border": "1px solid #eee",
        "fontSize": "13px",
        "display": "flex",
        "gap": "32px",
        "flexWrap": "wrap",
    }),

], style={
    "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "maxWidth": "1100px",
    "margin": "0 auto",
    "padding": "24px",
    "color": "#1a1a1a",
})


@app.callback(
    Output("scatter-plot", "figure"),
    Output("time-series", "figure"),
    Output("stats-bar", "children"),
    Input("ticker-filter", "value"),
    Input("feature-select", "value"),
    Input("return-horizon", "value"),
)
def update_charts(tickers, feature, return_col):
    filtered = df[df["ticker"].isin(tickers)].copy()

    if filtered.empty:
        empty = go.Figure()
        return empty, empty, "No data"

    feature_label = FEATURES[feature]
    return_label = return_col.replace("_pct", "").replace("return_", "") \
                             .replace("d", " day return %")

    # --- Scatter plot ---
    scatter = go.Figure()
    for ticker in tickers:
        t_df = filtered[filtered["ticker"] == ticker]
        scatter.add_trace(go.Scatter(
            x=t_df[feature],
            y=t_df[return_col],
            mode="markers",
            name=ticker,
            marker=dict(
                color=COLORS.get(ticker, "#999"),
                size=9,
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

    # Trendline across all filtered data
    clean = filtered[[feature, return_col]].dropna()
    if len(clean) > 3:
        m, b, r, p, _ = stats.linregress(clean[feature], clean[return_col])
        x_range = [clean[feature].min(), clean[feature].max()]
        y_range = [m * x + b for x in x_range]
        scatter.add_trace(go.Scatter(
            x=x_range, y=y_range,
            mode="lines",
            name=f"Trend (r={r:.2f})",
            line=dict(color="#999", width=1.5, dash="dot"),
            showlegend=True,
        ))

    scatter.update_layout(
        title=dict(text=f"{feature_label} vs {return_label.title()}",
                   font=dict(size=14)),
        xaxis_title=feature_label,
        yaxis_title=f"Return (%)",
        yaxis=dict(tickformat=".1f",
                   ticksuffix="%",
                   zeroline=True,
                   zerolinecolor="#ddd"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=50, r=20, t=50, b=60),
        hovermode="closest",
    )

    # --- Time series ---
    time_fig = go.Figure()
    for ticker in tickers:
        t_df = filtered[filtered["ticker"] == ticker].sort_values("filed_at")
        time_fig.add_trace(go.Scatter(
            x=t_df["filed_at"],
            y=t_df[feature],
            mode="lines+markers",
            name=ticker,
            line=dict(color=COLORS.get(ticker, "#999"), width=2),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                f"Date: %{{x|%Y-%m-%d}}<br>"
                f"{feature_label}: %{{y:.2f}}<extra></extra>"
            ),
        ))

    time_fig.update_layout(
        title=dict(text=f"{feature_label} Over Time", font=dict(size=14)),
        xaxis_title="Filing Date",
        yaxis_title=feature_label,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=50, r=20, t=50, b=60),
        hovermode="x unified",
    )

    # --- Stats bar ---
    corr, pvalue = stats.spearmanr(
        clean[feature], clean[return_col]
    ) if len(clean) > 3 else (0, 1)

    sig_text = "✓ Significant (p < 0.05)" if pvalue < 0.05 else "Not significant"
    sig_color = "#16a34a" if pvalue < 0.05 else "#dc2626"

    stats_children = [
        html.Div([
            html.Span("Observations  ", style={"color": "#666"}),
            html.Span(f"{len(clean)}", style={"fontWeight": "600"}),
        ]),
        html.Div([
            html.Span("Spearman r  ", style={"color": "#666"}),
            html.Span(f"{corr:.4f}", style={"fontWeight": "600"}),
        ]),
        html.Div([
            html.Span("P-value  ", style={"color": "#666"}),
            html.Span(f"{pvalue:.4f}", style={"fontWeight": "600"}),
        ]),
        html.Div([
            html.Span(sig_text, style={"fontWeight": "600", "color": sig_color}),
        ]),
        html.Div([
            html.Span("Companies  ", style={"color": "#666"}),
            html.Span(", ".join(tickers), style={"fontWeight": "600"}),
        ]),
    ]

    return scatter, time_fig, stats_children


if __name__ == "__main__":
    app.run(debug=True)