import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ALL, ctx, no_update
from scipy import stats
from ingestion.loader import get_connection
from features.hedging import (
    keyword_frequency, HEDGING_TERMS, UNCERTAINTY_TERMS, POSITIVE_SIGNAL_TERMS
)

_HEDGE_SET    = set(HEDGING_TERMS)
_UNCERT_SET   = set(UNCERTAINTY_TERMS)
_POSITIVE_SET = set(POSITIVE_SIGNAL_TERMS)

app = Dash(__name__)
server = app.server

BG_OBSIDIAN    = "#080D0A"
BG_CARD        = "#0F1411"
BORDER         = "#1F2922"
GREEN          = "#2ECC71"
RED            = "#E74C3C"
NEUTRAL        = "#6B7280"
TEXT_PRIMARY   = "#F0F0F0"
TEXT_SECONDARY = "#9CA3AF"
AXIS_TEXT      = "#6B7280"
ZERO_LINE      = "#1F2922"

CHART_COLORS = [
    "#2ECC71", "#E74C3C", "#3498DB", "#9B59B6",
    "#1ABC9C", "#F39C12", "#E67E22", "#E91E63",
    "#00BCD4", "#8BC34A", "#FF5722", "#607D8B",
]

# Market cap tiers based on approximate market caps (as of mid-2025).
# Mega >$500B, Large $50-500B, Mid $10-50B, Small <$10B
MARKET_CAP_TIERS = {
    "AAPL":  "Mega",   # ~$3.0T
    "MSFT":  "Mega",   # ~$3.1T
    "META":  "Mega",   # ~$1.4T
    "NVDA":  "Mega",   # ~$2.5T
    "GOOGL": "Mega",   # ~$2.0T
    "JPM":   "Mega",   # ~$650B
    "WMT":   "Mega",   # ~$680B
    "GS":    "Large",  # ~$175B
    "BAC":   "Large",  # ~$310B
    "JNJ":   "Large",  # ~$370B
    "UNH":   "Large",  # ~$480B
    "PFE":   "Large",  # ~$135B
    "XOM":   "Large",  # ~$480B
    "CVX":   "Large",  # ~$280B
    "COST":  "Large",  # ~$380B
    "CAT":   "Large",  # ~$175B
    "BA":    "Large",  # ~$90B
    "KO":    "Large",  # ~$260B
}

MKTCAP_ORDER = ["Mega", "Large", "Mid", "Small"]

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",        "MSFT": "Microsoft Corp.",
    "META": "Meta Platforms",    "NVDA": "NVIDIA Corp.",
    "GOOGL": "Alphabet Inc.",    "JPM": "JPMorgan Chase",
    "GS": "Goldman Sachs",       "BAC": "Bank of America",
    "JNJ": "Johnson & Johnson",  "UNH": "UnitedHealth Group",
    "PFE": "Pfizer Inc.",        "XOM": "Exxon Mobil",
    "CVX": "Chevron Corp.",      "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale",  "CAT": "Caterpillar Inc.",
    "BA": "Boeing Company",      "KO": "Coca-Cola Co.",
}

PLOT_LAYOUT = dict(
    plot_bgcolor=BG_OBSIDIAN,
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=TEXT_PRIMARY, size=12),
    hoverlabel=dict(
        bgcolor=BG_CARD,
        bordercolor=GREEN,
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
    gridcolor="rgba(31, 41, 34, 0.6)",
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
            font=dict(family="Inter", size=11, color=TEXT_SECONDARY),
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
            ps.close_price,
            ps.return_30d,
            ps.return_60d,
            ps.return_90d,
            t.word_count,
            t.prepared_remarks
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
    df["polarity"] = (df["finbert_positive"] - df["finbert_negative"]).round(4)
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

    # Derived columns
    df["market_cap_tier"] = df["ticker"].map(MARKET_CAP_TIERS).fillna("Unknown")

    diff = df["finbert_positive"] - df["finbert_negative"]
    df["sentiment_band"] = "NEUTRAL"
    df.loc[diff > 0.1,  "sentiment_band"] = "BULL"
    df.loc[diff < -0.1, "sentiment_band"] = "BEAR"

    return df


FEATURES = {
    "hedging_score":       "Hedging Score",
    "uncertainty_score":   "Uncertainty Score",
    "specificity_score":   "Specificity Score",
    "fog_index":           "Gunning Fog Index",
    "finbert_positive":    "FinBERT Positive",
    "finbert_negative":    "FinBERT Negative",
    "hedging_score_z":     "Hedging Score (z)",
    "uncertainty_score_z": "Uncertainty Score (z)",
    "specificity_score_z": "Specificity Score (z)",
    "fog_index_z":         "Fog Index (z)",
    "finbert_positive_z":  "FinBERT Positive (z)",
    "finbert_negative_z":  "FinBERT Negative (z)",
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


def build_ticker_tape_items(data: pd.DataFrame) -> list[tuple]:
    latest = data.sort_values("filed_at").groupby("ticker").last().reset_index()
    items = []
    for _, row in latest.sort_values("ticker").iterrows():
        is_bull = row["finbert_positive"] > row["finbert_negative"]
        items.append((
            row["ticker"],
            row.get("close_price", None),
            row["return_30d_pct"],
            is_bull,
        ))
    return items


def build_findings_table(findings: list[dict]) -> html.Div:
    if not findings:
        return html.Div(
            "No significant correlations found (p < 0.05).",
            style={"color": TEXT_SECONDARY, "fontSize": "13px"},
        )
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Feature",    style={"textAlign": "left"}),
            html.Th("Horizon",    style={"textAlign": "left"}),
            html.Th("Spearman r", style={"textAlign": "right"}),
            html.Th("P-Value",    style={"textAlign": "right"}),
            html.Th("N",          style={"textAlign": "right"}),
        ], style={
            "background": "rgba(46,204,113,0.08)",
            "color": GREEN,
            "fontFamily": "Inter",
            "fontSize": "10px",
            "fontWeight": "700",
            "letterSpacing": "0.1em",
            "textTransform": "uppercase",
        })),
        html.Tbody([
            html.Tr([
                html.Td(row["feature"],
                        style={"fontFamily": "Inter", "fontSize": "12px",
                               "padding": "10px 14px"}),
                html.Td(row["horizon"],
                        style={"fontFamily": "JetBrains Mono", "fontSize": "12px",
                               "padding": "10px 14px"}),
                html.Td(f"{row['r']:.4f}",
                        style={"fontFamily": "JetBrains Mono", "fontSize": "12px",
                               "textAlign": "right", "padding": "10px 14px",
                               "color": GREEN if row["r"] > 0 else RED}),
                html.Td(f"{row['p']:.4f}",
                        style={"fontFamily": "JetBrains Mono", "fontSize": "12px",
                               "textAlign": "right", "padding": "10px 14px",
                               "color": TEXT_SECONDARY}),
                html.Td(str(row["n"]),
                        style={"fontFamily": "JetBrains Mono", "fontSize": "12px",
                               "textAlign": "right", "padding": "10px 14px"}),
            ], style={"borderBottom": f"1px solid {BORDER}", "color": TEXT_PRIMARY})
            for row in findings
        ]),
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px"})


def build_metric_card(label: str, value: str, color: str = GREEN) -> html.Div:
    return html.Div([
        html.Div(label, style={
            "fontFamily": "Inter",
            "fontSize": "10px",
            "fontWeight": "600",
            "color": NEUTRAL,
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
        "borderBottom": f"2px solid {GREEN}",
        "borderRadius": "8px",
        "padding": "14px 18px",
        "flex": "1",
        "minWidth": "120px",
    })


def build_headline_metric_card(label: str, value: str, color: str = GREEN) -> html.Div:
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value", style={"color": color}),
    ], className="metric-card--headline")


def _panel_header(label: str, counter_id: str) -> html.Div:
    return html.Div([
        html.Span(label, style={
            "fontFamily": "JetBrains Mono",
            "fontSize": "10px",
            "fontWeight": "500",
            "color": TEXT_SECONDARY,
            "textTransform": "uppercase",
            "letterSpacing": "0.12em",
        }),
        html.Span(id=counter_id, style={
            "fontFamily": "JetBrains Mono",
            "fontSize": "10px",
            "color": GREEN,
            "letterSpacing": "0.06em",
        }),
    ], style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "marginBottom": "12px",
        "paddingBottom": "10px",
        "borderBottom": f"1px solid {BORDER}",
    })


# ---------------------------------------------------------------------------
# Cohort table helpers
# ---------------------------------------------------------------------------

COHORT_COLS = [
    ("SYM",         "sym"),
    ("NAME",        "name"),
    ("SECTOR",      "sector"),
    ("CAP",         "cap"),
    ("SENT",        "sent"),
    ("SPECIFICITY", "specificity"),
    ("DRIFT 30D",   "drift_30d"),
    ("FOG",         "fog"),
    ("PRICE",       "price"),
    ("Δ%",          "delta_pct"),
]


def build_cohort_df(filtered: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-company stats from the filtered dataframe."""
    latest = filtered.sort_values("filed_at").groupby("ticker").last().reset_index()
    agg = filtered.groupby("ticker").agg(
        specificity=("specificity_score", "mean"),
        drift_30d=("return_30d_pct", "mean"),
        fog=("fog_index", "mean"),
    ).reset_index()
    tbl = agg.merge(
        latest[["ticker", "sector", "finbert_positive", "finbert_negative",
                "close_price", "return_30d_pct"]],
        on="ticker", how="left",
    )
    tbl["sent"]      = tbl["finbert_positive"] - tbl["finbert_negative"]
    tbl["price"]     = tbl["close_price"]
    tbl["delta_pct"] = tbl["return_30d_pct"]
    tbl["cap"]       = tbl["ticker"].map(MARKET_CAP_TIERS).fillna("—")
    tbl["name"]      = tbl["ticker"].map(COMPANY_NAMES).fillna(tbl["ticker"])
    return tbl


def _sent_cell(sent: float, max_abs: float) -> html.Td:
    color = GREEN if sent >= 0 else RED
    pct = min(abs(sent) / max_abs * 100, 100) if max_abs > 0 else 0
    return html.Td([
        html.Div(f"{sent:+.2f}", style={
            "fontFamily": "JetBrains Mono", "fontSize": "11px",
            "color": color, "marginBottom": "4px",
        }),
        html.Div(
            html.Div(style={
                "width": f"{pct:.0f}%",
                "height": "100%",
                "background": color,
                "borderRadius": "1px",
            }),
            style={
                "background": "rgba(255,255,255,0.06)",
                "height": "3px",
                "borderRadius": "2px",
                "width": "72px",
            },
        ),
    ], style={"padding": "10px 12px", "verticalAlign": "middle", "minWidth": "90px"})


def build_cohort_rows(tbl: pd.DataFrame, sort_col: str, sort_dir: str,
                      selected_ticker: str | None) -> list:
    ascending = sort_dir == "asc"
    if sort_col in tbl.columns:
        tbl = tbl.sort_values(sort_col, ascending=ascending, na_position="last")

    max_abs_sent = tbl["sent"].abs().max() or 1

    rows = []
    for _, row in tbl.iterrows():
        ticker    = row["ticker"]
        is_sel    = ticker == selected_ticker
        drift     = row["drift_30d"]
        delta     = row["delta_pct"]
        drift_col = GREEN if drift >= 0 else RED
        delta_col = GREEN if delta >= 0 else RED
        delta_arr = "▲" if delta >= 0 else "▼"
        row_bg    = "rgba(46,204,113,0.06)" if is_sel else "transparent"

        tr = html.Tr([
            html.Td(ticker, style={
                "fontFamily": "Inter", "fontWeight": "700",
                "fontSize": "12px", "color": TEXT_PRIMARY,
                "padding": "10px 12px",
            }),
            html.Td(row["name"], style={
                "fontFamily": "Inter", "fontSize": "11px",
                "color": TEXT_SECONDARY, "padding": "10px 12px",
                "whiteSpace": "nowrap", "maxWidth": "140px",
                "overflow": "hidden", "textOverflow": "ellipsis",
            }),
            html.Td(row["sector"], style={
                "fontFamily": "Inter", "fontSize": "11px",
                "color": TEXT_SECONDARY, "padding": "10px 12px",
            }),
            html.Td(row["cap"], style={
                "fontFamily": "JetBrains Mono", "fontSize": "11px",
                "color": NEUTRAL, "padding": "10px 12px",
            }),
            _sent_cell(row["sent"], max_abs_sent),
            html.Td(f"{row['specificity']:.1f}", style={
                "fontFamily": "JetBrains Mono", "fontSize": "11px",
                "color": TEXT_PRIMARY, "padding": "10px 12px",
            }),
            html.Td(f"{drift:+.1f}%", style={
                "fontFamily": "JetBrains Mono", "fontSize": "12px",
                "color": drift_col, "padding": "10px 12px", "fontWeight": "600",
            }),
            html.Td(f"{row['fog']:.1f}", style={
                "fontFamily": "JetBrains Mono", "fontSize": "11px",
                "color": NEUTRAL, "padding": "10px 12px",
            }),
            html.Td(
                f"${row['price']:.2f}" if pd.notna(row["price"]) else "—",
                style={
                    "fontFamily": "JetBrains Mono", "fontSize": "11px",
                    "color": TEXT_PRIMARY, "padding": "10px 12px",
                },
            ),
            html.Td(f"{delta_arr} {abs(delta):.1f}%", style={
                "fontFamily": "JetBrains Mono", "fontSize": "11px",
                "color": delta_col, "padding": "10px 12px",
            }),
        ],
        id={"type": "cohort-row", "index": ticker},
        n_clicks=0,
        style={
            "borderBottom": f"1px solid {BORDER}",
            "background": row_bg,
            "cursor": "pointer",
            "transition": "background 120ms ease",
        },
        className="cohort-row",
        )
        rows.append(tr)
    return rows


def build_keyword_pills(freq: dict[str, int]) -> html.Div:
    if not freq:
        return html.Div("No term data available for this cohort.", style={
            "fontFamily": "Inter", "fontSize": "13px", "color": NEUTRAL,
            "padding": "20px",
        })
    max_count = max(freq.values()) if freq else 1
    pills = []
    for term, count in freq.items():
        if term in _POSITIVE_SET:
            color, border = GREEN, "rgba(46,204,113,0.35)"
        elif term in _UNCERT_SET:
            color, border = RED, "rgba(231,76,60,0.35)"
        else:
            color, border = NEUTRAL, "rgba(107,114,128,0.35)"
        # Font size 13–20px interpolated by relative frequency
        size = 13 + (count / max_count) * 7
        pills.append(html.Span(
            f"{term} ×{count}",
            style={
                "fontFamily": "JetBrains Mono",
                "fontSize": f"{size:.1f}px",
                "color": color,
                "border": f"1px solid {border}",
                "borderRadius": "4px",
                "padding": "4px 10px",
                "display": "inline-block",
                "lineHeight": "1.4",
                "background": f"rgba({','.join(str(int(color[i:i+2], 16)) for i in (1, 3, 5))},0.06)",
            },
        ))
    return html.Div(pills, style={
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "8px",
        "padding": "20px",
    })


# ---------------------------------------------------------------------------
# Bootstrap data
# ---------------------------------------------------------------------------
df = load_data()
TICKERS = sorted(df["ticker"].unique())
FINDINGS = compute_findings(df)
TAPE_ITEMS = build_ticker_tape_items(df)

ALL_SECTORS = sorted(df["sector"].dropna().unique())
ALL_MKTCAP_TIERS = [t for t in MKTCAP_ORDER if t in df["market_cap_tier"].unique()]

# Sector → sorted ticker list (for expandable tree)
SECTOR_TICKERS = {
    sector: sorted(group["ticker"].unique())
    for sector, group in df.groupby("sector")
}

# Headline stats (specificity_score vs return_30d_pct, all companies)
_hl_clean = df[["specificity_score", "return_30d_pct"]].dropna()
_hl_corr, _hl_p = (
    stats.spearmanr(_hl_clean["specificity_score"], _hl_clean["return_30d_pct"])
    if len(_hl_clean) > 3 else (0, 1)
)
_hl_median = _hl_clean["specificity_score"].median()
_hl_correct = (
    ((_hl_clean["specificity_score"] > _hl_median) & (_hl_clean["return_30d_pct"] > 0))
    | ((_hl_clean["specificity_score"] <= _hl_median) & (_hl_clean["return_30d_pct"] <= 0))
).sum()
_hl_win_rate = (_hl_correct / len(_hl_clean) * 100) if len(_hl_clean) > 0 else 0
_hl_p_color = GREEN if _hl_p < 0.05 else RED
_hl_direction = "positively" if _hl_corr > 0 else "negatively"

HEADLINE_SUMMARY = (
    f"When companies use more specific numeric language in 8-K filings, "
    f"their stock returns over the next 30 days are {_hl_direction} correlated "
    f"(Spearman r = {_hl_corr:.3f}, p = {_hl_p:.4f}, n = {len(_hl_clean)})."
)
HEADLINE_CARDS = [
    build_headline_metric_card("Spearman r", f"{_hl_corr:.4f}"),
    build_headline_metric_card("P-Value", f"{_hl_p:.4f}", _hl_p_color),
    build_headline_metric_card("Win Rate", f"{_hl_win_rate:.1f}%"),
    build_headline_metric_card("N", str(len(_hl_clean))),
]

# Research window
_min_q = df["filed_at"].min()
_max_q = df["filed_at"].max()

def _quarter_label(dt) -> str:
    return f"Q{(dt.month - 1) // 3 + 1} {dt.year}"

RESEARCH_WINDOW = f"{_quarter_label(_min_q)} - {_quarter_label(_max_q)}"
HERO_SUBHEAD = (
    f"{df['filed_at'].nunique()} SEC 8-K filings analyzed across "
    f"{df['ticker'].nunique()} companies since "
    f"{_min_q.strftime('%B %Y')}, scoring hedging language, numeric specificity, "
    f"and sentiment -- then mapped against 30/60/90-day forward returns."
)

# Ticker tape spans
def _make_tape_spans(items):
    spans = []
    for ticker, close, ret30, is_bull in items:
        arrow = "▲" if is_bull else "▼"
        css_cls = "ticker-item ticker-item--bull" if is_bull else "ticker-item ticker-item--bear"
        close_str = f"${close:.2f}" if close is not None else "--"
        ret_str = f"{ret30:+.1f}%" if ret30 is not None else "--"
        spans.append(html.Span(f"{arrow} {ticker}  {close_str}  {ret_str}", className=css_cls))
        spans.append(html.Span("  ·  ", className="ticker-separator"))
    return spans

_tape_spans = _make_tape_spans(TAPE_ITEMS) * 3

# ---------------------------------------------------------------------------
# Tab styles
# ---------------------------------------------------------------------------
TAB_STYLE = {
    "fontFamily": "JetBrains Mono",
    "fontSize": "11px",
    "letterSpacing": "0.1em",
    "color": TEXT_SECONDARY,
    "background": BG_CARD,
    "border": f"1px solid {BORDER}",
    "borderBottom": "none",
    "padding": "10px 24px",
    "textTransform": "uppercase",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "color": GREEN,
    "borderTop": f"2px solid {GREEN}",
    "background": BG_OBSIDIAN,
}

# ---------------------------------------------------------------------------
# Shared panel card style
# ---------------------------------------------------------------------------
PANEL_CARD = {
    "background": BG_CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "16px",
    "flex": "1",
    "minWidth": "180px",
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div([

    dcc.Store(id="selected-ticker", data=None),
    dcc.Store(id="cohort-sort", data={"col": "drift_30d", "dir": "desc"}),

    # Ticker tape
    html.Div(
        html.Div(_tape_spans, className="ticker-tape-track"),
        className="ticker-tape-wrap",
    ),

    # Hero block
    html.Div([
        html.Div([
            html.Div([
                html.Span("■ ", style={"fontSize": "9px"}),
                html.Span(f"RESEARCH WINDOW  ·  {RESEARCH_WINDOW}"),
            ], className="research-badge"),
            html.H1([
                html.Span("Read the filing.", className="hero-line-white"),
                html.Span([
                    "Trace the drift",
                    html.Span("_", className="blink-cursor"),
                ], className="hero-line-green"),
            ], className="hero-headline"),
            html.P(HERO_SUBHEAD, className="hero-subhead"),
        ]),
    ], className="hero-block dashboard-main"),

    # Tabs
    html.Div([
        dcc.Tabs(
            id="main-tabs",
            value="headline",
            children=[

                # ── TAB 1: Headline ──────────────────────────────────────
                dcc.Tab(
                    label="Headline",
                    value="headline",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[html.Div([

                        html.Div("01 / KEY FINDING", className="section-eyebrow"),

                        html.Div(HEADLINE_SUMMARY, style={
                            "fontFamily": "Inter",
                            "fontSize": "15px",
                            "color": TEXT_PRIMARY,
                            "lineHeight": "1.7",
                            "background": BG_CARD,
                            "border": f"1px solid {BORDER}",
                            "borderLeft": f"3px solid {GREEN}",
                            "borderRadius": "6px",
                            "padding": "16px 20px",
                            "marginBottom": "24px",
                        }),

                        html.Div(HEADLINE_CARDS, style={
                            "display": "flex",
                            "gap": "12px",
                            "flexWrap": "wrap",
                            "marginBottom": "12px",
                        }),

                        html.Div(
                            "Specificity Score vs 30-day return  ·  all companies  ·  p < 0.05 threshold",
                            style={
                                "fontFamily": "JetBrains Mono",
                                "fontSize": "10px",
                                "color": NEUTRAL,
                                "letterSpacing": "0.06em",
                                "marginTop": "8px",
                            }
                        ),

                    ], className="tab-fade-in", style={"paddingBottom": "40px", "paddingTop": "28px"})],
                ),

                # ── TAB 2: Explore ───────────────────────────────────────
                dcc.Tab(
                    label="Explore",
                    value="explore",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[html.Div([

                        # Section header
                        html.Div("01 / COHORT", className="section-eyebrow",
                                 style={"marginTop": "28px"}),
                        html.H2("Shape the cohort", style={
                            "fontFamily": "Inter",
                            "fontSize": "28px",
                            "fontWeight": "700",
                            "color": TEXT_PRIMARY,
                            "margin": "6px 0 8px 0",
                            "letterSpacing": "-0.01em",
                        }),
                        html.P(
                            "Filter the universe by sector, market cap, and sentiment band. "
                            "Charts update live.",
                            style={
                                "fontFamily": "Inter",
                                "fontSize": "16px",
                                "color": TEXT_SECONDARY,
                                "margin": "0 0 24px 0",
                            }
                        ),

                        # Four filter panels
                        html.Div([

                            # PANEL 1 — SECTOR (expandable company tree)
                            html.Div([
                                _panel_header("SECTOR", "sector-counter"),
                                html.Div([
                                    html.Details([
                                        html.Summary([
                                            html.Span(sector, className="sector-tree-label"),
                                            html.Span(
                                                f"{len(tickers)}",
                                                className="sector-tree-count",
                                            ),
                                        ], className="sector-tree-summary"),
                                        dcc.Checklist(
                                            id={"type": "company-filter", "index": sector},
                                            options=[{"label": f" {t}", "value": t}
                                                     for t in tickers],
                                            value=tickers,
                                            className="filter-checklist company-checklist",
                                            labelStyle={
                                                "display": "flex",
                                                "alignItems": "center",
                                                "color": TEXT_SECONDARY,
                                                "fontFamily": "JetBrains Mono",
                                                "fontSize": "11px",
                                                "marginBottom": "4px",
                                                "cursor": "pointer",
                                            },
                                            inputStyle={
                                                "accentColor": GREEN,
                                                "marginRight": "8px",
                                            },
                                        ),
                                    ], className="sector-tree-item")
                                    for sector, tickers in sorted(SECTOR_TICKERS.items())
                                ]),
                            ], style={**PANEL_CARD, "minWidth": "220px", "maxWidth": "280px"}),

                            # PANEL 2 — MARKET CAP
                            html.Div([
                                _panel_header("MARKET CAP", "mktcap-counter"),
                                dcc.Checklist(
                                    id="mktcap-filter",
                                    options=[{"label": f"  {t}", "value": t}
                                             for t in ALL_MKTCAP_TIERS],
                                    value=ALL_MKTCAP_TIERS,
                                    className="filter-checklist",
                                    labelStyle={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "color": TEXT_PRIMARY,
                                        "fontFamily": "Inter",
                                        "fontSize": "12px",
                                        "marginBottom": "6px",
                                        "cursor": "pointer",
                                    },
                                    inputStyle={"accentColor": GREEN, "marginRight": "8px"},
                                ),
                            ], style=PANEL_CARD),

                            # PANEL 3 — SENTIMENT BAND
                            html.Div([
                                html.Div([
                                    html.Span("SENTIMENT BAND", style={
                                        "fontFamily": "JetBrains Mono",
                                        "fontSize": "10px",
                                        "fontWeight": "500",
                                        "color": TEXT_SECONDARY,
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.12em",
                                    }),
                                ], style={
                                    "marginBottom": "12px",
                                    "paddingBottom": "10px",
                                    "borderBottom": f"1px solid {BORDER}",
                                }),
                                html.Div(
                                    dcc.RadioItems(
                                        id="sentiment-filter",
                                        options=[
                                            {"label": "ALL",     "value": "ALL"},
                                            {"label": "BULL",    "value": "BULL"},
                                            {"label": "NEUTRAL", "value": "NEUTRAL"},
                                            {"label": "BEAR",    "value": "BEAR"},
                                        ],
                                        value="ALL",
                                        inline=False,
                                    ),
                                    className="pill-filter",
                                ),
                            ], style=PANEL_CARD),

                            # PANEL 4 — RETURN HORIZON
                            html.Div([
                                html.Div([
                                    html.Span("HORIZON", style={
                                        "fontFamily": "JetBrains Mono",
                                        "fontSize": "10px",
                                        "fontWeight": "500",
                                        "color": TEXT_SECONDARY,
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.12em",
                                    }),
                                ], style={
                                    "marginBottom": "12px",
                                    "paddingBottom": "10px",
                                    "borderBottom": f"1px solid {BORDER}",
                                }),
                                html.Div(
                                    dcc.RadioItems(
                                        id="return-horizon",
                                        options=[
                                            {"label": "30D", "value": "return_30d_pct"},
                                            {"label": "60D", "value": "return_60d_pct"},
                                            {"label": "90D", "value": "return_90d_pct"},
                                        ],
                                        value="return_30d_pct",
                                        inline=False,
                                    ),
                                    className="pill-filter",
                                ),
                            ], style=PANEL_CARD),

                        ], style={
                            "display": "flex",
                            "gap": "12px",
                            "flexWrap": "wrap",
                            "marginBottom": "28px",
                        }),

                        # ── 02 / Cohort table ────────────────────────────
                        html.Div([
                            html.Div("02 / COHORT TABLE", className="section-eyebrow"),
                            html.P(
                                "Sortable readout of the active cohort. "
                                "Click a row to highlight it in the charts below.",
                                style={
                                    "fontFamily": "Inter", "fontSize": "13px",
                                    "color": TEXT_SECONDARY, "margin": "4px 0 16px 0",
                                }
                            ),
                            html.Div([
                                # Panel header row
                                html.Div([
                                    html.Span(
                                        id="cohort-universe-header",
                                        style={
                                            "fontFamily": "JetBrains Mono",
                                            "fontSize": "11px",
                                            "color": TEXT_PRIMARY,
                                            "letterSpacing": "0.08em",
                                        },
                                    ),
                                    html.Div([
                                        html.Span(
                                            RESEARCH_WINDOW,
                                            style={
                                                "fontFamily": "JetBrains Mono",
                                                "fontSize": "10px",
                                                "color": GREEN,
                                                "border": f"1px solid rgba(46,204,113,0.3)",
                                                "borderRadius": "3px",
                                                "padding": "2px 8px",
                                                "marginRight": "12px",
                                            },
                                        ),
                                        html.Button(
                                            "Show all companies",
                                            id="clear-selection-btn",
                                            n_clicks=0,
                                            style={
                                                "fontFamily": "JetBrains Mono",
                                                "fontSize": "10px",
                                                "color": NEUTRAL,
                                                "background": "none",
                                                "border": "none",
                                                "cursor": "pointer",
                                                "textDecoration": "underline",
                                                "padding": "0",
                                                "letterSpacing": "0.04em",
                                            },
                                        ),
                                    ], style={"display": "flex", "alignItems": "center"}),
                                ], style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "padding": "12px 16px",
                                    "borderBottom": f"1px solid {BORDER}",
                                }),
                                # Table
                                html.Div([
                                    html.Table([
                                        html.Thead(
                                            html.Tr([
                                                html.Th(
                                                    html.Button(
                                                        col_label,
                                                        id={"type": "sort-btn", "index": col_key},
                                                        n_clicks=0,
                                                        className="sort-btn",
                                                    ),
                                                    style={"padding": "0", "whiteSpace": "nowrap"},
                                                )
                                                for col_label, col_key in COHORT_COLS
                                            ], style={
                                                "background": "rgba(46,204,113,0.05)",
                                                "borderBottom": f"1px solid {BORDER}",
                                            }),
                                        ),
                                        html.Tbody(id="cohort-table-body"),
                                    ], style={"width": "100%", "borderCollapse": "collapse"}),
                                ], style={"overflowX": "auto"}),
                            ], style={
                                "background": BG_CARD,
                                "border": f"1px solid {BORDER}",
                                "borderRadius": "8px",
                                "overflow": "hidden",
                                "marginBottom": "28px",
                            }),
                        ]),

                        # ── 03 / Keyword pressure ────────────────────────
                        html.Div([
                            html.Div("03 / KEYWORD PRESSURE", className="section-eyebrow"),
                            html.P(
                                "Weighted frequency of hedging, uncertainty, and "
                                "sentiment-bearing terms across the active cohort's filings.",
                                style={
                                    "fontFamily": "Inter", "fontSize": "13px",
                                    "color": TEXT_SECONDARY, "margin": "4px 0 16px 0",
                                }
                            ),
                            html.Div([
                                html.Div([
                                    html.Span("■", style={
                                        "color": GREEN, "fontSize": "8px", "marginRight": "6px",
                                    }),
                                    html.Span("Positive signal", style={
                                        "fontFamily": "JetBrains Mono", "fontSize": "10px",
                                        "color": GREEN, "marginRight": "20px",
                                    }),
                                    html.Span("■", style={
                                        "color": RED, "fontSize": "8px", "marginRight": "6px",
                                    }),
                                    html.Span("Uncertainty", style={
                                        "fontFamily": "JetBrains Mono", "fontSize": "10px",
                                        "color": RED, "marginRight": "20px",
                                    }),
                                    html.Span("■", style={
                                        "color": NEUTRAL, "fontSize": "8px", "marginRight": "6px",
                                    }),
                                    html.Span("Hedging", style={
                                        "fontFamily": "JetBrains Mono", "fontSize": "10px",
                                        "color": NEUTRAL,
                                    }),
                                ], style={
                                    "padding": "10px 20px",
                                    "borderBottom": f"1px solid {BORDER}",
                                }),
                                html.Div(id="keyword-pills"),
                            ], style={
                                "background": BG_CARD,
                                "border": f"1px solid {BORDER}",
                                "borderRadius": "8px",
                                "overflow": "hidden",
                                "marginBottom": "28px",
                            }),
                        ]),

                        # ── 04 / Post-print drift ────────────────────────
                        html.Div([
                            html.Div("04 / POST-PRINT DRIFT", className="section-eyebrow"),
                            html.P(
                                "Average cumulative return at 30, 60, and 90 days following "
                                "each filing, by selected company.",
                                style={
                                    "fontFamily": "Inter", "fontSize": "13px",
                                    "color": TEXT_SECONDARY, "margin": "4px 0 16px 0",
                                }
                            ),
                            html.Div([

                                # Main drift chart
                                html.Div([
                                    html.Div([
                                        html.Span(
                                            id="drift-chart-header",
                                            style={
                                                "fontFamily": "JetBrains Mono",
                                                "fontSize": "11px",
                                                "fontWeight": "600",
                                                "color": TEXT_PRIMARY,
                                                "letterSpacing": "0.1em",
                                            },
                                        ),
                                    ], style={
                                        "padding": "12px 16px",
                                        "borderBottom": f"1px solid {BORDER}",
                                    }),
                                    dcc.Graph(
                                        id="drift-chart",
                                        config={"displayModeBar": False},
                                        style={"height": "300px"},
                                        figure=go.Figure(layout=dict(
                                            plot_bgcolor=BG_OBSIDIAN,
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            margin=dict(l=48, r=24, t=24, b=40),
                                        )),
                                    ),
                                ], style={
                                    "flex": "1",
                                    "background": BG_CARD,
                                    "border": f"1px solid {BORDER}",
                                    "borderRadius": "8px",
                                    "overflow": "hidden",
                                }),

                                # Cohort pick list sidebar
                                html.Div([
                                    html.Div([
                                        html.Span("COHORT // PICK", style={
                                            "fontFamily": "JetBrains Mono",
                                            "fontSize": "10px",
                                            "fontWeight": "600",
                                            "color": TEXT_SECONDARY,
                                            "letterSpacing": "0.14em",
                                            "textTransform": "uppercase",
                                        }),
                                    ], style={
                                        "padding": "12px 14px",
                                        "borderBottom": f"1px solid {BORDER}",
                                    }),
                                    html.Div(
                                        id="drift-picklist-body",
                                        style={
                                            "overflowY": "auto",
                                            "maxHeight": "280px",
                                        },
                                    ),
                                ], style={
                                    "width": "200px",
                                    "flexShrink": "0",
                                    "background": BG_CARD,
                                    "border": f"1px solid {BORDER}",
                                    "borderRadius": "8px",
                                    "overflow": "hidden",
                                }),

                            ], style={
                                "display": "flex",
                                "gap": "12px",
                                "marginBottom": "28px",
                            }),
                        ]),

                        # ── 05 / Distribution ────────────────────────────
                        html.Div([
                            html.Div("05 / DISTRIBUTION", className="section-eyebrow"),
                            html.P(
                                "Each dot = one filing. X is sentiment polarity, "
                                "Y is 30-day return, size = word count.",
                                style={
                                    "fontFamily": "Inter", "fontSize": "13px",
                                    "color": TEXT_SECONDARY, "margin": "4px 0 16px 0",
                                }
                            ),
                            html.Div([

                                # Scatter panel
                                html.Div([
                                    html.Div([
                                        html.Span("SCATTER // sentiment × return", style={
                                            "fontFamily": "JetBrains Mono", "fontSize": "10px",
                                            "fontWeight": "600", "color": TEXT_SECONDARY,
                                            "letterSpacing": "0.12em", "textTransform": "uppercase",
                                        }),
                                        html.Span(RESEARCH_WINDOW, style={
                                            "fontFamily": "JetBrains Mono", "fontSize": "10px",
                                            "color": GREEN,
                                            "border": "1px solid rgba(46,204,113,0.3)",
                                            "borderRadius": "3px", "padding": "2px 8px",
                                        }),
                                    ], style={
                                        "display": "flex", "justifyContent": "space-between",
                                        "alignItems": "center", "padding": "10px 14px",
                                        "borderBottom": f"1px solid {BORDER}",
                                    }),
                                    dcc.Graph(id="dist-scatter", config={"displayModeBar": False},
                                              style={"height": "340px"}),
                                ], style={
                                    "flex": "1", "background": BG_CARD,
                                    "border": f"1px solid {BORDER}", "borderRadius": "8px",
                                    "overflow": "hidden",
                                }),

                                # Histogram panel
                                html.Div([
                                    html.Div([
                                        html.Span("HISTOGRAM // tone polarity", style={
                                            "fontFamily": "JetBrains Mono", "fontSize": "10px",
                                            "fontWeight": "600", "color": TEXT_SECONDARY,
                                            "letterSpacing": "0.12em", "textTransform": "uppercase",
                                        }),
                                        html.Span(RESEARCH_WINDOW, style={
                                            "fontFamily": "JetBrains Mono", "fontSize": "10px",
                                            "color": GREEN,
                                            "border": "1px solid rgba(46,204,113,0.3)",
                                            "borderRadius": "3px", "padding": "2px 8px",
                                        }),
                                    ], style={
                                        "display": "flex", "justifyContent": "space-between",
                                        "alignItems": "center", "padding": "10px 14px",
                                        "borderBottom": f"1px solid {BORDER}",
                                    }),
                                    dcc.Graph(id="polarity-hist", config={"displayModeBar": False},
                                              style={"height": "340px"}),
                                ], style={
                                    "flex": "1", "background": BG_CARD,
                                    "border": f"1px solid {BORDER}", "borderRadius": "8px",
                                    "overflow": "hidden",
                                }),

                            ], style={"display": "flex", "gap": "12px", "marginBottom": "28px"}),
                        ]),

                        # Feature selector (what to measure)
                        html.Div([
                            html.Div("06 / MEASURE", className="section-eyebrow"),
                            html.Div([
                                html.Label("Linguistic Feature", style={
                                    "fontFamily": "Inter",
                                    "fontSize": "11px",
                                    "fontWeight": "600",
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
                                        "maxWidth": "320px",
                                    },
                                ),
                            ]),
                        ], style={"marginBottom": "24px"}),

                        # Dynamic metric cards
                        html.Div(id="stats-bar", style={
                            "display": "flex",
                            "gap": "12px",
                            "marginBottom": "28px",
                            "flexWrap": "wrap",
                        }),

                        html.Hr(style={
                            "border": "none",
                            "borderTop": "1px solid rgba(46,204,113,0.1)",
                            "margin": "0 0 20px 0",
                        }),

                        html.Div("07 / RESULTS", className="section-eyebrow"),

                        # Charts
                        html.Div([
                            html.Div([
                                dcc.Graph(id="scatter-plot", style={"height": "380px"}),
                            ], style={
                                "flex": "1",
                                "background": BG_CARD,
                                "border": f"1px solid {BORDER}",
                                "borderRadius": "8px",
                                "padding": "8px",
                            }),
                            html.Div([
                                dcc.Graph(id="time-series", style={"height": "380px"}),
                            ], style={
                                "flex": "1",
                                "background": BG_CARD,
                                "border": f"1px solid {BORDER}",
                                "borderRadius": "8px",
                                "padding": "8px",
                            }),
                        ], style={"display": "flex", "gap": "16px", "marginBottom": "40px"}),

                    ], className="tab-fade-in")],
                ),

                # ── TAB 3: Findings ──────────────────────────────────────
                dcc.Tab(
                    label="Findings",
                    value="findings",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[html.Div([

                        html.Div("04 / FINDINGS", className="section-eyebrow",
                                 style={"marginTop": "28px"}),

                        html.Div([
                            build_findings_table(FINDINGS),
                        ], style={
                            "background": BG_CARD,
                            "border": f"1px solid {BORDER}",
                            "borderRadius": "8px",
                            "padding": "16px 20px",
                            "marginBottom": "40px",
                        }),

                    ], className="tab-fade-in")],
                ),

            ],
            style={"borderBottom": f"1px solid {BORDER}"},
            colors={"border": BORDER, "primary": GREEN, "background": BG_CARD},
        ),
    ], style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "0 24px 40px 24px",
    }),

], style={
    "background": BG_OBSIDIAN,
    "minHeight": "100vh",
    "color": TEXT_PRIMARY,
    "fontFamily": "Inter, sans-serif",
})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("cohort-sort", "data"),
    Input({"type": "sort-btn", "index": ALL}, "n_clicks"),
    State("cohort-sort", "data"),
    prevent_initial_call=True,
)
def update_sort(_, current):
    triggered = ctx.triggered_id
    if not triggered:
        return current
    col = triggered["index"]
    if current["col"] == col:
        new_dir = "asc" if current["dir"] == "desc" else "desc"
    else:
        new_dir = "desc"
    return {"col": col, "dir": new_dir}


@app.callback(
    Output("selected-ticker", "data"),
    Input({"type": "cohort-row", "index": ALL}, "n_clicks"),
    Input({"type": "pick-row",   "index": ALL}, "n_clicks"),
    Input("clear-selection-btn", "n_clicks"),
    State("selected-ticker", "data"),
    prevent_initial_call=True,
)
def update_selected_ticker(row_clicks, pick_clicks, clear_clicks, current):
    # Guard: Dash fires when dynamic components are added to the DOM (all n_clicks=0).
    # Only act when at least one trigger has value > 0 (a real user click).
    if not ctx.triggered:
        return no_update
    if all((t["value"] or 0) == 0 for t in ctx.triggered):
        return no_update
    triggered = ctx.triggered_id
    if not triggered:
        return no_update
    if triggered == "clear-selection-btn":
        return None
    ticker = triggered["index"]
    return None if current == ticker else ticker


@app.callback(
    Output("cohort-table-body", "children"),
    Output("cohort-universe-header", "children"),
    Input({"type": "company-filter", "index": ALL}, "value"),
    Input("mktcap-filter",    "value"),
    Input("sentiment-filter", "value"),
    Input("cohort-sort",      "data"),
    Input("selected-ticker",  "data"),
)
def update_cohort_table(company_selections, mktcap_tiers, sentiment, sort, selected_ticker):
    selected_tickers = [t for g in (company_selections or []) if g for t in g]
    mask = pd.Series(True, index=df.index)
    if selected_tickers:
        mask &= df["ticker"].isin(selected_tickers)
    if mktcap_tiers:
        mask &= df["market_cap_tier"].isin(mktcap_tiers) | (df["market_cap_tier"] == "Unknown")
    if sentiment and sentiment != "ALL":
        mask &= df["sentiment_band"] == sentiment
    filtered = df[mask]

    tbl = build_cohort_df(filtered)
    sort_col = sort.get("col", "drift_30d") if sort else "drift_30d"
    sort_dir = sort.get("dir", "desc") if sort else "desc"
    rows = build_cohort_rows(tbl, sort_col, sort_dir, selected_ticker)
    header = f"UNIVERSE // {len(tbl)} SYMBOLS"
    return rows, header


@app.callback(
    Output("drift-chart",        "figure"),
    Output("drift-chart-header", "children"),
    Output("drift-picklist-body","children"),
    Input({"type": "company-filter", "index": ALL}, "value"),
    Input("mktcap-filter",    "value"),
    Input("sentiment-filter", "value"),
    Input("selected-ticker",  "data"),
)
def update_drift_section(company_selections, mktcap_tiers, sentiment, selected_ticker):
    # Build cohort mask (no ticker drill-down here — picklist shows the full cohort)
    selected_tickers = [t for g in (company_selections or []) if g for t in g]
    mask = pd.Series(True, index=df.index)
    if selected_tickers:
        mask &= df["ticker"].isin(selected_tickers)
    if mktcap_tiers:
        mask &= df["market_cap_tier"].isin(mktcap_tiers) | (df["market_cap_tier"] == "Unknown")
    if sentiment and sentiment != "ALL":
        mask &= df["sentiment_band"] == sentiment
    cohort = df[mask]

    # Pick list rows (all cohort tickers, not drill-down)
    latest = cohort.sort_values("filed_at").groupby("ticker").last().reset_index()
    pick_rows = []
    for _, row in latest.sort_values("ticker").iterrows():
        ticker    = row["ticker"]
        fp        = row.get("finbert_positive", 0) or 0
        fn        = row.get("finbert_negative", 0) or 0
        sent      = fp - fn
        drift_raw = row.get("return_30d_pct")
        drift     = float(drift_raw) if pd.notna(drift_raw) else None
        is_sel    = ticker == selected_ticker
        sent_col  = GREEN if sent >= 0 else RED
        drift_col = GREEN if (drift or 0) >= 0 else RED
        drift_str = f"{drift:+.1f}%" if drift is not None else "—"
        pick_rows.append(html.Div(
            [
                html.Span(ticker, style={
                    "fontFamily": "Inter", "fontWeight": "700",
                    "fontSize": "11px", "color": TEXT_PRIMARY,
                    "width": "44px", "display": "inline-block",
                }),
                html.Span(f"{sent:+.2f}", style={
                    "fontFamily": "JetBrains Mono", "fontSize": "10px",
                    "color": sent_col, "width": "50px", "display": "inline-block",
                }),
                html.Span(drift_str, style={
                    "fontFamily": "JetBrains Mono", "fontSize": "10px",
                    "color": drift_col,
                }),
            ],
            id={"type": "pick-row", "index": ticker},
            n_clicks=0,
            style={
                "padding": "8px 14px",
                "borderBottom": f"1px solid {BORDER}",
                "cursor": "pointer",
                "background": "rgba(46,204,113,0.06)" if is_sel else "transparent",
                "transition": "background 120ms ease",
                "display": "flex",
                "alignItems": "center",
                "gap": "4px",
            },
            className="pick-row",
        ))

    # Drift figure — apply ticker drill-down if set
    if selected_ticker and selected_ticker in cohort["ticker"].values:
        subset   = cohort[cohort["ticker"] == selected_ticker]
        hdr_name = selected_ticker
    else:
        subset   = cohort
        hdr_name = "ALL"

    def _safe_mean(series):
        v = series.dropna().mean()
        return float(v) if pd.notna(v) else None

    avg_30 = _safe_mean(subset["return_30d_pct"])
    avg_60 = _safe_mean(subset["return_60d_pct"])
    avg_90 = _safe_mean(subset["return_90d_pct"])

    base_layout = dict(
        **PLOT_LAYOUT,
        margin=dict(l=56, r=24, t=16, b=48),
        xaxis=dict(
            **AXIS_DEFAULTS,
            title=None,
            categoryorder="array",
            categoryarray=["30D", "60D", "90D"],
        ),
        yaxis=dict(
            **AXIS_DEFAULTS,
            title="Avg Return (%)",
            tickformat="+.1f",
            ticksuffix="%",
            zeroline=True,
        ),
        bargap=0.35,
    )

    fig = go.Figure()

    if avg_30 is None:
        fig.update_layout(
            **base_layout,
            annotations=[dict(
                text="No return data available for this cohort",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor="center", yanchor="middle", showarrow=False,
                font=dict(family="Inter", size=12, color=NEUTRAL),
            )],
        )
        return fig, f"DRIFT.{hdr_name}", pick_rows

    # Build bars — only for horizons that have actual data
    all_horizons = [("30D", avg_30), ("60D", avg_60), ("90D", avg_90)]
    available    = [(lbl, v) for lbl, v in all_horizons if v is not None]
    missing      = [lbl for lbl, v in all_horizons if v is None]

    x_labels = [p[0] for p in available]
    y_values = [p[1] for p in available]
    colors   = [GREEN if v >= 0 else RED for v in y_values]

    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_values,
        marker_color=colors,
        marker_opacity=0.85,
        marker_line=dict(width=0),
        text=[f"{v:+.2f}%" for v in y_values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color=TEXT_PRIMARY),
        hovertemplate="%{x}: <b>%{y:+.2f}%</b><extra></extra>",
    ))

    n = subset["filed_at"].nunique()
    annotations = [dict(
        text=f"n = {n} filings",
        xref="paper", yref="paper",
        x=1, y=1.08, xanchor="right", yanchor="top",
        showarrow=False,
        font=dict(family="JetBrains Mono", size=10, color=NEUTRAL),
    )]
    if missing:
        annotations.append(dict(
            text=f"{', '.join(missing)} unavailable — run price pipeline",
            xref="paper", yref="paper",
            x=0, y=1.08, xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(family="JetBrains Mono", size=9, color=NEUTRAL),
        ))

    fig.update_layout(**base_layout, annotations=annotations)
    return fig, f"DRIFT.{hdr_name}", pick_rows


@app.callback(
    Output("dist-scatter",  "figure"),
    Output("polarity-hist", "figure"),
    Input({"type": "company-filter", "index": ALL}, "value"),
    Input("mktcap-filter",    "value"),
    Input("sentiment-filter", "value"),
    Input("selected-ticker",  "data"),
)
def update_distribution(company_selections, mktcap_tiers, sentiment, selected_ticker):
    selected_tickers = [t for g in (company_selections or []) if g for t in g]
    mask = pd.Series(True, index=df.index)
    if selected_tickers:
        mask &= df["ticker"].isin(selected_tickers)
    if mktcap_tiers:
        mask &= df["market_cap_tier"].isin(mktcap_tiers) | (df["market_cap_tier"] == "Unknown")
    if sentiment and sentiment != "ALL":
        mask &= df["sentiment_band"] == sentiment
    if selected_ticker and selected_ticker in df.loc[mask, "ticker"].values:
        mask &= df["ticker"] == selected_ticker
    filtered = df[mask].dropna(subset=["polarity", "return_30d_pct"])

    empty_layout = dict(
        **PLOT_LAYOUT,
        margin=dict(l=48, r=16, t=16, b=48),
        xaxis=dict(**AXIS_DEFAULTS),
        yaxis=dict(**AXIS_DEFAULTS),
    )

    if filtered.empty:
        return go.Figure(layout=empty_layout), go.Figure(layout=empty_layout)

    # ── Scatter ──────────────────────────────────────────────────────────────
    wc       = filtered["word_count"].fillna(filtered["word_count"].median())
    wc_min, wc_max = wc.min(), wc.max()
    wc_range = (wc_max - wc_min) if wc_max > wc_min else 1
    sizes    = 5 + ((wc - wc_min) / wc_range * 18)
    colors   = [GREEN if p > 0 else RED for p in filtered["polarity"]]

    scatter = go.Figure()
    # Quadrant lines
    scatter.add_vline(x=0, line=dict(color=NEUTRAL, width=1, dash="dash"))
    scatter.add_hline(y=0, line=dict(color=NEUTRAL, width=1, dash="dash"))
    scatter.add_trace(go.Scatter(
        x=filtered["polarity"],
        y=filtered["return_30d_pct"],
        mode="markers",
        marker=dict(
            color=colors, size=sizes,
            opacity=0.75,
            line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
        ),
        customdata=filtered[["ticker", "word_count"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Polarity: %{x:.3f}<br>"
            "30D Return: %{y:+.1f}%<br>"
            "Word count: %{customdata[1]:,}<extra></extra>"
        ),
        showlegend=False,
    ))
    scatter.update_layout(
        **PLOT_LAYOUT,
        margin=dict(l=48, r=16, t=16, b=48),
        hovermode="closest",
        xaxis=dict(**AXIS_DEFAULTS, title="Sentiment Polarity  (positive − negative)",
                   range=[-1, 1], zeroline=False),
        yaxis=dict(**AXIS_DEFAULTS, title="30D Return (%)",
                   tickformat="+.1f", ticksuffix="%", zeroline=False),
    )

    # ── Histogram ────────────────────────────────────────────────────────────
    pol = filtered["polarity"].values
    counts, edges = np.histogram(pol, bins=8)
    centers = (edges[:-1] + edges[1:]) / 2

    # Gradient: red for negative bins, green for positive; intensity by distance from 0
    max_abs = max(abs(centers.min()), abs(centers.max())) or 1
    bar_colors = []
    for c in centers:
        t = abs(c) / max_abs        # 0 = near zero, 1 = extreme
        if c < 0:
            r = int(180 + t * 51)   # 180→231
            bar_colors.append(f"rgba({r},76,60,0.85)")
        else:
            g = int(150 + t * 54)   # 150→204
            bar_colors.append(f"rgba(46,{g},113,0.85)")

    hist = go.Figure()
    hist.add_trace(go.Bar(
        x=[f"{c:.2f}" for c in centers],
        y=counts,
        marker_color=bar_colors,
        marker_line=dict(width=0),
        hovertemplate="Polarity %{x}: <b>%{y} filings</b><extra></extra>",
    ))
    hist.update_layout(
        **PLOT_LAYOUT,
        margin=dict(l=48, r=16, t=16, b=48),
        bargap=0.08,
        xaxis=dict(**AXIS_DEFAULTS, title="Sentiment Polarity"),
        yaxis=dict(**AXIS_DEFAULTS, title="Filings"),
    )

    return scatter, hist


@app.callback(
    Output("scatter-plot",   "figure"),
    Output("time-series",    "figure"),
    Output("stats-bar",      "children"),
    Output("sector-counter", "children"),
    Output("mktcap-counter", "children"),
    Output("keyword-pills",  "children"),
    Input({"type": "company-filter", "index": ALL}, "value"),
    Input("mktcap-filter",   "value"),
    Input("sentiment-filter","value"),
    Input("feature-select",  "value"),
    Input("return-horizon",  "value"),
    Input("selected-ticker", "data"),
)
def update_charts(company_selections, mktcap_tiers, sentiment, feature, return_col,
                  selected_ticker):
    # Flatten all checked tickers across all sector groups
    selected_tickers = [t for group in (company_selections or []) if group for t in group]
    total_companies = sum(len(v) for v in SECTOR_TICKERS.values())
    sector_counter = f"{len(selected_tickers)}/{total_companies}"

    n_tiers = len(mktcap_tiers) if mktcap_tiers else 0
    total_tiers = len(ALL_MKTCAP_TIERS)
    mktcap_counter = f"{n_tiers}/{total_tiers}"

    # Apply AND filters
    mask = pd.Series(True, index=df.index)
    if selected_tickers:
        mask &= df["ticker"].isin(selected_tickers)
    if mktcap_tiers:
        mask &= df["market_cap_tier"].isin(mktcap_tiers) | (df["market_cap_tier"] == "Unknown")
    if sentiment and sentiment != "ALL":
        mask &= df["sentiment_band"] == sentiment
    # Row drill-down overrides cohort filters for charts.
    # Only apply if ticker actually exists in the current cohort (guards spurious store values).
    if selected_ticker and selected_ticker in df.loc[mask, "ticker"].values:
        mask &= df["ticker"] == selected_ticker

    filtered = df[mask].copy()

    if filtered.empty:
        empty = go.Figure()
        empty.update_layout(**PLOT_LAYOUT)
        empty_cards = [
            build_metric_card("Spearman r", "--"),
            build_metric_card("P-Value",    "--"),
            build_metric_card("Win Rate",   "--"),
            build_metric_card("N",          "0"),
        ]
        return empty, empty, empty_cards, sector_counter, mktcap_counter, html.Div()

    feature_label = FEATURES[feature]
    return_label = (
        return_col.replace("_pct", "")
                  .replace("return_", "")
                  .replace("d", " day return %")
    )

    TRANSITION = dict(duration=400, easing="cubic-in-out")

    visible_tickers = sorted(filtered["ticker"].unique())

    # Scatter plot — per-ticker dots + one overall trend line
    scatter = go.Figure()
    for idx, ticker in enumerate(visible_tickers):
        t_df = filtered[filtered["ticker"] == ticker]
        color = CHART_COLORS[idx % len(CHART_COLORS)]
        scatter.add_trace(go.Scatter(
            x=t_df[feature],
            y=t_df[return_col],
            mode="markers",
            name=ticker,
            marker=dict(color=color, size=8, opacity=0.85,
                        line=dict(width=1, color="rgba(255,255,255,0.2)")),
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
        m, b, r, *_ = stats.linregress(clean[feature], clean[return_col])
        x0, x1 = clean[feature].min(), clean[feature].max()
        trend_rgba = "rgba(46,204,113,0.6)" if r > 0 else "rgba(231,76,60,0.6)"
        scatter.add_trace(go.Scatter(
            x=[x0, x1], y=[m * x0 + b, m * x1 + b],
            mode="lines",
            name=f"Trend (r={r:.2f})",
            line=dict(color=trend_rgba, width=1.5, dash="dash"),
        ))

    scatter.update_layout(
        title=dict(
            text=f"{feature_label} vs {return_label.title()}",
            font=dict(family="Inter", size=13, color=TEXT_PRIMARY),
        ),
        xaxis=axis(feature_label),
        yaxis=axis("Return (%)", tickformat=".1f", ticksuffix="%", zeroline=True),
        margin=dict(l=56, r=24, t=52, b=64),
        hovermode="closest",
        transition=TRANSITION,
        **PLOT_LAYOUT,
    )

    # Time series — per-ticker lines
    time_fig = go.Figure()
    for idx, ticker in enumerate(visible_tickers):
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
            font=dict(family="Inter", size=13, color=TEXT_PRIMARY),
        ),
        xaxis=dict(title="Filing Date", **AXIS_DEFAULTS),
        yaxis=dict(title=feature_label, **AXIS_DEFAULTS),
        margin=dict(l=56, r=24, t=52, b=64),
        hovermode="x unified",
        transition=TRANSITION,
        **PLOT_LAYOUT,
    )

    # Metric cards
    clean = filtered[[feature, return_col]].dropna()
    corr, pvalue = (
        stats.spearmanr(clean[feature], clean[return_col])
        if len(clean) > 3 else (0, 1)
    )
    median = clean[feature].median() if len(clean) > 0 else 0
    correct = (
        ((clean[feature] > median) & (clean[return_col] > 0))
        | ((clean[feature] <= median) & (clean[return_col] <= 0))
    ).sum()
    win_rate = (correct / len(clean) * 100) if len(clean) > 0 else 0
    p_color = GREEN if pvalue < 0.05 else RED

    metric_cards = [
        build_metric_card("Spearman r", f"{corr:.4f}", GREEN if corr > 0 else RED),
        build_metric_card("P-Value",    f"{pvalue:.4f}", p_color),
        build_metric_card("Win Rate",   f"{win_rate:.1f}%"),
        build_metric_card("N",          str(len(clean))),
    ]

    # Keyword pressure — uses cohort-level filter (no ticker drill-down)
    cohort_texts = df[
        df["ticker"].isin(filtered["ticker"].unique())
    ]["prepared_remarks"].dropna().tolist()
    freq = keyword_frequency(cohort_texts)
    kw_pills = build_keyword_pills(freq)

    return scatter, time_fig, metric_cards, sector_counter, mktcap_counter, kw_pills



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
