"""leaklab.views - presentation layer.

Deliberately bespoke and restrained: a flat, high-density light UI with a
limited slate/ink palette, a system sans-serif stack, hairline dividers instead
of floating cards, and inline SVG icons in place of emoji. No gradients, no
neon accents, no decorative chrome.

Business logic stays in the other ``leaklab`` modules; this file only renders.
"""
from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from .data import EXCLUDED_FIELDS
from .metrics import Metric, format_inr
from .rules import Alert, RuleStatus

# --------------------------------------------------------------------------- #
# Palette (limited, professional, no neon)
# --------------------------------------------------------------------------- #
INK = "#161c24"        # primary text
SUB = "#5b6672"        # secondary text
MUTED = "#8a94a0"      # tertiary text
LINE = "#e4e7eb"       # hairline
LINE_SOFT = "#eef1f4"  # chart grid
ACCENT = "#22374f"     # brand / primary
CHART = "#4b6b8a"      # single data series
THRESH = "#9c2f2f"     # threshold reference
SEV = {"High": "#9c2f2f", "Medium": "#8a6412", "Low": "#33566f"}
BASIS = {"Observed": "#1f6b4a", "Estimated": "#8a6412", "Unavailable": "#6b7280"}


# --------------------------------------------------------------------------- #
# Inline SVG icon set (functional glyphs only)
# --------------------------------------------------------------------------- #
_ICONS = {
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13.5"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "database": '<ellipse cx="12" cy="5" rx="8.5" ry="3"/><path d="M20.5 12c0 1.66-3.8 3-8.5 3s-8.5-1.34-8.5-3"/><path d="M3.5 5v14c0 1.66 3.8 3 8.5 3s8.5-1.34 8.5-3V5"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8.5" y1="13" x2="15.5" y2="13"/><line x1="8.5" y1="17" x2="13" y2="17"/>',
    "trend": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "bars": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="15"/>',
    "info": '<circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16.5"/><line x1="12" y1="7.5" x2="12.01" y2="7.5"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "dot": '<circle cx="12" cy="12" r="6"/>',
    "search": '<circle cx="11" cy="11" r="7.5"/><line x1="21" y1="21" x2="16.7" y2="16.7"/>',
}


def icon(name: str, size: int = 15, color: str = SUB, stroke: float = 1.9) -> str:
    body = _ICONS.get(name, _ICONS["dot"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
        f'style="flex:0 0 auto;vertical-align:-2px">{body}</svg>'
    )


def _esc(v) -> str:
    return html.escape(str(v), quote=True)


def _dot(color: str, size: int = 7) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 8 8" aria-hidden="true" style="flex:0 0 auto">'
            f'<circle cx="4" cy="4" r="4" fill="{color}"/></svg>')


# --------------------------------------------------------------------------- #
# Stylesheet
# --------------------------------------------------------------------------- #
_CSS = """
<style>
:root{
  --ink:#161c24; --sub:#5b6672; --muted:#8a94a0;
  --line:#e4e7eb; --soft:#eef1f4; --accent:#22374f; --panel:#f7f8fa;
}
html, body, .stApp{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",
    Arial,"Noto Sans",sans-serif;
}
/* Restore Streamlit's icon font (do not let the app font stack hijack it). */
[class*="material-symbols"], [class*="material-icons"]{
  font-family:"Material Symbols Rounded","Material Symbols Outlined","Material Icons" !important;
}
.stApp{ background:#ffffff; color:var(--ink); }
header[data-testid="stHeader"]{ background:transparent; height:0; }
/* Chrome stripped, but Streamlit hosts the collapsed-sidebar expand control
   inside the toolbar. Keep the toolbar hidden normally (no chrome) and reveal
   it only while the sidebar is collapsed, so the control is reachable. */
[data-testid="stToolbar"], [data-testid="stStatusWidget"], #MainMenu, footer{ display:none !important; }
.stApp:has([data-testid="stSidebar"][data-collapsed="true"]) [data-testid="stToolbar"]{ display:flex !important; }
[data-testid="stExpandSidebarButton"]{
  position:fixed !important; top:2px !important; left:3px !important; z-index:1000002;
  color:var(--sub);
}
[data-testid="stExpandSidebarButton"]:hover{ color:var(--ink); }
/* The S-key listener is mounted in an off-flow iframe: kept alive (script runs)
   but removed from layout so it adds no whitespace or gap. */
div[data-testid="stElementContainer"]:has(> [data-testid="stIFrame"]){
  position:absolute !important; left:-9999px !important; top:0 !important;
  width:1px !important; height:1px !important; margin:0 !important; padding:0 !important;
  overflow:hidden !important; pointer-events:none !important;
}
.block-container{ padding:1.1rem 2.1rem 3rem 2.1rem; max-width:1560px; }

/* Sidebar: flat, hairline-separated */
[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child{ padding-top:1rem; }
[data-testid="stSidebar"] .stRadio > label, [data-testid="stSidebar"] label{ font-size:.82rem; color:var(--sub); }
[data-testid="stSidebar"] hr{ margin:.7rem 0; }

h1,h2,h3,h4,h5{ color:var(--ink); letter-spacing:-.01em; font-weight:600; }
hr{ border:0; border-top:1px solid var(--line); margin:.55rem 0; }

/* ---- brand, headers ---- */
.ll-brand{ display:flex; align-items:center; gap:.55rem; font-weight:600; font-size:1rem; color:var(--ink); }
.ll-brand small{ display:block; font-weight:400; font-size:.72rem; color:var(--muted); letter-spacing:.01em; }
.ll-head{ display:flex; justify-content:space-between; align-items:flex-start; gap:1.25rem; margin:.1rem 0 .5rem; }
.ll-head-main{ flex:1 1 auto; min-width:0; }
.ll-h1{ font-size:1.32rem; font-weight:650; letter-spacing:-.015em; color:var(--ink); line-height:1.15; word-wrap:break-word; }
.ll-sub{ font-size:.8rem; color:var(--sub); margin-top:.15rem; line-height:1.35; white-space:normal; word-wrap:break-word; }
.ll-chip{ flex:0 0 auto; align-self:flex-start; display:inline-block; font-size:.7rem; color:var(--sub);
  background:#fff; border:1px solid var(--line); border-radius:4px; padding:.22rem .5rem;
  white-space:normal; word-break:break-word; text-align:right; max-width:22rem; }
.ll-sec{ display:flex; align-items:center; gap:.42rem; font-size:.72rem; font-weight:600;
  letter-spacing:.07em; text-transform:uppercase; color:var(--sub); margin:.15rem 0 .45rem; }

/* ---- KPI strip: gap-as-hairline (no doubled borders at any wrap count) ---- */
.ll-strip{ display:flex; flex-wrap:wrap; gap:1px; background:var(--line);
  border:1px solid var(--line); border-radius:7px; overflow:hidden; margin:.35rem 0 .1rem; }
.ll-kpi{ flex:1 1 150px; min-width:130px; background:#fff; padding:.6rem .85rem; }
.ll-kpi-l{ font-size:.68rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.ll-kpi-v{ font-size:1.18rem; font-weight:600; color:var(--ink); margin-top:.12rem; line-height:1.2; }
.ll-kpi-s{ font-size:.66rem; color:var(--muted); margin-top:.1rem; }

/* ---- detail header ---- */
.ll-dhead{ display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }
.ll-dtitle{ font-size:1.06rem; font-weight:600; color:var(--ink); }
.ll-dmeta{ display:flex; align-items:center; gap:.6rem; font-size:.78rem; color:var(--sub); }

/* ---- badges ---- */
.ll-badge{ display:inline-flex; align-items:center; gap:.3rem; font-size:.66rem; font-weight:600;
  letter-spacing:.04em; text-transform:uppercase; padding:.12rem .42rem; border-radius:4px; border:1px solid; }
.ll-sev{ display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem; font-weight:600; }

/* ---- field grid ---- */
.ll-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:.1rem .2rem; border-top:1px solid var(--line); margin:.55rem 0 .35rem; }
.ll-cell{ padding:.45rem .1rem .4rem; }
.ll-k{ font-size:.66rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.ll-v{ font-size:.84rem; color:var(--ink); margin-top:.1rem; }

/* ---- inset panel (functional, not a floating card) ---- */
.ll-inset{ border-left:3px solid var(--accent); background:var(--panel);
  border-radius:0 6px 6px 0; padding:.55rem .75rem; margin:.2rem 0 .5rem; }
.ll-inset-t{ font-size:.68rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em;
  color:var(--sub); display:flex; align-items:center; gap:.35rem; margin-bottom:.3rem; }
.ll-inset-b{ font-size:.86rem; color:var(--ink); line-height:1.4; }
.ll-inset-m{ font-size:.74rem; color:var(--sub); margin-top:.35rem; }

.ll-impact{ font-size:1.35rem; font-weight:600; color:var(--ink); letter-spacing:-.01em; }
.ll-meta-line{ font-size:.76rem; color:var(--sub); margin-top:.15rem; line-height:1.35; white-space:normal; word-wrap:break-word; }
.ll-hint{ font-size:.68rem; color:var(--muted); margin-top:.7rem; line-height:1.4; white-space:normal; }
.ll-hint kbd{ font-family:inherit; font-size:.66rem; color:var(--sub); background:#fff;
  border:1px solid var(--line); border-bottom-width:2px; border-radius:3px; padding:0 .3rem; }
.ll-note{ display:flex; align-items:center; gap:.4rem; font-size:.78rem; color:var(--muted);
  padding:.35rem 0; }
.ll-bar{ height:5px; background:var(--soft); border-radius:3px; overflow:hidden; margin:.35rem 0 .3rem; }
.ll-bar > i{ display:block; height:100%; background:var(--accent); }
.ll-parts{ font-size:.75rem; color:var(--sub); line-height:1.55; }
.ll-parts div{ padding-left:.7rem; position:relative; }
.ll-parts div:before{ content:"·"; position:absolute; left:.15rem; color:var(--muted); }

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"]{ gap:1.6rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ padding:.36rem 0; font-size:.86rem; font-weight:500; color:var(--sub); }
.stTabs [aria-selected="true"]{ color:var(--accent) !important; font-weight:600; }
.stTabs [data-baseweb="tab-highlight"]{ background-color:var(--accent); height:2px; }

/* ---- widgets ---- */
.stButton > button{ border-radius:5px; border:1px solid var(--line); font-weight:500;
  font-size:.82rem; padding:.3rem .7rem; color:var(--ink); background:#fff; }
.stButton > button:hover{ border-color:var(--accent); color:var(--accent); }
.stButton > button[kind="primary"]{ background:var(--accent); border-color:var(--accent); color:#fff; }
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:6px; }
[data-testid="stExpander"] summary{ font-size:.82rem; font-weight:500; }
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"]{ font-size:.78rem; }
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:6px; }
[data-testid="stDataFrame"] table{ table-layout:fixed; word-wrap:break-word; }
[data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td{ overflow:hidden; text-overflow:ellipsis; }
.stRadio [role="radiogroup"]{ gap:.15rem; }
[data-testid="stMetricValue"]{ font-size:1.2rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def brand() -> None:
    mark = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 32 32" '
        'aria-hidden="true" style="flex:0 0 auto">'
        '<rect width="32" height="32" rx="7" fill="#22374f"/>'
        '<path d="M16 6.2c4.9 5.9 7.6 9.3 7.6 13.4a7.6 7.6 0 1 1-15.2 0c0-4.1 2.7-7.5 7.6-13.4z" fill="#ffffff"/>'
        "</svg>"
    )
    st.sidebar.markdown(
        f'<div class="ll-brand">{mark}<span>LeakLens<small>Leakage decision system · Round 2</small></span></div>',
        unsafe_allow_html=True,
    )


def sidebar_hotkey() -> None:
    """Bind the ``S`` key to collapse/expand the sidebar.

    Runs client-side inside a zero-height component iframe, so toggling the
    panel does not trigger a Streamlit rerun. The listener is attached once to
    the parent window and guarded by a flag, so reruns never stack duplicates.
    """
    st.iframe(
        """
        <script>
        (function () {
          var doc, win;
          try { doc = window.parent.document; win = window.parent; }
          catch (e) { return; }              /* cross-origin: do nothing */
          if (win.__llSidebarHotkey) return;
          win.__llSidebarHotkey = true;

          function toggle() {
            var sb = doc.querySelector('[data-testid="stSidebar"]');
            var collapsed = sb && sb.getAttribute('data-collapsed') === 'true';
            var btn;
            if (collapsed) {
              btn = doc.querySelector('[data-testid="stExpandSidebarButton"]');
            } else {
              /* the collapse control is a wrapper; its inner <button> is clickable */
              btn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
                 || doc.querySelector('button[aria-label="Collapse sidebar"]');
            }
            if (btn) btn.click();
          }
          doc.addEventListener('keydown', function (e) {
            if (e.key !== 's' && e.key !== 'S') return;
            if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
            var t = e.target || {};
            var tag = (t.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea' || tag === 'select' || t.isContentEditable) return;
            toggle();
            e.preventDefault();
          }, true);
        })();
        </script>
        """,
        height=1,
    )


def header(title: str, subtitle: str, badge: str = "") -> None:
    chip = f'<span class="ll-chip">{_esc(badge)}</span>' if badge else ""
    st.markdown(
        f'<div class="ll-head">'
        f'<div class="ll-head-main">'
        f'<div class="ll-h1">{_esc(title)}</div>'
        f'<div class="ll-sub">{_esc(subtitle)}</div>'
        f"</div>{chip}</div>",
        unsafe_allow_html=True,
    )


def section(text: str, ic: str | None = None) -> None:
    pre = icon(ic, color=SUB) if ic else ""
    st.markdown(f'<div class="ll-sec">{pre}{_esc(text)}</div>', unsafe_allow_html=True)


def note(msg: str) -> None:
    st.markdown(f'<div class="ll-note">{icon("info", color=MUTED)} {_esc(msg)}</div>',
                unsafe_allow_html=True)


def _basis_badge(basis: str) -> str:
    c = BASIS.get(basis, MUTED)
    return (f'<span class="ll-badge" style="color:{c};border-color:{c}55;background:{c}0d">'
            f'{_dot(c, 6)}{_esc(basis)}</span>')


def _sev(sev: str) -> str:
    c = SEV.get(sev, MUTED)
    return f'<span class="ll-sev" style="color:{c}">{_dot(c, 7)}{_esc(sev)}</span>'


# --------------------------------------------------------------------------- #
# KPI strip
# --------------------------------------------------------------------------- #
def kpi_strip(metrics: dict[str, Metric], alerts: list[Alert]) -> None:
    specs = [
        ("Net realized revenue", "realized_revenue"),
        ("Cancellation rate", "cancellation_rate"),
        ("Return rate (fulfilled)", "return_rate"),
        ("Discount rate", "discount_rate"),
        ("Products at reorder point", "low_stock_count"),
    ]
    cells = []
    for label, key in specs:
        m = metrics.get(key)
        val = m.display if m else "Unavailable"
        sub = m.availability if m else ""
        tip = f"{m.definition} Formula: {m.formula}".strip() if m else ""
        cells.append(
            f'<div class="ll-kpi" title="{_esc(tip)}">'
            f'<div class="ll-kpi-l">{_esc(label)}</div>'
            f'<div class="ll-kpi-v">{_esc(val)}</div>'
            f'<div class="ll-kpi-s">{_esc(sub)}</div></div>'
        )
    cells.append(
        f'<div class="ll-kpi" title="Deterministic rules that crossed their threshold.">'
        f'<div class="ll-kpi-l">Leaks flagged</div>'
        f'<div class="ll-kpi-v">{len(alerts)} of up to 3</div>'
        f'<div class="ll-kpi-s">ranked by evidence strength</div></div>'
    )
    st.markdown(f'<div class="ll-strip">{"".join(cells)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Alerts: dense selectable grid (row selection drives the investigation)
# --------------------------------------------------------------------------- #
def _fmt_rate(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "—"


def alert_grid(alerts: list[Alert]) -> int | None:
    rows = []
    for a in alerts:
        rate = _fmt_rate(a.rate) if a.rate is not None else "count rule"
        thr = _fmt_rate(a.threshold) if a.threshold is not None else "—"
        rows.append({
            "#": a.rank,
            "Leak": a.title,
            "Severity": a.severity,
            "Scope": a.scope,
            "Rate vs threshold": f"{rate} vs {thr}",
            "Impact": (f"{format_inr(a.impact_value)} · {a.impact_basis}"
                       if a.impact_value is not None else "—"),
            "Score": int(a.evidence_score),
        })
    df = pd.DataFrame(rows)
    event = st.dataframe(
        df, hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key="alert_grid",
        column_config={
            "#": st.column_config.NumberColumn("#", width="small", format="%d"),
            "Leak": st.column_config.TextColumn("Leak", width="medium"),
            "Severity": st.column_config.TextColumn("Severity", width="small"),
            "Scope": st.column_config.TextColumn("Scope", width="large"),
            "Rate vs threshold": st.column_config.TextColumn("Rate vs threshold", width="medium"),
            "Impact": st.column_config.TextColumn("Impact", width="medium"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                     format="%d", width="small"),
        },
    )
    try:
        sel = event.selection.rows
    except Exception:
        sel = []
    return sel[0] if sel else None


def render_monitored(statuses: list[RuleStatus]) -> None:
    with st.expander("Monitored rules — did not cross threshold", expanded=False):
        df = pd.DataFrame([{"Rule": s.title, "Status": s.status, "Detail": s.detail} for s in statuses])
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("These rules ran against the data and found nothing above their threshold. "
                   "Shown so the absence of a leak is explicit, not silent.")


# --------------------------------------------------------------------------- #
# Charts (muted, single series, no chrome)
# --------------------------------------------------------------------------- #
def _styled(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            grid=True, gridColor=LINE_SOFT, gridOpacity=1,
            domainColor=LINE, tickColor=LINE, labelColor=SUB, titleColor=MUTED,
            labelFontSize=11, titleFontSize=11, titleFontWeight="normal",
            labelPadding=6, titlePadding=8,
        )
        .configure_legend(labelColor=SUB, titleColor=MUTED, labelFontSize=11, titleFontSize=11)
        .configure(background="transparent")
    )


def render_trend(alert: Alert) -> None:
    t = alert.trend
    if t is None or len(t) == 0 or "period" not in t.columns:
        note("No trend available — a date column was not mapped for this rule.")
        return
    unit = "pct" if t.get("kind", pd.Series(["rate"])).iloc[0] == "rate" else "currency"
    if unit == "pct":
        y = alt.Y("value:Q", title="Rate", axis=alt.Axis(format="%"))
        tip = alt.Tooltip("value:Q", format=".1%")
    else:
        y = alt.Y("value:Q", title="Value (₹)")
        tip = alt.Tooltip("value:Q", format=",.0f")
    line = alt.Chart(t).mark_line(color=CHART, strokeWidth=2,
                                  point=alt.OverlayMarkDef(size=22, color=CHART, filled=True)
                                  ).encode(x=alt.X("period:N", title=None, sort=None), y=y,
                                           tooltip=["period", tip])
    layers = [line]
    if "threshold" in t.columns and t["threshold"].notna().any():
        thr = t[["period", "threshold"]].dropna()
        layers.append(alt.Chart(thr).mark_line(strokeDash=[4, 3], strokeWidth=1.4, color=THRESH)
                      .encode(x=alt.X("period:N", sort=None), y=alt.Y("threshold:Q")))
    st.altair_chart(_styled(alt.layer(*layers).properties(height=210)), width="stretch")
    if "threshold" in t.columns and t["threshold"].notna().any():
        st.caption(f"Dashed line = rule threshold ({_fmt_rate(alert.threshold)}).")


def render_bars(alert: Alert) -> None:
    b = alert.bars
    if b is None or len(b) == 0:
        note("No ranked breakdown available for this rule.")
        return
    unit = b["unit"].iloc[0]
    fmt = ".1%" if unit == "pct" else ",.0f"
    x = alt.X("value:Q", title=None, axis=alt.Axis(format=fmt))
    chart = alt.Chart(b).mark_bar(color=CHART, size=13).encode(
        y=alt.Y("label:N", sort="-x", title=None,
                axis=alt.Axis(grid=False, domain=False, ticks=False, labelColor=SUB)),
        x=x,
        tooltip=[alt.Tooltip("label:N", title="Segment"),
                 alt.Tooltip("value:Q", title="Value", format=fmt)],
    ).properties(height=max(190, 26 * len(b)))
    st.altair_chart(_styled(chart), width="stretch")


# --------------------------------------------------------------------------- #
# Investigation detail
# --------------------------------------------------------------------------- #
def _field_grid(pairs: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="ll-cell"><div class="ll-k">{_esc(k)}</div><div class="ll-v">{_esc(v)}</div></div>'
        for k, v in pairs
    )
    st.markdown(f'<div class="ll-grid">{cells}</div>', unsafe_allow_html=True)


def render_investigation(alert: Alert) -> None:
    st.markdown(
        f'<div class="ll-dhead">'
        f'<div><div class="ll-dtitle">{_esc(alert.title)}</div>'
        f'<div class="ll-sub">Rank #{alert.rank} · {_esc(alert.leak)} · {_esc(alert.availability)}</div></div>'
        f'<div class="ll-dmeta">{_sev(alert.severity)}'
        f'<span>Score <b>{alert.evidence_score}</b>/100</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    _field_grid([
        ("Rule", alert.rule_text),
        ("Scope", alert.scope),
        ("Comparison window", alert.comparison_window),
        ("Numerator", alert.numerator),
        ("Denominator", alert.denominator),
        ("Baseline", alert.baseline_note),
    ])

    left, right = st.columns([2, 1], gap="large")

    with left:
        section("Trend", "trend")
        render_trend(alert)
        section("Ranked breakdown — likely drivers", "bars")
        render_bars(alert)
        if alert.tables:
            section("Supporting breakdowns", "layers")
            for name, df in alert.tables.items():
                with st.expander(name, expanded=(len(alert.tables) == 1)):
                    st.dataframe(_dense(df), width="stretch", hide_index=True)

    with right:
        st.markdown(
            f'<div class="ll-inset"><div class="ll-inset-t">{icon("check", color=ACCENT)}Recommended action</div>'
            f'<div class="ll-inset-b">{_esc(alert.action)}</div>'
            f'<div class="ll-inset-m">Owner: {_esc(alert.owner)} · Validation metric: {_esc(alert.validation_metric)}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        section("Impact treatment", "shield")
        st.markdown(
            f'<div>{_basis_badge(alert.impact_basis)}</div>'
            f'<div class="ll-impact">{_esc(format_inr(alert.impact_value))}</div>'
            f'<div class="ll-meta-line">{_esc(alert.impact_note)}</div>',
            unsafe_allow_html=True,
        )

        section("Evidence strength", "activity")
        pct = max(0, min(100, int(alert.evidence_score)))
        st.markdown(
            f'<div class="ll-meta-line">{pct} / 100</div>'
            f'<div class="ll-bar"><i style="width:{pct}%"></i></div>'
            '<div class="ll-parts">'
            + "".join(f"<div>{_esc(p)}</div>" for p in alert.evidence_parts)
            + "</div>",
            unsafe_allow_html=True,
        )

        section("Why flagged", "alert")
        st.markdown(f'<div class="ll-inset-b">{_esc(alert.why)}</div>', unsafe_allow_html=True)

        if alert.assumptions:
            with st.expander("Assumptions for this alert"):
                for a in alert.assumptions:
                    st.markdown(f"- {a}")


def _dense(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if "rate" in str(c).lower() and out[c].dtype.kind == "f":
            out[c] = (out[c] * 100).round(2)
            out = out.rename(columns={c: f"{c} (%)"})
        elif out[c].dtype.kind == "f":
            out[c] = out[c].round(2)
    return out


# --------------------------------------------------------------------------- #
# Secondary panels
# --------------------------------------------------------------------------- #
def render_campaigns(model) -> None:
    camp = model.campaigns
    if camp is None:
        note("No campaign file mapped.")
        return
    st.markdown(
        '<div class="ll-meta-line">Campaign performance is shown independently — no verified key '
        "links campaigns to transactions, so no attribution is fabricated. Values are in ₹ (INR).</div>",
        unsafe_allow_html=True,
    )
    if "channel" in camp.columns:
        g = camp.groupby("channel")
        base = pd.DataFrame(index=g.size().index)
        base.index.name = "Channel"
        base["Campaigns"] = g.size()
        for col, lab in [("spend", "Spend"), ("impressions", "Impressions"),
                         ("clicks", "Clicks"), ("conversions", "Conversions")]:
            if col in camp.columns:
                base[lab] = g[col].sum()
        if "Spend" in base and "Conversions" in base:
            base["Cost / conversion"] = (base["Spend"] / base["Conversions"]).round(2)
        st.dataframe(_dense(base.reset_index()), width="stretch", hide_index=True)
    st.caption("Campaign spend is Observed. It cannot be joined to transactions by key (only by "
               "date/channel/category), so it is reported separately.")


def render_profile(files: dict, profiles: dict, mapping, model) -> None:
    section("Files", "database")
    rows = [{
        "File": name, "Role": p.role, "Rows": p.rows, "Columns": len(p.columns),
        "Max null rate": round(max((c.null_rate for c in p.columns), default=0), 4),
    } for name, p in profiles.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    section("Field mapping — name hint + type validation", "layers")
    map_rows = [{"File": f, "Concept": c, "Column": col}
                for f, m in mapping.by_file.items() for c, col in m.items() if col is not None]
    st.dataframe(pd.DataFrame(map_rows), width="stretch", hide_index=True)
    for w in mapping.warnings:
        st.warning(w)

    section("Join health — left joins from the transaction grain", "shield")
    if model.join_health:
        st.dataframe(pd.DataFrame([j.as_row() for j in model.join_health]),
                     width="stretch", hide_index=True)
    else:
        note("No dimension joins were applied.")

    with st.expander("Column detail — types, nulls, ranges, samples"):
        pick = st.selectbox("File", list(profiles.keys()), label_visibility="collapsed")
        prof = profiles[pick]
        detail = [{
            "Column": c.name, "dtype": c.dtype, "null_rate": c.null_rate, "n_unique": c.n_unique,
            "min": c.minimum, "max": c.maximum,
            "top_values": str(c.top_values) if c.top_values else "",
        } for c in prof.columns]
        st.dataframe(pd.DataFrame(detail), width="stretch", hide_index=True)
        st.caption("Sample rows")
        st.dataframe(prof.sample, width="stretch", hide_index=True)


def render_method(model, metrics: dict[str, Metric]) -> None:
    section("How money is measured — all values in ₹ (INR)", "info")
    st.markdown(
        "- **Net realized revenue** is the realized line value of *completed* orders only "
        "(`sum(quantity × unit_price)` where status is completed). Cancelled, returned and pending "
        "lines are excluded.\n"
        "- **Return rate** uses the PRD denominator — returned lines ÷ *fulfilled* lines "
        "(completed + returned). On all placed lines the rate is lower; both are shown so the "
        "denominator choice is transparent.\n"
        "- **Discount rate** comes from the observed `discount_pct` field weighted by gross line "
        "value, not derived from independently generated list prices."
    )

    section("Metric availability", "check")
    for k in ["realized_revenue", "cancellation_rate", "return_rate", "discount_rate",
              "low_stock_count", "campaign_spend"]:
        m = metrics.get(k)
        if m:
            st.markdown(f"- **{m.label}** — {m.availability}. {m.definition}")

    section("Data-quality flags — surfaced, not hidden", "alert")
    impossible = "—"
    if "__cost" in model.tx.columns and "__list_price" in model.tx.columns:
        impossible = f"{(model.tx['__cost'] > model.tx['__list_price']).sum():,}"
    st.markdown(
        f"- `products.unit_cost` and `products.list_price` are independently generated: for "
        f"**{impossible}** transaction lines the synthetic cost exceeds list price, producing "
        "impossible negative margins. **No profit or margin figure is presented as fact.**\n"
        "- `campaigns` shares no key with `transactions`; campaign metrics are independent and never "
        "attributed to orders.\n"
        "- Low-stock impact is an **Estimated** missed-demand proxy (`recent fulfilled units × list "
        "price`), never observed revenue.\n"
        "- Where no comparable prior window exists, the alert says so and uses the absolute threshold."
    )

    section("Round 1 scope", "shield")
    st.caption("No ML model, no causal claims, no forecasting, no automated inventory optimisation. "
               "Breakdowns are labelled likely drivers, never proven root causes.")


# =========================================================================== #
# ROUND 2 — LeakLens screens
# =========================================================================== #
# Palette additions for the decision layer.
OBS = "#22374f"          # observed realisation
EST = "#9a7326"          # estimated / scenario
TIER_COLOR = {"Review now": SEV["High"], "Watch": SEV["Medium"], "Routine": MUTED}


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _num(v, d: int = 2) -> str:
    try:
        return f"{float(v):,.{d}f}"
    except (TypeError, ValueError):
        return "—"


def kpi_row(cells: list[tuple[str, str, str]]) -> None:
    """cells = (label, value, sub)."""
    html = "".join(
        f'<div class="ll-kpi"><div class="ll-kpi-l">{_esc(l)}</div>'
        f'<div class="ll-kpi-v">{_esc(v)}</div>'
        f'<div class="ll-kpi-s">{_esc(s)}</div></div>'
        for l, v, s in cells
    )
    st.markdown(f'<div class="ll-strip">{html}</div>', unsafe_allow_html=True)


def banner(kind: str, title: str, body: str) -> None:
    """A functional, non-decorative inset used for honest verdicts."""
    col = {"ok": BASIS["Observed"], "warn": BASIS["Estimated"], "stop": SEV["High"]}.get(kind, SUB)
    ic = {"ok": "check", "warn": "info", "stop": "alert"}.get(kind, "info")
    st.markdown(
        f'<div class="ll-inset" style="border-left-color:{col}">'
        f'<div class="ll-inset-t" style="color:{col}">{icon(ic, color=col)} {_esc(title)}</div>'
        f'<div class="ll-inset-b">{body}</div></div>',
        unsafe_allow_html=True,
    )


def _plain(text: str) -> str:
    """Escape then re-allow a tiny whitelist of inline tags for banner bodies."""
    return html.escape(str(text), quote=False)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _base(chart: alt.Chart) -> alt.Chart:
    return _styled(chart)


def chart_exposure_time(df: pd.DataFrame) -> None:
    """Chart 1 — observed leak value vs estimated exposure over time."""
    if df is None or not len(df):
        note("No dated records available for a time series.")
        return
    long = df.melt(id_vars=["period"], value_vars=["observed_value", "expected_exposure"],
                   var_name="series", value_name="value")
    label = {"observed_value": "Observed leak value", "expected_exposure": "Estimated exposure"}
    long["series"] = long["series"].map(label)
    color = alt.Scale(domain=["Observed leak value", "Estimated exposure"], range=[OBS, EST])
    bars = alt.Chart(long).mark_bar(size=14).encode(
        x=alt.X("period:N", title=None, sort=None),
        y=alt.Y("value:Q", title="₹ value"),
        xOffset=alt.XOffset("series:N", sort=list(label.values())),
        color=alt.Color("series:N", title=None, scale=color,
                        legend=alt.Legend(orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip("period:N", title="Period"),
                 alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("value:Q", title="Value", format=",.0f")],
    ).properties(height=220)
    st.altair_chart(_base(bars), width="stretch")
    st.caption("Observed leak value is realised (status-verified). Estimated exposure is "
               "line value × predicted risk — a forward scenario, not revenue.")


def chart_segment_bars(df: pd.DataFrame, dim: str, metric: str = "expected_exposure") -> None:
    """Chart 2 — ranked driver / segment bars."""
    if df is None or not len(df):
        note("No ranked breakdown available.")
        return
    d = df.copy()
    is_money = metric in {"expected_exposure", "observed_value"}
    fmt = ",.0f" if is_money else ".1%"
    title = {"expected_exposure": "Estimated exposure (₹)",
             "observed_value": "Observed leak value (₹)",
             "observed_rate": "Observed adverse rate",
             "mean_risk": "Mean predicted risk"}.get(metric, metric)
    d = d.sort_values(metric, ascending=False).head(12)
    chart = alt.Chart(d).mark_bar(color=CHART, size=14).encode(
        y=alt.Y(f"{dim}:N", sort="-x", title=None,
                axis=alt.Axis(grid=False, domain=False, ticks=False, labelColor=SUB)),
        x=alt.X(f"{metric}:Q", title=title, axis=alt.Axis(format=fmt)),
        tooltip=[alt.Tooltip(f"{dim}:N", title=dim.replace("_", " ").title()),
                 alt.Tooltip(f"{metric}:Q", title=title, format=fmt),
                 alt.Tooltip("records:Q", title="Records", format=",.0f")],
    ).properties(height=max(190, 24 * len(d)))
    st.altair_chart(_base(chart), width="stretch")


def chart_pr(pr: pd.DataFrame) -> None:
    if pr is None or not len(pr):
        note("Precision–recall curve unavailable (need both classes in the hold-out).")
        return
    base = alt.Chart(pd.DataFrame({"y": [0, 1], "x": [0, 1]})).mark_line(
        strokeDash=[4, 3], color=MUTED, strokeWidth=1).encode(x="x:Q", y="y:Q")
    line = alt.Chart(pr).mark_line(color=CHART, strokeWidth=2).encode(
        x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("recall:Q", format=".2f"), alt.Tooltip("precision:Q", format=".2f")])
    st.altair_chart(_base(alt.layer(base, line).properties(height=210)), width="stretch")


def chart_calibration(cal: pd.DataFrame) -> None:
    if cal is None or not len(cal):
        note("Calibration table unavailable.")
        return
    diag = alt.Chart(pd.DataFrame({"x": [0, 0.6], "y": [0, 0.6]})).mark_line(
        strokeDash=[4, 3], color=MUTED, strokeWidth=1).encode(x="x:Q", y="y:Q")
    pts = alt.Chart(cal).mark_line(point=alt.OverlayMarkDef(size=28, color=CHART), color=CHART).encode(
        x=alt.X("predicted:Q", title="Mean predicted risk", scale=alt.Scale(domain=[0, 0.6])),
        y=alt.Y("observed:Q", title="Observed adverse rate", scale=alt.Scale(domain=[0, 0.6])),
        tooltip=[alt.Tooltip("bin:N", title="Bin"), alt.Tooltip("predicted:Q", format=".3f"),
                 alt.Tooltip("observed:Q", format=".3f"), alt.Tooltip("n:Q")])
    st.altair_chart(_base(alt.layer(diag, pts).properties(height=200)), width="stretch")
    st.caption("Dashed line = perfect calibration. A flat line near the base rate means the model "
               "cannot separate outcomes — so probabilities stay honest, not sharp.")


def chart_importance(imp: pd.DataFrame) -> None:
    if imp is None or not len(imp):
        note("Permutation importance unavailable.")
        return
    d = imp.copy()
    d = d[d["importance"] > 0].sort_values("importance", ascending=False).head(12)
    if not len(d):
        note("No feature improved the hold-out score when permuted — consistent with no signal.")
        return
    chart = alt.Chart(d).mark_bar(color=CHART, size=13).encode(
        y=alt.Y("label:N", sort="-x", title=None,
                axis=alt.Axis(grid=False, domain=False, ticks=False, labelColor=SUB)),
        x=alt.X("importance:Q", title="Drop in hold-out PR-AUC when permuted"),
        tooltip=[alt.Tooltip("label:N", title="Feature"),
                 alt.Tooltip("importance:Q", format=".4f"),
                 alt.Tooltip("kind:N", title="Type")])
    st.altair_chart(_base(chart).properties(height=max(190, 24 * len(d))), width="stretch")


def chart_signal_audit(audit: dict) -> None:
    if not audit or not audit.get("available"):
        note("Signal audit not available.")
        return
    perm = pd.DataFrame({"auc": audit.get("permuted_aucs", [])})
    obs = audit.get("observed_auc")
    layers = []
    if len(perm):
        layers.append(alt.Chart(perm).mark_bar(color="#c7ccd3", size=14).encode(
            x=alt.X("auc:Q", title="ROC-AUC", bin=alt.Bin(maxbins=12),
                    scale=alt.Scale(domain=[0.42, 0.58])),
            y=alt.Y("count()", title="Permuted models")))
    if obs is not None:
        layers.append(alt.Chart(pd.DataFrame({"x": [obs]})).mark_rule(
            color=SEV["High"], strokeWidth=2).encode(x="x:Q"))
    st.altair_chart(_base(alt.layer(*layers).properties(height=180)), width="stretch")
    st.caption("Grey bars = AUC of models trained on shuffled labels (the null). Red line = the "
               "model actually reported. If the red line sits inside the grey bars, there is no signal.")


def confusion_matrix_figure(cm: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"": f"Predicted positive (risk ≥ {cm['threshold']:.2f})", "Adverse (actual)": cm["tp"], "Not adverse (actual)": cm["fp"]},
        {"": "Predicted negative", "Adverse (actual)": cm["fn"], "Not adverse (actual)": cm["tn"]},
    ])


# --------------------------------------------------------------------------- #
# Screen 1 — Data trust
# --------------------------------------------------------------------------- #
def render_data_trust(files, profiles, mapping, model, ft, specs, grain, warnings) -> None:
    section("Files — role inferred, never hard-coded", "database")
    rows = [{
        "File": name, "Role": p.role, "Rows": p.rows, "Columns": len(p.columns),
        "Max null rate": round(max((c.null_rate for c in p.columns), default=0), 4),
    } for name, p in profiles.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    section("Base grain & join health", "shield")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base grain", grain["grain"])
    c2.metric("Rows", f"{grain['rows']:,}")
    c3.metric("Joins preserved grain", "Yes" if grain["joins_preserved"] else "No")
    c4.metric("Max unmatched", _pct(grain["max_unmatched_pct"]))
    if model.join_health:
        st.dataframe(pd.DataFrame([j.as_row() for j in model.join_health]),
                     width="stretch", hide_index=True)
    if grain.get("grain_note"):
        st.caption(grain["grain_note"])

    section("Field mapping — name hint + type validation", "layers")
    map_rows = [{"File": f, "Concept": c, "Column": col}
                for f, m in mapping.by_file.items() for c, col in m.items() if col is not None]
    st.dataframe(pd.DataFrame(map_rows), width="stretch", hide_index=True)
    for w in warnings:
        st.warning(w)

    section("Modelling eligibility — the label and the leak guards", "check")
    elig = int(ft["eligible"].sum())
    known = ft["target_known"].astype(bool)
    adverse = int((ft["target"] == 1).sum())
    cells = [
        ("Rows with a verified outcome", f"{elig:,}", "terminal status only"),
        ("Excluded (unverified)", f"{len(ft) - elig:,}", "e.g. pending — outcome unknown"),
        ("Adverse outcomes", f"{adverse:,}", "cancelled + returned"),
        ("Base adverse rate", _pct(adverse / elig if elig else float('nan')), "observed"),
    ]
    kpi_row(cells)
    st.markdown(
        f'<div class="ll-meta-line"><b>Target:</b> {_esc(specs_target_label(ft))}'
        '<br><b>Leakage guard:</b> only pre-outcome fields become features; the verified status '
        'field is used <i>only</i> to build the label.</div>', unsafe_allow_html=True)

    section("Pre-outcome features in the model", "bars")
    feats = pd.DataFrame([{
        "Feature": s.label, "Column": s.name, "Type": s.kind,
        "Source": s.source or "derived", "Leakage-safe": "yes" if s.pre_outcome else "NO",
        "Note": s.note,
    } for s in specs])
    st.dataframe(feats, width="stretch", hide_index=True)

    with st.expander("Excluded by design (post-outcome / identifiers)"):
        for f in EXCLUDED_FIELDS:
            st.markdown(f"- {f}")


def specs_target_label(ft: pd.DataFrame) -> str:
    vals = sorted(ft.loc[ft["eligible"], "status_raw"].dropna().astype(str).unique().tolist())
    return ("cancelled or returned (1) vs completed (0). Eligible statuses: " + ", ".join(vals))


# --------------------------------------------------------------------------- #
# Screen 2 — Train & evaluate
# --------------------------------------------------------------------------- #
def render_train(res, *, capacity: float, kind: str, train_button: bool = True) -> bool:
    """Render the model card. Returns True if the train button was clicked."""
    clicked = False
    c1, c2 = st.columns([3, 1])
    with c2:
        if train_button:
            clicked = st.button("Train model", type="primary", width="stretch",
                                help="Fit the baseline and final models on the provided data.")
    if res is None:
        with c1:
            banner("warn", "No model yet",
                   "Click <b>Train model</b> to fit a baseline and a final model on the loaded data. "
                   "Training runs locally — no external API.")
        return clicked

    with c1:
        banner("ok" if res.audit.get("usable") else "stop",
               "Signal audit: " + ("usable" if res.audit.get("usable") else "no detectable signal"),
               _plain(res.audit.get("verdict", "")))
    st.caption(f"Trained {res.trained_at} in {res.seconds:.1f}s · seed {res.seed} · "
               f"{res.final_label}")

    kpi_row([
        ("Labelled population", f"{res.population:,}", "verified outcomes"),
        ("Positive (adverse)", f"{res.positives:,}", f"base rate {_pct(res.base_rate)}"),
        ("Hold-out ROC-AUC", _num(res.final.get("roc_auc"), 3),
         f"95% CI {_num(res.auc_ci[0], 2)}–{_num(res.auc_ci[1], 2)}"),
        ("Signal p-value", _num(res.audit.get("p_value"), 3), "permutation test"),
        ("Operating threshold", _num(res.threshold, 3), f"capacity {_pct(capacity)}"),
    ])

    if not res.audit.get("usable", False):
        banner("warn", "How to read this",
               "The provided data does <b>not</b> support predictive prioritisation: discrimination is "
               "at chance. The queue below therefore ranks by <b>verified value and exposure</b>, and "
               "is labelled Estimated. Treat the model as a calibrated exposure estimate, not a "
               "risk oracle. See the limitations at the bottom of this screen.")

    section("How the model was built", "layers")
    grid = [
        ("Target", res.target_label),
        ("Split", f"{res.split_kind} — {res.split_note}"),
        ("Training window", res.train_span),
        ("Hold-out window", res.test_span),
        ("Fit / validation / test rows", f"{res.fit_n:,} / {res.val_n:,} / {res.test_n:,}"),
        ("Baseline", res.baseline_label),
        ("Final", res.final_label),
        ("Random seed", str(res.seed)),
    ]
    _field_grid([(k, str(v)) for k, v in grid])

    section("Baseline vs final — hold-out metrics", "activity")
    keys = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision", "Precision"),
            ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy"),
            ("brier", "Brier (lower better)"), ("lift", "Lift vs base rate")]
    mrows = []
    for k, lab in keys:
        mrows.append({"Metric": lab,
                      "Baseline (LogReg)": round(res.baseline.get(k, float("nan")), 4),
                      "Final (RF)": round(res.final.get(k, float("nan")), 4)})
    st.dataframe(pd.DataFrame(mrows), width="stretch", hide_index=True)
    st.caption("At the chosen operating point. The baseline is the interpretation floor: a final "
               "model must beat it to be worth the added complexity. Here it does not — reported honestly.")

    left, right = st.columns(2, gap="large")
    with left:
        section("Precision–recall curve", "trend")
        chart_pr(res.pr_curve)
    with right:
        section("Calibration — are probabilities honest?", "check")
        chart_calibration(res.calibration)

    left2, right2 = st.columns(2, gap="large")
    with left2:
        section("Confusion matrix at the operating point", "layers")
        st.dataframe(confusion_matrix_figure(res.final), width="stretch", hide_index=True)
    with right2:
        section("Signal audit — permutation test", "shield")
        chart_signal_audit(res.audit)

    section("Feature importance — association, not cause", "bars")
    chart_importance(res.importance)
    st.caption("Permutation importance measured on the hold-out set (drop in PR-AUC when a feature "
               "is shuffled). Near-zero for every feature, which is itself the finding.")

    with st.expander("Model card & limitations", expanded=False):
        for k, v in res.model_card().items():
            st.markdown(f"- **{k}**: {v}")
        st.markdown("**Limitations**")
        st.markdown(
            "- The provided dataset shows **no detectable signal** for the leakage outcome; the "
            "hold-out AUC sits inside the permutation null, so predictions should not be treated as "
            "a risk oracle.\n"
            "- Feature importances are **associations**, not causes — no causal claim is made.\n"
            "- No profit or margin is modelled: `unit_cost > list_price` for a large share of the "
            "synthetic catalogue.\n"
            "- `campaigns` cannot be joined to orders by key and is excluded from the model.\n"
            "- Thresholds are set by review capacity, not by a claim about optimal economics.")
        for n in res.notes:
            st.caption(n)
    return clicked


# --------------------------------------------------------------------------- #
# Screen 3 — Priority queue
# --------------------------------------------------------------------------- #
def render_queue(summary, scored: pd.DataFrame, seg: pd.DataFrame, dim: str,
                 alerts: list[Alert], statuses: list[RuleStatus]) -> int | None:
    """KPI strip + queue table + segment bars + verified rule signals."""
    kpi_row([
        ("Records scored", f"{summary.population:,}", "all grain rows"),
        ("Flagged for review", f"{summary.flagged:,}", f"top {_pct(summary.capacity)} by risk"),
        ("Observed leak value", format_inr(summary.observed_leak_value), "Observed · verified"),
        ("Estimated exposure", format_inr(summary.expected_exposure), "Estimated · forward"),
        ("Base adverse rate", _pct(summary.base_rate), "observed"),
    ])

    section("Verified deterministic signals (Round 1 rules)", "alert")
    if alerts:
        df = pd.DataFrame([{
            "#": a.rank, "Leak": a.title, "Severity": a.severity, "Scope": a.scope,
            "Rate vs threshold": (f"{_fmt_rate(a.rate)} vs {_fmt_rate(a.threshold)}"
                                  if a.rate is not None else "count rule"),
            "Impact": (f"{format_inr(a.impact_value)} · {a.impact_basis}"
                       if a.impact_value is not None else "—"),
            "Score": int(a.evidence_score),
        } for a in alerts])
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        note("No deterministic rule crossed its threshold in this window.")
    with st.expander("Monitored rules — did not cross threshold"):
        st.dataframe(pd.DataFrame([{"Rule": s.title, "Status": s.status, "Detail": s.detail}
                                   for s in statuses]), width="stretch", hide_index=True)

    section("Priority queue — ranked by expected exposure (Estimated)", "bars")
    show = scored.head(400).copy()
    cols = ["record_id", "date", "status", "tier", "value", "risk", "expected_exposure", dim]
    cols = [c for c in cols if c in show.columns]
    show = show[cols].rename(columns={
        "record_id": "Record", "date": "Date", "status": "Status", "tier": "Tier",
        "value": "Order value (₹)", "risk": "Risk", "expected_exposure": "Expected exposure (₹)",
    })
    event = st.dataframe(
        show, hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key="queue_grid",
        column_config={
            "Order value (₹)": st.column_config.NumberColumn(format="%.0f"),
            "Risk": st.column_config.NumberColumn(format="%.3f"),
            "Expected exposure (₹)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption("Queue is Estimated: order value × predicted risk. Observed leak value elsewhere is "
               "verified. Select a row to open it in *Why this risk*.")
    try:
        sel = event.selection.rows
    except Exception:
        sel = []
    return int(show.index[sel[0]]) if sel else None


# --------------------------------------------------------------------------- #
# Screen 4 — Why this risk
# --------------------------------------------------------------------------- #
def render_why(res, scored: pd.DataFrame, record: pd.Series | None, over_time: pd.DataFrame,
               seg: pd.DataFrame, dim: str, evidence: list[dict], templates) -> None:
    if record is None or not len(scored):
        note("Select a row in **Priority queue** to investigate it here.")
        return
    head = (
        f'<div class="ll-dhead"><div>'
        f'<div class="ll-dtitle">{_esc(record["record_id"])}</div>'
        f'<div class="ll-sub">{_esc(record.get("status", ""))} · '
        f'{_esc(str(pd.to_datetime(record.get("date"), errors="coerce"))[:10])} · '
        f'{_esc(dim.replace("_", " ").title())}: {_esc(str(record.get(dim, "—")))}</div></div>'
        f'<div class="ll-dmeta">'
        f'<span class="ll-sev" style="color:{TIER_COLOR.get(record["tier"], MUTED)}">'
        f'{_dot(TIER_COLOR.get(record["tier"], MUTED), 7)}{_esc(record["tier"])}</span>'
        f'<span>Risk <b>{_num(record["risk"], 3)}</b></span></div></div>'
    )
    st.markdown(head, unsafe_allow_html=True)

    left, right = st.columns([2, 1], gap="large")
    with left:
        section("Exposure over time — observed vs estimated", "trend")
        chart_exposure_time(over_time)
        section(f"Ranked drivers — estimated exposure by {dim.replace('_', ' ')}", "bars")
        chart_segment_bars(seg, dim, "expected_exposure")

    with right:
        section("Why this is flagged", "alert")
        parts = [
            ("Population", f"Scored among {len(scored):,} records; base adverse rate "
                           f"{_pct(scored.loc[scored['target_known'], 'observed_adverse'].mean())}."),
            ("Model result", f"Predicted risk {_num(record['risk'], 3)} vs population mean "
                             f"{_num(scored['risk'].mean(), 3)} (threshold {_num(res.threshold, 3)})."),
            ("Comparison", f"Order value {format_inr(record['value'])}; expected exposure "
                           f"{format_inr(record['expected_exposure'])} (Estimated)."),
            ("Signal caveat", "The permutation test found no detectable signal, so risk ≈ the base "
                              "rate for most records; the ranking is value-led."),
        ]
        st.markdown(
            '<div class="ll-parts">' + "".join(
                f'<div><b>{_esc(k)}:</b> {_esc(v)}</div>' for k, v in parts) + "</div>",
            unsafe_allow_html=True)
        if record.get("observed_adverse"):
            st.markdown(f'<div class="ll-meta-line">{_basis_badge("Observed")} This order is already '
                        'verified as a leakage outcome.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ll-meta-line">{_basis_badge("Estimated")} Outcome not yet '
                        'verified — this is a forward exposure estimate.</div>', unsafe_allow_html=True)

        if evidence:
            with st.expander("Driver comparison vs population", expanded=True):
                ev = pd.DataFrame(evidence).rename(columns={
                    "driver": "Driver", "value": "Value", "segment_risk": "Segment risk",
                    "population_risk": "Population risk", "segment_observed_rate": "Observed rate",
                    "segment_records": "Records"})
                st.dataframe(ev, width="stretch", hide_index=True)

        section("Recommended actions", "check")
        for t in templates[:2]:
            st.markdown(
                f'<div class="ll-inset"><div class="ll-inset-t">{icon("check", color=ACCENT)}'
                f'{_esc(t.leak_type)}</div><div class="ll-inset-b">{_esc(t.action)}</div>'
                f'<div class="ll-inset-m">Owner: {_esc(t.owner)} · Validates with: '
                f'{_esc(t.validation_metric)}</div></div>', unsafe_allow_html=True)
        st.caption("Record an action in the *Action lab* tab to close the loop.")


# --------------------------------------------------------------------------- #
# Screen 5 — Action lab
# --------------------------------------------------------------------------- #
def render_action_lab(templates, log_df: pd.DataFrame, default_scope: str,
                      expected_exposure: float, baseline_value: str) -> dict | None:
    section("Turn a leak into a measurable next step", "check")
    st.caption("Pick a leak type, confirm the action, set an owner and a recovery assumption. "
               "The recoverable figure is a scenario (Estimated), not a forecast.")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        tpl_labels = [f"{t.leak_type} — {t.action[:46]}…" for t in templates]
        pick = st.selectbox("Action to take", range(len(templates)),
                            format_func=lambda i: tpl_labels[i])
        t = templates[pick]
        owner = st.text_input("Owner", value=t.owner)
        due = st.date_input("Due date")
        scope = st.text_input("Scope", value=default_scope)
        status = st.selectbox("Status", ["Open", "In progress", "Done"])
    with c2:
        recovery = st.slider("Assume we recover this share of exposure", 0, 100, 30, 5,
                             format="%d%%")
        led = apply_recovery(expected_exposure, recovery / 100)
        kpi_row([
            ("Estimated exposure", format_inr(led["expected_exposure"]), "Estimated"),
            (f"Recoverable @ {recovery}%", format_inr(led["recoverable"]), "Estimated scenario"),
            ("Validation metric", "see card", t.validation_metric[:38]),
        ])
        st.markdown(
            f'<div class="ll-inset"><div class="ll-inset-t">{icon("info", color=SUB)}Expected direction</div>'
            f'<div class="ll-inset-b">{_esc(t.expected_direction)}</div>'
            f'<div class="ll-inset-m">Baseline now: {_esc(baseline_value)}</div></div>',
            unsafe_allow_html=True)
        submit = st.button("Log this action", type="primary", width="stretch")

    if submit:
        return {
            "scope": scope, "leak_type": t.leak_type, "action": t.action, "owner": owner,
            "expected_direction": t.expected_direction, "validation_metric": t.validation_metric,
            "baseline_value": baseline_value, "due_date": str(due), "status": status,
        }
    if log_df is not None and len(log_df):
        section("Action log", "layers")
        st.dataframe(log_df.iloc[::-1], width="stretch", hide_index=True)
    return None


def apply_recovery(expected_exposure: float, pct: float) -> dict:
    from .score import recoverable_scenario
    return recoverable_scenario(expected_exposure, pct)


# --------------------------------------------------------------------------- #
# Screen 6 — Measure
# --------------------------------------------------------------------------- #
def render_measure(log_df: pd.DataFrame, summary, metrics: dict[str, Metric],
                   model, policy_text: str) -> None:
    section("Validation loop — what changes after the action", "check")
    if log_df is None or not len(log_df):
        note("No actions logged yet. Log one in the *Action lab* to populate this screen.")
    else:
        from .actions import log_summary
        s = log_summary(log_df)
        kpi_row([
            ("Actions logged", f"{s['total']:,}", "this workspace"),
            ("Open", f"{s['open']:,}", "not yet done"),
            ("Owners", f"{s['owners']:,}", "named"),
            ("Leak types", f"{s['leak_types']:,}", "covered"),
        ])
        st.dataframe(log_df.iloc[::-1], width="stretch", hide_index=True)

    section("Before / after plan", "activity")
    val = pd.DataFrame([
        {"Metric": "Observed leak value (verified)", "Now": format_inr(summary.observed_leak_value),
         "Target": "falls as flagged records are prevented or recovered"},
        {"Metric": "Estimated exposure (forward)", "Now": format_inr(summary.expected_exposure),
         "Target": "recomputed each period on the refreshed data"},
        {"Metric": "Flagged adverse rate", "Now": _pct(summary.base_rate),
         "Target": "adverse rate among worked records below the base rate"},
        {"Metric": "Cancellation rate", "Now": (metrics.get('cancellation_rate').display
                                                 if metrics.get('cancellation_rate') else "—"),
         "Target": "down after payment/stock checks"},
        {"Metric": "Return rate (of fulfilled)", "Now": (metrics.get('return_rate').display
                                                          if metrics.get('return_rate') else "—"),
         "Target": "down after product-quality audit"},
    ])
    st.dataframe(val, width="stretch", hide_index=True)
    st.caption("Re-run the same command next period; the app recomputes these from the refreshed "
               "CSVs, so each action has a measurable before/after.")

    banner("warn", "Data-collection recommendation", policy_text)

    with st.expander("Method & assumptions (Round 1 metrics retained)"):
        render_method(model, metrics)

