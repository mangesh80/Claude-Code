#!/usr/bin/env python3
"""
Streamlit UI for NFL Combine Career Predictor
Run with: streamlit run app.py
"""

import os
import ssl
import datetime
from google import genai
import pandas as pd
import streamlit as st

# macOS Python 3.13 SSL cert fix
try:
    import certifi  # noqa: F401
    ssl._create_default_https_context = ssl.create_default_context
except ImportError:
    pass

from nfl_combine_analyzer import (
    load_combine_data,
    find_player,
    get_metrics_for_position,
    find_similar_players,
    height_to_inches,
    METRIC_LABELS,
    POSITION_METRICS,
)

CURRENT_YEAR = datetime.date.today().year

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Combine Career Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.player-header {
    background: linear-gradient(135deg, #0f2942 0%, #1a3a5c 100%);
    border-radius: 12px;
    padding: 22px 28px;
    border-left: 5px solid #3b82f6;
    margin-bottom: 8px;
}
.player-name {
    font-size: 28px;
    font-weight: 800;
    color: #f9fafb;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.player-meta { color: #9ca3af; font-size: 14px; margin-top: 2px; }
.badge {
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 5px;
    margin-top: 8px;
    margin-right: 6px;
    letter-spacing: 0.04em;
    vertical-align: middle;
}
.badge-pos    { background: #3b82f6; color: white; }
.badge-r1     { background: #059669; color: white; }
.badge-r2r3   { background: #0284c7; color: white; }
.badge-late   { background: #6b7280; color: white; }
.badge-manual { background: #7c3aed; color: white; }
.section-title {
    font-size: 13px;
    font-weight: 700;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 20px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("app_step", "idle"),       # idle | manual | results
    ("target_dict", None),
    ("searched_name", ""),
    ("is_manual_entry", False),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_data(start_year: int, end_year: int) -> pd.DataFrame:
    return load_combine_data(start_year, end_year)


@st.cache_data(show_spinner=False)
def get_school_list() -> list[str]:
    """Return sorted unique schools from the full combine dataset."""
    df = load_combine_data(2000, 2025)
    schools = sorted(df["school"].dropna().astype(str).unique().tolist())
    return [""] + schools  # blank first so user must pick


def inches_to_ft_in(inches) -> str:
    """Convert total inches to display like 3'4\" or 10'10\"."""
    try:
        total = int(round(float(inches)))
        return f"{total // 12}'{total % 12}\""
    except (TypeError, ValueError):
        return "N/A"


def draft_badge(dr, dp) -> str:
    if pd.notna(dr):
        r = int(dr)
        pick_str = f", Pick #{int(dp)}" if pd.notna(dp) else ""
        label = f"Round {r}{pick_str}"
        cls = "badge-r1" if r == 1 else ("badge-r2r3" if r <= 3 else "badge-late")
    else:
        label, cls = "Undrafted / N/A", "badge-late"
    return f'<span class="badge {cls}">{label}</span>'


def stream_analysis(target: pd.Series, similar: pd.DataFrame, metrics: list, api_key: str):
    """Generator that yields text chunks from Gemini."""
    client = genai.Client(api_key=api_key)
    pos = target.get("pos", "?")

    dr, dp = target.get("draft_round"), target.get("draft_ovr")
    draft_str = f"Round {int(dr)}, Pick #{int(dp)}" if pd.notna(dr) and pd.notna(dp) else "Undrafted / Unknown"
    year = int(target["season"]) if pd.notna(target.get("season")) else "?"

    target_lines = [
        f"Name: {target.get('player_name', 'Unknown')}",
        f"Position: {pos}",
        f"Year: {year}",
        f"Draft: {draft_str}",
        "Combine Metrics:",
    ]
    for m in metrics:
        label = METRIC_LABELS.get(m, m)
        if m == "ht":
            v = str(target.get("ht_display", "N/A"))
            val_str = v if v not in ("nan", "", "None") else "N/A"
        elif m in ("vertical", "broad_jump"):
            v = target.get(m)
            val_str = inches_to_ft_in(v) if pd.notna(v) and v != 0 else "N/A"
        else:
            v = target.get(m)
            val_str = str(v) if pd.notna(v) and v != 0 else "N/A"
        target_lines.append(f"  {label}: {val_str}")
    # Include 10-yard split if manually entered
    ten = target.get("ten_split")
    if ten is not None and pd.notna(ten) and float(ten) > 0:
        target_lines.append(f"  10-Yard Split: {ten}s")
    target_text = "\n".join(target_lines)

    comp_cards = []
    for _, row in similar.iterrows():
        s = int(row["season"]) if pd.notna(row.get("season")) else "?"
        rdr, rdp = row.get("draft_round"), row.get("draft_ovr")
        d = f"Round {int(rdr)}, Pick #{int(rdp)}" if pd.notna(rdr) and pd.notna(rdp) else "Undrafted"
        lines = [f"{row['player_name']} ({row.get('pos','?')}, {s} — {d})"]
        for m in metrics:
            if m == "ht":
                hv = str(row.get("ht_display", ""))
                if hv and hv not in ("nan", "None"):
                    lines.append(f"  {METRIC_LABELS['ht']}: {hv}")
            else:
                v = row.get(m)
                if pd.notna(v) and v != 0:
                    lines.append(f"  {METRIC_LABELS.get(m, m)}: {v}")
        comp_cards.append("\n".join(lines))
    comps_text = "\n\n---\n\n".join(comp_cards)

    prompt = f"""You are a veteran NFL scout and draft analyst with encyclopedic knowledge of NFL combine history, draft results, and player career trajectories from 2000 to the present.

Evaluate the TARGET PLAYER's combine profile, compare to historically similar players, and predict career trajectory.

═══ TARGET PLAYER ═══
{target_text}

═══ MOST SIMILAR HISTORICAL PLAYERS (by combine metrics at {pos}) ═══
{comps_text}

═══ ANALYSIS REQUIRED ═══

Draw on your knowledge of the listed comparable players' actual NFL careers.

**1. COMBINE PROFILE ASSESSMENT**
Grade each recorded metric (Elite / Above-Average / Average / Below-Average / Concern) vs. historical {pos} prospects. Note standout traits or red flags.

**2. HISTORICAL COMP ANALYSIS**
For each comparable player listed, briefly describe their actual NFL career outcome. Identify patterns: which combine traits predicted success vs. bust at {pos}?

**3. CAREER CEILING & FLOOR**
- Best-case scenario (most optimistic comp and why)
- Realistic expectation
- Worst-case scenario (cautionary tale)

**4. POSITION-SPECIFIC PREDICTORS**
Which metrics are most predictive of {pos} success, and how does this player grade out on those?

**5. FINAL VERDICT**
- Career outlook: Bust / Depth Player / Starter / Pro Bowl / Elite
- Confidence rating: Low / Medium / High (based on combine signal strength)
- One concise paragraph summary

Be specific, analytical, and use the historical comps as evidence."""

    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=prompt,
    ):
        if chunk.text:
            yield chunk.text


def show_results(target: pd.Series, df: pd.DataFrame, num_comps: int, effective_key: str, is_manual: bool):
    """Render player card, metrics grid, comps table, and Claude analysis."""
    pos     = str(target.get("pos", "?")).upper()
    metrics = get_metrics_for_position(pos)
    year    = int(target["season"]) if pd.notna(target.get("season")) else "?"
    school  = target.get("school", "")

    # Player header
    dr, dp = target.get("draft_round"), target.get("draft_ovr")
    pos_badge  = f'<span class="badge badge-pos">{pos}</span>'
    draft_html = draft_badge(dr, dp)
    manual_tag = '<span class="badge badge-manual">Manual Entry</span>' if is_manual else ""

    st.markdown(f"""
    <div class="player-header">
      <div class="player-name">{target["player_name"]} {pos_badge}{manual_tag}</div>
      <div class="player-meta">{year} NFL Combine{f' · {school}' if school and str(school) not in ('nan', '') else ''}</div>
      {draft_html}
    </div>
    """, unsafe_allow_html=True)

    # Combine metrics grid
    st.markdown('<div class="section-title">Combine Metrics</div>', unsafe_allow_html=True)
    display_metrics = [m for m in metrics if m in df.columns or m == "ht"]
    for i in range(0, len(display_metrics), 4):
        row_m = display_metrics[i : i + 4]
        cols  = st.columns(len(row_m))
        for col, m in zip(cols, row_m):
            label = METRIC_LABELS.get(m, m)
            if m == "ht":
                v = str(target.get("ht_display", ""))
                val_str = v if v not in ("nan", "", "None") else "N/A"
            elif m in ("vertical", "broad_jump"):
                v = target.get(m)
                val_str = inches_to_ft_in(v) if pd.notna(v) and v != 0 else "N/A"
            else:
                v = target.get(m)
                val_str = str(round(float(v), 2)) if pd.notna(v) and v != 0 else "N/A"
            col.metric(label=label, value=val_str)

    # Comps table
    st.divider()
    st.markdown(f'<div class="section-title">Top Historical {pos} Comps</div>', unsafe_allow_html=True)

    # Compute which metrics will actually drive the comparison
    valid_metrics_used = [
        m for m in metrics
        if m in df.columns
        and pd.notna(target.get(m))
        and float(target.get(m, 0) or 0) != 0
    ]

    if is_manual:
        if valid_metrics_used:
            label_list = ", ".join(METRIC_LABELS.get(m, m) for m in valid_metrics_used)
            st.info(
                f"**Comparing on {len(valid_metrics_used)} metric(s):** {label_list}",
                icon="📊",
            )
        else:
            st.warning(
                "⚠️ No combine stats were entered — unable to find statistical comps. "
                "Enter at least one metric (e.g. 40-Yard Dash) and click **Compare & Analyze** again.",
                icon="⚠️",
            )
            return

    with st.spinner(f"Finding {num_comps} closest {pos} comps..."):
        similar = find_similar_players(df, target, n=num_comps)

    if similar.empty:
        st.warning(f"No comparable {pos} players found with sufficient data.")
    else:
        table_cols = ["player_name", "season", "draft_round", "draft_ovr"]
        if "ht_display" in similar.columns:
            table_cols.append("ht_display")
        table_cols += [m for m in metrics if m in similar.columns and m != "ht"]
        table_cols = [c for c in table_cols if c in similar.columns]

        rename = {
            "player_name": "Player", "season": "Year",
            "draft_round": "Round", "draft_ovr": "Overall Pick",
            "ht_display": "Height",
            **{m: METRIC_LABELS.get(m, m) for m in metrics},
        }
        tbl = similar[table_cols].copy().rename(columns=rename)

        for col in tbl.select_dtypes(include="float").columns:
            tbl[col] = tbl[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "—")
        for col in ["Round", "Overall Pick", "Year"]:
            if col in tbl.columns:
                tbl[col] = tbl[col].apply(
                    lambda x: str(int(float(x))) if pd.notna(x) and x not in ("—", "") else "—"
                )
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Gemini analysis
    st.divider()
    st.markdown('<div class="section-title">🤖 Gemini AI Career Prediction</div>', unsafe_allow_html=True)

    if not effective_key:
        st.warning(
            "⚠️ No Google API key found. "
            "Set `GOOGLE_API_KEY` as an environment variable or enter it in the sidebar."
        )
    else:
        try:
            with st.spinner("Gemini is thinking..."):
                st.write_stream(stream_analysis(target, similar, metrics, effective_key))
        except Exception as e:
            if "API_KEY" in str(e).upper() or "401" in str(e) or "403" in str(e):
                st.error("❌ Invalid API key. Get your key at aistudio.google.com")
            elif "429" in str(e) or "quota" in str(e).lower():
                st.error("⚠️ Rate limited. Please wait a moment and try again.")
            else:
                st.error(f"Gemini API error: {e}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Read key from Streamlit secrets (Cloud) or env var (local)
    _default_key = st.secrets.get("GOOGLE_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        placeholder="AIza...",
        value=_default_key,
        help="Get your key at aistudio.google.com",
    )

    st.divider()
    num_comps = st.slider("Historical comps to find", min_value=5, max_value=20, value=10)

    col_a, col_b = st.columns(2)
    with col_a:
        start_year = st.number_input("From", min_value=2000, max_value=CURRENT_YEAR, value=2000, step=1)
    with col_b:
        end_year = st.number_input("To", min_value=2000, max_value=CURRENT_YEAR,
                                   value=min(CURRENT_YEAR - 1, 2025), step=1)

    st.divider()
    st.caption(f"Data: nflverse (2000–{end_year})\nModel: Gemini 2.0 Flash")
    st.caption("By Maxintech")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏈 NFL Combine Career Predictor")
st.markdown(
    "Search any NFL combine participant **or enter stats manually** for current combine players. "
    "Claude AI compares their metrics against thousands of historical players and predicts career trajectory."
)

# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Input mode",
    ["🔍 Search by name", "✍️ Enter stats manually"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("")  # spacing


# ══════════════════════════════════════════════════════════════════════════════
# MODE A: Search by name
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🔍 Search by name":

    c1, c2 = st.columns([5, 1])
    with c1:
        player_input = st.text_input(
            "player",
            placeholder="e.g.  Sonny Styles  ·  Caleb Williams  ·  Ja'Marr Chase",
            label_visibility="collapsed",
        )
    with c2:
        search_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if search_clicked and player_input.strip():
        st.session_state.searched_name   = player_input.strip()
        st.session_state.target_dict     = None
        st.session_state.is_manual_entry = False

        with st.spinner("Loading combine database..."):
            try:
                df = cached_data(int(start_year), int(end_year))
            except Exception as e:
                st.error(f"Failed to load combine data: {e}")
                st.stop()

        found = find_player(df, player_input.strip())

        if found is not None:
            st.session_state.target_dict = found.to_dict()
            st.session_state.app_step    = "results"
        else:
            st.session_state.app_step = "manual_hint"

    elif search_clicked and not player_input.strip():
        st.warning("Please enter a player name.")

    # Show "not found" hint suggesting manual mode
    if st.session_state.app_step == "manual_hint":
        st.error(
            f"❌ No combine record found for **{st.session_state.searched_name}**. "
            f"If they're in the {CURRENT_YEAR} combine class, switch to "
            f"**✍️ Enter stats manually** mode above."
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODE B: Enter stats manually  (covers current-year combine players)
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.caption(
        f"Use this for {CURRENT_YEAR} combine players whose data hasn't been uploaded yet. "
        "Look up their results on NFL.com or ESPN."
    )

    sorted_positions = sorted(POSITION_METRICS.keys())

    with st.form("manual_entry_form"):
        # ── Basic info ────────────────────────────────────────────────────────
        st.markdown("**Player Info**")
        bi1, bi2, bi3, bi4, bi5 = st.columns(5)
        with bi1:
            m_name = st.text_input("Player Name *", placeholder="e.g. Sonny Styles")
        with bi2:
            m_pos = st.selectbox("Position *", options=sorted_positions)
        with bi3:
            m_ht = st.text_input("Height (ft-in)", placeholder="e.g. 6-2")
        with bi4:
            m_wt = st.number_input("Weight (lbs)", min_value=0, max_value=400, value=0, step=1)
        with bi5:
            m_school = st.selectbox("School", options=get_school_list())

        st.divider()

        # ── Combine metrics ────────────────────────────────────────────────────
        st.markdown("**Combine Results**")
        st.info(
            "Enter only the stats you have — leave anything unknown at **0**. "
            "Only the stats you fill in are used for the historical comparison. "
            "Even a single metric (like the 40-yard dash) is enough to find comps.",
            icon="ℹ️",
        )

        pos_metrics = get_metrics_for_position(m_pos)
        # Metrics to show in the generic grid (excluding special-cased ones)
        other_metrics = [
            m for m in pos_metrics
            if m not in ("ht", "wt", "forty", "vertical", "broad_jump")
        ]

        # Speed row
        st.markdown("*Speed*")
        sp1, sp2, sp3, sp4 = st.columns(4)
        with sp1:
            m_forty = st.number_input("40-Yard Dash (s)", min_value=0.0, value=0.0,
                                      step=0.01, format="%.2f", help="0 = unknown")
        with sp2:
            m_ten = st.number_input("10-Yard Split (s)", min_value=0.0, value=0.0,
                                    step=0.01, format="%.2f", help="0 = unknown")

        # Agility / Strength (position-aware)
        if other_metrics:
            st.markdown("*Agility & Strength*")
            ag_cols = st.columns(4)
            m_vals: dict = {}
            for i, m in enumerate(other_metrics):
                with ag_cols[i % 4]:
                    is_time = m in ("shuttle", "cone")
                    m_vals[m] = st.number_input(
                        METRIC_LABELS.get(m, m),
                        min_value=0.0, value=0.0,
                        step=0.01 if is_time else 1.0,
                        format="%.2f" if is_time else "%.0f",
                        help="0 = not performed / unknown",
                        key=f"manual_{m}",
                    )
        else:
            m_vals = {}

        # Jump tests (ft-in text format, same as height)
        st.markdown("*Jump Tests*")
        j1, j2, j3, j4 = st.columns(4)
        with j1:
            m_vertical = st.text_input("Vertical Jump (ft-in)", placeholder="e.g. 3-9",
                                       help="Format: feet-inches, e.g. 3-9 for 3'9\"")
        with j3:
            m_broad = st.text_input("Broad Jump (ft-in)", placeholder="e.g. 10-6",
                                    help="Format: feet-inches, e.g. 10-6 for 10'6\"")

        st.divider()
        m_submitted = st.form_submit_button("🔍 Compare & Analyze", type="primary", use_container_width=True)

    if m_submitted:
        if not m_name.strip():
            st.warning("Please enter a player name.")
        else:
            ht_num      = height_to_inches(m_ht.strip())     if m_ht.strip()     else float("nan")
            vertical_in = height_to_inches(m_vertical.strip()) if m_vertical.strip() else float("nan")
            broad_in    = height_to_inches(m_broad.strip())    if m_broad.strip()    else float("nan")
            manual_data = {
                "player_name": m_name.strip(),
                "pos":         m_pos,
                "season":      float(CURRENT_YEAR),
                "ht":          ht_num,
                "ht_display":  m_ht.strip() or "N/A",
                "wt":          float(m_wt) if m_wt > 0 else float("nan"),
                "school":      m_school,
                "draft_round": float("nan"),
                "draft_ovr":   float("nan"),
                "forty":       float(m_forty) if m_forty > 0 else float("nan"),
                "ten_split":   float(m_ten) if m_ten > 0 else float("nan"),
                "vertical":    vertical_in,
                "broad_jump":  broad_in,
            }
            for m, v in m_vals.items():
                manual_data[m] = float(v) if v > 0 else float("nan")

            st.session_state.target_dict     = manual_data
            st.session_state.is_manual_entry = True
            st.session_state.app_step        = "results"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS (shared by both modes)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.app_step == "results" and st.session_state.target_dict:
    effective_key = api_key_input.strip() or os.getenv("GOOGLE_API_KEY", "")

    with st.spinner("Loading combine database..."):
        try:
            df = cached_data(int(start_year), int(end_year))
        except Exception as e:
            st.error(f"Failed to load combine data: {e}")
            st.stop()

    target = pd.Series(st.session_state.target_dict)

    show_results(
        target        = target,
        df            = df,
        num_comps     = num_comps,
        effective_key = effective_key,
        is_manual     = st.session_state.is_manual_entry,
    )
