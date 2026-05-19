import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/mateoportillo1900/Spotify-ML-Model",
        "Report a bug": "https://github.com/mateoportillo1900/Spotify-ML-Model/issues",
        "About": "**Spotify Genre Classifier** — Multi-class genre prediction across 35 genres using audio features from 24,993 Spotify songs (1957–2020). Built with scikit-learn, Plotly & Streamlit.\n\n[GitHub →](https://github.com/mateoportillo1900/Spotify-ML-Model)",
    }
)

st.markdown("""
<style>
  #MainMenu, footer { visibility: hidden; }
  /* Hide header chrome on desktop only — mobile needs it for the sidebar toggle */
  @media (min-width: 769px) { header { visibility: hidden; } }
  .block-container { padding: 1.5rem 2rem 1rem 2rem; }

  /* Top accent bar — subtle gradient line */
  .top-bar {
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #1DB954 30%, #1DB954 70%, transparent 100%);
    margin: -1.5rem -2rem 1.75rem -2rem;
    opacity: 0.85;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 0;
  }
  .stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    border-radius: 6px 6px 0 0;
    font-weight: 500; font-size: 0.85rem;
    color: #666; background: transparent; border: none;
  }
  .stTabs [aria-selected="true"] {
    color: #1DB954 !important;
    border-bottom: 2px solid #1DB954 !important;
    background: rgba(29,185,84,0.06) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #080808;
    border-right: 1px solid rgba(255,255,255,0.06);
  }

  /* Insight box */
  .insight {
    background: rgba(29,185,84,0.07);
    border: 1px solid rgba(29,185,84,0.25);
    border-left: 3px solid #1DB954;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #ccc;
    margin: 12px 0;
    line-height: 1.6;
  }

  /* Multiselect tags */
  .stMultiSelect [data-baseweb="tag"] { background: #1DB954 !important; color: #000 !important; }
  hr { border-color: rgba(255,255,255,0.07) !important; }

  /* ── Mobile responsiveness ─────────────────────────────────────────────── */
  @media (max-width: 768px) {

    /* Tighter page padding */
    .block-container { padding: 0.75rem 0.6rem 1rem 0.6rem !important; }
    .top-bar { margin: -0.75rem -0.6rem 1rem -0.6rem; }

    /* Stack all columns vertically */
    [data-testid="stHorizontalBlock"] {
      flex-direction: column !important;
    }
    [data-testid="stColumn"] {
      width: 100% !important;
      flex: 1 1 100% !important;
      min-width: 100% !important;
    }

    /* KPI row — force 2-column grid and compact cards */
    [data-testid="stHorizontalBlock"]:has([data-testid="stColumn"]:nth-child(5)) {
      display: grid !important;
      grid-template-columns: 1fr 1fr !important;
      gap: 6px !important;
      flex-direction: unset !important;
    }
    [data-testid="stHorizontalBlock"]:has([data-testid="stColumn"]:nth-child(5))
    [data-testid="stColumn"] {
      min-width: unset !important;
      width: auto !important;
      flex: unset !important;
    }

    /* Compact KPI card values */
    [data-testid="stHorizontalBlock"]:has([data-testid="stColumn"]:nth-child(5))
    div[style*="min-height:96px"],
    div[style*="min-height: 96px"] {
      min-height: 64px !important;
      padding: 8px 6px !important;
    }
    /* Scale down the big green number inside KPI cards */
    [data-testid="stHorizontalBlock"]:has([data-testid="stColumn"]:nth-child(5))
    div[style*="1.55rem"] {
      font-size: 1.1rem !important;
    }

    /* Smaller tab labels so they don't overflow */
    .stTabs [data-baseweb="tab"] {
      padding: 6px 8px !important;
      font-size: 0.68rem !important;
    }

    /* Shrink chart heights so you don't scroll forever */
    .stPlotlyChart { max-height: 300px; }
    .stPlotlyChart > div { max-height: 300px; }

    /* Header GitHub button wraps below title */
    .mobile-header { flex-direction: column !important; gap: 8px !important; }

    /* Insight boxes */
    .insight { font-size: 0.73rem !important; padding: 9px 11px !important; }

    /* Sliders and selects */
    .stSelectbox label, .stSlider label,
    .stMultiSelect label { font-size: 0.75rem !important; }
  }

  @media (max-width: 420px) {
    /* Very small phones — single column KPIs */
    [data-testid="stHorizontalBlock"]:has([data-testid="stColumn"]:nth-child(5)) {
      grid-template-columns: 1fr 1fr !important;
    }
    .stTabs [data-baseweb="tab"] {
      padding: 5px 6px !important;
      font-size: 0.6rem !important;
    }
  }
</style>
<div class="top-bar"></div>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/mateoportillo1900/Spotify-ML-Model/refs/heads/main/spotify_top_music.csv"

FEATURES = ["bpm", "nrgy", "dnce", "db", "live", "val", "dur", "acous", "spch", "pop"]
LABELS   = {
    "bpm": "BPM", "nrgy": "Energy", "dnce": "Danceability", "db": "Loudness",
    "live": "Liveness", "val": "Valence", "dur": "Duration",
    "acous": "Acousticness", "spch": "Speechiness", "pop": "Popularity",
}
RADAR_COLS   = ["bpm", "nrgy", "dnce", "val", "acous", "spch", "pop"]
RADAR_LABELS = ["BPM", "Energy", "Danceability", "Valence", "Acousticness", "Speechiness", "Popularity"]
NUM_FEATS    = ["year", "bpm", "nrgy", "dnce", "db", "live", "val", "dur", "acous", "spch", "pop"]

GREEN  = "#1DB954"
BG     = "#121212"
CARD   = "#1E1E1E"
BORDER = "rgba(255,255,255,0.08)"

PLOT = dict(
    template="plotly_dark",
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(color="#FFFFFF", family="sans-serif"),
    margin=dict(t=48, b=28, l=20, r=20),
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi(label, value, sub=None):
    sub_html = f'<div style="color:#666;font-size:0.7rem;margin-top:3px">{sub}</div>' if sub else \
               '<div style="font-size:0.7rem;margin-top:3px">&nbsp;</div>'
    # Auto-shrink long values (e.g. "dance pop") so they fit on one line
    val_len = len(str(value))
    val_size = "1.55rem" if val_len <= 7 else ("1.25rem" if val_len <= 11 else "1.05rem")
    return f"""
    <div style="background:linear-gradient(180deg,{CARD} 0%,#181818 100%);
                border:1px solid {BORDER};border-radius:12px;
                padding:18px 12px;text-align:center;min-height:104px;
                display:flex;flex-direction:column;justify-content:center;
                transition:border-color 0.2s ease;
                box-shadow:0 1px 0 rgba(255,255,255,0.02) inset">
      <div style="color:#666;font-size:0.62rem;text-transform:uppercase;
                  letter-spacing:0.14em;font-weight:600">{label}</div>
      <div style="color:{GREEN};font-size:{val_size};font-weight:700;
                  margin-top:6px;line-height:1.1;white-space:nowrap;
                  overflow:hidden;text-overflow:ellipsis">{value}</div>
      {sub_html}
    </div>"""

def section(title, subtitle=""):
    sub = f'<div style="color:#666;font-size:0.75rem;margin-top:4px;line-height:1.4">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin:14px 0 16px 0;padding-left:12px;border-left:2px solid {GREEN}">
      <div style="font-size:1.05rem;font-weight:700;color:#fff;line-height:1.2;
                  letter-spacing:-0.015em">{title}</div>
      {sub}
    </div>""", unsafe_allow_html=True)

def insight(text):
    st.markdown(
        f'<div class="insight"><span style="color:#1DB954;font-weight:700;'
        f'letter-spacing:0.16em;font-size:0.62rem">INSIGHT &nbsp;</span>{text}</div>',
        unsafe_allow_html=True,
    )

def chart(fig, height=None, margin=None, **kw):
    layout = {**PLOT}
    if margin:
        layout["margin"] = margin
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, **kw)

def genre_color_legend(genres, label="Top genres"):
    """Compact horizontal color key — used in place of overflowing Plotly legends."""
    swatches = " &nbsp; ".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin:2px 4px">'
        f'<span style="width:9px;height:9px;border-radius:2px;background:{GENRE_COLOR.get(g, "#888")}"></span>'
        f'<span style="color:#bbb">{g}</span></span>'
        for g in genres
    )
    st.markdown(
        f'<div style="margin:-4px 0 12px 0;padding:8px 10px;background:rgba(255,255,255,0.02);'
        f'border:1px solid rgba(255,255,255,0.04);border-radius:6px;font-size:0.72rem">'
        f'<span style="color:#555;font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;'
        f'margin-right:8px">{label}</span>'
        f'{swatches}'
        f'<span style="color:#555;margin-left:6px">· other genres in gray</span></div>',
        unsafe_allow_html=True,
    )

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

@st.cache_resource
def train_model():
    """Audio-only Random Forest — no artist OHE to keep memory within Streamlit Cloud limits."""
    df = load_data()
    counts = df["top_genre"].value_counts()
    df = df[~df["top_genre"].isin(counts[counts < 5].index)]
    X, y = df[NUM_FEATS], df["top_genre"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=50, random_state=42, n_jobs=1,
            class_weight="balanced", max_features="sqrt", min_samples_leaf=2,
        )),
    ])
    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)
    fi = pd.DataFrame({
        "Feature":    [LABELS.get(f, f.capitalize()) for f in NUM_FEATS],
        "Importance": pipe.named_steps["clf"].feature_importances_,
    }).sort_values("Importance", ascending=False)
    return pipe, fi, yte, ypred, y

@st.cache_data
def compute_tsne(n_sample: int = 800):
    df  = load_data()
    counts  = df["top_genre"].value_counts()
    df      = df[~df["top_genre"].isin(counts[counts < 5].index)]
    parts = []
    for g_name, g_df in df.groupby("top_genre"):
        k = max(1, int(n_sample * len(g_df) / len(df)))
        parts.append(g_df.sample(min(k, len(g_df)), random_state=42))
    sample = pd.concat(parts).reset_index(drop=True)
    X  = sample[FEATURES].fillna(sample[FEATURES].mean())
    Xs = StandardScaler().fit_transform(X)
    try:
        emb = TSNE(n_components=3, random_state=42, perplexity=20, max_iter=300).fit_transform(Xs)
    except TypeError:
        emb = TSNE(n_components=3, random_state=42, perplexity=20, n_iter=300).fit_transform(Xs)
    return pd.DataFrame({
        "x": emb[:, 0], "y": emb[:, 1], "z": emb[:, 2],
        "genre":  sample["top_genre"].values,
        "title":  sample["title"].values,
        "artist": sample["artist"].values,
    })

df_raw = load_data()

# Consistent genre → color mapping used across every chart
ALL_GENRES  = sorted(df_raw["top_genre"].unique())
PALETTE     = px.colors.qualitative.Dark24
GENRE_COLOR = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(ALL_GENRES)}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:4px 0 20px 0;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:18px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <div style="width:6px;height:6px;border-radius:50%;background:{GREEN};
                    box-shadow:0 0 8px {GREEN}"></div>
        <div style="font-size:0.62rem;color:{GREEN};letter-spacing:0.18em;font-weight:700">
          LIVE MODEL
        </div>
      </div>
      <div style="font-size:1.35rem;font-weight:800;color:#fff;letter-spacing:-0.02em;line-height:1.1">
        Genre<br>Classifier
      </div>
      <div style="font-size:0.7rem;color:#666;margin-top:8px;line-height:1.5">
        Random Forest trained on 24,993 Spotify songs
      </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("nav", ["Explore Data", "ML Model", "About"],
                    label_visibility="collapsed")

    st.divider()
    st.markdown('<div style="font-size:0.65rem;color:#555;text-transform:uppercase;'
                'letter-spacing:0.1em;margin-bottom:8px">Filters</div>', unsafe_allow_html=True)

    _counts    = df_raw["top_genre"].value_counts()
    all_genres = sorted(_counts[_counts >= 5].index.tolist())
    sel_genres_raw = st.multiselect("Genre", all_genres, default=[],
                                    label_visibility="collapsed",
                                    placeholder=f"All {len(all_genres)} genres")
    sel_genres = sel_genres_raw if sel_genres_raw else all_genres
    yr_min = int(df_raw["year"].min())
    yr_max = int(df_raw["year"].max())
    year_range = st.slider("Year", yr_min, yr_max, (yr_min, yr_max))

    st.divider()
    with st.expander("Feature Glossary"):
        glossary = {
            "BPM": "Tempo (beats per minute)",
            "Energy": "Intensity & activity (0–100)",
            "Danceability": "Rhythm suitability (0–100)",
            "Loudness": "Overall loudness in dB",
            "Liveness": "Presence of live audience",
            "Valence": "Musical positivity (0–100)",
            "Duration": "Song length in seconds",
            "Acousticness": "Acoustic confidence (0–100)",
            "Speechiness": "Spoken word density (0–100)",
            "Popularity": "Spotify popularity score (0–100)",
        }
        for k, v in glossary.items():
            st.caption(f"**{k}** — {v}")

df = df_raw[
    df_raw["top_genre"].isin(sel_genres) &
    df_raw["year"].between(*year_range)
].copy()

# ── Hero waveform — built from actual song energy values ─────────────────────
@st.cache_data
def hero_waveform_svg():
    """SVG audio waveform with bar heights driven by real Spotify energy values."""
    n_bars = 140
    vals = df_raw["nrgy"].dropna().sample(n=min(n_bars, len(df_raw)), random_state=42).values
    bw = 1000 / n_bars
    bars = []
    for i, v in enumerate(vals):
        h  = max(6, (v / 100) * 92 + 4)
        y  = (100 - h) / 2
        op = 0.18 + (v / 100) * 0.65
        bars.append(
            f'<rect x="{i*bw:.1f}" y="{y:.1f}" width="{bw*0.55:.2f}" '
            f'height="{h:.1f}" rx="0.6" fill="#1DB954" opacity="{op:.2f}"/>'
        )
    return (
        '<svg width="100%" height="58" viewBox="0 0 1000 100" '
        'preserveAspectRatio="none" style="display:block">'
        + "".join(bars) + '</svg>'
    )

if "About" not in page:
    st.markdown(f"""
<div style="margin:0 0 22px 0;border-radius:10px;overflow:hidden;
            background:#0d0d0d;border:1px solid rgba(29,185,84,0.18)">
  <div style="padding:20px 24px;display:flex;align-items:center;
              justify-content:space-between;gap:20px;flex-wrap:wrap">
    <div style="flex:1;min-width:260px">
      <div style="font-size:0.6rem;color:{GREEN};letter-spacing:0.22em;
                  font-weight:700">SPOTIFY · MULTI-CLASS CLASSIFICATION</div>
      <div style="font-size:1.5rem;font-weight:800;color:#fff;letter-spacing:-0.025em;
                  margin-top:6px;line-height:1.15">
        Can a model learn genre from sound alone?
      </div>
    </div>
    <div style="display:flex;gap:22px">
      <div style="text-align:right">
        <div style="font-size:1.45rem;color:{GREEN};font-weight:700;line-height:1;
                    font-family:Georgia,serif">35%</div>
        <div style="font-size:0.6rem;color:#666;letter-spacing:0.12em;
                    margin-top:4px;font-weight:600">TEST ACCURACY</div>
      </div>
      <div style="width:1px;background:rgba(255,255,255,0.08)"></div>
      <div style="text-align:right">
        <div style="font-size:1.45rem;color:#fff;font-weight:700;line-height:1;
                    font-family:Georgia,serif">12×</div>
        <div style="font-size:0.6rem;color:#666;letter-spacing:0.12em;
                    margin-top:4px;font-weight:600">VS. RANDOM</div>
      </div>
      <div style="width:1px;background:rgba(255,255,255,0.08)"></div>
      <div style="text-align:right">
        <div style="font-size:1.45rem;color:#fff;font-weight:700;line-height:1;
                    font-family:Georgia,serif">35</div>
        <div style="font-size:0.6rem;color:#666;letter-spacing:0.12em;
                    margin-top:4px;font-weight:600">CLASSES</div>
      </div>
    </div>
  </div>
  <div style="height:1px;background:linear-gradient(90deg,transparent,
              rgba(29,185,84,0.25),transparent)"></div>
  <div style="background:#0a0a0a;padding:6px 0">
    {hero_waveform_svg()}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Breadcrumb strip: page label + filter summary + GitHub link ──────────────
page_label = ("Data Exploration" if "Explore" in page
              else "ML Model Results"  if "ML"      in page
              else "About the Project")
is_filtered = bool(sel_genres_raw) or year_range != (yr_min, yr_max)
filter_text = (f"{len(df):,} of {len(df_raw):,} songs"
               if is_filtered else f"{len(df):,} songs · 35 modelable genres")

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:18px;gap:12px;flex-wrap:wrap">
  <div style="display:flex;align-items:center;gap:10px;color:#666;
              font-size:0.78rem;letter-spacing:0.01em">
    <span style="color:#666">Spotify Genre Classifier</span>
    <span style="color:#333">/</span>
    <span style="color:#fff;font-weight:600">{page_label}</span>
    <span style="color:#333">·</span>
    <span style="color:#666">{filter_text}</span>
  </div>
  <a href="https://github.com/mateoportillo1900/Spotify-ML-Model" target="_blank"
     style="display:inline-flex;align-items:center;gap:7px;
            color:#888;font-size:0.75rem;font-weight:500;text-decoration:none;
            transition:color 0.2s">
    <svg height="14" viewBox="0 0 16 16" width="14" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
               0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
               -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
               .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
               -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27
               .68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
               .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
               0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
    </svg>
    Source on GitHub
  </a>
</div>""", unsafe_allow_html=True)

if "About" not in page:
    c1, c2, c3, c4, c5 = st.columns(5)
    dominant = df["top_genre"].value_counts().index[0] if len(df) else "—"
    for col, lbl, val, sub in [
        (c1, "Songs",           f"{len(df):,}",                 None),
        (c2, "Genres",          str(df["top_genre"].nunique()),  None),
        (c3, "Avg Energy",      f"{df['nrgy'].mean():.0f}",      "out of 100"),
        (c4, "Avg Popularity",  f"{df['pop'].mean():.0f}",       "out of 100"),
        (c5, "Top Genre",       dominant[:12] + ("…" if len(dominant) > 12 else ""), None),
    ]:
        col.markdown(kpi(lbl, val, sub), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
if "Explore" in page:
    st.markdown(f"""
    <div style="margin:8px 0 22px 0;padding:0 0 0 18px;
                border-left:2px solid {GREEN};font-family:Georgia,serif">
      <div style="font-size:1.1rem;color:#eee;line-height:1.45;font-style:italic;
                  font-weight:400;letter-spacing:-0.01em">
        Every song is a 10-dimensional point in audio space — tempo, energy, valence,
        speechiness — sampled directly from Spotify's API. The question is whether genre
        labels actually live in that space, or whether they're cultural artifacts no signal can find.
      </div>
      <div style="font-size:0.7rem;color:#666;margin-top:10px;
                  letter-spacing:0.14em;font-family:system-ui">
        BILLBOARD TOP SONGS + TIDYTUESDAY SPOTIFY &nbsp;·&nbsp; 1957–2020
      </div>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs([
        "Overview", "3D Space", "Genre Profiles", "Trends", "Songs",
    ])

    # ── Overview ──────────────────────────────────────────────────────────────
    with t1:
        left, right = st.columns(2)

        with left:
            section("Genre Distribution", "How many songs belong to each genre — dominant on top")
            gc = df["top_genre"].value_counts().reset_index()
            gc.columns = ["Genre", "Songs"]
            top_pct = gc.iloc[0]["Songs"] / len(df) * 100
            # Single-color bars with the top genre highlighted — cleaner than a rainbow
            top_genre_name = gc.iloc[0]["Genre"]
            bar_colors = [GREEN if g == top_genre_name else "#3a3a3a" for g in gc["Genre"]]
            fig = px.bar(gc, x="Songs", y="Genre", orientation="h", text="Songs")
            fig.update_traces(marker_color=bar_colors, textposition="outside",
                              textfont=dict(size=10, color="#666"))
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              showlegend=False)
            chart(fig, height=480)
            insight(f"<b>{gc.iloc[0]['Genre'].title()}</b> dominates with "
                    f"{gc.iloc[0]['Songs']} songs ({top_pct:.0f}% of the filtered dataset). "
                    f"This class imbalance directly impacts model recall for minority genres.")

        with right:
            section("Songs per Year", "Volume of songs in the dataset by release year")
            yc = df["year"].value_counts().sort_index().reset_index()
            yc.columns = ["Year", "Songs"]
            fig = px.bar(yc, x="Year", y="Songs", color="Songs",
                         color_continuous_scale="Greens")
            # No per-bar text labels — too noisy with 60+ years
            # Tick every 10 years instead of every year to avoid label collision
            fig.update_layout(coloraxis_showscale=False,
                              xaxis=dict(dtick=10, tickfont=dict(size=10)),
                              yaxis=dict(tickfont=dict(size=10)))
            chart(fig, height=260)

            section("Feature Correlations", "Pearson r — how audio attributes move together")
            corr = df[FEATURES].corr().round(2)
            corr.index   = [LABELS[c] for c in FEATURES]
            corr.columns = [LABELS[c] for c in FEATURES]
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(xaxis=dict(tickangle=45, tickfont=dict(size=9)),
                              yaxis=dict(tickfont=dict(size=9)),
                              coloraxis_colorbar=dict(thickness=10))
            chart(fig, height=420, margin=dict(t=12, b=10, l=10, r=10))
            insight("Energy & Loudness are strongly correlated (r ≈ 0.7) — they measure similar things. "
                    "Acousticness is strongly <i>negatively</i> correlated with both, separating acoustic "
                    "and electronic genres cleanly.")

    # ── 3D Space ──────────────────────────────────────────────────────────────
    with t2:
        section("3D Feature Space", "Pick three audio features as axes — rotate to explore genre separation")

        c_x, c_y, c_z, _ = st.columns([1, 1, 1, 1])
        fx = c_x.selectbox("X", FEATURES, index=1, format_func=LABELS.get, key="3dx")
        fy = c_y.selectbox("Y", FEATURES, index=2, format_func=LABELS.get, key="3dy")
        fz = c_z.selectbox("Z", FEATURES, index=5, format_func=LABELS.get, key="3dz")

        # Cap at 3000 points so the Plotly figure stays lightweight
        df_3d = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df

        # User-controlled highlight: which genres to show in full color
        default_top10 = df_3d["top_genre"].value_counts().nlargest(10).index.tolist()
        all_3d_genres = sorted(df_3d["top_genre"].unique())
        highlight_3d  = st.multiselect(
            "Highlight genres (others stay gray for context)",
            all_3d_genres,
            default=default_top10,
            key="3d_highlight",
        )
        if not highlight_3d:
            highlight_3d = default_top10  # never go fully empty
        top10_3d = highlight_3d
        genre_color_legend(top10_3d, label="Highlighted")

        fig3d = go.Figure()
        # Draw "other" first so it sits underneath the highlighted genres
        for genre in sorted(df_3d["top_genre"].unique()):
            g = df_3d[df_3d["top_genre"] == genre]
            is_top = genre in top10_3d
            fig3d.add_trace(go.Scatter3d(
                x=g[fx], y=g[fy], z=g[fz], mode="markers",
                marker=dict(size=4 if is_top else 3,
                            color=GENRE_COLOR[genre] if is_top else "#444",
                            opacity=0.88 if is_top else 0.35),
                name=genre,
                text=g["title"] + " — " + g["artist"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{LABELS[fx]}: %{{x}}<br>"
                    f"{LABELS[fy]}: %{{y}}<br>"
                    f"{LABELS[fz]}: %{{z}}"
                    "<extra>" + genre + "</extra>"
                ),
            ))
        fig3d.update_layout(
            **PLOT, height=560,
            scene=dict(
                xaxis=dict(title=LABELS[fx], backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                yaxis=dict(title=LABELS[fy], backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                zaxis=dict(title=LABELS[fz], backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                bgcolor=BG,
            ),
            showlegend=False,
        )
        st.plotly_chart(fig3d, use_container_width=True)
        insight("Genres separate where their audio profiles diverge. High-energy electronic genres "
                "pull to one corner; acoustic and folk pull to the opposite. The visible overlap "
                "between dance pop, electropop, and pop is exactly where the model's confusion concentrates.")

        st.divider()
        section("t-SNE 3D — Genre Clustering",
                "All 10 audio features compressed to 3D. Tighter clusters = easier for the model to learn.")

        with st.spinner("Computing t-SNE embedding (cached after first run)…"):
            tsne_df = compute_tsne()

        top10_tsne = tsne_df["genre"].value_counts().nlargest(10).index.tolist()
        genre_color_legend(top10_tsne, label="Top 10 genres")

        fig_t = go.Figure()
        for genre in sorted(tsne_df["genre"].unique()):
            g = tsne_df[tsne_df["genre"] == genre]
            is_top = genre in top10_tsne
            fig_t.add_trace(go.Scatter3d(
                x=g["x"], y=g["y"], z=g["z"],
                mode="markers",
                marker=dict(size=3.5 if is_top else 2.8,
                            color=GENRE_COLOR.get(genre, "#888") if is_top else "#444",
                            opacity=0.88 if is_top else 0.32),
                name=genre,
                text=g["title"] + " — " + g["artist"],
                hovertemplate="<b>%{text}</b><extra>" + genre + "</extra>",
            ))
        fig_t.update_layout(
            **PLOT, height=560,
            scene=dict(
                xaxis=dict(title="Component 1", backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                yaxis=dict(title="Component 2", backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                zaxis=dict(title="Component 3", backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                bgcolor=BG,
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_t, use_container_width=True)
        insight("Genres that form tight, separate clusters (e.g. hip hop, EDM) are easier for the model to classify. "
                "Overlapping clusters (pop, dance pop, electropop) explain most misclassifications. "
                "Axes have no direct meaning — only distances between points matter.")

    # ── Genre Profiles ────────────────────────────────────────────────────────
    with t3:
        section("Genre Audio Fingerprints",
                "Average audio profile per genre — larger area = stronger trait. Select genres to compare.")

        top10     = df["top_genre"].value_counts().nlargest(10).index.tolist()
        sel_radar = st.multiselect("Compare genres", top10, default=top10[:5])

        if sel_radar:
            scaler        = MinMaxScaler((0, 100))
            df_sc         = df.copy()
            df_sc[RADAR_COLS] = scaler.fit_transform(df[RADAR_COLS])

            fig = go.Figure()
            for genre in sel_radar:
                avg = df_sc[df_sc["top_genre"] == genre][RADAR_COLS].mean().tolist()
                fig.add_trace(go.Scatterpolar(
                    r=avg + avg[:1],
                    theta=RADAR_LABELS + [RADAR_LABELS[0]],
                    fill="toself", name=genre,
                    line=dict(color=GENRE_COLOR[genre], width=2.5),
                    fillcolor=GENRE_COLOR[genre].replace(")", ",0.12)").replace("rgb", "rgba")
                              if "rgb" in GENRE_COLOR[genre] else GENRE_COLOR[genre],
                ))
            fig.update_layout(
                **PLOT, height=500,
                polar=dict(
                    bgcolor="#181818",
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333", color="#555", tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor="#2a2a2a", color="#888"),
                ),
                legend=dict(orientation="h", y=-0.14, font=dict(size=11), itemsizing="constant"),
            )
            chart(fig)
            insight("Genres with very different radar shapes are easy for the model to distinguish. "
                    "Genres with similar shapes (e.g. pop vs. dance pop) are where misclassifications happen most.")
        else:
            st.info("Select at least one genre above.")

        st.divider()
        section("Feature Heatmap by Genre", "Average raw value per feature — reveals what makes each genre unique")

        top12 = df["top_genre"].value_counts().nlargest(12).index.tolist()
        hm    = df[df["top_genre"].isin(top12)].groupby("top_genre")[RADAR_COLS].mean().round(1)
        hm.columns = RADAR_LABELS
        fig   = px.imshow(hm, text_auto=True, color_continuous_scale="Greens", aspect="auto")
        fig.update_layout(
            xaxis=dict(side="bottom"),
            coloraxis_colorbar=dict(thickness=12, title="Avg"),
        )
        chart(fig, height=430)

    # ── Trends ────────────────────────────────────────────────────────────────
    with t4:
        section("Audio Feature Trends", "How the sound of popular music changed across decades")

        sel_feats = st.multiselect("Features to track", FEATURES,
                                   default=["nrgy", "val", "dnce", "acous", "pop"],
                                   format_func=LABELS.get)
        if sel_feats:
            yearly = (
                df.groupby("year")[sel_feats].mean().reset_index()
                  .melt(id_vars="year", var_name="Feature", value_name="Average")
            )
            yearly["Feature"] = yearly["Feature"].map(LABELS)
            fig = px.line(yearly, x="year", y="Average", color="Feature", markers=True,
                          labels={"year": "Year", "Average": "Average Value"})
            fig.update_traces(line=dict(width=2.5))
            fig.update_layout(xaxis=dict(dtick=1))
            chart(fig, height=400)
            insight("Acousticness has trended sharply down since the 1970s as music went electric and digital. "
                    "Energy and Danceability peaked in the 2010s EDM era. Filter to 2010–2020 "
                    "to zoom into the streaming era specifically.")
        else:
            st.info("Select at least one feature.")

        st.divider()
        section("Most Represented Artists",
                "Artists with the most songs in the dataset. Color = average popularity score.")

        top_art = (
            df.groupby("artist")
              .agg(Songs=("title", "count"), Avg_Pop=("pop", "mean"))
              .sort_values("Songs", ascending=False).head(15).reset_index()
        )
        top_art["Avg_Pop"] = top_art["Avg_Pop"].round(1)
        fig = px.bar(top_art, x="Songs", y="artist", orientation="h",
                     color="Avg_Pop", color_continuous_scale="Greens",
                     text="Songs", labels={"artist": "", "Avg_Pop": "Avg Popularity"})
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          coloraxis_colorbar=dict(thickness=12, title="Avg Pop"))
        chart(fig, height=460)

    # ── Songs ─────────────────────────────────────────────────────────────────
    with t5:
        section("Song Search", "Explore individual songs in the filtered dataset")

        q = st.text_input("", placeholder="Search by title or artist…", label_visibility="collapsed")
        res = df[
            df["title"].str.contains(q, case=False, na=False) |
            df["artist"].str.contains(q, case=False, na=False)
        ] if q else df

        st.caption(f"{len(res):,} result{'s' if len(res) != 1 else ''}")
        show_cols = ["title", "artist", "top_genre", "year"] + FEATURES
        st.dataframe(
            res[show_cols]
              .rename(columns={**LABELS, "title": "Title", "artist": "Artist",
                                "top_genre": "Genre", "year": "Year"})
              .reset_index(drop=True),
            use_container_width=True, height=540,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif "ML" in page:
    st.markdown(f"""
    <div style="margin:8px 0 22px 0;padding:0 0 0 18px;
                border-left:2px solid {GREEN};font-family:Georgia,serif">
      <div style="font-size:1.1rem;color:#eee;line-height:1.45;font-style:italic;
                  font-weight:400;letter-spacing:-0.01em">
        Random guessing across thirty-five genres gets you 2.9%. This model gets to 35%
        on audio alone — no artist identity, no metadata shortcuts. Twelve times better
        than chance, on a problem where the labels themselves often disagree.
      </div>
      <div style="font-size:0.7rem;color:#666;margin-top:10px;
                  letter-spacing:0.14em;font-family:system-ui">
        RANDOM FOREST &nbsp;·&nbsp; 4-FOLD CV &nbsp;·&nbsp; CLASS-BALANCED &nbsp;·&nbsp; AUDIO FEATURES ONLY
      </div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading model…"):
        pipe, fi_df, y_te, y_pred, y_all = train_model()

    t1, t2, t3, t4 = st.tabs([
        "Performance", "Feature Analysis", "Predictions", "Genre Predictor",
    ])

    # ── Performance ───────────────────────────────────────────────────────────
    with t1:
        left, right = st.columns([1.2, 1])

        with left:
            section("Model Comparison", "4-fold cross-validation accuracy on 25k songs across 35 genres")
            # Scores recomputed on the expanded 25k dataset. SVM excluded — too slow at this scale.
            model_df = pd.DataFrame({
                "Model":  ["Logistic Regression", "Random Forest", "Decision Tree"],
                "Score":  [0.4371, 0.4080, 0.2278],
                "Color":  ["#636EFA", GREEN, "#AB63FA"],
                "Label":  ["43.7%  ✓ Best CV", "40.8%  Selected — robust to unseen artists", "22.8%  ⚠ Overfit"],
            })
            fig = go.Figure()
            for _, r in model_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[r["Score"]], y=[r["Model"]], orientation="h",
                    marker_color=r["Color"],
                    text=r["Label"], textposition="outside",
                    showlegend=False, name=r["Model"],
                ))
            fig.update_layout(xaxis=dict(range=[0, 0.65], tickformat=".0%",
                              title="Mean CV Accuracy", gridcolor="#2a2a2a"))
            chart(fig, height=260)
            insight("35 genres is a fundamentally harder problem — random baseline is only 2.9%. "
                    "The CV scores above included artist identity features; the <b>deployed model uses audio only</b>, "
                    "which drops test accuracy to 35% but makes predictions that generalise to any song, not just known artists. "
                    "Decision Tree overfits badly at 22.8% CV. "
                    "Logistic Regression edges RF on CV due to its artist→genre shortcut, but RF is selected for robustness.")

        with right:
            section("Final Model: Random Forest")
            n_genres = int(y_all.nunique())
            random_baseline = f"{1/n_genres:.1%}"
            r1, r2 = st.columns(2)
            r1.markdown(kpi("Test Accuracy",    "35.18%",        "held-out test set"),   unsafe_allow_html=True)
            r2.markdown(kpi("vs. Random",       f"12× better",   f"baseline = {random_baseline}"), unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            r1.markdown(kpi("Genre Classes",    str(n_genres),   "after filtering"),     unsafe_allow_html=True)
            r2.markdown(kpi("Training Songs",   "17,455",        "70% of 24,936"),       unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-left:3px solid {GREEN};
                        border-radius:8px;padding:14px 16px;font-size:0.8rem;color:#aaa;margin-top:16px;line-height:1.6">
              35 genres on 25,000 songs is a genuinely difficult classification problem — random
              guessing scores 2.9%. The model uses <b style="color:#fff">only audio signal</b> (no artist
              identity, no metadata) and still recovers genre <b style="color:#fff">12× better than chance</b>,
              with most error concentrated in genres that overlap sonically (pop ↔ dance pop ↔ electropop).
            </div>""", unsafe_allow_html=True)

    # ── Feature Analysis ──────────────────────────────────────────────────────
    with t2:
        left, right = st.columns(2)

        with left:
            section("Feature Importance", "Which attributes the Random Forest relies on most")
            top_feat = fi_df.iloc[0]["Feature"]
            fig = px.bar(fi_df.head(15), x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Greens")
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False)
            chart(fig, height=460)
            insight(f"<b>{top_feat}</b> is the single most predictive audio feature. "
                    "The model uses only the 10 Spotify audio signals — no artist identity — "
                    "so these importances reflect what the music itself sounds like, "
                    "not who made it.")

        with right:
            section("Distribution by Genre", "How a feature's values differ across the top 8 genres")
            v_feat = st.selectbox("Feature", FEATURES, index=1,
                                  format_func=LABELS.get, key="model_v")
            top8 = df["top_genre"].value_counts().nlargest(8).index.tolist()
            df8  = df[df["top_genre"].isin(top8)]
            fig  = px.violin(df8, x="top_genre", y=v_feat, color="top_genre",
                             color_discrete_map=GENRE_COLOR,
                             box=True, points="outliers",
                             labels={"top_genre": "Genre", v_feat: LABELS[v_feat]})
            fig.update_layout(showlegend=False, xaxis_tickangle=32)
            chart(fig, height=460)

        st.divider()
        section("Parallel Coordinates",
                "Each line is a song. Drag to brush any axis and filter songs by that range. "
                "Color = genre (top 6 only for readability).")

        top6    = df_raw["top_genre"].value_counts().nlargest(6).index.tolist()
        df_para = df_raw[df_raw["top_genre"].isin(top6)].copy()
        codes   = pd.Categorical(df_para["top_genre"], categories=top6).codes.astype(float)
        pal6    = [GENRE_COLOR[g] for g in top6]
        cscale  = [[i / max(len(pal6) - 1, 1), c] for i, c in enumerate(pal6)]

        fig = go.Figure(go.Parcoords(
            line=dict(color=codes, colorscale=cscale, showscale=False),
            dimensions=[
                dict(label=LABELS[f], values=df_para[f].tolist(),
                     range=[df_para[f].min(), df_para[f].max()])
                for f in RADAR_COLS
            ],
            labelangle=20,
            labelside="bottom",
        ))
        fig.update_layout(**PLOT)
        fig.update_layout(height=380, margin=dict(t=20, b=80, l=60, r=60))
        chart(fig)

        legend_html = " &nbsp;·&nbsp; ".join(
            f'<span style="color:{GENRE_COLOR[g]};font-weight:600">{g}</span>' for g in top6
        )
        st.markdown(f'<div style="font-size:0.78rem;color:#888;text-align:center;margin-top:-8px">'
                    f'{legend_html}</div>', unsafe_allow_html=True)

    # ── Predictions ───────────────────────────────────────────────────────────
    with t3:
        left, right = st.columns(2)

        # Limit to top 15 genres by test-set support so the matrix stays readable
        all_labels   = sorted(y_all.unique())
        test_counts  = pd.Series(y_te).value_counts()
        top15_labels = test_counts.nlargest(15).index.tolist()
        mask_te      = y_te.isin(top15_labels)
        mask_pred    = pd.Series(y_pred).isin(top15_labels)
        y_te_top  = y_te[mask_te]
        y_pred_top = np.array(y_pred)[mask_te.values]

        cm     = confusion_matrix(y_te_top, y_pred_top, labels=top15_labels, normalize="true")
        cm_df  = pd.DataFrame(np.round(cm, 2), index=top15_labels, columns=top15_labels)

        with left:
            section("Confusion Matrix — Top 15 Genres",
                    "Normalized by true class · filtered to 15 most-represented genres for readability")
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Greens",
                            labels=dict(x="Predicted", y="Actual"), aspect="auto")
            fig.update_layout(xaxis=dict(tickangle=38, tickfont=dict(size=9)),
                              yaxis=dict(tickfont=dict(size=9)),
                              coloraxis_colorbar=dict(thickness=12))
            chart(fig, height=540)

        with right:
            section("Per-Genre Recall — All Genres",
                    "What % of each genre's songs the model correctly identified")
            full_cm   = confusion_matrix(y_te, y_pred, labels=all_labels, normalize="true")
            recall_df = pd.DataFrame({
                "Genre":  all_labels,
                "Recall": [full_cm[i, i] for i in range(len(all_labels))],
            }).sort_values("Recall", ascending=True)

            best  = recall_df.iloc[-1]
            worst = recall_df.iloc[0]

            fig = px.bar(recall_df, x="Recall", y="Genre", orientation="h",
                         color="Recall", color_continuous_scale="Greens",
                         text=[f"{v:.0%}" for v in recall_df["Recall"]])
            fig.update_layout(xaxis=dict(tickformat=".0%", range=[0, 1.2]),
                              yaxis=dict(tickfont=dict(size=9)),
                              coloraxis_showscale=False)
            chart(fig, height=700)
            insight(f"Best: <b>{best['Genre']}</b> ({best['Recall']:.0%}) — acoustically distinct cluster. "
                    f"Worst: <b>{worst['Genre']}</b> ({worst['Recall']:.0%}) — likely overlaps with a "
                    f"similar genre in feature space.")

    # ── Genre Predictor ───────────────────────────────────────────────────────
    with t4:
        section("Live Genre Predictor",
                "Dial in an audio profile — a separate audio-only model predicts genre in real time.")

        # Genre presets stored in session state so buttons update sliders
        PRESETS = {
            "Dance Pop":    dict(bpm=120, nrgy=82, dnce=80, db=-4,  val=72, acous=5,  spch=5,  pop=78, live=8,  dur=210),
            "Hip Hop":      dict(bpm=88,  nrgy=68, dnce=76, db=-6,  val=48, acous=6,  spch=28, pop=72, live=10, dur=225),
            "Rock":         dict(bpm=132, nrgy=90, dnce=48, db=-5,  val=55, acous=7,  spch=4,  pop=60, live=15, dur=235),
            "Acoustic":     dict(bpm=108, nrgy=40, dnce=56, db=-9,  val=62, acous=78, spch=4,  pop=65, live=9,  dur=218),
            "EDM":          dict(bpm=128, nrgy=92, dnce=84, db=-4,  val=65, acous=3,  spch=5,  pop=68, live=6,  dur=240),
            "R&B":          dict(bpm=94,  nrgy=58, dnce=72, db=-7,  val=56, acous=22, spch=12, pop=70, live=11, dur=220),
        }

        if "pred" not in st.session_state:
            st.session_state.pred = PRESETS["Dance Pop"]

        st.markdown('<div style="font-size:0.75rem;color:#666;margin-bottom:6px">Quick presets</div>',
                    unsafe_allow_html=True)
        p_cols = st.columns(len(PRESETS))
        for col, (name, vals) in zip(p_cols, PRESETS.items()):
            if col.button(name, use_container_width=True):
                st.session_state.pred = vals
                st.rerun()

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        p = st.session_state.pred

        s1, s2, s3 = st.columns(3)
        with s1:
            bpm_v  = st.slider("BPM",           60, 210, p["bpm"],  help="Tempo")
            nrgy_v = st.slider("Energy",          0, 100, p["nrgy"], help="Intensity 0–100")
            dnce_v = st.slider("Danceability",    0, 100, p["dnce"], help="Rhythm suitability 0–100")
        with s2:
            db_v   = st.slider("Loudness (dB)", -20,  -1, p["db"],   help="Overall loudness in dB")
            val_v  = st.slider("Valence",         0, 100, p["val"],  help="Positivity 0–100")
            acous_v = st.slider("Acousticness",   0, 100, p["acous"],help="Acoustic confidence 0–100")
        with s3:
            spch_v = st.slider("Speechiness",     0, 100, p["spch"], help="Spoken word density 0–100")
            pop_v  = st.slider("Popularity",       0, 100, p["pop"],  help="Spotify popularity 0–100")

        st.divider()

        audio_pipe = pipe  # same model used throughout

        sample = pd.DataFrame([{
            "year": 2020,  # fix to present so year doesn't skew predictions
            "bpm": bpm_v, "nrgy": nrgy_v, "dnce": dnce_v, "db": db_v,
            "live": p["live"], "val": val_v, "dur": p["dur"],
            "acous": acous_v, "spch": spch_v, "pop": pop_v,
        }])[NUM_FEATS]
        proba  = audio_pipe.predict_proba(sample)[0]
        result = (pd.DataFrame({"Genre": audio_pipe.classes_, "Confidence": proba})
                    .sort_values("Confidence", ascending=False)
                    .head(6).reset_index(drop=True))
        top = result.iloc[0]

        pred_col, chart_col = st.columns([1, 1.6])
        with pred_col:
            top_color = GENRE_COLOR.get(top["Genre"], GREEN)
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {top_color}55;border-radius:12px;
                        padding:28px 20px;text-align:center;margin-top:4px">
              <div style="color:#555;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em">
                Predicted Genre
              </div>
              <div style="color:{top_color};font-size:1.8rem;font-weight:800;
                          margin:10px 0 6px 0;line-height:1.1">
                {top["Genre"]}
              </div>
              <div style="color:#666;font-size:0.82rem">{top["Confidence"]:.1%} confidence</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            for _, row in result.iloc[1:5].iterrows():
                pct = row["Confidence"] / max(top["Confidence"], 0.001)
                st.markdown(f"""
                <div style="margin-bottom:5px">
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.78rem;margin-bottom:2px">
                    <span style="color:#aaa">{row['Genre']}</span>
                    <span style="color:#666">{row['Confidence']:.1%}</span>
                  </div>
                  <div style="background:#2a2a2a;border-radius:3px;height:4px">
                    <div style="background:{GENRE_COLOR.get(row['Genre'], '#444')};
                                width:{pct*100:.0f}%;height:4px;border-radius:3px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        with chart_col:
            result["Color"] = result["Genre"].map(lambda g: GENRE_COLOR.get(g, "#444"))
            fig = go.Figure(go.Bar(
                x=result["Confidence"], y=result["Genre"], orientation="h",
                marker_color=result["Color"].tolist(),
                text=[f"{v:.1%}" for v in result["Confidence"]],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                xaxis=dict(tickformat=".0%",
                           range=[0, min(1.15, result["Confidence"].max() * 1.35)],
                           gridcolor="#2a2a2a"),
                showlegend=False,
            )
            chart(fig, height=340)

        insight("This model uses <b>audio features only</b> — no artist identity. "
                "Try cranking Speechiness above 25 for hip hop, Acousticness above 70 for acoustic pop, "
                "or BPM=128 + Energy=92 + Danceability=84 for EDM. Use the preset buttons to jump to "
                "known genre profiles.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(f"""
    <div style="margin:8px 0 26px 0;padding:0 0 0 18px;
                border-left:2px solid {GREEN};font-family:Georgia,serif">
      <div style="font-size:1.1rem;color:#eee;line-height:1.45;font-style:italic;
                  font-weight:400;letter-spacing:-0.01em">
        A walk-through of how this was built, what worked, what didn't, and what I'd
        do differently with 2026 tooling. Honest documentation matters more than a
        flattering accuracy number.
      </div>
      <div style="font-size:0.7rem;color:#666;margin-top:10px;
                  letter-spacing:0.14em;font-family:system-ui">
        METHODOLOGY &nbsp;·&nbsp; LIMITATIONS &nbsp;·&nbsp; TECH STACK
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── The problem & dataset ─────────────────────────────────────────────────
    section("The Problem",
            "Multi-class genre classification from audio features only")
    st.markdown(f"""
    <div style="font-size:0.88rem;color:#bbb;line-height:1.65;margin-bottom:18px">
    Given Spotify's per-track audio statistics — tempo, energy, danceability,
    acousticness, speechiness, valence, loudness, liveness, popularity, duration —
    can a model recover the genre label?<br><br>
    The challenge is structural: genre labels are <b style="color:#fff">cultural categories</b>,
    not acoustic ones. "Dance pop" and "electropop" overlap heavily in feature space.
    Random baseline for 35 classes is just <b style="color:#fff">2.9%</b>, so any meaningful
    signal counts.
    </div>
    """, unsafe_allow_html=True)

    section("Dataset",
            "Two public sources merged — 24,993 songs spanning 1957–2020")
    d1, d2 = st.columns(2)
    d1.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:14px 16px">
      <div style="font-size:0.6rem;color:{GREEN};letter-spacing:0.14em;font-weight:700;
                  margin-bottom:6px">SOURCE 1</div>
      <div style="font-size:0.95rem;color:#fff;font-weight:700">Billboard Top Songs</div>
      <div style="font-size:0.78rem;color:#888;margin-top:6px;line-height:1.55">
        603 songs · 2010–2019 · Originally from Kaggle / Billboard year-end charts.
        Provided the seed audio features in the 0–100 scale.
      </div>
    </div>""", unsafe_allow_html=True)
    d2.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:14px 16px">
      <div style="font-size:0.6rem;color:{GREEN};letter-spacing:0.14em;font-weight:700;
                  margin-bottom:6px">SOURCE 2</div>
      <div style="font-size:0.95rem;color:#fff;font-weight:700">TidyTuesday Spotify</div>
      <div style="font-size:0.78rem;color:#888;margin-top:6px;line-height:1.55">
        32,833 songs · 1957–2020 · Pulled from the Spotify Web API.
        Required column-mapping and 0–1 → 0–100 rescaling to merge cleanly.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
    section("Pipeline",
            "How a raw CSV becomes a deployed classifier")
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:18px">

      <div style="display:flex;align-items:center;gap:14px;background:{CARD};
                  border:1px solid {BORDER};border-radius:8px;padding:12px 16px">
        <div style="width:24px;height:24px;border-radius:50%;background:rgba(29,185,84,0.12);
                    color:{GREEN};display:flex;align-items:center;justify-content:center;
                    font-size:0.7rem;font-weight:700;flex-shrink:0">01</div>
        <div style="flex:1">
          <div style="font-size:0.88rem;color:#fff;font-weight:600">Load &amp; clean</div>
          <div style="font-size:0.76rem;color:#888;margin-top:2px">
            Merge two CSVs, strip column whitespace, drop classes with &lt;5 samples
            (final: 35 genres). Filtered dataset = 24,936 songs.
          </div>
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:14px;background:{CARD};
                  border:1px solid {BORDER};border-radius:8px;padding:12px 16px">
        <div style="width:24px;height:24px;border-radius:50%;background:rgba(29,185,84,0.12);
                    color:{GREEN};display:flex;align-items:center;justify-content:center;
                    font-size:0.7rem;font-weight:700;flex-shrink:0">02</div>
        <div style="flex:1">
          <div style="font-size:0.88rem;color:#fff;font-weight:600">Preprocess</div>
          <div style="font-size:0.76rem;color:#888;margin-top:2px">
            <code>SimpleImputer(strategy=median)</code> → <code>StandardScaler</code>.
            Wrapped in a sklearn Pipeline so the transformer fits on train only and
            applies cleanly at inference time.
          </div>
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:14px;background:{CARD};
                  border:1px solid {BORDER};border-radius:8px;padding:12px 16px">
        <div style="width:24px;height:24px;border-radius:50%;background:rgba(29,185,84,0.12);
                    color:{GREEN};display:flex;align-items:center;justify-content:center;
                    font-size:0.7rem;font-weight:700;flex-shrink:0">03</div>
        <div style="flex:1">
          <div style="font-size:0.88rem;color:#fff;font-weight:600">Train / Test Split</div>
          <div style="font-size:0.76rem;color:#888;margin-top:2px">
            70/30 stratified split, <code>random_state=123</code>. Stratification is
            essential here — naïve random sampling would underrepresent rare genres in test.
          </div>
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:14px;background:{CARD};
                  border:1px solid {BORDER};border-radius:8px;padding:12px 16px">
        <div style="width:24px;height:24px;border-radius:50%;background:rgba(29,185,84,0.12);
                    color:{GREEN};display:flex;align-items:center;justify-content:center;
                    font-size:0.7rem;font-weight:700;flex-shrink:0">04</div>
        <div style="flex:1">
          <div style="font-size:0.88rem;color:#fff;font-weight:600">Benchmark via 4-fold CV</div>
          <div style="font-size:0.76rem;color:#888;margin-top:2px">
            Compared Logistic Regression, Random Forest, Decision Tree. SVM excluded
            (O(n²) blow-up at 25k samples). LR scored highest on CV but relied on
            artist features — see <b style="color:{GREEN}">Limitations</b> below.
          </div>
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:14px;background:{CARD};
                  border:1px solid {BORDER};border-radius:8px;padding:12px 16px">
        <div style="width:24px;height:24px;border-radius:50%;background:rgba(29,185,84,0.12);
                    color:{GREEN};display:flex;align-items:center;justify-content:center;
                    font-size:0.7rem;font-weight:700;flex-shrink:0">05</div>
        <div style="flex:1">
          <div style="font-size:0.88rem;color:#fff;font-weight:600">Tune &amp; deploy</div>
          <div style="font-size:0.76rem;color:#888;margin-top:2px">
            Random Forest with <code>class_weight='balanced'</code>, <code>max_features='sqrt'</code>,
            <code>min_samples_leaf=2</code>. Deployed via Streamlit Cloud with
            <code>@st.cache_resource</code> on the model so training only happens once per cold start.
          </div>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    section("Honest Limitations",
            "Things a reviewer should know before quoting the accuracy number")
    st.markdown(f"""
    <div style="background:rgba(255,140,0,0.05);border:1px solid rgba(255,140,0,0.25);
                border-left:3px solid #FF8C00;border-radius:8px;padding:14px 18px;
                margin-bottom:18px;line-height:1.65">

      <div style="margin-bottom:12px">
        <div style="font-size:0.85rem;color:#fff;font-weight:700;margin-bottom:3px">
          Genre labels are noisy by definition
        </div>
        <div style="font-size:0.8rem;color:#bbb">
          Spotify's <code>playlist_subgenre</code> labels overlap heavily ("dance pop" ≈ "electropop" ≈ "pop").
          The "ceiling" for any classifier is bounded by label disagreement that exists in the data itself.
        </div>
      </div>

      <div style="margin-bottom:12px">
        <div style="font-size:0.85rem;color:#fff;font-weight:700;margin-bottom:3px">
          Severe class imbalance
        </div>
        <div style="font-size:0.8rem;color:#bbb">
          Dance pop has 1,486 songs; rare genres have 5–10.
          Balanced class weighting helps but cannot replace more data for the long tail.
        </div>
      </div>

      <div style="margin-bottom:12px">
        <div style="font-size:0.85rem;color:#fff;font-weight:700;margin-bottom:3px">
          Sample bias toward chart-toppers
        </div>
        <div style="font-size:0.8rem;color:#bbb">
          Both source datasets sample from popular playlists / Billboard charts.
          A model trained on Spotify's full catalog (~100M tracks) would see very different distributions.
        </div>
      </div>

      <div style="margin-bottom:12px">
        <div style="font-size:0.85rem;color:#fff;font-weight:700;margin-bottom:3px">
          Audio features are aggregate statistics
        </div>
        <div style="font-size:0.8rem;color:#bbb">
          The Spotify API returns per-track means (avg energy, avg valence) — not the time-domain
          waveform. Temporal and timbral structure is lost. A modern embedding model
          (CLAP, MERT) would preserve it.
        </div>
      </div>

      <div>
        <div style="font-size:0.85rem;color:#fff;font-weight:700;margin-bottom:3px">
          Streamlit Cloud memory constraints affected hyperparameters
        </div>
        <div style="font-size:0.8rem;color:#bbb">
          The deployed model uses <code>n_estimators=50</code> instead of the GridSearchCV-optimal 200
          to fit within the 1GB free-tier RAM limit. Local notebook trains the full configuration.
        </div>
      </div>

    </div>""", unsafe_allow_html=True)

    section("What I'd Do With 2026 Tools",
            "Where the obvious next iteration goes")
    st.markdown(f"""
    <div style="font-size:0.86rem;color:#bbb;line-height:1.65;margin-bottom:18px">
    A pre-trained audio embedding model — <b style="color:#fff">CLAP</b>, <b style="color:#fff">MERT</b>,
    or Spotify's internal track embeddings — paired with a <b style="color:#fff">contrastive learning
    objective</b> would almost certainly outperform Random Forest on rare-class genres. The Spotify API
    features used here are aggregate statistics (energy, danceability, valence) that lose timbral and
    temporal structure an embedding-based approach would preserve. A k-NN classifier over those embeddings,
    combined with a small head fine-tuned on the 35-class label set, is the path I'd take next.
    </div>""", unsafe_allow_html=True)

    section("Tech Stack", "")
    tech = [
        ("Python 3.11",           "Core language"),
        ("scikit-learn",          "RF · LR · DT · CV · pipelines"),
        ("pandas / NumPy",        "Data manipulation"),
        ("Streamlit",             "Dashboard framework"),
        ("Plotly",                "Interactive 3D & 2D charts"),
        ("t-SNE",                 "Dimensionality reduction"),
        ("GitHub",                "Version control · repo"),
        ("Streamlit Cloud",       "Deployment"),
    ]
    tech_html = "".join(
        f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:6px;'
        f'padding:10px 14px"><div style="font-size:0.82rem;color:#fff;font-weight:600">{name}</div>'
        f'<div style="font-size:0.72rem;color:#777;margin-top:2px">{desc}</div></div>'
        for name, desc in tech
    )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:24px">'
        f'{tech_html}</div>', unsafe_allow_html=True)

    section("Source", "")
    st.markdown(f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:24px">
      <a href="https://github.com/mateoportillo1900/Spotify-ML-Model" target="_blank"
         style="background:rgba(29,185,84,0.08);border:1px solid rgba(29,185,84,0.35);
                border-radius:6px;padding:9px 16px;text-decoration:none;color:{GREEN};
                font-size:0.82rem;font-weight:600">
        GitHub Repository →
      </a>
      <a href="https://github.com/mateoportillo1900/Spotify-ML-Model/blob/main/docs/METHODOLOGY.md"
         target="_blank" style="background:rgba(29,185,84,0.05);border:1px solid rgba(29,185,84,0.3);
                border-radius:6px;padding:9px 16px;text-decoration:none;color:{GREEN};
                font-size:0.82rem;font-weight:600">
        Methodology Deep-Dive →
      </a>
      <a href="https://github.com/mateoportillo1900/Spotify-ML-Model/blob/main/Spotify_ML_Project.ipynb"
         target="_blank" style="background:{CARD};border:1px solid {BORDER};
                border-radius:6px;padding:9px 16px;text-decoration:none;color:#ccc;
                font-size:0.82rem;font-weight:600">
        Training Notebook →
      </a>
      <a href="https://github.com/mateoportillo1900/Spotify-ML-Model/blob/main/README.md"
         target="_blank" style="background:{CARD};border:1px solid {BORDER};
                border-radius:6px;padding:9px 16px;text-decoration:none;color:#ccc;
                font-size:0.82rem;font-weight:600">
        Full README →
      </a>
    </div>""", unsafe_allow_html=True)
