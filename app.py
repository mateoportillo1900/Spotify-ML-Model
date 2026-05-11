import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Genre Intelligence",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.5rem 2rem 1rem 2rem; }

  /* Top accent bar */
  .top-bar {
    height: 3px;
    background: linear-gradient(90deg, #1DB954, #158a3e, #1DB954);
    border-radius: 0 0 2px 2px;
    margin: -1.5rem -2rem 1.5rem -2rem;
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
CAT_FEATS    = ["artist"]

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
    sub_html = f'<div style="color:#666;font-size:0.7rem;margin-top:2px">{sub}</div>' if sub else ""
    return f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
                padding:16px 10px;text-align:center;">
      <div style="color:#666;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em">{label}</div>
      <div style="color:{GREEN};font-size:1.55rem;font-weight:700;margin-top:4px;line-height:1.1">{value}</div>
      {sub_html}
    </div>"""

def section(title, subtitle=""):
    sub = f'<div style="color:#666;font-size:0.78rem;margin-top:1px">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:10px;margin:12px 0 14px 0">
      <div style="width:3px;min-height:28px;background:{GREEN};border-radius:2px;margin-top:4px;flex-shrink:0"></div>
      <div>
        <div style="font-size:1.05rem;font-weight:700;color:#F0F0F0;line-height:1.2">{title}</div>
        {sub}
      </div>
    </div>""", unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)

def chart(fig, height=None, **kw):
    if height:
        fig.update_layout(height=height)
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True, **kw)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

@st.cache_resource
def train_model():
    df = load_data()
    counts = df["top_genre"].value_counts()
    df = df[~df["top_genre"].isin(counts[counts < 5].index)]

    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="constant")), ("s", StandardScaler())]), NUM_FEATS),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATS),
    ])
    X, y = df[NUM_FEATS + CAT_FEATS], df["top_genre"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ))])
    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)
    ohe   = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["o"]
    fi    = pd.DataFrame({
        "Feature":    NUM_FEATS + ohe.get_feature_names_out(CAT_FEATS).tolist(),
        "Importance": pipe.named_steps["clf"].feature_importances_,
    }).sort_values("Importance", ascending=False)
    return pipe, fi, yte, ypred, y

@st.cache_resource
def train_audio_model():
    """Audio-only Random Forest — no artist features so the predictor reflects pure sound."""
    df = load_data()
    counts = df["top_genre"].value_counts()
    df = df[~df["top_genre"].isin(counts[counts < 5].index)]
    X  = df[FEATURES]
    y  = df["top_genre"]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight="balanced", max_features="sqrt", min_samples_leaf=2,
        )),
    ])
    pipe.fit(Xtr, ytr)
    return pipe

@st.cache_data
def compute_tsne():
    df  = load_data()
    X   = df[FEATURES].fillna(df[FEATURES].mean())
    Xs  = StandardScaler().fit_transform(X)
    # max_iter replaces n_iter in sklearn >= 1.4
    try:
        emb = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=500).fit_transform(Xs)
    except TypeError:
        emb = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=500).fit_transform(Xs)
    return pd.DataFrame({
        "x": emb[:, 0], "y": emb[:, 1], "z": emb[:, 2],
        "genre":  df["top_genre"].values,
        "title":  df["title"].values,
        "artist": df["artist"].values,
    })

df_raw = load_data()

# Consistent genre → color mapping used across every chart
ALL_GENRES  = sorted(df_raw["top_genre"].unique())
PALETTE     = px.colors.qualitative.Dark24
GENRE_COLOR = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(ALL_GENRES)}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:4px 0 18px 0">
      <div style="font-size:1.25rem;font-weight:800;color:{GREEN};letter-spacing:-0.01em">🎵 Spotify Intel</div>
      <div style="font-size:0.7rem;color:#555;margin-top:1px">Billboard Top Songs · 2010–2019</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("nav", ["🔍  Explore Data", "🤖  ML Model"],
                    label_visibility="collapsed")

    st.divider()
    st.markdown('<div style="font-size:0.65rem;color:#555;text-transform:uppercase;'
                'letter-spacing:0.1em;margin-bottom:8px">Filters</div>', unsafe_allow_html=True)

    all_genres = sorted(df_raw["top_genre"].unique())
    sel_genres = st.multiselect("Genre", all_genres, default=all_genres,
                                label_visibility="collapsed", placeholder="All genres")
    year_range = st.slider("Year", 2010, 2019, (2010, 2019))

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

# ── Header ────────────────────────────────────────────────────────────────────
page_title = "🔍 Data Exploration" if "Explore" in page else "🤖 ML Model Results"
st.markdown(f"""
<div style="margin-bottom:18px">
  <div style="font-size:1.6rem;font-weight:800;letter-spacing:-0.02em;line-height:1.2">{page_title}</div>
  <div style="color:#555;font-size:0.82rem;margin-top:3px">
    {len(df):,} songs &nbsp;·&nbsp; {df["top_genre"].nunique()} genres &nbsp;·&nbsp;
    {year_range[0]}–{year_range[1]}
  </div>
</div>""", unsafe_allow_html=True)

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
    t1, t2, t3, t4, t5 = st.tabs([
        "📊  Overview", "🌐  3D Space", "🎯  Genre Profiles", "📈  Trends", "🔎  Songs",
    ])

    # ── Overview ──────────────────────────────────────────────────────────────
    with t1:
        left, right = st.columns(2)

        with left:
            section("Genre Distribution", "How many Billboard charting songs belong to each genre")
            gc = df["top_genre"].value_counts().reset_index()
            gc.columns = ["Genre", "Songs"]
            top_pct = gc.iloc[0]["Songs"] / len(df) * 100
            fig = px.bar(gc, x="Songs", y="Genre", orientation="h",
                         color="Genre", color_discrete_map=GENRE_COLOR,
                         text="Songs")
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              showlegend=False)
            chart(fig, height=480)
            insight(f"<b>{gc.iloc[0]['Genre'].title()}</b> dominates with "
                    f"{gc.iloc[0]['Songs']} songs ({top_pct:.0f}% of the filtered dataset). "
                    f"This class imbalance directly impacts model recall for minority genres.")

        with right:
            section("Songs per Year", "Volume of charting songs across the decade")
            yc = df["year"].value_counts().sort_index().reset_index()
            yc.columns = ["Year", "Songs"]
            fig = px.bar(yc, x="Year", y="Songs", color="Songs",
                         color_continuous_scale="Greens", text="Songs")
            fig.update_layout(coloraxis_showscale=False, xaxis=dict(dtick=1))
            chart(fig, height=210)

            section("Feature Correlations", "Pearson r — how audio attributes move together")
            corr = df[FEATURES].corr().round(2)
            corr.index   = [LABELS[c] for c in FEATURES]
            corr.columns = [LABELS[c] for c in FEATURES]
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(margin=dict(t=10, b=0, l=0, r=0))
            chart(fig, height=230)
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

        fig3d = go.Figure()
        for genre in sorted(df["top_genre"].unique()):
            g = df[df["top_genre"] == genre]
            fig3d.add_trace(go.Scatter3d(
                x=g[fx], y=g[fy], z=g[fz],
                mode="markers",
                marker=dict(size=4, color=GENRE_COLOR[genre], opacity=0.85),
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
            **PLOT, height=570,
            scene=dict(
                xaxis=dict(title=LABELS[fx], backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                yaxis=dict(title=LABELS[fy], backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                zaxis=dict(title=LABELS[fz], backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                bgcolor=BG,
            ),
            legend=dict(orientation="v", x=1.01, y=0.5, font=dict(size=9), itemsizing="constant"),
        )
        st.plotly_chart(fig3d, use_container_width=True)
        insight(f"Try <b>Energy × Danceability × Valence</b> — dance pop clusters high on all three, "
                f"while acoustic and folk genres pull to the low-energy, high-acousticness corner. "
                f"Clear separation here explains why the model performs well on those genres.")

        st.divider()
        section("t-SNE 3D — Genre Clustering",
                "All 10 audio features compressed to 3D. Tighter clusters = easier for the model to learn.")

        with st.spinner("Computing t-SNE embedding (cached after first run)…"):
            tsne_df = compute_tsne()

        fig_t = go.Figure()
        for genre in sorted(tsne_df["genre"].unique()):
            g = tsne_df[tsne_df["genre"] == genre]
            fig_t.add_trace(go.Scatter3d(
                x=g["x"], y=g["y"], z=g["z"],
                mode="markers",
                marker=dict(size=3.5, color=GENRE_COLOR.get(genre, "#888"), opacity=0.88),
                name=genre,
                text=g["title"] + " — " + g["artist"],
                hovertemplate="<b>%{text}</b><extra>" + genre + "</extra>",
            ))
        fig_t.update_layout(
            **PLOT, height=570,
            scene=dict(
                xaxis=dict(title="Component 1", backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                yaxis=dict(title="Component 2", backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                zaxis=dict(title="Component 3", backgroundcolor=CARD, gridcolor="#2a2a2a", showbackground=True),
                bgcolor=BG,
            ),
            legend=dict(orientation="v", x=1.01, y=0.5, font=dict(size=9), itemsizing="constant"),
        )
        st.plotly_chart(fig_t, use_container_width=True)
        insight("Dance pop (green cluster) is tightly packed and well-separated — the model achieves "
                "99% recall on it. Overlapping genre clusters (pop, electropop) explain the model's "
                "lower recall on those classes. Axes have no direct meaning; only distances matter.")

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
        section("Audio Feature Trends", "How the sound of Billboard hits changed year by year")

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
            insight("Acousticness has trended down while Energy held steady — pop music got louder and "
                    "more electronic over the decade. Valence dipped mid-decade, suggesting a shift toward "
                    "more melancholic or emotionally complex hits around 2015–2017.")
        else:
            st.info("Select at least one feature.")

        st.divider()
        section("Most Charted Artists",
                "Artists with the most songs in the Billboard Top 100 (2010–2019). Color = average popularity.")

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
else:
    with st.spinner("Loading model…"):
        pipe, fi_df, y_te, y_pred, y_all = train_model()

    t1, t2, t3, t4 = st.tabs([
        "📊  Performance", "🔬  Feature Analysis", "🗺  Predictions", "🎵  Genre Predictor",
    ])

    # ── Performance ───────────────────────────────────────────────────────────
    with t1:
        left, right = st.columns([1.2, 1])

        with left:
            section("Model Comparison", "4-fold cross-validation accuracy — how each model generalises")
            model_df = pd.DataFrame({
                "Model":  ["Logistic Regression", "Random Forest", "SVM", "Decision Tree"],
                "Score":  [0.7842, 0.8026, 0.6079, 0.9184],
                "Color":  ["#636EFA", GREEN, "#EF553B", "#AB63FA"],
                "Label":  ["78.4%", "80.3%  ✓ Selected", "60.8%", "91.8%  ⚠ Overfit"],
            })
            fig = go.Figure()
            for _, r in model_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[r["Score"]], y=[r["Model"]], orientation="h",
                    marker_color=r["Color"],
                    text=r["Label"], textposition="outside",
                    showlegend=False, name=r["Model"],
                ))
            fig.update_layout(xaxis=dict(range=[0, 1.15], tickformat=".0%",
                              title="Mean CV Accuracy", gridcolor="#2a2a2a"))
            chart(fig, height=300)
            insight("The Decision Tree's 91.8% CV score is misleading — it memorises the training folds. "
                    "Random Forest's bagging averages 100 trees, giving a more honest 80.3% that actually "
                    "holds on unseen data (confirmed at 83.5% test accuracy).")

        with right:
            section("Final Model: Random Forest")
            r1, r2 = st.columns(2)
            r1.markdown(kpi("Test Accuracy",  "83.54%", "held-out test set"),  unsafe_allow_html=True)
            r2.markdown(kpi("Best CV Score",  "80.26%", "cross-validation"),   unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            r1.markdown(kpi("Top Recall",     "99%",    "dance pop"),          unsafe_allow_html=True)
            r2.markdown(kpi("Genre Classes",  "17",     "after filtering"),    unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-left:3px solid {GREEN};
                        border-radius:8px;padding:14px 16px;font-size:0.8rem;color:#aaa;margin-top:16px;line-height:1.6">
              Random Forest with <code>class_weight='balanced'</code> compensates for genre imbalance
              by upweighting rare genres during training — critical for a dataset where dance pop
              has 10× more samples than some other genres.
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
            insight(f"<b>{top_feat}</b> is the single most predictive feature. "
                    "Artist identity features appear near the top — the model has partially "
                    "learned that certain artists are synonymous with certain genres, which "
                    "may not generalise to new artists.")

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

        with left:
            section("Confusion Matrix",
                    "Normalized by true class. Diagonal = correct. Off-diagonal = what the model confused it with.")
            labels = sorted(y_all.unique())
            cm     = confusion_matrix(y_te, y_pred, labels=labels, normalize="true")
            cm_df  = pd.DataFrame(np.round(cm, 2), index=labels, columns=labels)
            fig    = px.imshow(cm_df, text_auto=True, color_continuous_scale="Greens",
                               labels=dict(x="Predicted", y="Actual"), aspect="auto")
            fig.update_layout(xaxis=dict(tickangle=35, tickfont=dict(size=9)),
                              yaxis=dict(tickfont=dict(size=9)),
                              coloraxis_colorbar=dict(thickness=12))
            chart(fig, height=520)

        with right:
            section("Per-Genre Recall", "What % of each genre's songs the model correctly identified")
            recall_df = pd.DataFrame({
                "Genre":  labels,
                "Recall": [cm[i, i] for i in range(len(labels))],
            }).sort_values("Recall", ascending=True)

            best  = recall_df.iloc[-1]
            worst = recall_df.iloc[0]

            fig = px.bar(recall_df, x="Recall", y="Genre", orientation="h",
                         color="Recall", color_continuous_scale="Greens",
                         text=[f"{v:.0%}" for v in recall_df["Recall"]])
            fig.update_layout(xaxis=dict(tickformat=".0%", range=[0, 1.15]),
                              yaxis=dict(tickfont=dict(size=10)),
                              coloraxis_showscale=False)
            chart(fig, height=520)
            insight(f"Best: <b>{best['Genre']}</b> ({best['Recall']:.0%}) — large, distinct cluster. "
                    f"Worst: <b>{worst['Genre']}</b> ({worst['Recall']:.0%}) — fewer training examples "
                    f"or acoustically similar to another genre.")

    # ── Genre Predictor ───────────────────────────────────────────────────────
    with t4:
        section("Live Genre Predictor",
                "Dial in an audio profile — a separate audio-only model predicts genre in real time.")

        # Genre presets stored in session state so buttons update sliders
        PRESETS = {
            "🎉 Dance Pop":    dict(bpm=120, nrgy=82, dnce=80, db=-4,  val=72, acous=5,  spch=5,  pop=78, live=8,  dur=210),
            "🎤 Hip Hop":      dict(bpm=88,  nrgy=68, dnce=76, db=-6,  val=48, acous=6,  spch=28, pop=72, live=10, dur=225),
            "🎸 Rock":         dict(bpm=132, nrgy=90, dnce=48, db=-5,  val=55, acous=7,  spch=4,  pop=60, live=15, dur=235),
            "🎹 Acoustic Pop": dict(bpm=108, nrgy=40, dnce=56, db=-9,  val=62, acous=78, spch=4,  pop=65, live=9,  dur=218),
            "🌊 EDM":          dict(bpm=128, nrgy=92, dnce=84, db=-4,  val=65, acous=3,  spch=5,  pop=68, live=6,  dur=240),
            "🎷 R&B":          dict(bpm=94,  nrgy=58, dnce=72, db=-7,  val=56, acous=22, spch=12, pop=70, live=11, dur=220),
        }

        if "pred" not in st.session_state:
            st.session_state.pred = PRESETS["🎉 Dance Pop"]

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

        with st.spinner("Loading audio model…"):
            audio_pipe = train_audio_model()

        sample = pd.DataFrame([{
            "bpm": bpm_v, "nrgy": nrgy_v, "dnce": dnce_v, "db": db_v,
            "live": p["live"], "val": val_v, "dur": p["dur"],
            "acous": acous_v, "spch": spch_v, "pop": pop_v,
        }])
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
                title="Top 6 Genre Predictions",
                showlegend=False,
            )
            chart(fig, height=340)

        insight("This model uses <b>audio features only</b> — no artist identity. "
                "Try cranking Speechiness above 25 for hip hop, Acousticness above 70 for acoustic pop, "
                "or BPM=128 + Energy=92 + Danceability=84 for EDM. Use the preset buttons to jump to "
                "known genre profiles.")
