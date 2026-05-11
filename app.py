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

# ── Config ────────────────────────────────────────────────────────────────────
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

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(255,255,255,0.08); }
  .stTabs [data-baseweb="tab"] {
    padding: 8px 18px; border-radius: 6px 6px 0 0;
    font-weight: 500; font-size: 0.88rem; color: #888;
    background: transparent; border: none;
  }
  .stTabs [aria-selected="true"] { color: #1DB954 !important; border-bottom: 2px solid #1DB954 !important; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0A0A0A;
    border-right: 1px solid rgba(255,255,255,0.06);
  }

  /* Dividers */
  hr { border-color: rgba(255,255,255,0.08) !important; }

  /* Selectbox / multiselect */
  .stMultiSelect [data-baseweb="tag"] { background: #1DB954 !important; color: #000 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/mateoportillo1900/Spotify-ML-Model/refs/heads/main/spotify_top_music.csv"

FEATURES = ["bpm", "nrgy", "dnce", "db", "live", "val", "dur", "acous", "spch", "pop"]
LABELS = {
    "bpm": "BPM", "nrgy": "Energy", "dnce": "Danceability", "db": "Loudness",
    "live": "Liveness", "val": "Valence", "dur": "Duration",
    "acous": "Acousticness", "spch": "Speechiness", "pop": "Popularity",
}
RADAR_COLS   = ["bpm", "nrgy", "dnce", "val", "acous", "spch", "pop"]
RADAR_LABELS = ["BPM", "Energy", "Danceability", "Valence", "Acousticness", "Speechiness", "Popularity"]

BG    = "#121212"
CARD  = "#1E1E1E"
GREEN = "#1DB954"
BORDER = "rgba(255,255,255,0.08)"

PLOT = dict(
    template="plotly_dark",
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(color="#FFFFFF", family="sans-serif"),
    margin=dict(t=50, b=30, l=20, r=20),
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi(label, value, sub=None):
    sub_html = f'<div style="color:{GREEN};font-size:0.72rem;margin-top:3px">{sub}</div>' if sub else ""
    return f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;
                padding:18px 12px;text-align:center;">
      <div style="color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em">{label}</div>
      <div style="color:{GREEN};font-size:1.65rem;font-weight:700;margin-top:5px;line-height:1.1">{value}</div>
      {sub_html}
    </div>"""

def section(title, subtitle=""):
    sub = f'<div style="color:#888;font-size:0.82rem;margin-top:2px">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:12px;margin:4px 0 16px 0">
      <div style="width:4px;min-height:32px;background:{GREEN};border-radius:2px;margin-top:3px"></div>
      <div>
        <div style="font-size:1.15rem;font-weight:700;color:#FFF">{title}</div>
        {sub}
      </div>
    </div>""", unsafe_allow_html=True)

def chart(fig, **kwargs):
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True, **kwargs)

# ── Data & models ─────────────────────────────────────────────────────────────
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

    num_f = ["year", "bpm", "nrgy", "dnce", "db", "live", "val", "dur", "acous", "spch", "pop"]
    cat_f = ["artist"]
    pre   = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="constant")), ("s", StandardScaler())]), num_f),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat_f),
    ])
    X, y = df[num_f + cat_f], df["top_genre"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ))])
    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)
    ohe   = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["o"]
    fi    = pd.DataFrame({
        "Feature": num_f + ohe.get_feature_names_out(cat_f).tolist(),
        "Importance": pipe.named_steps["clf"].feature_importances_,
    }).sort_values("Importance", ascending=False)
    return fi, yte, ypred, y

@st.cache_data
def compute_tsne():
    df = load_data()
    X  = df[FEATURES].fillna(df[FEATURES].mean())
    Xs = StandardScaler().fit_transform(X)
    emb = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=500).fit_transform(Xs)
    return pd.DataFrame({
        "x": emb[:, 0], "y": emb[:, 1], "z": emb[:, 2],
        "genre": df["top_genre"].values,
        "title": df["title"].values,
        "artist": df["artist"].values,
    })

df_raw = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 0 20px 0">
      <div style="font-size:1.3rem;font-weight:800;color:{GREEN}">🎵 Spotify Intel</div>
      <div style="font-size:0.72rem;color:#666;margin-top:2px">Billboard Top Songs · 2010–2019</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio(
        "Section",
        ["🔍  Explore Data", "🤖  ML Model"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div style="font-size:0.72rem;color:#777;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">Filters</div>', unsafe_allow_html=True)

    all_genres = sorted(df_raw["top_genre"].unique())
    sel_genres = st.multiselect("Genre", all_genres, default=all_genres, label_visibility="collapsed",
                                placeholder="All genres")
    year_range = st.slider("Year", 2010, 2019, (2010, 2019))

    st.divider()
    st.markdown('<div style="font-size:0.72rem;color:#777;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">Feature Glossary</div>', unsafe_allow_html=True)
    for feat, label in LABELS.items():
        st.caption(f"**{label}** ({feat})" + {
            "bpm": " — Tempo", "nrgy": " — Intensity 0–100", "dnce": " — Rhythm fit 0–100",
            "db": " — Loudness dB", "live": " — Live audience likelihood",
            "val": " — Positivity 0–100", "dur": " — Duration (sec)",
            "acous": " — Acoustic confidence", "spch": " — Spoken word density",
            "pop": " — Spotify popularity 0–100",
        }.get(feat, ""))

df = df_raw[
    df_raw["top_genre"].isin(sel_genres) &
    df_raw["year"].between(*year_range)
].copy()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:20px">
  <div style="font-size:1.7rem;font-weight:800;letter-spacing:-0.02em">
    {"🔍 Data Exploration" if "Explore" in page else "🤖 ML Model Results"}
  </div>
  <div style="color:#888;font-size:0.85rem;margin-top:3px">
    {len(df):,} songs &nbsp;·&nbsp; {df["top_genre"].nunique()} genres &nbsp;·&nbsp;
    {year_range[0]}–{year_range[1]}
  </div>
</div>""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, val, sub in [
    (c1, "Songs",        f"{len(df):,}",               None),
    (c2, "Genres",       str(df["top_genre"].nunique()), None),
    (c3, "Avg Energy",   f"{df['nrgy'].mean():.0f}",    "out of 100"),
    (c4, "Avg Popularity", f"{df['pop'].mean():.0f}",   "out of 100"),
    (c5, "Avg BPM",      f"{df['bpm'].mean():.0f}",     "beats / min"),
]:
    col.markdown(kpi(label, val, sub), unsafe_allow_html=True)

st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════
if "Explore" in page:
    t1, t2, t3, t4, t5 = st.tabs([
        "📊  Overview", "🌐  3D Space", "🎯  Genre Profiles", "📈  Trends", "🔎  Songs",
    ])

    # ── Overview ──────────────────────────────────────────────────────────────
    with t1:
        left, right = st.columns(2)

        with left:
            section("Genre Distribution", "Songs per genre in the current filter")
            gc = df["top_genre"].value_counts().reset_index()
            gc.columns = ["Genre", "Songs"]
            fig = px.bar(gc, x="Songs", y="Genre", orientation="h",
                         color="Songs", color_continuous_scale="Greens",
                         text="Songs")
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False, height=480)
            chart(fig)

        with right:
            section("Songs per Year", "Volume of charting songs by year")
            yc = df["year"].value_counts().sort_index().reset_index()
            yc.columns = ["Year", "Songs"]
            fig = px.bar(yc, x="Year", y="Songs", color="Songs",
                         color_continuous_scale="Greens", text="Songs")
            fig.update_layout(coloraxis_showscale=False, xaxis=dict(dtick=1), height=220)
            chart(fig)

            section("Feature Correlations", "Pearson r between audio attributes")
            corr = df[FEATURES].corr().round(2)
            corr.index   = [LABELS[c] for c in FEATURES]
            corr.columns = [LABELS[c] for c in FEATURES]
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1)
            fig.update_layout(height=235, margin=dict(t=10, b=0, l=0, r=0))
            chart(fig)

    # ── 3D Space ──────────────────────────────────────────────────────────────
    with t2:
        section("3D Feature Space", "Rotate, zoom, and hover to explore individual songs")

        cx, cy, cz, _ = st.columns([1, 1, 1, 1])
        fx = cx.selectbox("X Axis", FEATURES, index=1, format_func=LABELS.get, key="3dx")
        fy = cy.selectbox("Y Axis", FEATURES, index=2, format_func=LABELS.get, key="3dy")
        fz = cz.selectbox("Z Axis", FEATURES, index=5, format_func=LABELS.get, key="3dz")

        genres_in_view  = df["top_genre"].unique()
        palette         = px.colors.qualitative.Dark24
        genre_color_map = {g: palette[i % len(palette)] for i, g in enumerate(sorted(df_raw["top_genre"].unique()))}

        fig3d = go.Figure()
        for genre in sorted(genres_in_view):
            gdf = df[df["top_genre"] == genre]
            fig3d.add_trace(go.Scatter3d(
                x=gdf[fx], y=gdf[fy], z=gdf[fz],
                mode="markers",
                marker=dict(size=4, color=genre_color_map[genre], opacity=0.82),
                name=genre,
                text=gdf["title"] + " — " + gdf["artist"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{LABELS[fx]}: %{{x}}<br>"
                    f"{LABELS[fy]}: %{{y}}<br>"
                    f"{LABELS[fz]}: %{{z}}<extra></extra>"
                ),
            ))

        fig3d.update_layout(
            **PLOT,
            height=580,
            scene=dict(
                xaxis=dict(title=LABELS[fx], backgroundcolor=CARD, gridcolor="#333", showbackground=True),
                yaxis=dict(title=LABELS[fy], backgroundcolor=CARD, gridcolor="#333", showbackground=True),
                zaxis=dict(title=LABELS[fz], backgroundcolor=CARD, gridcolor="#333", showbackground=True),
                bgcolor=BG,
            ),
            legend=dict(orientation="v", x=1.01, y=0.5, font=dict(size=10)),
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.divider()
        section("t-SNE 3D Genre Clustering",
                "All 10 audio features reduced to 3D — rotate to see how well genres separate")

        with st.spinner("Computing t-SNE (cached after first run)…"):
            tsne_df = compute_tsne()

        fig_tsne = go.Figure()
        for genre in sorted(tsne_df["genre"].unique()):
            gdf = tsne_df[tsne_df["genre"] == genre]
            fig_tsne.add_trace(go.Scatter3d(
                x=gdf["x"], y=gdf["y"], z=gdf["z"],
                mode="markers",
                marker=dict(size=3.5, color=genre_color_map.get(genre, "#888"), opacity=0.85),
                name=genre,
                text=gdf["title"] + " — " + gdf["artist"],
                hovertemplate="<b>%{text}</b><br>Genre: " + genre + "<extra></extra>",
            ))

        fig_tsne.update_layout(
            **PLOT,
            height=580,
            scene=dict(
                xaxis=dict(title="Component 1", backgroundcolor=CARD, gridcolor="#333", showbackground=True),
                yaxis=dict(title="Component 2", backgroundcolor=CARD, gridcolor="#333", showbackground=True),
                zaxis=dict(title="Component 3", backgroundcolor=CARD, gridcolor="#333", showbackground=True),
                bgcolor=BG,
            ),
            legend=dict(orientation="v", x=1.01, y=0.5, font=dict(size=10)),
        )
        st.plotly_chart(fig_tsne, use_container_width=True)
        st.caption("Tighter clusters = more distinctive genre. Overlapping regions = genres the model finds hard to separate.")

    # ── Genre Profiles ────────────────────────────────────────────────────────
    with t3:
        section("Genre Audio Fingerprints", "Each shape is a genre's average audio profile — larger area = stronger trait")

        top10     = df["top_genre"].value_counts().nlargest(10).index.tolist()
        sel_radar = st.multiselect("Compare genres", top10, default=top10[:6])

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
                    line=dict(width=2),
                ))
            fig.update_layout(
                **PLOT,
                polar=dict(
                    bgcolor=CARD,
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333", color="#666"),
                    angularaxis=dict(gridcolor="#333", color="#aaa"),
                ),
                height=520,
                legend=dict(orientation="h", y=-0.12, font=dict(size=11)),
            )
            chart(fig)
        else:
            st.info("Select at least one genre above.")

        st.divider()
        section("Feature Heatmap by Genre", "Average value of each audio feature across top 12 genres")

        top12 = df["top_genre"].value_counts().nlargest(12).index.tolist()
        hm    = df[df["top_genre"].isin(top12)].groupby("top_genre")[RADAR_COLS].mean().round(1)
        hm.columns = RADAR_LABELS
        fig   = px.imshow(hm, text_auto=True, color_continuous_scale="Greens", aspect="auto")
        fig.update_layout(**PLOT, height=420)
        chart(fig)

    # ── Trends ────────────────────────────────────────────────────────────────
    with t4:
        section("Audio Feature Trends", "How Billboard songs changed over the decade")

        sel_feats = st.multiselect(
            "Features", FEATURES,
            default=["nrgy", "val", "dnce", "acous", "pop"],
            format_func=LABELS.get,
        )
        if sel_feats:
            yearly = (
                df.groupby("year")[sel_feats].mean().reset_index()
                .melt(id_vars="year", var_name="Feature", value_name="Average")
            )
            yearly["Feature"] = yearly["Feature"].map(LABELS)
            fig = px.line(yearly, x="year", y="Average", color="Feature",
                          markers=True, labels={"year": "Year", "Average": "Average Value"})
            fig.update_traces(line=dict(width=2.5))
            fig.update_layout(**PLOT, xaxis=dict(dtick=1), height=420)
            chart(fig)
        else:
            st.info("Select at least one feature.")

        st.divider()
        section("Most Charted Artists", "Artists with the most songs in Billboard Top 100 (2010–2019)")

        top_art = (
            df.groupby("artist")
            .agg(Songs=("title", "count"), Avg_Pop=("pop", "mean"))
            .sort_values("Songs", ascending=False).head(15).reset_index()
        )
        top_art["Avg_Pop"] = top_art["Avg_Pop"].round(1)
        fig = px.bar(top_art, x="Songs", y="artist", orientation="h",
                     color="Avg_Pop", color_continuous_scale="Greens",
                     text="Songs", labels={"artist": "", "Avg_Pop": "Avg Popularity"})
        fig.update_layout(**PLOT, yaxis={"categoryorder": "total ascending"},
                          coloraxis_colorbar=dict(title="Avg Pop", thickness=12), height=480)
        chart(fig)

    # ── Songs ─────────────────────────────────────────────────────────────────
    with t5:
        section("Song Search", "Search the filtered dataset by title or artist")

        query   = st.text_input("", placeholder="Search title or artist…", label_visibility="collapsed")
        results = df[
            df["title"].str.contains(query, case=False, na=False) |
            df["artist"].str.contains(query, case=False, na=False)
        ] if query else df

        st.caption(f"{len(results):,} result{'s' if len(results) != 1 else ''}")
        cols_show = ["title", "artist", "top_genre", "year"] + FEATURES
        st.dataframe(
            results[cols_show]
            .rename(columns={**LABELS, "title": "Title", "artist": "Artist",
                              "top_genre": "Genre", "year": "Year"})
            .reset_index(drop=True),
            use_container_width=True, height=520,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
else:
    with st.spinner("Training Random Forest model…"):
        fi_df, y_te, y_pred, y_all = train_model()

    t1, t2, t3 = st.tabs(["📊  Performance", "🔬  Feature Analysis", "🗺  Predictions"])

    # ── Performance ───────────────────────────────────────────────────────────
    with t1:
        left, right = st.columns([1.1, 1])

        with left:
            section("Model Comparison", "4-fold cross-validation accuracy across all candidate models")
            model_df = pd.DataFrame({
                "Model":  ["Logistic Regression", "Random Forest", "SVM", "Decision Tree"],
                "Score":  [0.7842, 0.8026, 0.6079, 0.9184],
                "Color":  ["#636EFA", GREEN, "#EF553B", "#AB63FA"],
                "Note":   ["", "✓ Selected", "", "Overfit"],
            })
            fig = go.Figure()
            for _, row in model_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row["Score"]], y=[row["Model"]],
                    orientation="h",
                    marker_color=row["Color"],
                    text=f'{row["Score"]:.1%}  {row["Note"]}',
                    textposition="outside",
                    name=row["Model"],
                    showlegend=False,
                ))
            fig.update_layout(**PLOT, xaxis=dict(range=[0, 1.12], tickformat=".0%",
                              title="Mean CV Accuracy"), height=310)
            chart(fig)

        with right:
            section("Final Model Metrics", "Random Forest evaluated on 30% held-out test set")
            r1, r2 = st.columns(2)
            for col, lbl, val, sub in [
                (r1, "Test Accuracy",    "83.54%",  "held-out data"),
                (r2, "Best CV Score",    "80.26%",  "cross-validation"),
                (r1, "Top Genre Recall", "99%",     "dance pop"),
                (r2, "Genre Classes",    "17",      "after filtering"),
            ]:
                col.markdown(kpi(lbl, val, sub), unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-left:3px solid {GREEN};
                        border-radius:8px;padding:14px 16px;font-size:0.82rem;color:#ccc;margin-top:8px">
            Decision Tree's 91.8% CV score looks impressive but doesn't hold on unseen data —
            a classic sign of overfitting. Random Forest's bagging gives a more honest 83.5%
            that generalises to new songs.
            </div>""", unsafe_allow_html=True)

    # ── Feature Analysis ──────────────────────────────────────────────────────
    with t2:
        left, right = st.columns(2)

        with left:
            section("Feature Importance", "Which attributes drive genre prediction most")
            fig = px.bar(
                fi_df.head(15), x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Greens",
            )
            fig.update_layout(**PLOT, yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False, height=480)
            chart(fig)

        with right:
            section("Audio Feature Distributions", "Violin plots show spread within the filtered data")
            v_feat = st.selectbox("Feature", FEATURES, index=1,
                                  format_func=LABELS.get, key="model_violin")
            top8   = df["top_genre"].value_counts().nlargest(8).index.tolist()
            df8    = df[df["top_genre"].isin(top8)]
            fig    = px.violin(df8, x="top_genre", y=v_feat, color="top_genre",
                               box=True, points="outliers",
                               labels={"top_genre": "Genre", v_feat: LABELS[v_feat]})
            fig.update_layout(**PLOT, showlegend=False, xaxis_tickangle=30, height=480)
            chart(fig)

        st.divider()
        section("Parallel Coordinates",
                "Each line is a song — brush any axis to filter. Reveals how features cluster by genre")

        top6    = df_raw["top_genre"].value_counts().nlargest(6).index.tolist()
        df_para = df_raw[df_raw["top_genre"].isin(top6)].copy()
        codes   = pd.Categorical(df_para["top_genre"], categories=top6).codes
        palette = [GREEN, "#636EFA", "#EF553B", "#AB63FA", "#FFA15A", "#19D3F3"]

        fig = go.Figure(go.Parcoords(
            line=dict(
                color=codes,
                colorscale=[[i / (len(palette) - 1), c] for i, c in enumerate(palette)],
                showscale=False,
            ),
            dimensions=[
                dict(label=LABELS[f], values=df_para[f],
                     range=[df_para[f].min(), df_para[f].max()])
                for f in RADAR_COLS
            ],
        ))
        fig.update_layout(**PLOT, height=360)
        chart(fig)
        st.caption(f"Colors: {', '.join(f'<span style=\"color:{palette[i]}\">{g}</span>' for i, g in enumerate(top6))}", unsafe_allow_html=True)

    # ── Predictions ───────────────────────────────────────────────────────────
    with t3:
        section("Confusion Matrix",
                "Normalized by true class — diagonal = correct predictions, off-diagonal = misclassifications")

        labels  = sorted(y_all.unique())
        cm      = confusion_matrix(y_te, y_pred, labels=labels, normalize="true")
        cm_df   = pd.DataFrame(np.round(cm, 2), index=labels, columns=labels)
        fig     = px.imshow(cm_df, text_auto=True, color_continuous_scale="Greens",
                            labels=dict(x="Predicted Genre", y="True Genre"), aspect="auto")
        fig.update_layout(**PLOT, height=560,
                          xaxis=dict(tickangle=35, tickfont=dict(size=10)),
                          yaxis=dict(tickfont=dict(size=10)))
        chart(fig)

        st.divider()
        section("Per-Genre Recall", "How reliably the model identifies each genre")

        recall_data = pd.DataFrame({
            "Genre":  labels,
            "Recall": [cm[i, i] for i in range(len(labels))],
        }).sort_values("Recall", ascending=True)

        fig = px.bar(recall_data, x="Recall", y="Genre", orientation="h",
                     color="Recall", color_continuous_scale="Greens",
                     text=[f"{v:.0%}" for v in recall_data["Recall"]])
        fig.update_layout(**PLOT, xaxis=dict(tickformat=".0%", range=[0, 1.1]),
                          coloraxis_showscale=False, height=480)
        chart(fig)
        st.caption("Genres with few training samples (< 10 songs) tend to have lower recall — more data would help.")
