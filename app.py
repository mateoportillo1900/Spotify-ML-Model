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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Genre Intelligence",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; color: #1DB954; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { letter-spacing: -0.02em; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px 6px 0 0; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/mateoportillo1900/Spotify-ML-Model/refs/heads/main/spotify_top_music.csv"

FEATURES = ["bpm", "nrgy", "dnce", "db", "live", "val", "dur", "acous", "spch", "pop"]
LABELS   = {
    "bpm": "BPM", "nrgy": "Energy", "dnce": "Danceability",
    "db": "Loudness", "live": "Liveness", "val": "Valence",
    "dur": "Duration", "acous": "Acousticness", "spch": "Speechiness", "pop": "Popularity",
}
RADAR_COLS   = ["bpm", "nrgy", "dnce", "val", "acous", "spch", "pop"]
RADAR_LABELS = ["BPM", "Energy", "Dance", "Valence", "Acousticness", "Speechiness", "Popularity"]
GREEN_SCALE  = "Greens"

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

    num_feats = ["year", "bpm", "nrgy", "dnce", "db", "live", "val", "dur", "acous", "spch", "pop"]
    cat_feats  = ["artist"]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="constant")), ("sc", StandardScaler())]), num_feats),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
    ])

    X, y = df[num_feats + cat_feats], df["top_genre"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ))])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    ohe      = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
    all_feat = num_feats + ohe.get_feature_names_out(cat_feats).tolist()
    fi_df    = pd.DataFrame({"Feature": all_feat,
                              "Importance": pipe.named_steps["clf"].feature_importances_}
                            ).sort_values("Importance", ascending=False)
    return pipe, fi_df, y_te, y_pred, y

df_raw = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Spotify Explorer")
    st.caption("Billboard Top Songs · 2010–2019")
    st.divider()

    all_genres = sorted(df_raw["top_genre"].unique())
    sel_genres = st.multiselect("Genre", all_genres, default=all_genres)

    year_range = st.slider("Year Range", 2010, 2019, (2010, 2019))

    st.divider()
    st.markdown("**Feature Glossary**")
    st.markdown(
        "**BPM** Tempo · **Energy** Intensity · **Danceability** Rhythm fit · "
        "**Loudness** dB · **Valence** Positivity · **Acousticness** Acoustic confidence · "
        "**Speechiness** Spoken words · **Popularity** Spotify score"
    )

df = df_raw[
    df_raw["top_genre"].isin(sel_genres) &
    df_raw["year"].between(*year_range)
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎵 Spotify Genre Intelligence")
st.caption(
    f"Analyzing **{len(df):,} songs** across **{df['top_genre'].nunique()} genres** "
    f"from **{year_range[0]}** to **{year_range[1]}**"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Songs",        f"{len(df):,}")
c2.metric("Genres",       df["top_genre"].nunique())
c3.metric("Avg Energy",   f"{df['nrgy'].mean():.0f} / 100")
c4.metric("Avg Popularity", f"{df['pop'].mean():.0f} / 100")
c5.metric("Avg BPM",      f"{df['bpm'].mean():.0f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_explorer, tab_profiles, tab_trends, tab_model = st.tabs([
    "📊 Overview", "🔍 Feature Explorer", "🎯 Genre Profiles", "📈 Trends", "🤖 Model Results",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    left, right = st.columns(2)

    with left:
        genre_counts = df["top_genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Songs"]
        fig = px.bar(
            genre_counts, x="Songs", y="Genre", orientation="h",
            title="Songs per Genre",
            color="Songs", color_continuous_scale=GREEN_SCALE, text="Songs",
        )
        fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"},
                          coloraxis_showscale=False, height=520)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        year_counts = df["year"].value_counts().sort_index().reset_index()
        year_counts.columns = ["Year", "Songs"]
        fig = px.bar(
            year_counts, x="Year", y="Songs",
            title="Songs per Year",
            color="Songs", color_continuous_scale=GREEN_SCALE, text="Songs",
        )
        fig.update_layout(template="plotly_white", coloraxis_showscale=False,
                          xaxis=dict(dtick=1), height=240)
        st.plotly_chart(fig, use_container_width=True)

        corr = df[FEATURES].corr().round(2)
        corr.index = [LABELS[c] for c in FEATURES]
        corr.columns = [LABELS[c] for c in FEATURES]
        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Feature Correlation Matrix",
        )
        fig.update_layout(template="plotly_white", height=260,
                          margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — FEATURE EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab_explorer:
    st.markdown("### Interactive Scatter Explorer")
    st.caption("Hover over any song to see its title, artist, and year.")

    col_x, col_y, col_sz = st.columns(3)
    x_feat  = col_x.selectbox("X Axis",      FEATURES, index=1, format_func=LABELS.get)
    y_feat  = col_y.selectbox("Y Axis",       FEATURES, index=2, format_func=LABELS.get)
    sz_feat = col_sz.selectbox("Bubble Size", FEATURES, index=9, format_func=LABELS.get)

    fig = px.scatter(
        df, x=x_feat, y=y_feat, color="top_genre", size=sz_feat,
        hover_data={"title": True, "artist": True, "year": True,
                    x_feat: False, y_feat: False, sz_feat: False},
        title=f"{LABELS[x_feat]} vs {LABELS[y_feat]} (sized by {LABELS[sz_feat]})",
        labels={x_feat: LABELS[x_feat], y_feat: LABELS[y_feat], "top_genre": "Genre"},
        opacity=0.75, size_max=18,
    )
    fig.update_layout(template="plotly_white", height=500,
                      legend=dict(orientation="v", x=1.01, y=1))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Distribution by Genre")

    top8    = df["top_genre"].value_counts().nlargest(8).index.tolist()
    df_top8 = df[df["top_genre"].isin(top8)]
    v_feat  = st.selectbox("Feature", FEATURES, index=1,
                           format_func=LABELS.get, key="violin")

    fig = px.violin(
        df_top8, x="top_genre", y=v_feat, color="top_genre",
        box=True, points="outliers",
        title=f"{LABELS[v_feat]} Distribution — Top 8 Genres",
        labels={"top_genre": "Genre", v_feat: LABELS[v_feat]},
    )
    fig.update_layout(showlegend=False, template="plotly_white",
                      xaxis_tickangle=30, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Song Search")
    query = st.text_input("Search by title or artist", placeholder="e.g. Katy Perry, Blinding Lights…")
    results = df[
        df["title"].str.contains(query, case=False, na=False) |
        df["artist"].str.contains(query, case=False, na=False)
    ] if query else df.head(20)

    display_cols = ["title", "artist", "top_genre", "year"] + FEATURES
    st.dataframe(
        results[display_cols].rename(columns=LABELS).rename(
            columns={"title": "Title", "artist": "Artist",
                     "top_genre": "Genre", "year": "Year"}
        ).reset_index(drop=True),
        use_container_width=True, height=300,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GENRE PROFILES
# ─────────────────────────────────────────────────────────────────────────────
with tab_profiles:
    st.markdown("### Genre Audio Fingerprints")
    st.caption("Each shape shows the average audio profile of a genre — larger = higher value.")

    top10          = df["top_genre"].value_counts().nlargest(10).index.tolist()
    sel_radar      = st.multiselect("Compare Genres", top10, default=top10[:6])

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
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Genre Audio Fingerprints",
            template="plotly_white", height=560,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one genre above.")

    st.divider()
    st.markdown("### Average Feature Values by Genre")

    top12  = df["top_genre"].value_counts().nlargest(12).index.tolist()
    hm_df  = (df[df["top_genre"].isin(top12)]
              .groupby("top_genre")[RADAR_COLS]
              .mean().round(1))
    hm_df.columns = RADAR_LABELS

    fig = px.imshow(
        hm_df, text_auto=True, color_continuous_scale=GREEN_SCALE,
        title="Average Audio Features — Top 12 Genres", aspect="auto",
    )
    fig.update_layout(template="plotly_white", height=460)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — TRENDS
# ─────────────────────────────────────────────────────────────────────────────
with tab_trends:
    st.markdown("### Audio Feature Trends Over Time")

    sel_feats = st.multiselect(
        "Features to Track", FEATURES,
        default=["nrgy", "val", "dnce", "acous", "pop"],
        format_func=LABELS.get,
    )

    if sel_feats:
        yearly = (
            df.groupby("year")[sel_feats].mean().reset_index()
            .melt(id_vars="year", var_name="Feature", value_name="Average")
        )
        yearly["Feature"] = yearly["Feature"].map(LABELS)
        fig = px.line(
            yearly, x="year", y="Average", color="Feature", markers=True,
            title="How Billboard Songs Changed (2010–2019)",
            labels={"year": "Year", "Average": "Average Value"},
        )
        fig.update_layout(template="plotly_white", xaxis=dict(dtick=1), height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one feature above.")

    st.divider()
    st.markdown("### Most Charted Artists")

    top_artists = (
        df.groupby("artist")
        .agg(Songs=("title", "count"), Avg_Popularity=("pop", "mean"))
        .sort_values("Songs", ascending=False).head(15)
        .reset_index()
    )
    top_artists["Avg_Popularity"] = top_artists["Avg_Popularity"].round(1)

    fig = px.bar(
        top_artists, x="Songs", y="artist", orientation="h",
        color="Avg_Popularity", color_continuous_scale=GREEN_SCALE,
        title="Top 15 Most Charted Artists (2010–2019)",
        text="Songs",
        labels={"artist": "Artist", "Avg_Popularity": "Avg Popularity"},
    )
    fig.update_layout(template="plotly_white",
                      yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    left, right = st.columns([1, 1])

    with left:
        model_df = pd.DataFrame({
            "Model":       ["Logistic Regression", "Random Forest", "SVM", "Decision Tree"],
            "CV Accuracy": [0.7842, 0.8026, 0.6079, 0.9184],
        })
        fig = px.bar(
            model_df, x="CV Accuracy", y="Model", orientation="h",
            color="Model",
            color_discrete_map={
                "Logistic Regression": "#636EFA",
                "Random Forest":       "#1DB954",
                "SVM":                 "#EF553B",
                "Decision Tree":       "#AB63FA",
            },
            text=[f"{v:.1%}" for v in model_df["CV Accuracy"]],
            title="Model Comparison — 4-Fold CV Accuracy",
        )
        fig.update_layout(
            template="plotly_white",
            xaxis=dict(range=[0, 1.05], tickformat=".0%"),
            showlegend=False, height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Final Model: Random Forest")
        m1, m2 = st.columns(2)
        m1.metric("Test Accuracy",   "83.54%")
        m2.metric("Best CV Score",   "80.26%")
        m1.metric("Top Genre Recall", "99%", delta="Dance Pop")
        m2.metric("Genre Classes",   "17")
        st.markdown("""
        > Random Forest was selected over Decision Tree despite lower CV accuracy
        > because DT shows signs of overfitting — its 91.8% CV score doesn't
        > hold on held-out data the way RF's 83.5% does.
        """)

    st.divider()

    with st.spinner("Training model to compute feature importance…"):
        _, fi_df, y_te, y_pred, y_all = train_model()

    left2, right2 = st.columns(2)

    with left2:
        fig = px.bar(
            fi_df.head(15), x="Importance", y="Feature", orientation="h",
            title="Top 15 Most Important Features",
            color="Importance", color_continuous_scale=GREEN_SCALE,
        )
        fig.update_layout(
            template="plotly_white",
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False, height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        labels = sorted(y_all.unique())
        cm     = confusion_matrix(y_te, y_pred, labels=labels, normalize="true")
        cm_df  = pd.DataFrame(np.round(cm, 2), index=labels, columns=labels)
        fig    = px.imshow(
            cm_df, text_auto=True, color_continuous_scale=GREEN_SCALE,
            title="Confusion Matrix (Normalized by True Class)",
            labels=dict(x="Predicted", y="Actual"), aspect="auto",
        )
        fig.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)
