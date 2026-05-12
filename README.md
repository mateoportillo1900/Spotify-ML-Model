# 🎵 Spotify Genre Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> Multi-class genre classification across 35 genres using audio features from 24,993 Spotify songs spanning 1957–2020.

---

## Table of Contents

- [Overview](#overview)
- [Dashboard](#dashboard)
- [Dataset](#dataset)
- [Models & Results](#models--results)
- [Feature Importance](#feature-importance)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Findings](#key-findings)
- [License](#license)

---

## Overview

This project builds a **multi-class genre classifier** trained on Spotify audio attributes. Three supervised learning models were evaluated using 4-fold cross-validation on 24,993 songs across 35 genres. The deployed Random Forest model uses **audio features only** (no artist identity) and achieves **35.2% test accuracy** — 12× better than the random baseline of 2.9% for 35 classes.

**Goal:** Given song-level audio features (energy, danceability, BPM, etc.), predict the genre of the song.

---

## Dashboard

An interactive Streamlit dashboard lets you explore the full dataset and interact with the trained model live.

**Features:**
- 3D scatter plot and t-SNE genre clustering (interactive, rotate in browser)
- Genre audio fingerprint radar chart
- Audio feature trends across decades (1957–2020)
- **Live Genre Predictor** — adjust audio sliders and get real-time genre predictions

> To run locally: `streamlit run app.py`

---

## Dataset

| Property | Value |
|---|---|
| Sources | Billboard Top Songs (2010–2019) + TidyTuesday Spotify dataset |
| Size | 24,993 songs |
| Year range | 1957–2020 |
| Target classes | 35 genres (after filtering genres with <5 samples) |
| Key features | BPM, Energy, Danceability, Loudness, Liveness, Valence, Acousticness, Speechiness, Popularity |

The dataset was standardized before model training. Class imbalance across genres was addressed with balanced class weighting.

---

## Models & Results

| Model | CV Accuracy | Notes |
|---|---|---|
| Logistic Regression | 43.71% | Best CV score — benefits from artist one-hot encoding |
| Random Forest ✅ | 40.80% | Selected — more robust to unseen artists than LR |
| Decision Tree | 22.78% | Severe overfitting |

> Random baseline (35 classes) = **2.9%**. All models are well above chance.
> SVM excluded from comparison — prohibitively slow at 25k samples.

**Final model (Random Forest, audio features only, balanced class weighting):**
- Test accuracy: **35.18%** (12× above random baseline)
- Trained on 17,455 songs, evaluated on 7,481
- CV scores above used artist identity features; deployed model intentionally excludes them for generalisability

---

## Feature Importance

Top audio-only predictors (artist features excluded):

| Rank | Feature | Role |
|---|---|---|
| 1 | Energy (`nrgy`) | Separates high-intensity EDM/rock from acoustic genres |
| 2 | Acousticness (`acous`) | Strong negative signal for electronic genres |
| 3 | Speechiness (`spch`) | Key driver for hip hop / rap classification |
| 4 | Danceability (`dnce`) | Distinguishes rhythm-driven genres |
| 5 | Valence (`val`) | Separates emotionally positive vs. darker genres |

---

## Project Structure

```
Spotify-ML-Model/
├── app.py                     # Streamlit dashboard
├── Spotify_ML_Project.ipynb   # Full analysis and model training
├── spotify_top_music.csv      # Merged dataset (24,993 songs)
├── requirements.txt
├── .streamlit/
│   └── config.toml            # Dark theme configuration
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
git clone https://github.com/mateoportillo1900/Spotify-ML-Model.git
cd Spotify-ML-Model
jupyter notebook Spotify_ML_Project.ipynb
```

### Run the dashboard

```bash
streamlit run app.py
```

---

## Key Findings

- **35 genres is a genuinely hard problem.** Random baseline is 2.9% — the model at 35% is 12× better than chance. The earlier 84% accuracy (on 17 genres, 600 songs) was partly inflated by artist identity leaking into predictions.
- **Audio-only signals still have real power.** The Live Genre Predictor uses no artist data and still separates hip hop (high speechiness), EDM (high energy + danceability), and acoustic pop (high acousticness) reliably.
- **Music got louder and less acoustic over 60 years.** The trends chart shows acousticness cratering from the 1970s onward as music went electric, while energy peaked with the EDM era of the 2010s.
- **Class imbalance matters more at scale.** Dance pop has 1,486 songs vs. niche genres with 5–10. Balanced class weighting in Random Forest partially compensates, but more data for rare genres remains the highest-impact improvement.
- **Decision Tree overfits badly** — 22.8% CV vs higher training accuracy, confirming that ensemble methods (Random Forest) are essential for this problem.

---

## License

[MIT](LICENSE)
