# 🎵 Spotify Genre Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> Predicting a song's genre from audio features using Billboard Top Songs data (2010–2019).

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models & Results](#models--results)
- [Feature Importance](#feature-importance)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Findings](#key-findings)
- [License](#license)

---

## Overview

This project builds a **multi-class genre classifier** trained on Spotify audio attributes. Four supervised learning models were evaluated using 4-fold cross-validation, with hyperparameter tuning applied to the best performer. The final model achieves **83.5% test accuracy** on 17 genre classes.

**Goal:** Given song-level audio features (energy, danceability, BPM, etc.), predict the genre of the song.

---

## Dataset

| Property | Value |
|---|---|
| Source | Billboard Top Songs (2010–2019) |
| Size | 603 songs |
| Target classes | 17 genres |
| Key features | BPM, Energy, Danceability, Loudness, Liveness, Valence, Acousticness, Speechiness, Popularity |

The dataset (`spotify_top_music.csv`) includes one-hot encoded artist features and was standardized before model training. Class imbalance was addressed with weighted training.

---

## Models & Results

| Model | Cross-Val Accuracy |
|---|---|
| Logistic Regression | 78.42% |
| Random Forest ✅ | **80.26%** |
| Support Vector Machine | 60.79% |
| Decision Tree | 91.84%* |

> *Decision Tree shows signs of overfitting. Random Forest was selected as the final model for its better generalization.

**Final model (Random Forest with balanced class weighting):**
- Test accuracy: **83.54%**
- Strongest genre: *dance pop* — 99% recall

---

## Feature Importance

Top predictors ranked by Random Forest feature importance:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Energy (`nrgy`) | 5.01% |
| 2 | Popularity (`pop`) | 4.79% |
| 3 | Acousticness (`acous`) | 4.51% |
| 4 | Valence | 4.34% |

Artist identity features also contribute meaningfully (~4.5% each), suggesting that certain artists strongly anchor to specific genres.

---

## Project Structure

```
Spotify-ML-Model/
├── Spotify_ML_Project.ipynb   # Full analysis and model training
├── spotify_top_music.csv      # Processed dataset
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

```bash
git clone https://github.com/mateoportillo1900/Spotify-ML-Model.git
cd Spotify-ML-Model
jupyter notebook Spotify_ML_Project.ipynb
```

---

## Key Findings

- **Random Forest outperforms SVM and Logistic Regression** on this multi-class problem, likely due to the non-linear decision boundaries between genres.
- **Energy and Acousticness** are the strongest audio-only signals — louder, high-energy tracks cluster toward pop/dance genres while acoustic songs skew toward country and folk.
- **Class imbalance** (some genres have <5 samples) limits recall on rare genres; collecting more data per genre would be the highest-impact next step.
- A **Decision Tree's 91.8% CV accuracy** is misleading — it memorizes training splits. Random Forest's bagging corrects this to a more honest 83.5% on held-out data.

---

## License

[MIT](LICENSE)
