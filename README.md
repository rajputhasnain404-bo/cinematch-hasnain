# CineMatch AI — Movie Recommendation System

An AI-based movie recommendation system that suggests movies according to a
user's interests, favourite genres and ratings — built on the real
**MovieLens ml-latest-small** dataset (9,742 movies, 100,836 ratings, 610 users).

## How to run (Windows)

1. Extract `cinematch.zip` anywhere.
2. Double-click **`run.bat`** (it installs requirements, then starts the server).
3. The app opens automatically at **http://127.0.0.1:8002**.

If port 8002 is busy, stop whatever is using it first:
`netstat -ano | findstr :8002` then `taskkill /PID <pid> /F`.

Manual start (either style works):

```bat
pip install -r requirements.txt
uvicorn backend.main:app --port 8002
```

or from inside `backend/`:

```bat
uvicorn main:app --port 8002
```

## What it does

- **Discover** — trending + top-rated rows, title search, genre / year /
  minimum-rating filters, 6 sort orders, pagination over all 9,742 movies.
- **Movie detail** — click any poster: synopsis, year, genres, director,
  cast, popular tags, average rating, similar movies. Rate it 1–5 stars or
  save it to your watchlist.
- **For You** — personalised recommendations with a **match %** and a
  **"Because you liked X"** explanation for every pick.
- **Onboarding** — rate 5 popular movies to unlock your matches, or try one
  of 3 demo personas with pre-existing ratings:
  - 🚀 **Ayaan** — Action & Sci-Fi Fan (34 ratings)
  - 🎭 **Sara** — Drama & Classics Lover (31 ratings)
  - 🍿 **Danish** — Family & Feel-Good Fan (30 ratings)
- **Cold start** — no ratings yet? Pick a mood (Feel-good, Edge of your
  seat, Mind-bending, Heartfelt, Family night, Late-night scare, Epic
  adventure) and get instant picks ranked by Bayesian average rating.
- **My Ratings / Watchlist** — everything you rated or saved, stored
  server-side per guest session (no login; the session id lives in
  localStorage).
- **Analytics** — dataset stats, genre & rating distributions (Chart.js,
  bundled locally — no CDN), top movies, and the model evaluation table.

## How the recommendation works

Three models, all real and explainable:

1. **Content-based** — TF-IDF over genres + MovieLens user tags + director +
   cast, cosine similarity to the movies you liked (rated ≥ 4).
2. **Collaborative filtering** — item-item cosine similarity on the
   mean-centred user–movie ratings matrix (top-50 neighbours per movie),
   with significance weighting (similarity shrunk by the number of users who
   rated both movies) and shrinkage toward your mean rating, so one weak
   neighbour can't produce an extreme prediction.
3. **Hybrid** — 45% content + 45% collaborative + 10% popularity (Bayesian
   average rating), each min-max normalised over the candidate set.

Match % is the blended hybrid score scaled to 55–98. "Because you liked X"
names the liked movie most similar to the recommendation (half content
similarity, half item-item similarity) plus the genres they share.

## Evaluation (hold-out test)

120 users (≥ 20 ratings each); 20% of each user's ratings held out
(1,026 ratings); relevant = held-out rating ≥ 4.0; candidates = movies with
≥ 10 training ratings. Seed 42. Recompute with
`python backend/evaluation.py` (writes `data/evaluation.json`).

| Model | Precision@10 | Recall@10 | RMSE |
|---|---|---|---|
| Content-based (TF-IDF) | 0.009 | 0.017 | 0.968 |
| Collaborative (item-item CF) | 0.049 | 0.110 | 0.680 |
| **Hybrid (45/45/10)** | **0.047** | **0.116** | 0.910 |
| Popularity baseline | 0.039 | 0.083 | — |

Reading the numbers: the hybrid finds more of what you like than any single
model (best Recall@10, beats the popularity baseline on both ranking
metrics), while pure collaborative filtering is the most accurate at
predicting exact star ratings (lowest RMSE). Pure content matching is weak
on its own — genres alone don't capture taste — which is exactly why the
system blends models. Random guessing would score Precision@10 ≈ 0.002.

## Data

- `data/movies.csv`, `data/ratings.csv`, `data/tags.csv` — MovieLens
  ml-latest-small (GroupLens). Real data.
- `data/metadata.json` — director, cast and a one-line synopsis (written for
  this project) for the 130 most-rated movies, plus their top MovieLens tags.
  Other titles show genres and community ratings.
- `data/personas.json` — the 3 demo personas (synthetic ratings, labelled as
  demo personas in the UI).
- `data/evaluation.json` — the hold-out metrics above.
- `data/guest_data.json` — created at runtime; guest sessions' ratings and
  watchlists.

## Tech

FastAPI + vanilla HTML/CSS/JS SPA (no framework, no CDN — Chart.js 4.4.1 is
bundled in `frontend/chart.min.js`). scikit-learn (TF-IDF, nearest
neighbours), scipy sparse matrices, pandas. Default port **8002** (8000 is
NoteWise/SupervisorMatch, 8001 is Polyglot). The browser auto-opens only
after port 8002 answers, so it never opens a stale server on another port;
set `AUTO_OPEN_BROWSER=0` to disable.

## Project layout

```
cinematch/
  README.md  requirements.txt  run.bat  run.sh
  backend/
    main.py            # FastAPI app + API + session store
    recommender.py     # ML engine (TF-IDF, item-item CF, hybrid)
    evaluation.py      # hold-out evaluation -> data/evaluation.json
    data_prep.py       # builds metadata.json / personas.json
    metadata_curated.py# director/cast/synopsis for top 130 movies
  data/                # MovieLens CSVs + generated JSON files
  frontend/
    index.html  styles.css  app.js  chart.min.js
```
