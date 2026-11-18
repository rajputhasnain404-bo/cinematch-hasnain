"""CineMatch AI — recommendation engine.

Real, explainable models on the MovieLens ml-latest-small dataset
(9,742 movies, 100,836 ratings, 610 users):

1. Content-based: TF-IDF over genres + user tags + director + cast,
   cosine similarity between a movie and the user's liked movies.
2. Collaborative: item-item cosine similarity on the mean-centred
   user-movie ratings matrix (top-50 neighbours per movie, scipy sparse).
3. Hybrid: 0.5 * content + 0.5 * collaborative (both min-max normalised
   over the candidate set), with a "Because you liked X" explanation and
   a match % for every recommendation.

Cold start: mood/genre picker ranks movies by a Bayesian average rating.
"""
import json
import math
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

MOODS = {
    "feel-good": {"label": "Feel-good", "genres": ["Comedy", "Animation", "Children", "Musical"],
                  "exclude": ["Crime", "Horror", "Thriller", "War", "Film-Noir"]},
    "thrilling": {"label": "Edge of your seat", "genres": ["Thriller", "Action", "Crime"],
                  "exclude": []},
    "mind-bending": {"label": "Mind-bending", "genres": ["Sci-Fi", "Mystery"], "exclude": []},
    "heartfelt": {"label": "Heartfelt", "genres": ["Drama", "Romance"], "exclude": ["Horror"]},
    "family": {"label": "Family night", "genres": ["Children", "Animation", "Adventure"],
               "exclude": ["Horror", "Crime", "Thriller"]},
    "scary": {"label": "Late-night scare", "genres": ["Horror"], "exclude": []},
    "epic": {"label": "Epic adventure", "genres": ["Adventure", "Fantasy", "War"], "exclude": []},
}

TOP_N_NEIGHBOURS = 50


def clean_title(raw):
    """'Shawshank Redemption, The (1994)' -> ('The Shawshank Redemption', 1994)."""
    title = raw.strip()
    m = YEAR_RE.search(title)
    year = int(m.group(1)) if m else None
    if m:
        title = title[: m.start()].strip()
    # Strip trailing sort-key parentheticals, e.g. "Amelie (Fabuleux ..., Le)".
    while True:
        m2 = re.search(r", (Le|La|The|A)\)\s*$", title)
        if not m2:
            break
        open_idx = title.rfind("(", 0, m2.start())
        if open_idx < 0:
            break
        title = title[:open_idx].strip()
    if title.endswith(", The"):
        title = "The " + title[:-5]
    elif title.endswith(", A"):
        title = "A " + title[:-3]
    return title, year


class Engine:
    def __init__(self, data_dir=DATA):
        self.data_dir = data_dir
        movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
        ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
        with open(os.path.join(data_dir, "metadata.json")) as f:
            self.metadata = json.load(f)

        movies["title_clean"], movies["year"] = zip(*movies["title"].map(clean_title))
        movies["genre_list"] = movies["genres"].apply(
            lambda g: [x for x in g.split("|") if x not in ("(no genres listed)", "IMAX")]
        )
        self.movies = movies.reset_index(drop=True)
        self.movie_ids = self.movies["movieId"].astype(int).tolist()
        self.mid_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.idx_to_mid = {i: mid for mid, i in self.mid_to_idx.items()}
        self.n_movies = len(self.movies)

        stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        self.movies["avg_rating"] = self.movies["movieId"].map(stats["mean"]).fillna(0.0)
        self.movies["rating_count"] = self.movies["movieId"].map(stats["count"]).fillna(0).astype(int)
        self.global_mean = float(ratings["rating"].mean())
        # Bayesian average (m = 50 prior ratings at the global mean).
        n = self.movies["rating_count"].astype(float)
        self.movies["bayes"] = (n * self.movies["avg_rating"] + 50 * self.global_mean) / (n + 50)
        # "Trending": ratings in the most recent part of the dataset (>= 2015).
        recent = ratings[ratings["timestamp"] >= 1420070400]  # 2015-01-01
        recent_counts = recent.groupby("movieId").size()
        self.movies["recent_count"] = self.movies["movieId"].map(recent_counts).fillna(0).astype(int)

        # ---- user-movie matrix (mean-centred per user) ----
        self.user_ids = sorted(ratings["userId"].unique().tolist())
        self.uid_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        rows = ratings["userId"].map(self.uid_to_idx).to_numpy()
        cols = ratings["movieId"].map(self.mid_to_idx).to_numpy()
        vals = ratings["rating"].to_numpy(dtype=float)
        self.R = sparse.csr_matrix((vals, (rows, cols)), shape=(len(self.user_ids), self.n_movies))
        user_means = np.asarray(self.R.sum(axis=1)).ravel() / np.maximum(
            np.asarray((self.R != 0).sum(axis=1)).ravel(), 1)
        self.user_means = user_means
        Rc = self.R.copy().tocoo()
        Rc.data = Rc.data - user_means[Rc.row]
        self.Rc = Rc.tocsr()
        self.rated_mask = (self.R != 0)

        # ---- item-item neighbours (cosine on mean-centred item vectors) ----
        nn = NearestNeighbors(n_neighbors=TOP_N_NEIGHBOURS + 1, metric="cosine", algorithm="brute")
        nn.fit(self.Rc.T.tocsr())
        dist, idx = nn.kneighbors(self.Rc.T.tocsr())
        sim = 1.0 - dist
        # Drop the self-neighbour (first column, sim == 1.0).
        self.item_nb_idx = idx[:, 1:]
        raw_sim = sim[:, 1:]
        # Significance weighting: cosine on a handful of shared raters is
        # spuriously high for obscure movies, so shrink each similarity by
        # the number of users who rated BOTH movies: sim * co / (co + 10).
        # A pair with 2 co-raters keeps ~17% of its similarity; 100 keep 91%.
        B = (self.R != 0).toarray().astype(bool)  # users x movies
        co = np.zeros_like(raw_sim)
        for i in range(self.n_movies):
            nb = self.item_nb_idx[i]
            co[i] = (B[:, nb] & B[:, i][:, None]).sum(axis=0)
        self.item_nb_sim = raw_sim * co / (co + 10.0)
        self.item_nb_co = co
        # Sparse similarity matrix for fast vectorised scoring.
        r = np.repeat(np.arange(self.n_movies), TOP_N_NEIGHBOURS)
        c = self.item_nb_idx.ravel()
        v = self.item_nb_sim.ravel()
        keep = v > 0.05
        self.S = sparse.csr_matrix((v[keep], (r[keep], c[keep])),
                                    shape=(self.n_movies, self.n_movies))

        # ---- content TF-IDF ----
        docs = []
        for _, row in self.movies.iterrows():
            mid = str(int(row["movieId"]))
            meta = self.metadata.get(mid, {})
            parts = [g.lower().replace("-", "") for g in row["genre_list"]]
            parts += [t.replace(" ", "_") for t in meta.get("tags", [])]
            if meta.get("director"):
                parts.append("dir_" + meta["director"].lower().replace(" ", "").replace(",", ""))
            for actor in meta.get("cast", []):
                parts.append("cast_" + actor.lower().replace(" ", ""))
            # Genres count double so the genre signal is not drowned out.
            docs.append(" ".join(parts + parts[: len(row["genre_list"])]))
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2)
        self.tfidf = self.vectorizer.fit_transform(docs)
        # Normalised TF-IDF rows for cosine via dot product.
        norms = sparse.linalg.norm(self.tfidf, axis=1)
        self.tfidf_n = sparse.diags(1.0 / np.maximum(norms, 1e-9)) @ self.tfidf

    # ---------- helpers ----------
    def movie_dict(self, idx):
        row = self.movies.iloc[idx]
        mid = int(row["movieId"])
        meta = self.metadata.get(str(mid), {})
        return {
            "id": mid,
            "title": row["title_clean"],
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "genres": row["genre_list"],
            "avg_rating": round(float(row["avg_rating"]), 2),
            "rating_count": int(row["rating_count"]),
            "director": meta.get("director"),
            "cast": meta.get("cast", []),
            "synopsis": meta.get("synopsis"),
            "tags": meta.get("tags", []),
        }

    def _rated_arrays(self, ratings):
        """ratings: {movieId: stars} -> (indices, values) for known movies."""
        idxs, vals = [], []
        for mid, r in ratings.items():
            i = self.mid_to_idx.get(int(mid))
            if i is not None:
                idxs.append(i)
                vals.append(float(r))
        return np.array(idxs, dtype=int), np.array(vals, dtype=float)

    # ---------- content model ----------
    def content_scores(self, ratings):
        """Cosine similarity of every movie to the user's liked movies."""
        idxs, vals = self._rated_arrays(ratings)
        scores = np.zeros(self.n_movies)
        if len(idxs) == 0:
            return scores
        liked = idxs[vals >= 4.0]
        if len(liked) == 0:
            liked = idxs[np.argsort(-vals)[:3]]
        profile = np.asarray(self.tfidf_n[liked].mean(axis=0)).ravel()
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
            scores = np.asarray(self.tfidf_n @ profile).ravel()
        return np.clip(scores, 0, 1)

    def content_predict(self, ratings):
        """Rating prediction: similarity-weighted average of the user's ratings."""
        idxs, vals = self._rated_arrays(ratings)
        pred = np.full(self.n_movies, np.nan)
        if len(idxs) == 0:
            return pred
        # Weighted average using the top-20 most similar rated movies per item.
        sim_mat = self.tfidf_n @ self.tfidf_n[idxs].T  # n_movies x n_rated (sparse)
        sim_mat = sim_mat.toarray()
        order = np.argsort(-sim_mat, axis=1)[:, :20]
        top_sim = np.take_along_axis(sim_mat, order, axis=1)
        top_r = vals[order]
        wsum = np.where(top_sim > 0.05, top_sim, 0.0)
        denom = wsum.sum(axis=1)
        numer = (wsum * top_r).sum(axis=1)
        # Shrinkage toward the user's own mean rating: a prediction backed by
        # only one weak neighbour must not jump to an extreme value.
        umean = float(vals.mean())
        ok = denom >= 0.3
        pred[ok] = (numer[ok] + 0.5 * umean) / (denom[ok] + 0.5)
        return pred

    # ---------- collaborative model ----------
    def cf_predict(self, ratings):
        """Item-item rating prediction for every movie (NaN where not possible)."""
        idxs, vals = self._rated_arrays(ratings)
        pred = np.full(self.n_movies, np.nan)
        if len(idxs) == 0:
            return pred
        umean = float(vals.mean())
        centered = vals - umean
        rvec = sparse.csr_matrix((centered, (np.zeros(len(idxs), dtype=int), idxs)),
                                  shape=(1, self.n_movies))
        numer = np.asarray((rvec @ self.S).todense()).ravel()
        denom = np.asarray(((rvec != 0).astype(float) @ np.abs(self.S)).todense()).ravel()
        # Shrinkage (beta = 0.5) + minimum similarity mass: a prediction backed
        # by one weak neighbour must not jump to an extreme value.
        ok = denom >= 0.3
        pred[ok] = umean + numer[ok] / (denom[ok] + 0.5)
        return np.clip(pred, 0.5, 5.0)

    # ---------- hybrid blending (shared by live recs and evaluation) ----------
    def blend_scores(self, ratings, mask=None):
        """Min-max normalised content / CF / popularity scores + hybrid blend.

        Weights: 0.45 content + 0.45 collaborative + 0.10 popularity
        (Bayesian average rating). The popularity term is a standard guard
        so the hybrid never does worse than recommending popular titles.
        `mask` selects the candidate set the normalisation runs over.
        """
        if mask is None:
            mask = np.ones(self.n_movies, dtype=bool)
        c_scores = self.content_scores(ratings)
        cf_pred = self.cf_predict(ratings)
        cf_scores = np.where(np.isnan(cf_pred), np.nan, (cf_pred - 0.5) / 4.5)
        pop = self.movies["bayes"].to_numpy(dtype=float)

        def _norm(a):
            vals = a[mask]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                return np.zeros(self.n_movies)
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-9:
                return np.where(np.isnan(a), 0.0, 0.5)
            return np.where(np.isnan(a), 0.0, (a - lo) / (hi - lo))

        n_c = _norm(c_scores)
        n_f = _norm(cf_scores)
        n_p = _norm(pop)
        hybrid = 0.45 * n_c + 0.45 * n_f + 0.10 * n_p
        return {"content": n_c, "cf": n_f, "popularity": n_p, "hybrid": hybrid,
                "content_raw": c_scores, "cf_pred": cf_pred}

    # ---------- hybrid recommendation ----------
    def recommend(self, ratings, limit=12, weights=(0.5, 0.5)):
        """Return ranked recommendations with match % and explanations."""
        idxs, _ = self._rated_arrays(ratings)
        if len(idxs) == 0:
            return []  # no profile yet — caller should use cold_start instead
        rated_set = set(idxs.tolist())
        mask = np.ones(self.n_movies, dtype=bool)
        mask[list(rated_set)] = False
        blend = self.blend_scores(ratings, mask=mask)
        hybrid = blend["hybrid"].copy()
        hybrid[list(rated_set)] = -1  # never recommend already-rated movies
        order = np.argsort(-hybrid)[: max(limit * 4, 50)]
        liked = [i for i in idxs if ratings.get(self.idx_to_mid[i], 0) >= 4.0] or idxs.tolist()

        results = []
        for j in order:
            if hybrid[j] < 0:
                continue
            # Explanation: the liked movie most similar to this recommendation.
            best, best_sim = None, -1.0
            for li in liked:
                s = 0.5 * float((self.tfidf_n[j] @ self.tfidf_n[li].T).toarray()[0, 0]) \
                    + 0.5 * float(self.S[j, li])
                if s > best_sim:
                    best_sim, best = s, li
            d = self.movie_dict(int(j))
            shared = [g for g in d["genres"] if best is not None
                      and g in self.movies.iloc[best]["genre_list"]]
            d["match"] = int(round(55 + 43 * float(hybrid[j])))
            d["content_score"] = round(float(blend["content_raw"][j]), 3)
            d["cf_score"] = None if np.isnan(blend["cf_pred"][j]) else round(
                float((blend["cf_pred"][j] - 0.5) / 4.5), 3)
            d["because"] = self.movies.iloc[best]["title_clean"] if best is not None else None
            d["because_genres"] = shared[:2]
            results.append(d)
            if len(results) >= limit:
                break
        return results

    # ---------- cold start / browsing ----------
    def cold_start(self, mood=None, genres=None, limit=12):
        df = self.movies[self.movies["rating_count"] >= 20].copy()
        if mood and mood in MOODS:
            want = set(MOODS[mood]["genres"])
            drop = set(MOODS[mood].get("exclude", []))
            df = df[df["genre_list"].apply(
                lambda gs: bool(want & set(gs)) and not (drop & set(gs)))]
        elif genres:
            want = set(genres)
            df = df[df["genre_list"].apply(lambda gs: bool(want & set(gs)))]
        df = df.sort_values("bayes", ascending=False).head(limit)
        return [self.movie_dict(int(i)) for i in df.index]

    def trending(self, limit=12):
        df = self.movies.sort_values(["recent_count", "bayes"], ascending=False).head(limit)
        return [self.movie_dict(int(i)) for i in df.index]

    def top_rated(self, limit=12):
        df = self.movies[self.movies["rating_count"] >= 50].sort_values(
            "bayes", ascending=False).head(limit)
        return [self.movie_dict(int(i)) for i in df.index]

    def similar_movies(self, movie_id, limit=8):
        i = self.mid_to_idx.get(int(movie_id))
        if i is None:
            return []
        sims = (self.tfidf_n[i] @ self.tfidf_n.T).toarray().ravel()
        sims[i] = -1
        order = np.argsort(-sims)[:limit]
        out = []
        for j in order:
            d = self.movie_dict(int(j))
            d["similarity"] = round(float(sims[j]), 3)
            out.append(d)
        return out

    def dataset_stats(self):
        return {
            "movies": int(self.n_movies),
            "ratings": int(self.R.nnz),
            "users": int(len(self.user_ids)),
            "global_mean_rating": round(self.global_mean, 2),
            "genres": GENRES,
            "curated_movies": len(self.metadata),
        }
