"""CineMatch AI — offline hold-out evaluation (run once at build time).

For 120 sampled users (>= 20 ratings each), hold out 20% of their ratings
(4-10 items per user). Relevant = held-out rating >= 4.0.
Candidates = movies with >= 10 training ratings, not in the user's train set.

Reports, per model (content-based, collaborative, hybrid):
  Precision@10, Recall@10
and RMSE of rating predictions (collaborative, content, hybrid).

Writes data/evaluation.json.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommender import Engine

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MIN_TRAIN_COUNT = 10
N_TEST_USERS = 120
K = 10


def main():
    t0 = time.time()
    eng = Engine(DATA)
    ratings = pd.read_csv(os.path.join(DATA, "ratings.csv"))
    rng = np.random.default_rng(42)

    eligible = ratings.groupby("userId").size()
    eligible = eligible[eligible >= 20].index.tolist()
    test_users = sorted(rng.choice(eligible, size=min(N_TEST_USERS, len(eligible)),
                                   replace=False).tolist())

    # Train counts per movie (used for the candidate universe).
    train_counts = ratings.groupby("movieId").size()
    cand_idx = np.array([eng.mid_to_idx[m] for m in train_counts[train_counts >= MIN_TRAIN_COUNT].index
                         if m in eng.mid_to_idx], dtype=int)

    metrics = {"content": {"p": [], "r": []}, "cf": {"p": [], "r": []},
               "hybrid": {"p": [], "r": []}, "popularity": {"p": [], "r": []}}
    rmse = {"content": [], "cf": [], "hybrid": []}
    n_held = 0

    for u in test_users:
        ur = ratings[ratings["userId"] == u]
        idx = np.arange(len(ur))
        rng.shuffle(idx)
        n_hold = int(np.clip(round(0.2 * len(ur)), 4, 10))
        hold_idx, train_idx = idx[:n_hold], idx[n_hold:]
        hold = ur.iloc[hold_idx]
        train = ur.iloc[train_idx]
        train_ratings = {int(mid): float(r) for mid, r in zip(train["movieId"], train["rating"])}
        hold_true = {int(mid): float(r) for mid, r in zip(hold["movieId"], hold["rating"])}
        relevant = {mid for mid, r in hold_true.items() if r >= 4.0}
        if not relevant:
            continue
        n_held += len(hold_true)

        # --- ranking scores for every candidate (same blending as live recs) ---
        rated_set = set(eng.mid_to_idx[m] for m in train_ratings if m in eng.mid_to_idx)
        cand_mask = np.zeros(eng.n_movies, dtype=bool)
        cand_mask[cand_idx] = True
        cand_mask[list(rated_set)] = False
        blend = eng.blend_scores(train_ratings, mask=cand_mask)
        co_pred = eng.content_predict(train_ratings)
        cf_pred = blend["cf_pred"]

        scores = {"content": blend["content"].copy(), "cf": blend["cf"].copy(),
                  "hybrid": blend["hybrid"].copy(),
                  "popularity": blend["popularity"].copy()}
        for s in scores.values():
            s[~cand_mask] = -1

        for name, sc in scores.items():
            top = np.argsort(-sc)[:K]
            hits = sum(1 for j in top if eng.idx_to_mid[int(j)] in relevant)
            metrics[name]["p"].append(hits / K)
            metrics[name]["r"].append(hits / len(relevant))

        # --- RMSE on held-out pairs ---
        for name, pred_arr in (("cf", cf_pred), ("content", co_pred)):
            errs = []
            for mid, true in hold_true.items():
                j = eng.mid_to_idx.get(mid)
                if j is not None and not np.isnan(pred_arr[j]):
                    errs.append((pred_arr[j] - true) ** 2)
            if errs:
                rmse[name].append(float(np.mean(errs)))
        errs = []
        for mid, true in hold_true.items():
            j = eng.mid_to_idx.get(mid)
            if j is None:
                continue
            c, f = co_pred[j], cf_pred[j]
            if np.isnan(c) and np.isnan(f):
                continue
            p = f if np.isnan(c) else (c if np.isnan(f) else 0.6 * f + 0.4 * c)
            errs.append((p - true) ** 2)
        if errs:
            rmse["hybrid"].append(float(np.mean(errs)))

    out = {
        "method": "Hold-out: 120 users (>=20 ratings), 20% of each user's ratings held out "
                  "(4-10 items), relevant = held-out rating >= 4.0, candidates = movies with "
                  ">= 10 training ratings. Seed 42.",
        "n_test_users": len(test_users),
        "n_held_out_ratings": n_held,
        "models": {},
    }
    for name in ("content", "cf", "hybrid", "popularity"):
        entry = {
            "precision_at_10": round(float(np.mean(metrics[name]["p"])), 3),
            "recall_at_10": round(float(np.mean(metrics[name]["r"])), 3),
        }
        if name in rmse:
            entry["rmse"] = round(float(np.sqrt(np.mean(rmse[name]))), 3) if rmse[name] else None
        out["models"][name] = entry
    out["computed_in_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(DATA, "evaluation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
