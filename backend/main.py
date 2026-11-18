"""CineMatch AI — FastAPI backend.

Serves the SPA and a JSON API over the MovieLens ml-latest-small dataset.
No login: a guest session id (localStorage) owns the user's ratings and
watchlist, stored server-side in data/guest_data.json. Three demo personas
come with pre-existing ratings.

Run from the project root:  uvicorn backend.main:app --port 8002
Run from inside backend/:   uvicorn main:app --port 8002
Or just double-click run.bat (Windows) / run.sh.
"""
import json
import os
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    # When run from inside backend/ as `uvicorn main:app`
    from recommender import GENRES, MOODS, Engine
except ImportError:
    # When run from the project root as `uvicorn backend.main:app`
    from backend.recommender import GENRES, MOODS, Engine

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FRONTEND = os.path.join(BASE, "frontend")
if os.environ.get("VERCEL") == "1":
    # Vercel's filesystem is read-only except /tmp: guest sessions live
    # there (fresh on every cold start, which is fine for guest sessions).
    GUEST_FILE = os.path.join("/tmp", "guest_data.json")
else:
    GUEST_FILE = os.path.join(DATA, "guest_data.json")
PORT = 8002

engine = Engine(DATA)
with open(os.path.join(DATA, "personas.json")) as f:
    PERSONAS = json.load(f)
with open(os.path.join(DATA, "onboarding.json")) as f:
    ONBOARDING_IDS = json.load(f)
try:
    with open(os.path.join(DATA, "evaluation.json")) as f:
        EVALUATION = json.load(f)
except FileNotFoundError:
    EVALUATION = {}

_lock = threading.Lock()
_sessions = {}
if os.path.exists(GUEST_FILE):
    try:
        with open(GUEST_FILE) as f:
            _sessions = json.load(f)
    except (json.JSONDecodeError, OSError):
        _sessions = {}


def _save_sessions():
    with _lock:
        with open(GUEST_FILE, "w") as f:
            json.dump(_sessions, f)


def _get_session(sid):
    s = _sessions.get(sid)
    if s is None:
        raise HTTPException(404, "Session not found. Create one with POST /api/sessions.")
    s.setdefault("ratings", {})
    s.setdefault("watchlist", [])
    return s


def _check_movie(movie_id):
    if int(movie_id) not in engine.mid_to_idx:
        raise HTTPException(404, "Movie not found")
    return int(movie_id)


app = FastAPI(title="CineMatch AI")


class RateIn(BaseModel):
    movie_id: int
    rating: float


class WatchIn(BaseModel):
    movie_id: int


@app.get("/api/health")
def health():
    return {"status": "ok", "movies": engine.n_movies,
            "ratings": int(engine.R.nnz), "users": len(engine.user_ids),
            "port": PORT}


@app.get("/api/stats")
def stats():
    return {**engine.dataset_stats(), "evaluation": EVALUATION}


@app.get("/api/genres")
def genres():
    return {"genres": GENRES}


@app.get("/api/moods")
def moods():
    return {k: {"label": v["label"], "genres": v["genres"]} for k, v in MOODS.items()}


@app.get("/api/movies")
def list_movies(search: str = "", genre: str = "", year_min: int = 0,
                year_max: int = 0, min_rating: float = 0.0,
                sort: str = "trending", page: int = 1, per_page: int = 24):
    df = engine.movies
    if search:
        q = search.lower()
        df = df[df["title_clean"].str.lower().str.contains(q, regex=False)]
    if genre:
        df = df[df["genre_list"].apply(lambda gs: genre in gs)]
    if year_min:
        df = df[df["year"] >= year_min]
    if year_max:
        df = df[df["year"] <= year_max]
    if min_rating:
        df = df[df["avg_rating"] >= min_rating]
    if sort == "rating":
        df = df[df["rating_count"] >= 20].sort_values("bayes", ascending=False)
    elif sort == "newest":
        df = df.sort_values("year", ascending=False)
    elif sort == "oldest":
        df = df.sort_values("year", ascending=True)
    elif sort == "title":
        df = df.sort_values("title_clean")
    elif sort == "popular":
        df = df.sort_values("rating_count", ascending=False)
    else:  # trending
        df = df.sort_values(["recent_count", "bayes"], ascending=False)
    total = len(df)
    page = max(page, 1)
    per_page = max(1, min(per_page, 60))
    start = (page - 1) * per_page
    items = [engine.movie_dict(int(i)) for i in df.index[start:start + per_page]]
    return {"total": total, "page": page, "per_page": per_page, "items": items}


@app.get("/api/movies/{movie_id}")
def movie_detail(movie_id: int):
    i = engine.mid_to_idx.get(_check_movie(movie_id))
    d = engine.movie_dict(i)
    d["similar"] = engine.similar_movies(movie_id, limit=8)
    return d


@app.get("/api/trending")
def trending(limit: int = 12):
    return engine.trending(limit=max(1, min(limit, 30)))


@app.get("/api/top-rated")
def top_rated(limit: int = 12):
    return engine.top_rated(limit=max(1, min(limit, 30)))


@app.get("/api/onboarding")
def onboarding():
    return [engine.movie_dict(engine.mid_to_idx[mid]) for mid in ONBOARDING_IDS]


@app.get("/api/personas")
def personas():
    return [{"id": p["id"], "name": p["name"], "tagline": p["tagline"],
             "emoji": p["emoji"], "color": p["color"],
             "n_ratings": len(p["ratings"])} for p in PERSONAS]


@app.get("/api/coldstart")
def coldstart(mood: str = "", genre: str = "", limit: int = 12):
    return engine.cold_start(mood=mood or None,
                             genres=[genre] if genre else None,
                             limit=max(1, min(limit, 30)))


@app.post("/api/sessions")
def create_session():
    sid = uuid.uuid4().hex[:16]
    _sessions[sid] = {"ratings": {}, "watchlist": [], "persona": None}
    _save_sessions()
    return {"session_id": sid}


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    s = _get_session(sid)
    return {"session_id": sid, "persona": s.get("persona"),
            "n_ratings": len(s["ratings"]), "n_watchlist": len(s["watchlist"])}


@app.post("/api/sessions/{sid}/rate")
def rate_movie(sid: str, body: RateIn):
    s = _get_session(sid)
    mid = _check_movie(body.movie_id)
    if not (0.5 <= body.rating <= 5.0) or (body.rating * 2) % 1 != 0:
        raise HTTPException(400, "Rating must be 0.5-5.0 in 0.5 steps")
    s["ratings"][str(mid)] = body.rating
    _save_sessions()
    return {"ok": True, "n_ratings": len(s["ratings"])}


@app.delete("/api/sessions/{sid}/rate/{movie_id}")
def unrate_movie(sid: str, movie_id: int):
    s = _get_session(sid)
    s["ratings"].pop(str(_check_movie(movie_id)), None)
    _save_sessions()
    return {"ok": True, "n_ratings": len(s["ratings"])}


@app.get("/api/sessions/{sid}/ratings")
def my_ratings(sid: str):
    s = _get_session(sid)
    out = []
    for mid, r in sorted(s["ratings"].items(), key=lambda kv: -kv[1]):
        i = engine.mid_to_idx.get(int(mid))
        if i is not None:
            d = engine.movie_dict(i)
            d["my_rating"] = r
            out.append(d)
    return out


@app.post("/api/sessions/{sid}/watchlist")
def watch_add(sid: str, body: WatchIn):
    s = _get_session(sid)
    mid = _check_movie(body.movie_id)
    if mid not in s["watchlist"]:
        s["watchlist"].append(mid)
    _save_sessions()
    return {"ok": True, "n_watchlist": len(s["watchlist"])}


@app.delete("/api/sessions/{sid}/watchlist/{movie_id}")
def watch_remove(sid: str, movie_id: int):
    s = _get_session(sid)
    mid = _check_movie(movie_id)
    if mid in s["watchlist"]:
        s["watchlist"].remove(mid)
    _save_sessions()
    return {"ok": True, "n_watchlist": len(s["watchlist"])}


@app.get("/api/sessions/{sid}/watchlist")
def watch_list(sid: str):
    s = _get_session(sid)
    out = []
    for mid in s["watchlist"]:
        i = engine.mid_to_idx.get(int(mid))
        if i is not None:
            out.append(engine.movie_dict(i))
    return out


@app.post("/api/sessions/{sid}/persona/{persona_id}")
def apply_persona(sid: str, persona_id: str):
    s = _get_session(sid)
    p = next((x for x in PERSONAS if x["id"] == persona_id), None)
    if p is None:
        raise HTTPException(404, "Persona not found")
    s["ratings"] = {k: float(v) for k, v in p["ratings"].items()}
    s["persona"] = p["id"]
    _save_sessions()
    return {"ok": True, "persona": p["name"], "n_ratings": len(s["ratings"])}


@app.get("/api/sessions/{sid}/recommendations")
def recommendations(sid: str, limit: int = 12):
    s = _get_session(sid)
    ratings = {int(k): float(v) for k, v in s["ratings"].items()}
    if len(ratings) < 1:
        return {"cold_start": True, "items": engine.cold_start(limit=limit),
                "message": "Rate a few movies (or pick a mood or persona) to unlock personalised picks."}
    recs = engine.recommend(ratings, limit=max(1, min(limit, 30)))
    return {"cold_start": False, "items": recs, "n_ratings": len(ratings)}


@app.get("/api/analytics")
def analytics():
    df = engine.movies
    genre_counts = {}
    genre_avg = {}
    for g in GENRES:
        sub = df[df["genre_list"].apply(lambda gs: g in gs)]
        genre_counts[g] = int(len(sub))
        genre_avg[g] = round(float(sub["avg_rating"].mean()), 2) if len(sub) else 0.0
    import numpy as np
    hist, edges = np.histogram(engine.R.data, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    return {
        "stats": engine.dataset_stats(),
        "evaluation": EVALUATION,
        "genre_counts": genre_counts,
        "genre_avg_rating": genre_avg,
        "rating_histogram": {"labels": ["0.5-1", "1.5-2", "2.5-3", "3.5-4", "4.5-5"],
                             "counts": [int(x) for x in hist]},
        "top_movies": engine.top_rated(10),
        "most_rated": [engine.movie_dict(int(i)) for i in
                       df.sort_values("rating_count", ascending=False).head(10).index],
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")


def _open_browser_when_ready():
    """Open the app once OUR port (8002) answers. Never fall back to another
    port: an old project (NoteWise/SupervisorMatch on 8000, Polyglot on 8001)
    may still be listening there, and opening it would show the wrong app."""
    if os.environ.get("AUTO_OPEN_BROWSER", "1") == "0":
        return
    import socket
    import time
    import webbrowser
    for _ in range(20):  # up to ~10s waiting for port 8002
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
                webbrowser.open(f"http://127.0.0.1:{PORT}")
                return
        except OSError:
            time.sleep(0.5)


@app.on_event("startup")
def _startup():
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
