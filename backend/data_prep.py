"""CineMatch AI — data preparation (run once at build time).

Reads the MovieLens ml-latest-small CSVs in data/, merges the hand-curated
metadata (backend/metadata_curated.py) and the top user tags, and writes:
  data/metadata.json   — director / cast / synopsis / tags for the 130
                         most-rated movies (validated against movies.csv)
  data/personas.json   — 3 demo personas with hand-picked ratings
  data/onboarding.json — 8 popular, diverse movies shown at onboarding
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metadata_curated import CURATED

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

PERSONAS = [
    {
        "id": "ayaan",
        "name": "Ayaan",
        "tagline": "Action & Sci-Fi Fan — big explosions, bigger ideas",
        "emoji": "\U0001f680",
        "color": "#e50914",
        "ratings": {"2571": 5, "589": 5, "79132": 5, "58559": 5, "260": 4.5, "480": 4,
                    "3578": 4.5, "1200": 4.5, "541": 4, "5445": 4, "1036": 4.5, "648": 4,
                    "377": 4, "33794": 4, "4993": 4.5, "6539": 4, "1527": 3.5, "780": 3.5,
                    "1580": 4, "3793": 4, "5349": 4, "5418": 4.5, "6874": 4, "6": 4,
                    "733": 3.5, "150": 4, "110": 4.5, "2028": 4.5, "356": 4, "1721": 3,
                    "597": 2.5, "539": 2, "500": 3, "587": 3},
    },
    {
        "id": "sara",
        "name": "Sara",
        "tagline": "Drama & Classics Lover — stories that stay with you",
        "emoji": "\U0001f3ad",
        "color": "#f5a623",
        "ratings": {"318": 5, "858": 5, "527": 5, "356": 5, "1213": 4.5, "296": 4.5,
                    "2959": 4.5, "2858": 4.5, "1704": 5, "4995": 4.5, "3147": 4.5,
                    "1193": 5, "1221": 5, "2329": 4.5, "48516": 4.5, "912": 4.5,
                    "111": 4, "1208": 4.5, "4973": 4.5, "7361": 4.5, "4226": 4,
                    "110": 4, "590": 4, "1721": 4, "587": 3.5, "539": 3.5, "608": 4,
                    "593": 4, "47": 4, "50": 4.5, "2762": 4},
    },
    {
        "id": "danish",
        "name": "Danish",
        "tagline": "Family & Feel-Good Fan — movies for everyone on the couch",
        "emoji": "\U0001f37f",
        "color": "#2fbf71",
        "ratings": {"1": 5, "6377": 5, "364": 5, "4306": 5, "588": 4.5, "4886": 5,
                    "8961": 4.5, "68954": 5, "60069": 5, "595": 4.5, "4896": 4.5,
                    "5816": 4, "1197": 4.5, "1265": 4.5, "500": 4, "586": 4,
                    "1270": 4.5, "1097": 4.5, "2": 4, "34": 4, "1073": 4, "356": 4.5,
                    "1732": 3.5, "1682": 4, "539": 4, "597": 3.5, "1196": 4, "260": 4,
                    "480": 3.5, "4993": 4},
    },
]

ONBOARDING_IDS = [356, 318, 2571, 1, 58559, 79132, 1721, 296]


def main():
    movies = pd.read_csv(os.path.join(DATA, "movies.csv"))
    valid_ids = set(movies["movieId"].astype(int))
    missing = [mid for mid in CURATED if mid not in valid_ids]
    assert not missing, f"curated ids missing from movies.csv: {missing}"

    # Top user tags per movie (real MovieLens tags, lowercased, top 5).
    tags = pd.read_csv(os.path.join(DATA, "tags.csv"))
    tags["tag"] = tags["tag"].astype(str).str.strip().str.lower()
    tag_counts = (
        tags.groupby(["movieId", "tag"]).size().reset_index(name="n")
        .sort_values(["movieId", "n"], ascending=[True, False])
    )
    top_tags = {}
    for mid, grp in tag_counts.groupby("movieId"):
        top_tags[int(mid)] = [t for t in grp["tag"].head(5) if len(t) > 1]

    metadata = {}
    for mid, info in CURATED.items():
        metadata[str(mid)] = {
            "director": info["director"],
            "cast": info["cast"],
            "synopsis": info["synopsis"],
            "tags": top_tags.get(mid, []),
        }
    with open(os.path.join(DATA, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=1)
    print(f"metadata.json: {len(metadata)} movies")

    for p in PERSONAS:
        bad = [mid for mid in p["ratings"] if int(mid) not in valid_ids]
        assert not bad, f"persona {p['id']} bad ids: {bad}"
    with open(os.path.join(DATA, "personas.json"), "w") as f:
        json.dump(PERSONAS, f, indent=1)
    print(f"personas.json: {len(PERSONAS)} personas, "
          f"{[len(p['ratings']) for p in PERSONAS]} ratings each")

    with open(os.path.join(DATA, "onboarding.json"), "w") as f:
        json.dump(ONBOARDING_IDS, f)
    print(f"onboarding.json: {len(ONBOARDING_IDS)} movies")


if __name__ == "__main__":
    main()
