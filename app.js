/* CineMatch AI — SPA frontend (no login; guest session in localStorage). */
const GENRE_STYLE = {
  "Action": ["linear-gradient(135deg,#ff416c,#ff4b2b)", "🔥"],
  "Adventure": ["linear-gradient(135deg,#11998e,#38ef7d)", "🧭"],
  "Animation": ["linear-gradient(135deg,#8e2de2,#4a00e0)", "🎨"],
  "Children": ["linear-gradient(135deg,#f7971e,#ffd200)", "🧸"],
  "Comedy": ["linear-gradient(135deg,#f7971e,#f5a623)", "😂"],
  "Crime": ["linear-gradient(135deg,#2c3e50,#4ca1af)", "🕵️"],
  "Documentary": ["linear-gradient(135deg,#2193b0,#6dd5ed)", "🎥"],
  "Drama": ["linear-gradient(135deg,#41295a,#2f0743)", "🎭"],
  "Fantasy": ["linear-gradient(135deg,#7b2ff7,#f107a3)", "🧚"],
  "Film-Noir": ["linear-gradient(135deg,#000000,#434343)", "🌃"],
  "Horror": ["linear-gradient(135deg,#0f9b0f,#000000)", "👻"],
  "Musical": ["linear-gradient(135deg,#ec008c,#fc6767)", "🎵"],
  "Mystery": ["linear-gradient(135deg,#1f4037,#99f2c8)", "🔍"],
  "Romance": ["linear-gradient(135deg,#ff0844,#ffb199)", "❤️"],
  "Sci-Fi": ["linear-gradient(135deg,#00c6ff,#0072ff)", "🚀"],
  "Thriller": ["linear-gradient(135deg,#8e0e00,#1f1c18)", "😱"],
  "War": ["linear-gradient(135deg,#757f9a,#d7dde8)", "⚔️"],
  "Western": ["linear-gradient(135deg,#a73737,#7a2828)", "🤠"],
};
const DEFAULT_STYLE = ["linear-gradient(135deg,#33334a,#55556a)", "🎬"];

const state = {
  sessionId: null, ratings: {}, watchlist: [],
  page: 1, perPage: 24, browseTotal: 0,
  moods: {}, genres: [], currentMood: null,
  onboardingRated: 0,
};

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}
function $(id) { return document.getElementById(id); }
function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
  setTimeout(() => t.classList.add("hidden"), 2600);
}
async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...(opts || {}),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || ("HTTP " + res.status));
  }
  return res.json();
}
function posterStyle(movie) {
  const g = (movie.genres && movie.genres[0]) || "";
  return GENRE_STYLE[g] || DEFAULT_STYLE;
}
function starsHtml(n, max = 5) {
  let out = "";
  for (let i = 1; i <= max; i++) out += i <= Math.round(n) ? "★" : "☆";
  return out;
}

/* ---------- session ---------- */
async function ensureSession() {
  let sid = localStorage.getItem("cm_session");
  if (sid) {
    try { await api(`/api/sessions/${sid}`); state.sessionId = sid; }
    catch (e) { sid = null; }
  }
  if (!sid) {
    const r = await api("/api/sessions", { method: "POST" });
    state.sessionId = r.session_id;
    localStorage.setItem("cm_session", state.sessionId);
  }
  await refreshMyData();
}
async function refreshMyData() {
  const [ratings, watch] = await Promise.all([
    api(`/api/sessions/${state.sessionId}/ratings`),
    api(`/api/sessions/${state.sessionId}/watchlist`),
  ]);
  state.ratings = {};
  ratings.forEach(r => { state.ratings[r.id] = r.my_rating; });
  state.watchlist = watch.map(m => m.id);
  $("ratingsPill").textContent = Object.keys(state.ratings).length;
  $("watchPill").textContent = state.watchlist.length;
}

/* ---------- cards ---------- */
function cardHtml(m) {
  const [grad, emoji] = posterStyle(m);
  const inWatch = state.watchlist.includes(m.id);
  const myR = state.ratings[m.id];
  const because = m.because
    ? `<div class="because">Because you liked <b>${esc(m.because)}</b>${m.because_genres && m.because_genres.length ? ` · ${esc(m.because_genres.join(", "))}` : ""}</div>`
    : "";
  return `<article class="card" data-id="${m.id}">
    <div class="poster" style="background:${grad}">
      ${inWatch ? '<span class="watch-flag">🔖</span>' : ""}
      ${m.match ? `<span class="match-badge">${m.match}% match</span>` : ""}
      <span class="p-emoji">${emoji}</span>
      <span class="p-genre">${esc((m.genres && m.genres[0]) || "Movie")}</span>
    </div>
    <div class="card-body">
      <div class="card-title">${esc(m.title)}</div>
      <div class="card-sub">${m.year || "—"} · ${esc((m.genres || []).slice(0, 2).join(", "))}</div>
      <div class="card-meta">
        <span class="stars-inline">★ ${m.avg_rating} <span style="color:var(--muted)">(${m.rating_count})</span></span>
        ${myR ? `<span class="stars-inline">Your: ${starsHtml(myR)}</span>` : ""}
      </div>
      ${because}
    </div>
  </article>`;
}
function bindCards(root) {
  root.querySelectorAll(".card").forEach(c =>
    c.addEventListener("click", () => openMovie(+c.dataset.id)));
}
function renderInto(el, movies) {
  el.innerHTML = movies.map(cardHtml).join("");
  bindCards(el);
}

/* ---------- discover ---------- */
async function loadDiscover() {
  const [tr, top] = await Promise.all([api("/api/trending?limit=14"), api("/api/top-rated?limit=14")]);
  renderInto($("trendingRail"), tr);
  renderInto($("topRail"), top);
  await loadBrowse();
}
async function loadBrowse() {
  const q = new URLSearchParams({
    search: $("searchInput").value.trim(),
    genre: $("genreFilter").value,
    year_min: $("yearMin").value || 0,
    year_max: $("yearMax").value || 0,
    min_rating: $("minRatingFilter").value,
    sort: $("sortFilter").value,
    page: state.page, per_page: state.perPage,
  });
  const r = await api(`/api/movies?${q}`);
  state.browseTotal = r.total;
  renderInto($("browseGrid"), r.items);
  const pages = Math.max(1, Math.ceil(r.total / state.perPage));
  $("pageInfo").textContent = `Page ${r.page} of ${pages} · ${r.total.toLocaleString()} movies`;
  $("prevPage").disabled = r.page <= 1;
  $("nextPage").disabled = r.page >= pages;
}
async function loadMoods() {
  const [moods, genres] = await Promise.all([api("/api/moods"), api("/api/genres")]);
  state.moods = moods;
  state.genres = genres.genres;
  const chips = Object.entries(moods).map(([k, v]) =>
    `<button class="mood-chip" data-mood="${k}">🎭 ${esc(v.label)}</button>`).join("");
  $("moodRow").innerHTML = chips;
  $("foryouMoods").innerHTML = chips;
  document.querySelectorAll(".mood-chip").forEach(ch =>
    ch.addEventListener("click", () => applyMood(ch.dataset.mood)));
  const gf = $("genreFilter");
  genres.genres.forEach(g => {
    const o = document.createElement("option"); o.value = g; o.textContent = g; gf.appendChild(o);
  });
}
async function applyMood(moodKey) {
  const items = await api(`/api/coldstart?mood=${moodKey}&limit=24`);
  state.currentMood = moodKey;
  renderInto($("browseGrid"), items);
  $("pageInfo").textContent = `Mood: ${esc(state.moods[moodKey].label)} · top picks by Bayesian average rating`;
  showTab("discover");
  document.getElementById("browseGrid").scrollIntoView({ behavior: "smooth", block: "start" });
  toast(`Mood: ${state.moods[moodKey].label} — no ratings needed`);
}

/* ---------- for you ---------- */
async function loadForYou() {
  const r = await api(`/api/sessions/${state.sessionId}/recommendations?limit=12`);
  if (r.cold_start || r.items.length === 0) {
    $("foryouEmpty").classList.remove("hidden");
    $("foryouContent").classList.add("hidden");
    const cold = await api("/api/coldstart?mood=feel-good&limit=12");
    renderInto($("coldGrid"), cold);
    renderPersonas($("foryouPersonas"));
    return;
  }
  $("foryouEmpty").classList.add("hidden");
  $("foryouContent").classList.remove("hidden");
  renderInto($("recGrid"), r.items);
  // Taste profile: top genres among liked movies.
  const liked = Object.entries(state.ratings).filter(([, v]) => v >= 4).map(([k]) => +k);
  const counts = {};
  const all = await api(`/api/sessions/${state.sessionId}/ratings`);
  all.forEach(m => { if (m.my_rating >= 4) (m.genres || []).forEach(g => counts[g] = (counts[g] || 0) + 1); });
  const topGenres = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 4);
  $("tasteBar").innerHTML = `<span>🎬 Your taste: </span>` +
    (topGenres.map(([g, n]) => `<span class="tag">${esc(g)} ×${n}</span>`).join("") ||
      `<span class="muted">rate more movies to sharpen your profile</span>`) +
    `<span class="muted" style="margin-left:auto">${r.n_ratings} ratings · hybrid model</span>`;
}
function renderPersonas(root) {
  api("/api/personas").then(list => {
    root.innerHTML = list.map(p =>
      `<button class="persona-card" data-persona="${p.id}">
        <div class="p-emoji">${p.emoji}</div>
        <div class="p-name">${esc(p.name)}</div>
        <div class="p-tag">${esc(p.tagline)}</div>
        <div class="p-n">${p.n_ratings} ratings ready</div>
      </button>`).join("");
    root.querySelectorAll(".persona-card").forEach(c =>
      c.addEventListener("click", () => applyPersona(c.dataset.persona)));
  });
}
async function applyPersona(pid) {
  const r = await api(`/api/sessions/${state.sessionId}/persona/${pid}`, { method: "POST" });
  await refreshMyData();
  localStorage.setItem("cm_onboarded", "1");
  closeModal("onboardModal");
  toast(`${r.persona}'s taste loaded — ${r.n_ratings} ratings`);
  showTab("foryou");
  await loadForYou();
}

/* ---------- ratings & watchlist tabs ---------- */
async function loadRatingsTab() {
  const items = await api(`/api/sessions/${state.sessionId}/ratings`);
  $("ratingsEmpty").classList.toggle("hidden", items.length > 0);
  renderInto($("ratingsGrid"), items);
}
async function loadWatchTab() {
  const items = await api(`/api/sessions/${state.sessionId}/watchlist`);
  $("watchEmpty").classList.toggle("hidden", items.length > 0);
  renderInto($("watchGrid"), items);
}

/* ---------- analytics ---------- */
let chartsDrawn = false;
async function loadAnalytics() {
  const a = await api("/api/analytics");
  const s = a.stats;
  $("statCards").innerHTML = [
    [s.movies.toLocaleString(), "movies in the catalogue"],
    [s.ratings.toLocaleString(), "ratings from real users"],
    [s.users, "MovieLens users"],
    ["★ " + s.global_mean_rating, "average rating overall"],
  ].map(([n, l]) => `<div class="stat-card"><div class="num">${n}</div><div class="lbl">${l}</div></div>`).join("");
  $("evalMethod").textContent = a.evaluation.method || "";
  const m = a.evaluation.models || {};
  const rows = [["Content-based (TF-IDF)", m.content], ["Collaborative (item-item CF)", m.cf],
                ["Hybrid (45/45/10 blend)", m.hybrid], ["Popularity baseline", m.popularity]];
  $("metricsTable").innerHTML = `<tr><th>Model</th><th>Precision@10</th><th>Recall@10</th><th>RMSE</th></tr>` +
    rows.map(([name, v]) => v
      ? `<tr><td>${name}</td><td>${v.precision_at_10}</td><td>${v.recall_at_10}</td><td>${v.rmse ?? "—"}</td></tr>`
      : "").join("");
  $("topList").innerHTML = a.top_movies.map(mv =>
    `<li><b>${esc(mv.title)}</b> (${mv.year}) — ★ ${mv.avg_rating} from ${mv.rating_count.toLocaleString()} ratings</li>`).join("");
  if (!chartsDrawn) {
    chartsDrawn = true;
    try {
      const gc = $("genreChart").getContext("2d");
      new Chart(gc, { type: "bar",
        data: { labels: Object.keys(a.genre_counts),
          datasets: [{ data: Object.values(a.genre_counts), backgroundColor: "#e50914" }] },
        options: { plugins: { legend: { display: false } },
          scales: { x: { ticks: { color: "#9a9ab0" } }, y: { ticks: { color: "#9a9ab0" } } } } });
      const rc = $("ratingChart").getContext("2d");
      new Chart(rc, { type: "bar",
        data: { labels: a.rating_histogram.labels,
          datasets: [{ data: a.rating_histogram.counts, backgroundColor: "#f5a623" }] },
        options: { plugins: { legend: { display: false } },
          scales: { x: { ticks: { color: "#9a9ab0" } }, y: { ticks: { color: "#9a9ab0" } } } } });
    } catch (e) { /* canvas unavailable (e.g. headless test) — table still renders */ }
  }
}

/* ---------- movie modal ---------- */
let modalMovie = null;
async function openMovie(id) {
  const m = await api(`/api/movies/${id}`);
  modalMovie = m;
  const [grad, emoji] = posterStyle(m);
  $("modalPoster").style.background = grad;
  $("modalPoster").innerHTML = `<span class="p-emoji">${emoji}</span>`;
  $("modalTitle").textContent = `${m.title}${m.year ? ` (${m.year})` : ""}`;
  $("modalMeta").textContent = `${(m.genres || []).join(" · ")}   ★ ${m.avg_rating} (${m.rating_count.toLocaleString()} ratings)`;
  $("modalSynopsis").textContent = m.synopsis || "No synopsis available for this title — the catalogue lists its genres and community ratings.";
  $("modalFacts").innerHTML =
    (m.director ? `<div><b>Director:</b> ${esc(m.director)}</div>` : "") +
    (m.cast && m.cast.length ? `<div><b>Cast:</b> ${esc(m.cast.join(", "))}</div>` : "") +
    (m.tags && m.tags.length ? `<div><b>Popular tags:</b> ${esc(m.tags.join(", "))}</div>` : "");
  renderStars($("modalStars"), state.ratings[m.id] || 0, async v => { await rateMovie(m.id, v); });
  updateWatchBtn();
  renderInto($("modalSimilar"), m.similar || []);
  $("movieModal").classList.remove("hidden");
}
function updateWatchBtn() {
  const b = $("modalWatch");
  const inW = state.watchlist.includes(modalMovie.id);
  b.textContent = inW ? "🔖 In your watchlist — tap to remove" : "🔖 Add to watchlist";
}
function renderStars(el, current, onPick) {
  el.innerHTML = "";
  for (let i = 1; i <= 5; i++) {
    const s = document.createElement("span");
    s.textContent = i <= current ? "★" : "☆";
    s.className = i <= current ? "on" : "";
    s.addEventListener("click", () => onPick(i));
    s.addEventListener("mouseenter", () => {
      [...el.children].forEach((c, j) => { c.textContent = j < i ? "★" : "☆"; c.className = j < i ? "on" : ""; });
    });
    s.addEventListener("mouseleave", () => renderStars(el, state.ratings[modalMovie ? modalMovie.id : 0] || current, onPick));
    el.appendChild(s);
  }
}
async function rateMovie(id, v) {
  await api(`/api/sessions/${state.sessionId}/rate`, {
    method: "POST", body: JSON.stringify({ movie_id: id, rating: v }),
  });
  state.ratings[id] = v;
  $("ratingsPill").textContent = Object.keys(state.ratings).length;
  toast(`Rated ★${v}`);
  if (!$("movieModal").classList.contains("hidden") && modalMovie && modalMovie.id === id)
    renderStars($("modalStars"), v, vv => rateMovie(id, vv));
  if (!$("onboardModal").classList.contains("hidden")) updateOnboardProgress();
}
async function toggleWatch() {
  const id = modalMovie.id;
  if (state.watchlist.includes(id)) {
    await api(`/api/sessions/${state.sessionId}/watchlist/${id}`, { method: "DELETE" });
    state.watchlist = state.watchlist.filter(x => x !== id);
    toast("Removed from watchlist");
  } else {
    await api(`/api/sessions/${state.sessionId}/watchlist`, {
      method: "POST", body: JSON.stringify({ movie_id: id }),
    });
    state.watchlist.push(id);
    toast("Added to watchlist 🔖");
  }
  $("watchPill").textContent = state.watchlist.length;
  updateWatchBtn();
}

/* ---------- onboarding ---------- */
function updateOnboardProgress() {
  const n = Object.keys(state.ratings).length;
  $("onboardCount").textContent = `${Math.min(n, 5)} / 5 rated`;
  $("onboardBar").style.width = `${Math.min(n, 5) / 5 * 100}%`;
  $("onboardDone").disabled = n < 5;
}
async function openOnboarding() {
  const items = await api("/api/onboarding");
  const list = $("onboardList");
  list.innerHTML = "";
  items.forEach(m => {
    const [grad, emoji] = posterStyle(m);
    const row = document.createElement("div");
    row.className = "onboard-item";
    row.innerHTML = `<div class="mini-poster" style="background:${grad}">${emoji}</div>
      <div><div class="o-title">${esc(m.title)}</div>
      <div class="o-sub">${m.year || ""} · ${esc((m.genres || []).slice(0, 2).join(", "))} · ★ ${m.avg_rating}</div></div>`;
    const stars = document.createElement("div");
    stars.className = "stars";
    const paint = () => {
      const cur = state.ratings[m.id] || 0;
      stars.innerHTML = "";
      for (let i = 1; i <= 5; i++) {
        const s = document.createElement("span");
        s.textContent = i <= cur ? "★" : "☆";
        s.className = i <= cur ? "on" : "";
        s.addEventListener("click", async () => { await rateMovie(m.id, i); paint(); });
        stars.appendChild(s);
      }
    };
    paint();
    row.appendChild(stars);
    list.appendChild(row);
  });
  renderPersonas($("onboardPersonas"));
  updateOnboardProgress();
  $("onboardModal").classList.remove("hidden");
}
function closeModal(id) { $(id).classList.add("hidden"); }

/* ---------- tabs ---------- */
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-page").forEach(p =>
    p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "foryou") loadForYou();
  if (name === "ratings") loadRatingsTab();
  if (name === "watchlist") loadWatchTab();
  if (name === "analytics") loadAnalytics();
}

document.addEventListener("DOMContentLoaded", async () => {
  document.querySelectorAll("#tabs button").forEach(b =>
    b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("searchBtn").addEventListener("click", () => { state.page = 1; loadBrowse(); });
  $("searchInput").addEventListener("keydown", e => { if (e.key === "Enter") { state.page = 1; loadBrowse(); } });
  $("applyFilters").addEventListener("click", () => { state.page = 1; loadBrowse(); });
  $("clearFilters").addEventListener("click", () => {
    $("searchInput").value = ""; $("genreFilter").value = ""; $("yearMin").value = "";
    $("yearMax").value = ""; $("minRatingFilter").value = "0"; $("sortFilter").value = "trending";
    state.page = 1; state.currentMood = null; loadBrowse();
  });
  $("prevPage").addEventListener("click", () => { if (state.page > 1) { state.page--; loadBrowse(); } });
  $("nextPage").addEventListener("click", () => { state.page++; loadBrowse(); });
  $("onboardBtn").addEventListener("click", openOnboarding);
  $("emptyOnboard").addEventListener("click", openOnboarding);
  $("emptyMoods").addEventListener("click", () => { showTab("discover"); document.getElementById("moodRow").scrollIntoView({ behavior: "smooth" }); });
  $("modalClose").addEventListener("click", () => { closeModal("movieModal"); loadBrowse(); loadForYou(); });
  $("movieModal").addEventListener("click", e => { if (e.target === $("movieModal")) { closeModal("movieModal"); loadBrowse(); } });
  $("onboardClose").addEventListener("click", () => closeModal("onboardModal"));
  $("onboardDone").addEventListener("click", () => {
    localStorage.setItem("cm_onboarded", "1");
    closeModal("onboardModal");
    showTab("foryou");
  });
  $("modalUnrate").addEventListener("click", async () => {
    await api(`/api/sessions/${state.sessionId}/rate/${modalMovie.id}`, { method: "DELETE" });
    delete state.ratings[modalMovie.id];
    $("ratingsPill").textContent = Object.keys(state.ratings).length;
    renderStars($("modalStars"), 0, v => rateMovie(modalMovie.id, v));
    toast("Rating removed");
  });
  $("modalWatch").addEventListener("click", toggleWatch);

  await ensureSession();
  await loadMoods();
  await loadDiscover();
  if (!localStorage.getItem("cm_onboarded") && Object.keys(state.ratings).length < 5) {
    setTimeout(openOnboarding, 600);
  }
});
