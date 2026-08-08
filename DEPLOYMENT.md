# Deploying Hokm as a Telegram Mini App

This repo ships a single production web service (`server.py`) that serves
the Mini App UI, the per-user game API, and the bot webhook. One container,
no separate frontend host, no polling process.

```
Telegram client ──▶ Mini App UI  (GET /)
                └─▶ Game API     (/api/*, auth = signed initData)
Telegram servers ─▶ Bot webhook  (/telegram/webhook/<secret>)
```

The legacy pieces are unchanged: `app.py` is still the local dev UI +
training console, and `hokm-mini-app/` is an earlier React prototype that
the Flask-served UI supersedes.

## 1. Create the bot

1. Open [@BotFather](https://t.me/BotFather) → `/newbot` → pick a name and
   username. Save the **bot token**.
2. Nothing else is needed in BotFather — the menu button and commands are
   set by `scripts/set_webhook.py` below.

## 2. Deploy the service

Any Docker host works. The service needs HTTPS with a public URL
(a Telegram requirement for Mini Apps).

### Option A — Render (one click-ish)

1. Push this repo to GitHub and create a new **Blueprint** on
   [render.com](https://render.com) pointing at it — `render.yaml` defines
   the service.
2. In the dashboard set `BOT_TOKEN` (from BotFather). After the first
   deploy, set `WEBAPP_URL` to the service URL Render assigned
   (e.g. `https://hokm-mini-app.onrender.com`) and let it redeploy.

### Option B — any Docker host (Fly.io, Railway, a VPS…)

```bash
docker build -t hokm .
docker run -d -p 8080:8080 \
  -e BOT_TOKEN=123456:ABC... \
  -e WEBAPP_URL=https://hokm.example.com \
  -e WEBHOOK_SECRET="$(openssl rand -hex 24)" \
  -e ALLOW_GUESTS=0 \
  hokm
```

Put it behind HTTPS (a platform-provided cert or a reverse proxy).

### Environment variables

| Variable         | Required | Purpose |
|------------------|----------|---------|
| `BOT_TOKEN`      | yes      | Verifies Mini App `initData` (auth) and sends bot replies. |
| `WEBAPP_URL`     | yes      | Public HTTPS URL of this service; used in the /start Play button. |
| `WEBHOOK_SECRET` | yes      | Random string; path + header secret for the webhook route. |
| `ALLOW_GUESTS`   | no       | `1` allows unauthenticated browser play (dev). Defaults to on **only** when `BOT_TOKEN` is unset. Set `0` in production. |
| `AI_KIND`        | no       | Opponent type: `pimc` (default — determinized Monte-Carlo search, the strongest), `heuristic` (rule-based), or `checkpoint` (greedy NFSP net from `MODEL_PATH`). |
| `PIMC_DETERMINIZATIONS` | no | Search width for `pimc` (default 32). Higher = stronger + slower; 32 costs a few ms per AI decision. |
| `MODEL_PATH`     | no       | Path to an NFSP checkpoint (`.pth`) for `AI_KIND=checkpoint`. Auto-detects the newest `.pth` in `models_release/` when unset. |
| `MATCH_TARGET`   | no       | Hands a team must win to take the match (default 7). A Kot (7-0 hand) counts 2. |
| `SESSION_TTL_SECONDS` | no  | Idle session eviction (default 7200). |
| `MAX_SESSIONS`   | no       | Concurrent user cap (default 500). |
| `DATABASE_URL`   | no       | Postgres connection string. Set it and every finished hand plus every player who launches the app is stored off-box — **no persistent disk needed**. Unset = nothing is persisted across redeploys. |
| `GAME_DATA_DIR`  | no       | Local JSONL fallback for recorded hands (default `game_data`). Needs a persistent disk; off by default when `DATABASE_URL` is set. |
| `ADMIN_TOKEN`    | no       | Enables `/api/admin/stats`, `/api/admin/players` and `/api/admin/export?token=…`. Unset = endpoints return 404. |
| `GAME_RECORDING` | no       | Forces the local JSONL sink on (`1`) or off (`0`). |

## Training data and the player log

Every finished hand is stored as one JSON document: the full deal, who was
Hakem, the trump choice, the exact play sequence, and the outcome —
everything needed to reconstruct each decision for imitation learning or RL
(see `game_recorder.replay_hand`).

There are two sinks and each is a no-op when unconfigured, so a deployment
can use either or both:

* **Postgres (`DATABASE_URL`)** — the disk-free option, and the recommended
  one on Render. Works with any managed Postgres: Neon, Supabase, Render
  Postgres. See "Setting up Postgres" below.
* **Local JSONL (`GAME_DATA_DIR`)** — the original behaviour. The container
  filesystem is ephemeral, so this needs a Render Disk mounted at `/data`
  with `GAME_DATA_DIR=/data/game_data`. The server logs an ERROR at startup
  if the directory is not writable.

### Setting up Postgres (no disk required)

1. Create a free database at [neon.tech](https://neon.tech) (or Supabase).
2. Copy its connection string — it looks like
   `postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname?sslmode=require`.
3. In the Render dashboard set `DATABASE_URL` to that string and redeploy.

The two tables are created automatically on first use:

| Table | One row per | Key columns |
|-------|-------------|-------------|
| `players` | user who has launched the app | `user_id`, `username` (the Telegram @handle, when they have one), `first_name`, `first_seen`, `last_seen`, `launches`, `hands_played` |
| `hands`   | finished hand | `ts`, `user_id`, `record` (JSONB — the full hand document) |

Writes go through a background thread with a bounded queue, so a sleeping
serverless database delays the write, never a player's move. If the database
is unreachable the hand is dropped and counted; play is unaffected. The
startup log line tells you which state you are in.

### Getting the data out

```bash
# every recorded hand, as JSONL (reads Postgres when configured)
curl -o hands.jsonl "https://<your-app>/api/admin/export?token=$ADMIN_TOKEN"

# who has played, newest activity first (handles, launch counts, hands played)
curl "https://<your-app>/api/admin/players?token=$ADMIN_TOKEN"

# totals and write-health counters for both sinks
curl "https://<your-app>/api/admin/stats?token=$ADMIN_TOKEN"
```

Set `ADMIN_TOKEN` in the environment to enable these; keep it secret. You can
also query the database directly — `SELECT count(*) FROM players;` answers
"how many people have played" without going through the app.

**On collecting handles:** `username` is only present for users who have set
one on Telegram, and it is the handle as it was at the time of their last
launch — people rename themselves. Note also that a bot may only message
users who have started it, so `/start` users are the ones you can actually
reach later.

## 3. Wire Telegram to the deployment

From any machine:

```bash
BOT_TOKEN=123456:ABC... \
WEBAPP_URL=https://hokm.example.com \
WEBHOOK_SECRET=<same value as the server env> \
python scripts/set_webhook.py
```

This sets the webhook, makes the bot's **menu button** open the Mini App,
and registers `/start`. Open your bot in Telegram, tap **▶️ Play Hokm**,
and play.

## 4. Shipping a trained AI (optional but recommended)

Checkpoints are gitignored, so the image plays with the heuristic AI by
default. To ship a trained opponent:

1. Train locally: `python train_hokm.py --num-games 20000 --seed 42`
   (see README for evaluation against baselines).
2. Copy the chosen checkpoint into the build context, uncomment the
   `COPY models/...` line in the `Dockerfile`, and set
   `MODEL_PATH=models/checkpoint.pth` in the service environment.

## Local smoke test (no Telegram needed)

```bash
pip install -r requirements-prod.txt
python server.py
# open http://localhost:8080 — guest mode is on because BOT_TOKEN is unset
```

## Scaling notes / current limits

* **Single worker.** Sessions are in-process (`SessionStore`), so gunicorn
  must run `--workers 1` (threads provide concurrency). One CPU-bound
  worker comfortably handles hundreds of casual sessions; past that, move
  `SessionStore` to Redis (serialize the engine state, or persist per-move
  events) before raising worker counts or replicas.
* **Single hand per game.** A game ends when a team takes 7 tricks —
  match play (best-of series, rotating Hakem between hands) is engine-
  supported but not yet exposed in the API.
* **Security posture.** initData is HMAC-verified server-side with a 24 h
  freshness window; the webhook is protected by a path secret and
  Telegram's `X-Telegram-Bot-Api-Secret-Token` header. Game state never
  trusts the client — legal moves are validated by the engine.
