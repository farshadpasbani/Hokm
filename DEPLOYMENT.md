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
training console. An earlier React prototype, `hokm-mini-app/`, has been
removed — the Flask-served UI superseded it.

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
| `GAME_DATA_DIR`  | no       | Where finished hands are recorded as JSONL training data (default `game_data`). Point at a persistent disk mount in production. |
| `ADMIN_TOKEN`    | no       | Enables `/api/admin/stats` and `/api/admin/export?token=…` to inspect/download recorded games. Unset = endpoints return 404. |
| `GAME_RECORDING` | no       | `0` disables hand recording (default on). |
| `BOT_USERNAME`   | no       | Your bot's username, without the `@`. Builds the `t.me` link that a table shares. Unset = no link; the 6-character join code still works. |
| `MINI_APP_SHORT_NAME` | no  | Mini App short name from BotFather. Set it and the share link becomes `t.me/<bot>/<name>?startapp=<code>`, which opens the Mini App directly on that table. Unset and the link becomes `t.me/<bot>?start=<code>`, which sends the code to the bot, and the bot answers with a button. |
| `TABLE_IDLE_SECONDS` | no   | Silence from a seated player before the AI covers that seat (default 45). The player reclaims the seat on their next request. |
| `TABLE_POLL_TIMEOUT_SECONDS` | no | How long `GET /api/table/state?since=…` parks before answering unchanged (default 3). |
| `TABLE_TTL_SECONDS` | no    | Idle table eviction (default 7200). |
| `MAX_TABLES`     | no       | Concurrent table cap (default 200). |
| `DIRECTORY_TTL_SECONDS` | no | How long a learned `@handle` stays usable for invites (default 30 days). |
| `MAX_DIRECTORY_ENTRIES` | no | Cap on the `@handle` directory (default 5000). The least recently seen entries are dropped first. |

## Playing with friends (shared tables)

Two to four people can play one match together. The AI fills every seat
nobody takes.

* A player creates a table and gets a 6-character **join code** and a share
  link. A friend joins with either.
* Seats 1 and 3 are one team, seats 2 and 4 the other. The lobby lets a
  player move to any free seat before the deal, so two friends can choose to
  be partners.
* A player can invite an `@handle`. Telegram publishes **no API that turns a
  username into a user id**, so the service keeps its own directory: it
  records `@handle → user id` for every request whose Telegram `initData`
  carries a username. An invited handle the service has never seen gets no
  DM — the response says so and returns the share link instead. It never
  reports a message it did not send.
* Tables are in-memory. A redeploy ends every table in flight. This is
  deliberate; there is no table persistence layer.

### Manual check of bot DM delivery

Automated tests cover the invite logic against a faked Bot API. Delivery
itself needs a live token, so run this check once after you set a real
`BOT_TOKEN` and `WEBAPP_URL`:

1. Open the Mini App as user **B** and let it load. This is what puts B's
   `@handle` in the directory — an invite cannot find a handle before its
   owner has opened the app at least once.
2. Open the Mini App as user **A**, tap **Play with friends**, then
   **Create a table**.
3. Type B's `@handle` in the invite box and tap **Invite**.
4. Expected: A sees `Invite sent to @<handle>.`, and B receives a bot DM
   with a **▶️ Join the table** button. Tapping it opens the Mini App on
   that table.
5. Repeat with a handle that has never opened the app. Expected: A sees
   `@<handle> has not opened Hokm yet — send them this link.` and **no DM is
   sent**.
6. Repeat with a real handle whose owner has opened the Mini App but has
   never started a chat with the bot. Expected: A sees `Could not message
   @<handle> …`, because Telegram refuses a DM into a chat that does not
   exist. The share link is the fallback in both failure cases.

## Training data from real games

Every finished hand is appended to `GAME_DATA_DIR` as one JSON line: the
full deal, who was Hakem, the trump choice, the exact play sequence, and
the outcome — everything needed to reconstruct each decision for
imitation learning or RL (see `game_recorder.replay_hand`).

Two operational notes:

1. **The container filesystem is ephemeral** — without a disk, recordings
   vanish on every redeploy. `render.yaml` provisions a 1 GB persistent
   disk at `/data` with `GAME_DATA_DIR=/data/game_data`. If your service
   was created manually (not from the blueprint), add a Disk in the Render
   dashboard (mount path `/data`) and set `GAME_DATA_DIR=/data/game_data`
   yourself. The server logs an ERROR at startup if the directory is not
   writable.
2. **Download your data** any time:
   `curl -o hands.jsonl "https://<your-app>/api/admin/export?token=$ADMIN_TOKEN"`
   (`/api/admin/stats?token=…` shows counts). Set `ADMIN_TOKEN` in the
   environment to enable these; keep it secret.

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

* **Single worker.** Sessions and tables are in-process (`SessionStore`,
  `TableStore`, `UserDirectory`), so gunicorn must run `--workers 1`
  (threads provide concurrency). One CPU-bound worker comfortably handles
  hundreds of casual sessions; past that, move that state to Redis
  (serialize the engine state, or persist per-move events) before raising
  worker counts or replicas.
* **Polling, not streaming.** Table clients poll
  `GET /api/table/state?since=<version>`. The worker runs eight threads, so
  a held-open stream would pin one thread per connected player and starve
  every other request. Do not swap this for SSE or WebSockets without
  changing the worker model first.
* **Single hand per game.** A game ends when a team takes 7 tricks —
  match play (best-of series, rotating Hakem between hands) is engine-
  supported but not yet exposed in the API.
* **Security posture.** initData is HMAC-verified server-side with a 24 h
  freshness window; the webhook is protected by a path secret and
  Telegram's `X-Telegram-Bot-Api-Secret-Token` header. Game state never
  trusts the client — legal moves are validated by the engine.
