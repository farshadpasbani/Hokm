# Hokm on the Pi — deployment runbook

The production Mini App runs on the Raspberry Pi (ssh alias `pi`) as a Docker
container behind the machine's Cloudflare tunnel. This file records the actual
arrangement and the update procedure. There is no deploy script: Docker with
`--restart unless-stopped` is the process supervisor.

## Topology

| Piece | Value |
|---|---|
| Container | `hokm` (image `hokm:latest`), gunicorn on internal `:8080` |
| Port map | `127.0.0.1:8110 -> 8080` |
| Public hostname | `https://hokm.feristal.com` (entry in `/etc/cloudflared/config.yml`) |
| Build checkout | `/home/raspberry/hokm` (this repo, branch `main`) |
| Game data | host `/home/raspberry/hokm-data` -> container `/data`; `GAME_DATA_DIR=/data/game_data` |
| Bot | `@Hakem_kot_bot`; webhook + menu button set by `scripts/set_webhook.py` |
| Sibling | `hokm-mp` on `127.0.0.1:8111` (multiplayer branch) — not covered here |

Public reachability follows the Pi's tunnel switch (`pi-exposure on|off`).
Secrets (`BOT_TOKEN`, `WEBHOOK_SECRET`, `ADMIN_TOKEN`) live only in the
container environment. Recover them from the running container; never commit
them:

    docker inspect hokm --format '{{range .Config.Env}}{{println .}}{{end}}'

## Update procedure

1. Pull the code and keep a rollback image:

       cd ~/hokm
       git pull --ff-only origin main
       docker tag hokm:latest hokm:prev
       docker build -t hokm:latest .

2. Recreate the container with the same environment. Read the current values
   first (command above), then:

       docker stop hokm && docker rm hokm
       docker run -d --name hokm --restart unless-stopped \
         -p 127.0.0.1:8110:8080 \
         -v /home/raspberry/hokm-data:/data \
         -e GAME_DATA_DIR=/data/game_data \
         -e AI_KIND=pimc -e ALLOW_GUESTS=0 \
         -e WEBAPP_URL=https://hokm.feristal.com \
         -e BOT_TOKEN=... -e BOT_USERNAME=Hakem_kot_bot \
         -e WEBHOOK_SECRET=... -e ADMIN_TOKEN=... \
         hokm:latest

3. Verify before you hand it to anyone:

       curl -fsS https://hokm.feristal.com/healthz
       docker logs hokm 2>&1 | grep -i "recording"   # must say active, not unwritable

4. Webhook (only needed when the token, secret, or hostname changed — it
   survives container recreation):

       docker exec hokm python scripts/set_webhook.py

## Rollback

    docker stop hokm && docker rm hokm
    # re-run the `docker run` above with hokm:prev

## Pulling training data

With `ADMIN_TOKEN` set, recorded hands (including `flagged_tricks` and
amendment lines — see `game_recorder.py` for the format) stream from:

    curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
      https://hokm.feristal.com/api/admin/export > hands.jsonl

Without the token, the data sits on the Pi at
`/home/raspberry/hokm-data/game_data/hands_*.jsonl`.
