# Onebox Nginx + Let's Encrypt In The Container

This image can now start Nginx automatically when the container starts. Nginx listens on `80/443`, proxies to the unified app on `8780`, and can request and renew a Let's Encrypt certificate from inside the same container.

## Required DNS

Point your domain to the onebox server public IP before starting the container.

Example:

```text
avatar.example.com -> 34.61.97.125
```

## Required environment

Set these in your `.env`:

```env
ONEBOX_PUBLIC_HOST=avatar.example.com
ONEBOX_NGINX_ENABLE=1
ONEBOX_LETSENCRYPT_ENABLE=1
LETSENCRYPT_EMAIL=you@example.com
ONEBOX_TLS_ENABLE=0
TURN_PUBLIC_HOST=avatar.example.com
```

Notes:

- `ONEBOX_TLS_ENABLE=0` keeps the Python app on plain HTTP internally.
- Nginx handles the public certificate and TLS termination.
- Persist `/etc/letsencrypt` so the certificate survives container restarts.

## Docker run

Use published `80/443` ports and mount a persistent cert directory:

```bash
docker run --rm -it \
  --name musetalk-onebox \
  --gpus all \
  --env-file .env \
  -p 80:80 \
  -p 443:443 \
  -p 127.0.0.1:8780:8780 \
  -p 3478:3478/tcp \
  -p 3478:3478/udp \
  -p 5349:5349/tcp \
  -p 49160-49200:49160-49200/udp \
  -v "$(pwd)/letsencrypt:/etc/letsencrypt" \
  -v "$(pwd)/models:/opt/musetalk/models" \
  -v "$(pwd)/results:/opt/musetalk/results" \
  -v "$(pwd)/.cache:/root/.cache" \
  -v "$(pwd)/turn-native:/opt/onebox/turn-native:ro" \
  coldslim/musetalk-onebox:blackwell
```

## Startup behavior

When the container starts:

1. The unified app starts on `8780`.
2. Nginx starts on `80/443`.
3. If no real cert exists yet, the container uses a temporary self-signed cert so Nginx can boot.
4. `certbot certonly --webroot` requests the real certificate for `ONEBOX_PUBLIC_HOST`.
5. Nginx reloads onto the valid certificate.
6. A background renewal loop runs every 12 hours.

## Important requirements

- Ports `80` and `443` must be reachable from the internet for Let's Encrypt.
- No other service on the host can already be using `80/443`.
- If certificate issuance fails on first boot, check DNS and firewall first.
