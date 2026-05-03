# turn-native in onebox-deploy

Use this folder to store TURN credentials/config used by the onebox container.

## Files

- `turn_credentials.env.example`: template
- `turn_credentials.env`: real credentials (do not commit)

The container auto-loads:

`/opt/onebox/turn-native/turn_credentials.env`

because `run_onebox.sh` mounts:

`./turn-native -> /opt/onebox/turn-native:ro`

and `start_services.sh` sources `TURN_ENV_FILE` if present.

## Quick setup

```bash
cd /teamspace/studios/this_studio/onebox-deploy/turn-native
cp turn_credentials.env.example turn_credentials.env
```

Edit `turn_credentials.env` values for your host and credentials.

## Priority

- If `turn_credentials.env` exists, its values are loaded.
- You can still override with container `.env` variables.
