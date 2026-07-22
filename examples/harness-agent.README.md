# Harness agent conversation eval — environment profiles

Example-only utility for running [`harness-agent-conversation.eval.yaml`](harness-agent-conversation.eval.yaml) against QA or prod Harness instances. This stays under `examples/` because the core `harness-evals` package is vendor-neutral.

## Setup

```bash
mkdir -p .harness-evals
cp examples/harness-environments.example.yaml .harness-evals/environments.yaml
```

Edit `.harness-evals/environments.yaml` (gitignored). Each environment needs:

| Field | Purpose |
| --- | --- |
| `url` | Harness base URL only (e.g. `https://qa.harness.io`) |
| `org` | Organization identifier |
| `project` | Project identifier |
| `account_id` | Harness account ID |
| `username` | Login username |
| `password` | Login password |
| `token` | Optional cached bearer token (written by `--save`) |
| `sse_path_template` | Optional override for SSE endpoint derivation |

Default SSE URL:

```text
{base_url}/gateway/chat/unified?orgIdentifier={org}&projectIdentifier={project}&accountIdentifier={account_id}
```

Confirm this path against your gateway on first live run; override `sse_path_template` per environment if needed.

## Commands

```bash
# List configured environments
python examples/load_harness_env.py list

# Print shell exports (uses cached token if present)
python examples/load_harness_env.py export qa

# Login, save token to profile, export vars
eval "$(python examples/load_harness_env.py export qa --login --save)"

# Login only
python examples/load_harness_env.py login qa --save

# Show resolved values (TOKEN masked)
python examples/load_harness_env.py show qa
```

Custom profile path:

```bash
python examples/load_harness_env.py list --profile /path/to/environments.yaml
python examples/load_harness_env.py export prod1 --profile /path/to/environments.yaml --login
```

## Run the eval

Load Harness credentials into the shell, then run the eval the same way you normally do:

```bash
eval "$(python examples/load_harness_env.py export qa --login --save)"
export OPENAI_API_KEY=...
PYTHONPATH=src poetry run harness-evals run examples/harness-agent-conversation.eval.yaml --log-level info
```

Optional wrapper (loads env and forwards to the same command):

```bash
python examples/load_harness_env.py run qa examples/harness-agent-conversation.eval.yaml -- --log-level info
python examples/load_harness_env.py run --login --save qa examples/harness-agent-conversation.eval.yaml -- --log-level info
```

The export step sets:

- `SSE_ENDPOINT_URL`
- `HARNESS_ACCOUNT`
- `HARNESS_ORG`
- `HARNESS_PROJECT`
- `TOKEN`

## Login API

Token generation uses:

```http
POST {base_url}/gateway/api/users/login
Content-Type: application/json

{"authorization": "Basic <base64(username:password)>"}
```

The response `token` field becomes `TOKEN` for the eval config's `Authorization: Bearer ${TOKEN}` header.

## Files

| File | Role |
| --- | --- |
| [`examples/harness_env.py`](harness_env.py) | Profile load, login API, env resolution |
| [`examples/load_harness_env.py`](load_harness_env.py) | CLI wrapper |
| [`examples/harness-environments.example.yaml`](harness-environments.example.yaml) | Profile template |
