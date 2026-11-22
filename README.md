# platodoom

Standalone open-source terminal AI coding, reasoning, and multi-model GPU CLI powered by Google Colab and HuggingFace open weights (`qwen3.8-27b` default).

## Requirements

- **Node.js 18+**
- **Python 3.10+**
- **Chrome or Brave** (for optional one-click Google Colab browser sign-in)

## Install via npm

```bash
npm install -g platodoom
```

On install, `platodoom` creates a tiny isolated Python virtual environment at `~/.platodoom/venv` with its local runtime dependencies (`selenium`, `webdriver-manager`, `PyYAML`, `pydantic`, `httpx`).

## Quick Start

Run `platodoom` from **any project directory**:

```bash
cd your-project
platodoom
```

On your first launch:
1. A self-contained `.platodoom/` folder is initialized inside the current directory (automatically excluded from git via `.git/info/exclude`).
2. A **skippable 3-step setup** walks you through:
   - **Step 1 — Google Colab Browser Sign-In:** Opens your browser once to sign into Google Colab (no Google Cloud OAuth app required).
   - **Step 2 — HuggingFace Token (Optional):** Paste your token or press `[o]` to open `huggingface.co/settings/tokens`.
   - **Step 3 — Tunnel Provider (Optional):** Uses free, zero-account **Cloudflare Quick Tunnels** by default, or lets you add an `NGROK_AUTHTOKEN`.
3. Every step can be skipped immediately and revisited anytime via:
   - `platodoom auth` (from your shell)
   - `/auth` (inside chat)

## Common Commands

```bash
platodoom                     # Launch interactive chat (default model: qwen3.8-27b)
platodoom auth                # Sign into Google Colab or update HF / tunnel tokens
platodoom auth status         # Show redacted credential & tunnel status
platodoom models              # List all 14 open-source Qwen & DeepSeek models
platodoom -m qwen-9b          # Launch with a specific model alias
platodoom --plan              # Start in read-only Plan Mode
platodoom --build             # Start in Build Mode (confirm-writes)
platodoom --resume            # Resume a saved conversation
platodoom --endpoint URL      # Connect to any OpenAI-compatible /v1 server directly
```

## Inside the REPL

- `/auth` — sign in or update HuggingFace / ngrok credentials
- `/model` — interactive numbered model picker (`qwen3.8-27b`, `qwen-9b`, `r1-14b`, `qwen-27b`, `qwen-35b-moe`, `r1-32b`, and more)
- `/plan` / `/build` — switch between read-only analysis and file editing
- `/gpu` / `/stack` — select Colab hardware tier (`G4`, `H100`, `A100-80GB`, `A100`, `L4`, `T4`, `v6e-1`, `v5e-1`) or co-resident multi-model stacks
- `/sessions` / `/resume` — inspect and resume saved conversations
- `/help` — show all slash commands
- `/exit` or `Ctrl+C` — save session state and exit

## Privacy & Security

- Credentials (`auth.hf_token`, `auth.ngrok_authtoken`) are saved strictly in `.platodoom/config.json` with `0600` file permissions and are never written to git-tracked `.platodoom.json` files or printed in logs.
- Built-in Zero-Trust Guardrails block model tools from reading sensitive files (`~/.ssh`, `~/.aws`, `.platodoom/config.json`, `*.pem`, `*.key`) and redact `.env` secrets automatically.

## License

MIT — see [LICENSE](./LICENSE).
