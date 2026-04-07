# Sommelier — Netflix recommendation assistant

AI-assisted recommender for movies and TV shows. It reads a remote PostgreSQL catalog (Netflix-style titles), extracts preferences with Claude, and returns suggestions through a terminal chat or an **MCP** (Model Context Protocol) HTTP server.

- **Python:** 3.12 or newer  
- **Stack:** Anthropic API, scikit-learn (similarity), Rich (CLI), FastMCP (MCP over HTTP)

---

## Quick start

### 1. Clone and environment

```bash
git clone <repository-url>
cd sommelier-netflix-ai-sdd
python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. Install dependencies

Choose one approach (they are equivalent in intent; use what fits your workflow).

**Option A — `pyproject.toml` (recommended)**

```bash
python -m pip install -U pip
pip install -e .
```

**Option B — `requirements.txt`**

```bash
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

The second line installs libraries; `pip install -e .` registers the `sommelier` package and console scripts (`sommelier`, `sommelier-mcp`) from `src/`.

**Optional extras**

| Goal | Command |
|------|---------|
| Run tests | `pip install -r requirements-dev.txt` then `pip install -e .` |
| MCP server only (includes `mcp[cli]`) | `pip install -r requirements-mcp.txt` then `pip install -e .` |
| Everything from pyproject | `pip install -e ".[dev,mcp]"` |

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at least:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Key from [Anthropic Console](https://console.anthropic.com/) |
| `DATABASE_URL` | Yes | PostgreSQL URL; database must expose the expected catalog (see below) |
| `EXTRACTION_MODEL` | No | Claude model for JSON preference extraction (default suited for **Claude API**) |
| `GENERATION_MODEL` | No | Claude model for natural-language replies |
| `MAX_HISTORY_TURNS` | No | Session history limit (default in `.env.example`) |
| `DEBUG` | No | `true` for verbose logs on stderr |
| `HOST` / `PORT` | No | MCP server bind address (defaults `0.0.0.0` / `8000`) |

Use **Claude API** model IDs (e.g. `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`). AWS Bedrock uses different IDs; this project uses the official `anthropic` Python SDK against the Anthropic API unless you customize the client.

### 4. Database

The app expects a PostgreSQL database reachable via `DATABASE_URL`, with a table compatible with the project’s loader (see `src/sommelier/infrastructure/dataset_store.py` and specs under `.sdd/specs/`). Without a valid catalog, startup will fail.

---

## Run locally — terminal chat

From the repository root (with venv active and `.env` filled):

```bash
python -m sommelier
```

Or, if your `Scripts` directory is on `PATH`:

```bash
sommelier
```

You should see startup info (database source redacted, title count) and a greeting. Type what you want to watch; use `quit` or `exit` to leave.

**Run tests** (after `pip install -r requirements-dev.txt` and `pip install -e .`):

```bash
pytest
```

---

## Run locally — MCP server (HTTP)

The MCP server loads the app once at startup and exposes tools over **streamable HTTP** (FastMCP).

Install MCP dependencies:

```bash
pip install -r requirements-mcp.txt
pip install -e .
```

Start the server:

```bash
python -m sommelier.mcp_server
```

Or:

```bash
sommelier-mcp
```

Default URL for MCP HTTP clients: **`http://127.0.0.1:8000/mcp`**  
(override with `HOST` and `PORT` in `.env`.)

### Exposed tools

| Tool | Purpose |
|------|---------|
| `recommend` | Natural-language query → recommendation reply (uses the same orchestration as the CLI) |
| `get_title_details` | Lookup by title name in the loaded catalog (metadata + description) |

### Connect an MCP client

Configuration depends on the client. Point it at the **HTTP MCP endpoint** above (not stdio). Example shape for a generic MCP-over-HTTP setup:

```json
{
  "mcpServers": {
    "sommelier": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Your client may require headers, OAuth, or a different JSON layout—follow its documentation for “HTTP” or “streamable HTTP” MCP servers. After the server is running, the client should list `recommend` and `get_title_details`.

**Deploy note:** `render.yaml` in this repo sketches a Render deployment using `sommelier-mcp`; set secrets (`DATABASE_URL`, `ANTHROPIC_API_KEY`, etc.) in the host’s dashboard.

---

## Project layout (short)

| Path | Role |
|------|------|
| `src/sommelier/` | Application code |
| `pyproject.toml` | Package metadata, dependencies, scripts |
| `requirements*.txt` | Pip-friendly dependency lists |
| `.env.example` | Environment template |
| `.sdd/` | Spec-driven development artifacts |

---

## License / contributing

Add your license and contribution guidelines here if applicable.
