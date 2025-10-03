# Repository Guidelines

## Project Structure & Module Organization
The repo contains two top-level apps:
- `croatian-stt/` holds the Next.js 15 frontend. Routes live under `src/app`, shared UI in `src/components`, and static assets in `public/`.
- `croatian-stt-backend/` runs the FastAPI WhisperX service. Entry point is `main.py`, runtime artifacts land in `out/`, and tests belong in `tests/`.
Keep frontend environment files in `.env.local`; backend secrets stay outside version control.

## Build, Test, and Development Commands
Frontend (run inside `croatian-stt/`):
- `pnpm dev` — launch Turbopack dev server at `http://localhost:3000`.
- `pnpm build` / `pnpm start` — create and serve the production bundle.
- `pnpm lint` / `pnpm format` — run Biome linting and formatting.
Backend (run inside `croatian-stt-backend/`):
- `uv sync` then `uv run python main.py` — install dependencies and start the API on `:7860`.
- Alternately, create a venv, `pip install -e .`, and run `python main.py`.

## Coding Style & Naming Conventions
TypeScript and React components use 2-space indentation, PascalCase for components, and `useCamelCase` for hooks. Favor colocated styles or `globals.css` for shared rules. Python modules follow PEP 8 with snake_case files and functions, PascalCase classes, and type hints where practical. Run Biome before committing and stick to pnpm; do not regenerate the npm lockfile.

## Testing Guidelines
Add frontend tests with Vitest or Jest alongside components or under `__tests__/`, following `*.test.tsx` naming. Backend tests belong in `croatian-stt-backend/tests/` with `pytest`. Target coverage on speech parsing, API error handling, and UI interaction flows.

## Commit & Pull Request Guidelines
Write Conventional Commit-style messages (`feat:`, `fix:`, `chore:`) capped at 72 characters. PRs should link issues, summarize changes, list test steps, and include screenshots or terminal output for UI or CLI updates. Keep changes focused and small to simplify review.

## Security & Configuration Tips
Match `AUTH_SECRET` across frontend `.env.local` and backend environment variables. Configure `NEXT_PUBLIC_WS_URL` to point to the backend WebSocket (e.g., `ws://localhost:7860/ws`). Expect large WhisperX model downloads on first run; ensure adequate disk space and prefer Apple Silicon `mps` acceleration when available.
