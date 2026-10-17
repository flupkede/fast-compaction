# fast-compaction

Repo instruction doc. Vendor-named variants (`CLAUDE.md`) mean the same.

## Judge infrastructure

- Training, quantization and agreement measurement run on the game pc (`ollama.lan`, RTX 3060 Ti 8 GB); the ZBook only serves the live judge. Headless ollama there: `C:\Users\daan\serve-ollama.cmd`, launched detached via `gguf-via-wmi.ps1` (WMI children survive ssh disconnect).
- `clef-mix` is NOT a library model — it is built locally: `ollama create clef-mix` from a Modelfile `FROM clef-flash` + `num_gpu 24` + `num_ctx 16384` (partial CPU offload so 10 GB weights never block the 8 GB card). The teacher is re-creatable anytime via `ollama pull clef-flash`, which needs ollama >= 0.40.4 (game pc upgraded 2026-10-16; clef-flash then answers ~7–9 s/request there).
- Live judge on the ZBook: ollama model `fast-compaction` (qwen3 1.7B, Q4_K_S, 1.1 GB, distilled from clef-mix decisions) behind `scripts/systemone-shim.ts` on port 11435 (autostart and manual restart: Startup `start-fast-compaction-shim.cmd` → `scripts/start-shim.ps1`, idempotent — kills the old listener first, hidden window, self-verifies). Plugin options: `baseUrl: http://localhost:11435/v1/systemone`, `model: fast-compaction`.
- Why the shim exists: ollama's `/v1/systemone` route only serves decision-arch models (clef/nimble/tev1 families — pull-only, no public training recipe; Typesafe explicitly offers no customer fine-tuning), so a fine-tuned qwen3 judge needs this protocol bridge. Keep it when moving between machines.
- Quant ladder (game pc, 40 val examples / 826 questions vs teacher labels): Q4_K_S 98.5%, Q8 98.5%, Q4_K_M 98.2%, Q3_K_M 72.9%, Q2_K broken (endless generation). Judge Modelfiles therefore carry `num_predict 4096` — a degenerate reply can never run away.
- Data pipeline: `npm run prepare-training` builds `~/.fast-compaction/training/fine-tune/{train,val}.jsonl` from stored compaction records; `npm run agreement` scores any candidate judge against val without re-asking the teacher.
- Plugin release: `npm run ship-plugin` = push + refresh the local marketplace clone (`claude plugin marketplace update`) + `claude plugin update`; then resume the Claude Code session (plugins load once per process). Bump the version in `.claude-plugin/plugin.json` before shipping.

