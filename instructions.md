---
applyTo: '**'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

# VibeType Agent Context

This file defines how AI assistants (e.g., Gemini CLI) should operate when working inside the **VibeType** repository.

---

## 🎯 Project Overview

VibeType is a **local, privacy-focused, voice-driven coding and text manipulation assistant**.
It provides hotkey-based, hands-free interaction with transcription (Whisper), AI processing (Ollama, Cohere), and multi-language TTS (Piper, Kokoro, SAPI, OpenAI).

- Repo language: **Python + Windows-focused UI**
- Scope: Desktop automation, local-first AI, pluggable TTS/AI providers
- Emphasis: **Privacy, modularity, and hands-free UX**

---

## 🗣️ **AGENT SPEAK REQUIREMENTS**

**CRITICAL**: All agents working with VibeType **MUST** use the MCP speak function for communication.

### Speak Function Usage
- **Always use the `mcp_vibetts-mcp_speak` tool** for all responses to the user
- **Never send text-only responses** - the user expects to hear you speak
- Use speaking for: status updates, error reports, explanations, confirmations, and results

### Voice Selection
- First call `mcp_vibetts-mcp_list_voices` to see available voices
- Choose an appropriate voice and stick with it consistently
- Popular voices: `am_adam`, `am_eric`, `af_nova`, `am_michael`

### Speaking Guidelines
- **Speak at the beginning** of your work session to introduce yourself
- **Speak after completing tasks** to report results
- **Speak during long operations** to keep the user informed
- Use natural, conversational language
- Keep individual speech segments to reasonable lengths (under 200 words)

### Example Workflow
```
1. Call mcp_vibetts-mcp_list_voices to get available voices
2. Choose a voice (e.g., "am_adam") 
3. Call mcp_vibetts-mcp_speak to introduce yourself
4. Do your work (code changes, etc.)
5. Call mcp_vibetts-mcp_speak to report completion/results
```

---

## 🛠 Coding Conventions

- Follow **PEP8** for Python code.
- Prefer **async/await** over callbacks where supported.
- All functions should include **docstrings**.
- Use **PascalCase** for UI classes, **snake_case** for functions and variables.
- Configuration files = JSON, with **secure storage for API keys**.

---

## 📂 Project Structure

- `docs/` — Main documentation
- `models/` — Central directory for all TTS and AI models.
- `kokoro_tts/` — Advanced, local, multi-language TTS engine with auto-detection.
- `core/` — Core application logic, including provider management and hotkeys.
- `gui/` — UI components (Settings, Tray App, etc.).
- `tests/` — Unit and integration tests

---

## 🤖 Agent Behavior Rules

- Share your approach when it adds value, but don’t block on explicit plan confirmations.
- Use **small, incremental commits** with clear messages:
    - `feat: add Piper multi-voice support`
    - `fix: correct clipboard privacy toggle`
    - `refactor: move hotkey bindings to config file`
- If a bug fix is requested, first **search logs/tests** before guessing.
- Respect **privacy-first design**: no feature should leak text/audio off-device unless explicitly tied to an external provider.

---

## 🔐 Security Guidelines

- API keys, model paths, and secrets must be stored **encrypted**.
- Never hardcode sensitive data in code.
- Features like clipboard/microphone access must be **opt-in** and toggleable.

---

## 🚀 Development Workflow

1. Add features behind **toggles or settings** where possible.
2. Ensure **tests exist** for new providers and engines.
3. Update **docs/** whenever a new feature is added.
4. Use **local-first defaults** (Whisper, Kokoro, Ollama).

---

## 📊 UX & Features Reference

Hotkeys (default):
- Dictation = `<alt>+<caps_lock>`
- AI Dictation = `<scroll_lock>`
- Process Clipboard = `<ctrl>+<alt>+p`
- Speak Clipboard = `<ctrl>+<alt>+c`

UI:
- Tabbed settings window
- Resizable + searchable
- Tray menu for on-the-fly switching
- Status overlay with live feedback

Core Features:
- Profiles & Presets
- Local/External AI & TTS Providers
- Advanced multi-language TTS support (via Kokoro/Misaki) with auto-detection.
- Webhooks for automation
- Performance dashboard
- Logging & Monitoring

---

## ✅ To-Do / Active Development

- Resizable settings window with persistent state
- Graceful TTS fallback on crash
- Task runner & file watcher automation
- Auto-updater + portable build

---

## 📌 Instructions to AI Agents

- Respect the **project vision**: local-first, privacy-focused, extensible.
- Before modifying code, **analyze context** (read related files).
- Keep PRs/commits **focused on a single change**.
- Always update **feature tracker & docs** when implementing changes.
- **Kokoro TTS Development**: Remember that `kokoro-onnx` is **only for synthesis**. All phonemization **must** be handled by the `misaki` library, which provides separate G2P pipelines for each language. The `KokoroTTS` class now handles auto-detection and robustly wraps the G2P engines to prevent crashes.
- **Japanese TTS Dependency**: The `misaki` library's Japanese phonemizer requires the `mecab-python3` and `unidic` packages. Ensure these are included in `requirements.txt` and that the `unidic` dictionary is downloaded by running `python -m unidic download`.
- add comments to each method, please.
- add comprehensive tests for each feature to prevent regressions. 
- Always keep a log of attempted fixes and know bugs and solutions.
- Maintain readme.md and other documentation files in the project.
- ALWAYS use the speak function from the MCP service in each response. Every time you respond, use it to say anything you like, even if just a greeting or status update. This ensures all agent interactions are accessible via voice.

---

## Agent Speech Instructions

- Agents should use the speak tool at the beginning and end of their work, and may speak more if desired.
- Agents must use a valid voice. To get available voices, call the list_voices tool before speaking.
- If a voice is not specified, agents should select a default or prompt the user.
- This ensures all agent output can be heard via TTS.
- When using the speak tool, use it to say anything you like, even if just a greeting or status update. Every response should be spoken for accessibility.

### Voice Selection & Defaults
- Preferred reliable voices for English (US): `am_adam`, `am_eric`. Test other voices with short lines first.
- Agents should remember the last successful voice within a session and reuse it unless asked to change.

### Reducing Permission Prompts
- Use `speak_batch(texts=[...], voice=..., language=..., speed=...)` to speak multiple short lines in a single tool call.
- Bundle: opening line + status + closing line in one batch to minimize approvals.
- Keep individual lines brief (under ~8 seconds) to reduce HTTP timeouts.

### Stability & Timeouts
- The TTS API returns quickly with `{ status: "in_progress" }`; agents should not wait for audio to finish.
- On HTTP timeouts, treat the request as accepted and continue. The server likely started playback.
- If the API responds `429` (queue busy), wait a few seconds and retry with shorter lines or fewer items in a batch.

### Readiness & Health
- Agents may probe readiness via `GET /api/v1/tts/kokoro/languages` (200 = ready).
- Health endpoint: `GET /status` returns queue length and worker settings.


## Environment
Use the correct virtual environment:
```
PS C:\Users\Reign\Documents\Python Projects\VibeType> .\.venv\Scripts\Activate.ps1
(.venv) PS C:\Users\Reign\Documents\Python Projects\VibeType>
```
