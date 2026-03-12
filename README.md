# Book to Audiobook Converter

Turn PDFs and EPUBs into spoken audio from a single local web app. Pick a page range, choose a TTS engine, preview what is available, and export either one MP3 or a ZIP of chapter files.

This repo is optimized for practical use, not as a framework. It gives you one place to:
- convert books from `input/` or direct uploads
- switch between local and cloud speech engines
- clone voices with reference audio
- save outputs back into the repo for repeatable runs
- come back later and still remember how the whole thing works

## Why This Exists

Most TTS tools make you choose between flexibility and convenience. This app keeps both:
- a clean browser UI instead of one-off scripts
- multiple backends instead of one engine lock-in
- chapter-aware conversion instead of only full-book output
- local-first options when you want privacy or offline experimentation
- cloud backends when you want convenience or specific voices

## What You Can Do

| Capability | Details |
| --- | --- |
| Input formats | PDF and EPUB |
| Source options | Upload from your machine or pick files from `input/` |
| Output formats | Single MP3 or chapter ZIP |
| Save mode | Download immediately or save into `output/` |
| Voice cloning | Upload reference audio and synthesize with XTTS |
| Chapter mode | Works for both PDF and EPUB |
| Progress tracking | Browser polls server-side job progress during conversion |
| Language filtering | UI can filter backends by English / Romanian |

## Backend Matrix

| Backend | Type | Languages | Best For | First Run / Dependency Notes |
| --- | --- | --- | --- | --- |
| Kokoro | Local | English | Best default local quality-to-speed balance | Requires `espeak-ng`; downloads model on first use |
| Piper | Local | English | Lightweight local ONNX voices and tunable pacing | Needs a `.onnx` model plus sidecar JSON in `models/` |
| Supertonic | Local | English | ONNX-based local synthesis with simple controls | Downloads about 305MB on first use |
| Hugging Face | Local | English, Romanian | Trying HF TTS models locally | Downloads model weights on first use |
| SpeechT5 | Local | English | Multi-speaker preset voices | Downloads model, vocoder, and speaker embeddings |
| XTTS-v2 | Local | English + multilingual | Voice cloning with a reference sample | Large model download, reference audio required |
| XTTS-v2 Romanian | Local | Romanian | Romanian voice cloning | Large Romanian fine-tune download, reference audio required |
| Amazon Polly | Cloud | English in current UI | Reliable cloud voices and billing visibility | Requires valid AWS credentials |
| HF Inference API | Cloud | English, Romanian | Quick cloud inference experiments | Requires `HF_TOKEN`; free tier can rate-limit |

## Quick Start

The shortest clean setup is two commands:

```bash
./scripts/bootstrap.sh
uv run book-to-audiobook
```

Then open `http://localhost:1234`.

If you want every optional backend as well:

```bash
./scripts/bootstrap.sh --all
uv run book-to-audiobook
```

### What the bootstrap script does

`./scripts/bootstrap.sh`:
- installs `uv` if missing
- installs `ffmpeg` and `espeak-ng`
- installs Python 3.12 via `uv`
- runs `uv sync --dev`
- creates `.env` from `.env.example` if needed

It currently supports:
- macOS with Homebrew
- Ubuntu / Debian style systems with `apt-get`

### Manual setup

If you prefer not to use the bootstrap script, the equivalent manual flow is below.

### 1. Install `uv`

macOS or Ubuntu:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you already use Homebrew on macOS:

```bash
brew install uv
```

### 2. Pin the project Python version

This repo is set up for Python 3.12 because several TTS backends are still fragile on newer interpreters.

```bash
uv python install 3.12
```

The repo includes [`.python-version`](./.python-version), so `uv` will use Python 3.12 automatically.

### 3. Install non-Python system dependencies

This project needs a couple of tools that are outside Python dependency management.

| Tool | Required? | Why |
| --- | --- | --- |
| `ffmpeg` | Yes | `pydub` uses it for MP3 import/export and audio format conversion |
| `espeak-ng` | Required for Kokoro | Kokoro depends on it for phonemization |
| `piper` CLI | Optional | Only needed if Piper falls back from the Python API to the CLI binary |

`pydub` relies on `ffmpeg` for MP3 import/export, and Kokoro relies on `espeak-ng`.

macOS:

```bash
brew install ffmpeg espeak-ng
```

If you want the optional Piper CLI available as a fallback:

```bash
brew install piper
```

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg espeak-ng
```

If you want the optional Piper CLI available as a fallback:

```bash
sudo apt-get install -y piper
```

Quick sanity check:

```bash
ffmpeg -version
espeak-ng --version
```

### 4. Sync Python dependencies with `uv`

For the default app experience, including the web app, Kokoro, Polly support, and the fast test suite:

```bash
uv sync --dev
```

If you want every optional backend available locally:

```bash
uv sync --dev --extra all
```

### Optional backend extras

| Extra | Adds |
| --- | --- |
| `piper` | Piper Python backend support |
| `supertonic` | Supertonic local ONNX backend |
| `huggingface` | Local Transformers TTS backend |
| `xtts` | XTTS / XTTS-RO voice cloning backends |
| `speecht5` | SpeechT5 multi-speaker backend |
| `ocr` | EasyOCR fallback for scanned PDFs |
| `all` | All optional backends above |

Examples:

```bash
uv sync --dev --extra piper --extra supertonic
uv sync --dev --extra xtts --extra ocr
```

### 5. Create your local environment file

```bash
cp .env.example .env
```

You do not need to fill everything in up front. For a first successful run, the minimum usually is:
- leave the default `TTS_BACKEND=kokoro`
- keep Kokoro defaults as-is
- skip cloud credentials until you want Polly or HF cloud

### 6. Start the app

```bash
uv run book-to-audiobook
```

Open:

```text
http://localhost:1234
```

### 7. First successful conversion

1. Open the **Convert** page.
2. Upload a PDF or EPUB, or place one inside `input/` and pick it from the dropdown.
3. Keep **Kokoro** selected for the simplest local path.
4. Leave page range as `all` or choose a range such as `1-10`.
5. Click **Convert to MP3**.

## Common `uv` Commands

```bash
./scripts/bootstrap.sh
./scripts/bootstrap.sh --all
uv sync --dev
uv sync --dev --extra all
uv lock
uv sync --frozen --dev
uv run book-to-audiobook
uv run pytest tests/test_smoke.py -q
```

## Configuration

The full configuration surface lives in [`./.env.example`](./.env.example).

Recommended workflow:

```bash
cp .env.example .env
```

Then edit only the values you care about.

### Most Important Settings

| Setting | Why It Matters |
| --- | --- |
| `TTS_BACKEND` | Sets the default backend shown on page load |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | Required for Polly readiness and conversion |
| `PIPER_MODEL` | Points to the default Piper voice model |
| `HF_TOKEN` | Required for HF cloud inference |
| `KOKORO_VOICE` / `KOKORO_SPEED` | Sets the default local voice and pace |
| `SUPERTONIC_*` | Sets default Supertonic voice, language, speed, and silence |

### Config Philosophy

- If a variable is not in `.env.example`, the app does not currently rely on it.
- `.env.example` uses safe placeholders and code defaults only.
- Keep real secrets in `.env` only.
- The app loads `.env` automatically through `python-dotenv`.

## How To Use The App

### Convert Page

The main workflow is a three-step wizard:

1. **Source**
   - Upload a `.pdf` or `.epub`
   - Or pick a file already present in `input/`

2. **Pages**
   - Use `all`, `5`, or `1-10`
   - The app fetches metadata and tries to detect chapters
   - If there are blank front/back pages, the UI suggests a likely text range

3. **Engine & Voice**
   - Pick a backend
   - The UI loads voice lists dynamically where needed
   - The status badge shows whether a backend is ready or still needs setup/downloads

### Chapter Mode

If chapters are detected, you can enable **Split into chapter files**.

Important behavior:
- chapter mode converts the whole document by detected chapters
- the result is a ZIP archive of MP3 files
- this works for both PDF and EPUB
- the typed page range is not the controlling unit when chapter mode is on

### Save Mode

Enable **Save to `output/` folder** if you want the results kept in the repo instead of downloaded directly by the browser.

### Voice Cloning Page

Use the **Voice Cloning** page to upload short reference audio:
- WAV, MP3, M4A, FLAC, OGG
- max 10MB
- 3 to 10 seconds is ideal

Those files are stored in `reference_audio/` and become selectable when using XTTS backends.

## Supported Inputs And Outputs

### Inputs

- PDF with extractable text
- EPUB with a valid content spine
- reference audio for XTTS voice cloning

### Outputs

- single MP3
- ZIP of per-chapter MP3 files
- optional saved artifacts under `output/`

### Output Naming

Generated filenames include:
- UTC timestamp
- source book name
- page or chapter label
- backend
- voice/model detail
- relevant prosody settings when applicable

## Project Layout

```text
.
├── app.py
├── pyproject.toml
├── .python-version
├── scripts/
│   └── bootstrap.sh
├── templates/
│   └── index.html
├── input/
├── output/
├── reference_audio/
├── tests/
├── models/
├── .env.example
└── uv.lock
```

What each piece does:
- `app.py`: the full Flask app, document extraction pipeline, backend dispatch, and routes
- `pyproject.toml`: dependency definitions, optional backend extras, and pytest config for `uv`
- `.python-version`: pins the interpreter version used by `uv`
- `scripts/bootstrap.sh`: one-command local setup for supported macOS and Ubuntu/Debian environments
- `templates/index.html`: the UI, styling, and browser-side JavaScript in one template
- `input/`: books you want selectable from the dropdown
- `output/`: saved MP3s and chapter ZIP directories
- `reference_audio/`: uploaded voice-cloning samples
- `models/`: Piper `.onnx` models and their `.json` sidecars
- `tests/`: fast regression coverage plus focused route and extraction tests
- `uv.lock`: fully resolved dependency lockfile generated by `uv`

## Architecture At A Glance

The app is intentionally simple:

- **Flask server**
  - serves the UI
  - accepts uploads and convert requests
  - exposes helper APIs for backend status, chapter metadata, and saved reference voices

- **Inline frontend**
  - HTML, CSS, and JavaScript live in `templates/index.html`
  - dynamic UI behavior uses `fetch`
  - progress is polled from `/progress/<job_id>`

- **Document processing**
  - PDFs are read with `PyMuPDF` / `fitz`
  - EPUBs are parsed from the content spine
  - extracted text is cleaned before TTS

- **Speech synthesis**
  - backend-specific wrappers normalize output into MP3 buffers
  - heavy models are cached lazily in process
  - chapter mode writes one file per chapter, then zips when needed

- **Local cache management**
  - Hugging Face and Supertonic caches are redirected into a project-local `.cache/`

## API Routes Worth Knowing

| Route | Purpose |
| --- | --- |
| `/` | Main app UI |
| `/convert` | Starts a conversion job and returns MP3, ZIP, or saved-result JSON |
| `/progress/<job_id>` | Current extraction / synthesis progress |
| `/api/backend-status` | Readiness notes for local and cloud backends |
| `/api/pdf-info` | Page and chapter metadata for PDFs and EPUBs |
| `/api/reference-voices` | Lists uploaded XTTS reference samples |
| `/api/upload-reference` | Uploads a new reference audio file |

## Text Processing Notes

Before synthesis, the app tries to make extracted book text sound better:

- joins hard-wrapped lines
- expands common abbreviations
- converts some numbers to words
- normalizes ligatures and punctuation
- strips URLs and citation-style brackets
- preserves normal bracketed prose such as `[aside]`
- detects and strips likely running headers in PDFs

This is a practical cleanup layer, not a perfect document normalizer.

## Testing

Run the fast regression suite:

```bash
uv run pytest tests/test_smoke.py -q
```

The current tests focus on:
- non-destructive text preprocessing
- EPUB chapter support
- Piper CLI parameter propagation
- Supertonic pause handling
- Polly readiness status behavior
- utility and route-level regressions

## Troubleshooting

### Kokoro fails or sounds unavailable

Check:
- `espeak-ng` is installed and available on `PATH`
- the backend status badge is not showing a missing dependency problem

### MP3 export or audio decode fails

Check:
- `ffmpeg` is installed
- your shell can run `ffmpeg -version`

### Piper works inconsistently or fallback errors mention the CLI

Check:
- `uv sync --extra piper` has been run if you want the Python Piper backend
- if the app falls back to the CLI, `piper` is installed and available on `PATH`
- your shell can run `piper --help`

### Polly shows as not ready

Check:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- outbound access to AWS STS / Polly

The app validates Polly readiness with a lightweight AWS call, so auth or network problems show up before conversion starts.

### Piper conversion fails

Check:
- the configured `.onnx` file exists
- the sidecar `.json` exists next to it
- `PIPER_BINARY` points to a real binary if the Python API falls back to CLI
- your CLI supports `--noise-scale` and `--noise-w`

### EPUB conversion says no readable content

Check:
- the EPUB has a valid content spine
- the relevant spine items contain real text, not only images or decorative markup

### PDF conversion says no extractable text

That usually means the PDF is scanned pages rather than embedded text. This app does not currently perform OCR.

### HF cloud errors or rate limits

Check:
- `HF_TOKEN` is set
- the model ID exists
- you are not hitting free-tier throttling

### XTTS says reference audio is missing

Check:
- the audio file was uploaded on the Voice Cloning page
- it appears in the saved reference voice list
- the selected filename still exists in `reference_audio/`

## Operational Notes

### First-Run Downloads

Several backends download large assets the first time you use them:
- Kokoro: model download
- Supertonic: about 305MB
- XTTS / XTTS-RO: very large, around 1.8GB-class downloads
- SpeechT5: model + vocoder + embeddings
- Hugging Face local backends: model dependent

If a backend is slow the first time, that is expected.

### Local-First Caches

Model caches are redirected into a repo-local `.cache/` directory so experiments stay self-contained.

### Polly Cost Visibility

For Polly conversions, the app tracks billed character counts and estimated cost from backend pricing constants. This is useful for quick budgeting, but it is not a billing statement.

## Limitations

- No OCR for scanned PDFs
- Frontend and styling live in a single template file by design
- Backends are powerful but heavy; some require significant first-run downloads
- Chapter detection is heuristic for many PDFs
- Romanian support is narrower than English support

## Good Defaults

If you want a practical starting point:

- use **Kokoro** for local English
- use **XTTS-RO** or HF Romanian models for Romanian
- use **chapter mode** for long books where navigation matters
- save important runs into `output/`
- keep a few short, clean reference clips in `reference_audio/`

## Security Reminder

Do not commit your real `.env`.

Keep:
- AWS credentials
- Hugging Face tokens
- any other private values

only in local, untracked environment files.

## License

This project is licensed under the GNU Affero General Public License v3.0.

That means if you modify it and make the modified version available to users over a network, you must also make the corresponding source available under the same license.

See [`LICENSE`](./LICENSE) for the full text.
