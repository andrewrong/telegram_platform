# Telegram Multimedia Bot

Telegram bot for image/PDF OCR, ingredient-list analysis, and voice transcription.

## OCR usage

Send any of the following to the bot:

- A Telegram photo.
- An image as a document to preserve its original resolution.
- A PDF document.

The caption can select an OCR engine and PDF pages:

```text
#glm
/ocr qwen
/ocr paddle pages=2-7
```

Supported engines are `auto`, `paddle`, `glm`, `qwen`, and explicit `jina`.
Use `#jina` or `/ocr jina pages=2-7` for Jina OCR. It uploads images/PDF pages to Jina's
cloud and may incur charges. Configure `OCR_JINA_API_KEY` only in the OCR service's
private deployment environment, never in Telegram or Git. `auto` stays local, and
ingredient analysis stays GLM. `/ocr_status` displays Jina configuration and metadata
reachability, which does not validate the Key or guarantee inference capacity.
Jina updates are not automatically replayed by the durable queue on failure, including
result-delivery network failures; manually resend if needed to avoid duplicate cloud charges.
The OCR server retains its own bounded retry/local fallback policy.
Short OCR results are returned as
messages; results longer than Telegram's message limit are returned as `ocr-result.md`.

OCR frontend commands:

- `/ocr` shows upload, engine, and page-range usage.
- `/ocr_status` shows the OCR service and Paddle, GLM, and Qwen availability.

Set `OCR_CHANNEL_ID` and optionally `OCR_TOPIC_ID` to turn one Telegram group/forum topic into a
dedicated OCR frontend. When they are unset, supported media continues to work in every chat.

Photos sent to `INGREDIENT_ANALYSIS_CHANNEL_ID` and the optional configured topic continue to use
ingredient analysis. Other supported images and PDFs use generic OCR.

## Docker deployment

Prerequisites:

- A Docker Compose runtime (OrbStack, Docker Desktop, ...).
- The OCR service available on the host at `http://127.0.0.1:18100`.
- Typeless running on the host at `http://127.0.0.1:28111` when voice transcription is needed.

Configure and start:

```bash
cp .env.example .env
# Set TELEGRAM_BOT_TOKEN and any optional feature credentials in .env.
make build
make up
make logs
```

The Compose service connects to the OCR service and to Typeless through
`OCR_SERVICE_URL_DOCKER` and `ASR_SERVICE_URL_DOCKER` (defaults
`http://host.docker.internal:18100` and `http://host.docker.internal:28111`). Voice and audio messages use Typeless's
stable `POST /v1/audio/transcriptions` contract with local `standard` post-processing. Set
`TYPELESS_API_KEY` only when Typeless authentication is required; it is sent as `X-API-Key`.

Incoming Telegram updates are first stored in `./data/inbox.db` before they are processed. This
directory must be retained when moving the deployment: it lets the bot resume downloads and
transcriptions after a restart or a short DNS/network outage without requiring users to resend.

Useful commands:

| Command | Description |
|---|---|
| `make build` | Build the bot image |
| `make up` | Start the bot |
| `make down` | Stop the bot |
| `make restart` | Rebuild and restart |
| `make logs` | Follow bot logs |

## Local development

```bash
uv sync
uv run python test_ocr_connection.py
uv run python src/main.py
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | required | Telegram bot token |
| `LLM_MODEL` | `gemini-3.6-flash` | Model used for ingredient analysis |
| `OCR_SERVICE_URL` | `http://127.0.0.1:18100` | OCR HTTP service for local runs |
| `OCR_DEFAULT_ENGINE` | `auto` | Default OCR engine |
| `OCR_MAX_CONCURRENT` | `4` | Maximum concurrent OCR requests |
| `OCR_REQUEST_TIMEOUT_SECONDS` | `900` | OCR request timeout |
| `OCR_MAX_FILE_SIZE_MB` | `20` | Maximum Telegram image/PDF size |
| `OCR_CHANNEL_ID` | unset | Dedicated OCR channel or group |
| `OCR_TOPIC_ID` | unset | Optional forum topic inside the OCR channel |
| `ASR_SERVICE_URL` | `http://127.0.0.1:28111` | Typeless base URL (without `/v1/...`) |
| `TYPELESS_API_KEY` | unset | Optional Typeless `X-API-Key` value |
| `ALLOWED_GROUP_IDS` | `[]` | Groups allowed for voice transcription |
| `INGREDIENT_ANALYSIS_CHANNEL_ID` | unset | Channel enabled for ingredient analysis |
| `INGREDIENT_ANALYSIS_TOPIC_ID` | unset | Optional ingredient-analysis topic |
| `INGREDIENT_OCR_ENGINE` | `glm` | OCR engine for ingredient photos |
| `UPDATE_STORE_PATH` | `/app/data/inbox.db` | Durable Telegram update inbox path |
| `UPDATE_WORKER_COUNT` | `2` | Concurrent durable message workers |

## ASR interface contract

Voice transcription expects an HTTP service at `ASR_SERVICE_URL` (the code appends the path).
The bot sends `POST {ASR_SERVICE_URL}/v1/audio/transcriptions` as `multipart/form-data` with:

- `file`: the audio, filename `audio.ogg`, media type `audio/ogg`.
- Form fields `model=default`, `language=auto` and `postprocess_mode=standard`.
- Header `X-API-Key: <TYPELESS_API_KEY>` only when that variable is set.

The service must answer `200` with JSON `{"text": "...", "raw_text": "..."}`. The bot uses `text`
and falls back to `raw_text`; an empty transcript or a non-2xx status counts as a failure
(timeout 300 s).

## License

MIT, see [LICENSE](LICENSE).
