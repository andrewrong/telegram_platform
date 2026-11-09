# Modules of telegram_platform

## Modules

| module | paths | entry | tests | depends |
|---|---|---|---|---|
| bot | src/bot/** src/main.py | src/main.py | tests/test_audio_handler.py tests/test_bot_core_ocr_routing.py tests/test_document_ocr_handler.py tests/test_ingredient_handler.py tests/test_logging.py tests/test_reliable_updates.py tests/test_voice_handler.py | |
| audio | src/services/audio/** | src/services/audio/__init__.py | tests/test_asr_service.py | |
| ocr | src/services/ocr/** test_ocr_connection.py | src/services/ocr/__init__.py | tests/test_ocr_service.py | |
| llm | src/services/llm/** | src/services/llm/__init__.py | tests/test_llm_service.py | |
| ingredient | src/services/ingredient/** | src/services/ingredient/__init__.py | tests/test_ingredient_analyzer.py | |
| core | src/models/** src/utils/** src/exceptions.py src/__init__.py src/services/__init__.py src/services/base.py | src/models/schemas.py | tests/test_media_group_cache.py tests/__init__.py | |
| config | src/config/** | src/config/settings.py | tests/test_settings.py | |
| ops | .github/** Dockerfile pyproject.toml scripts/** .dockerignore .env.example .gitignore .python-version LICENSE Makefile docker-compose.yml docker-compose.ocr.yml uv.lock | Dockerfile | | |
| docs | docs/** *.md AGENTS.md CLAUDE.md | AGENTS.md | | |
