# Rules for agents working in this repository

Read `CLAUDE.md` first for the toolchain (`uv`, `pytest`, `ruff`) and the working process.

## Modules

- The packages are the modules. `docs/design/modules.md` lists each with its paths, its entry file,
  its tests and the modules it depends on. A path belongs to the first row whose glob matches.
- A module issue edits only the paths of its module, as `docs/design/modules.md` and the `Files:` line of the
  issue give them. Needing another file is reported, not edited: name the file and the reason in the pull
  request description and leave it unchanged.

## Size

- A new source file stays under 600 lines. A file over 800 lines is listed with a ceiling in
  `docs/design/ratchet.json` and stays under it; a ceiling is never raised by an agent (the file is owner-only).
  When a change does not fit, report it in the pull request description; do not split the file yourself.
- `scripts/ci_ratchet.py` checks this in every pull request.

## Tests

- Every behaviour change comes with a test. Run `uv run --frozen pytest -q` and `uv run --frozen ruff check .`
  before committing.
- A test uses a temporary directory for every file it writes and never calls `time.sleep`.

## Libraries

- Prefer a mainstream, maintained library over code of our own: one that is widely used, has a stable API,
  a permissive license and few transitive dependencies. Write our own only when no such library fits, and say
  so in the pull request description ("own implementation, because ..."). Name a new direct dependency and its
  license in the description.

## Pull requests

- The `test` workflow runs the tests and, on a pull request, the ratchet. The scheduled run executes the full suite.
- Commit with the identity the history uses. No secrets, tokens, chat ids or personal data in code, commits,
  fixtures or documents.
