
# 🤖 Project Guidelines & Context for Claude Code

## 🛠️ Tech Stack & Toolchain
- **Package Manager**: `uv` (Fast Python package installer and resolver)
- **Language**: Python 3.12+ (Strict Type Hints Required)
- **Testing**: `pytest` + `pytest-cov`
- **Linting/Formatting**: `ruff`
- **Version Control**: Git

## ⚡ Operational Commands
All commands must be executed via `uv`.
- **Run Tests**: `uv run pytest` (or `uv run pytest tests/test_specific.py`)
- **Lint Code**: `uv run ruff check .`
- **Format Code**: `uv run ruff format .`
- **Run App**: `uv run python src/main.py`
- **Add Dependency**: `uv add <package>` (Ask before adding!)

## 🗺️ Workflow & Planning

### When to Enter Plan Mode
- **ALWAYS** enter plan mode for non-trivial tasks (3+ steps or architectural decisions)
- If something goes sideways, **STOP** and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### Task Management Process
1. **Plan First**: Write a plan with checkable items
2. **Verify Plan**: Check in before starting implementation (use ExitPlanMode)
3. **Track Progress**: Mark items complete as you go using SetTodoList
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add a review section to the plan
6. **Capture Lessons**: Record lessons after user corrections

## 🧬 Development Standards

### Development Workflow (TDD)
You must follow the **Red-Green-Refactor** cycle:

1. **PLAN**: Briefly explain your implementation plan and file structure changes.
2. **RED**: Write a failing test case in `tests/`. Run it to confirm failure.
3. **GREEN**: Write the *minimal* code in `src/` to pass the test.
4. **REFACTOR**: Optimize code, add type hints, docstrings. Run `uv run ruff check .`.
5. **VERIFY**: Prove it works - run tests, check logs, demonstrate correctness.

### Coding Standards (Non-negotiable)

#### Type Hints & Docstrings
- **Strict Typing**: All function arguments and return values MUST have type hints.
  - *Bad*: `def process(data):`
  - *Good*: `def process(data: dict[str, Any]) -> list[int]:`
- **Docstrings**: Use Google Style docstrings for all public modules, classes, and functions.

#### Architecture & Design
- **Modularity**: Keep functions small (Single Responsibility Principle).
- **No Global State**: Avoid global variables; use dependency injection if needed.
- **Path Handling**: Always use `pathlib.Path`, never string concatenation for paths.
- **Error Handling**: Use custom exceptions in `src/exceptions.py` rather than generic `Exception`.

#### Testing Rules
- **Structure**: Use `Arrange-Act-Assert` pattern in tests.
- **Independence**: Tests must not depend on each other or external execution order.
- **Mocking**: External APIs/DBs must be mocked. Do not make real network calls during tests.

## 🧩 Subagent Strategy
Use subagents liberally to keep main context window clean:

- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution
- Use Task tool for long-running operations to avoid blocking

## ✅ Code Quality & Verification

### Verification Before Done
- **Never** mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: *"Would a staff engineer approve this?"*
- Run tests, check logs, demonstrate correctness

### Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: *"Knowing everything I know now, implement the elegant solution"*
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.

## 🔧 Autonomous Execution

### Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### Context Isolation
- Use subagents for fixes to keep main context clean
- If code doesn't work as expected, spawn a subagent to fix it
- Don't clutter main context with detailed debugging steps

## 🔄 Continuous Improvement

### Self-Improvement Loop
- After **ANY** correction from the user: record the pattern as a lesson
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### Lesson Categories
- **Process**: TDD, planning, subagent usage patterns
- **Technical**: Python specifics, architecture decisions
- **Communication**: How to ask, when to confirm, what to explain

## 📦 Git Commit Convention
Follow **Conventional Commits** format:
- `feat: description` (New feature)
- `fix: description` (Bug fix)
- `docs: description` (Documentation only)
- `refactor: description` (Code change that neither fixes a bug nor adds a feature)
- `test: description` (Adding missing tests or correcting existing tests)

## 🚫 Constraints (Don'ts)
- **DO NOT** remove existing comments unless they are obsolete.
- **DO NOT** leave `print()` statements for debugging; use `logging`.
- **DO NOT** hallucinate dependencies. If you need a library, ask permission to `uv add` it.
- **DO NOT** skip verification - prove your changes work before saying "done".

## 🔒 TDD Enforcement (MANDATORY)

### Pre-commit Hook Protection
A git pre-commit hook enforces TDD compliance automatically:

```bash
# Hook location: .github/hooks/pre-commit (symlinked to .git/hooks/pre-commit)
# Manual check: chmod +x scripts/tdd_check.py && ./scripts/tdd_check.py
```

**Commit will be BLOCKED if:**
1. ❌ Python files have syntax errors
2. ❌ `src/services/*.py` files lack corresponding `tests/services/test_*.py` files
3. ❌ `src/repositories/*.py` files lack corresponding `tests/repositories/test_*.py` files

### TDD Checklist (Mental Model)
Before every commit, verify:

- [ ] **Test File Exists**: Every new service/repository has a matching test file
- [ ] **Tests Pass**: `uv run pytest` completes successfully  
- [ ] **Code Compiles**: `uv run ruff check .` shows no syntax errors
- [ ] **Coverage Maintained**: New code has test coverage

### Bypass (Emergency Only)
```bash
# NEVER bypass in normal development
# Only for hotfixes when tests are temporarily broken
git commit --no-verify -m "HOTFIX: description"
```

### TDD Violation Recovery
If you accidentally commit without tests:
1. Immediately add test file
2. Run `uv run pytest` to verify
3. Commit with `git commit --amend` or new commit
