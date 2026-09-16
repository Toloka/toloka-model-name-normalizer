# Contributing to toloka-model-name-normalizer

Thanks for contributing.

## Development setup

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Python 3.10 to 3.13 are supported; CI runs the matrix, and 3.10 is the only interpreter that exercises the `tomli`
branch, so run it locally when you touch rule loading.

## Pull request guidelines

1. Keep changes scoped and atomic.
2. Add or adjust tests with every behaviour change. A rule change regenerates `tests/data/golden.json`; review the
   fixture diff together with the rule change and explain it in the PR.
3. Never add a model catalogue, prices, capabilities or per-deployment names to `rules.toml`: the shipped default is
   the grammar plus stable route conventions. Per-name exceptions go to `[lookup]` with a `why`.
4. Update `README.md` when user-facing behaviour changes.
5. Do not include private or internal data (deployment names, internal hostnames, credentials) in this repository.
6. Use [Conventional Commits](https://www.conventionalcommits.org/) for commit and PR titles (`feat(scope): ...`,
   `fix(scope): ...`, `chore: ...`); the release tooling derives the version bump and the CHANGELOG from them.

## Issues

Use the issue forms under `.github/ISSUE_TEMPLATE/` (bug, enhancement, chore). Every form asks for a priority so
the backlog stays triaged.
