# Changelog

All notable changes to this project are documented in this file. The format follows
[Conventional Commits](https://www.conventionalcommits.org/); the release tooling derives the version bump from them.

## Unreleased

### Feat

- rule-driven model-name normalizer: one identity (`vendor`, `model`, `route`) and rule-derived descriptive fields for
  every LLM model reference; every parsing rule lives in `rules.toml` and any deployment overrides it from outside;
  golden-corpus tests assert every field of every row.
