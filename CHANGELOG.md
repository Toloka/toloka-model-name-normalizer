# Changelog

All notable changes to this project are documented in this file. The format follows
[Conventional Commits](https://www.conventionalcommits.org/); the release tooling derives the version bump from them.

## Unreleased

### Feat

- rule-driven model-name normalizer: one identity (`vendor`, `model`, `route`) and rule-derived descriptive fields for
  every LLM model reference; every parsing rule lives in `rules.toml` and any deployment overrides it from outside;
  golden-corpus tests assert every field of every row.
- the default rules (schema 2, rules version `2026.10.30.1`) read the common providers with no override and no code:
  native APIs, OpenRouter and the OpenAI-compatible hosts, Azure OpenAI and AI Foundry, AWS Bedrock and OCI dotted
  ids, Vertex AI, Fireworks and Cloudflare paths, proxies in front of a route. Measured on the LiteLLM 1.93.0
  catalogue: 97.5% of its chat models (24% before).
- new rule groups `families`, `family_vendor_aliases`, `canonical`; new route keys `also_vendor`, `dotted_vendor`,
  `regions`, `skip_segments`, `skip_paths`, `strip`, `decimal_p`; new policies `vendor_case` and `family_vendor`.
- `ModelName.wire` (the name as sent after the route chain) and `Provenance.vendor_source` (where the vendor came
  from; `family` marks the one guess the default makes).

### Breaking

- a bare name takes its family's vendor (`claude-opus-4-7` is `anthropic/claude-opus-4.7`) where it raised
  `UnknownVendor`; a resolver's empty answer falls through to the families table. `policy.family_vendor = "off"`
  restores the strict reading.
- `model` is the canonical spelling: Anthropic, Z.ai and AI21 minor versions take a dot (`claude-opus-4-6` is
  `claude-opus-4.6`), and deployment detail leaves the model. Code that sends a name to a provider reads `wire` or
  `routed`. Canonical identities, `model:` tags and default slugs move for these names.
- vendors fold to their OpenRouter slug: `xai`, `zai`, `zai-org`, `moonshot`, `mistral`, `deepseek-ai`, `ibm` and the
  other aliases in `rules.toml`, the vendor's case (`Qwen` is `qwen`), and `meta` for the Llama families. A price
  stays keyed by the key as spelled: `moonshot/kimi-k2.5` and `moonshotai/kimi-k2.5` are one model at two prices.
- `routed` reproduces the input's spelling and `same_wire` compares it.
- these names are routes now, so `parse_canonical` refuses them as a vendor: `azure`, `azure_ai`, `bedrock`,
  `bedrock_mantle`, `sagemaker`, `oci`, `vertex_ai`, `vertex_ai_beta`, `watsonx`, `cloudflare`, `fireworks_ai`,
  `together_ai`, `deepinfra`, `groq`, `novita`, `nebius`, `hyperbolic`, `replicate`, `vercel_ai_gateway`,
  `anyscale`, `sambanova`, `cerebras`, `lambda_ai`, `ollama`, `ollama_chat`, `hosted_vllm`, `vllm`, `huggingface`,
  `github`, `github_copilot`, `baseten`, `nscale`, `featherless_ai`, `friendliai`, `gmi`, `wandb`, `ovhcloud`,
  `scaleway`, `llamagate`, `gradient_ai`, `publicai`, `crusoe`, `lemonade`, `heroku`, `libertai`, `tensormesh`,
  `darkbloom`, `nlp_cloud`, `nvidia_nim`, `dashscope`, `volcengine`, `meta_llama`, `pinstripes`, `litellm_proxy`,
  `codestral`, `text-completion-codestral`, `chatgpt`, `text-completion-openai`, `cohere_chat`, `amazon-nova`,
  `amazon_nova`, `palm`, `v0`. A deployment that stored identities under one of them keeps them with a lookup.
  `perplexity`, `tencent`, `databricks` and `snowflake` are routes and vendors at once.
- the default no longer carries the fields-only lookup `azure_ai/cohere-command-a-plus-05-2026`: the name reads as
  `cohere/cohere-command-a-plus-05-2026` through `azure_ai`.
