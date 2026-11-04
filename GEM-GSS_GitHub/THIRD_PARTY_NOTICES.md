# Sources and attribution

This is a compact release of the user-supplied local GEM-GSS research workspace.
It does not assert that every included concept, source or dataset is newly
original. The package does not invent a blanket MIT/Apache license for materials
whose local source did not provide that grant.

- The weighted-window numerical setting and five 1D benchmark observation arrays
  originate from the local LLM-PDESR reproduction materials. Background paper:
  [LLM-PDESR, arXiv:2607.10546](https://arxiv.org/abs/2607.10546).
  The arrays are repacked as observation-only NPZ; analytic targets and fitted
  equations are not embedded in those arrays. No separate redistribution license
  for the original arrays was identified in the supplied local files.
- The 2D observation field is the project's generated advected Fisher–KPP case
  (periodic Fourier/RK4 generation, 65×65 saved layout, 129 times). Historical
  folder naming used AllenCahn, but that is not this data's equation family.
- The expression grammar, graph construction, conservative alias logic and
  persistent set search are consolidated from the local GEM52/55/66/67 research
  implementation. Original source SHA-256 values are in `release_provenance.json`.
  Old LLM-PDESR experience-buffer vendor files are **not** required or bundled.
- The bundled tokenizer is from
  [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash),
  revision `314113f8a7dadb388215396ba5fbee2c610a04d6`.
  Uncompressed tokenizer JSON SHA-256:
  `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`.
  Compression changes storage only. This does not grant rights to model weights
  or establish that another provider uses the same tokenization/billing.
- The node-subset viewpoint is inspired by
  [GraphComBO](https://arxiv.org/abs/2405.15119). No GraphComBO code, Gaussian-process
  surrogate or acquisition optimizer is bundled.
- `docs/method_overview.png` is the English conceptual illustration generated for
  this project. It is a schematic, not a measured graph or benchmark plot.

Third-party Python libraries are installed as dependencies rather than vendored.
Their original package licenses apply. Existing source/data rights are not
replaced by this source-and-data attribution note.
