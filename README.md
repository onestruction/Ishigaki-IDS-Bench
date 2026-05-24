# Ishigaki-IDS-Bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20362616.svg)](https://doi.org/10.5281/zenodo.20362616)

Minimal evaluation code for Ishigaki-IDS-Bench. The dataset is distributed on Hugging Face:

https://huggingface.co/datasets/ONESTRUCTION/Ishigaki-IDS-Bench

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python `3.12.7`. Dependencies are pinned in `requirements.txt`.

Set `OPENROUTER_API_KEY` for OpenRouter models.

The paper-reported Stage 1 audit metrics require the buildingSMART IDS-Audit-Tool
`ids-tool`; without it, the evaluator can still write facet scores, but the IDS
audit metrics are skipped and are not comparable to the paper. The reported
results used `ids-tool 1.0.96+e2c96c23`:

- NuGet package: https://www.nuget.org/packages/ids-tool.CommandLine/1.0.96
- Source commit: https://github.com/buildingSMART/IDS-Audit-tool/tree/e2c96c23

Check the installed version before reproducing the paper results:

```bash
ids-tool version
```

## Run

```bash
python scripts/run_eval.py --config config/eval-template.yaml
```

The template config uses the Hugging Face dataset `ONESTRUCTION/Ishigaki-IDS-Bench`.

For local pre-upload checks:

```bash
python scripts/run_eval.py --config config/eval-template.yaml --dataset-path /path/to/test.jsonl --limit 2
```

## Outputs

```text
results/predictions.jsonl
results/summary.json
results/openrouter_metadata.json
results/generated_ids/
results/gold_ids/
results/raw_outputs/
results/response_metadata/
results/failed_generations/
```

## Evaluation Notes

- If a property set is not specified, or if the input indicates that any custom property set is acceptable, the prompt instructs the model to represent `<ids:propertySet>` with the XML Schema regex pattern `^(?!(Pset_|Qto_)).+`.
- The facet scorer intentionally uses a compact comparison target and does not score every IDS attribute. For example, applicability occurrence attributes such as `minOccurs` and `maxOccurs` are checked by `ids-tool audit`, but they are not part of the facet matching score.
- Before scoring, the evaluator removes `<think>...</think>` blocks and, when the output contains a fenced code block, scores the first fenced block content as the generated IDS. Failed generations are saved for auditability; the evaluator does not repair, rerank, or regenerate outputs by default. The template uses `retries: 0` and `api_workers: 1`.

## Citation

```bibtex
@misc{kanazawa2026ishigakiidsbench,
  title = {Ishigaki-IDS-Bench: A Benchmark for Generating Information Delivery Specification from BIM Information Requirements},
  author = {Ryo Kanazawa and Koyo Hidaka and Teppei Miyamoto and Takayuki Kato and Tomoki Ando and Chenguang Wang and Dayuan Jiang and Naofumi Fujita and Shuhei Saitoh and Atomu Kondo and Koki Arakawa and Daiho Nishioka},
  year = {2026},
  eprint = {2605.22079},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2605.22079},
  url = {https://arxiv.org/abs/2605.22079}
}
```
