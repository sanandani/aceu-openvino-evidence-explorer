# ACE-U Evidence Explorer

Professional identity evidence is scattered across claims and artifacts, and text similarity alone cannot establish identity or trust.
This independent demo uses synthetic profiles and transparent rules to organize evidence around conceptual ACE-U dimensions.
OpenVINO runs the sentence embedding model used to retrieve relevant evidence; it does not produce a trust score.

Inspired by Shubham Anandani, “Trust-Aware Professional Identity Representation in AI-Mediated Systems,” SoutheastCon 2026. This repository is a new demonstration, not an empirical validation of the paper or framework.

## Run in 3 commands

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install -e .
streamlit run app.py
```

## Paper concepts and demo choices

| Paper concept | New demo choice in this repository |
| --- | --- |
| ACE-U: authenticity, credibility, empathy, uniqueness | Inspectable rules and illustrative thresholds organize evidence; they are not a validated score. |
| Observable evidence and provenance | Fictional profiles carry synthetic provenance labels. |
| Uncertainty and contextual explanation | Retrieval shows similarity, evidence coverage, and gaps for discussion. |
| Consent and responsible evaluation | Local demo with synthetic data; no real identities or employment decisions. |
| Professional identity representation | OpenVINO-backed semantic retrieval over a public embedding model. |

## Limitations

All data is synthetic. Embedding similarity does not verify identity, authenticity, credibility, or trustworthiness. ACE-U evidence rules are transparent illustrations, not validated measurements. This is not a hiring, credentialing, or employment tool, and the demo does not establish research validity.

## Run the app

The app uses `sentence-transformers/all-MiniLM-L6-v2` through Sentence Transformers' OpenVINO backend on CPU. OpenVINO initialization errors are surfaced as errors. The weights are downloaded to the local Hugging Face cache on first run and are not committed.

## Benchmark

Run one command to compare PyTorch CPU and OpenVINO CPU using the same model, texts, native pooling, L2 normalization, and batch sizes. It performs warm-up and repeated runs, reports p50/p95 latency, throughput, embedding cosine parity, hardware and installed versions, prints the result, and saves `benchmark_results.json`.

```bash
python -m aceu_evidence_explorer.benchmark
```

Reported numbers are machine-specific. No Intel hardware measurements are implied by a Mac run.

## Tests

```bash
pytest -q
RUN_OPENVINO_INTEGRATION=1 pytest -q tests/test_integration_openvino.py
```

## Model and code licensing

The project code is MIT licensed. The model is `sentence-transformers/all-MiniLM-L6-v2`; see its [Hugging Face model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for the checkpoint license and terms. Review those terms before redistributing model weights. The model card for this demo is in [`docs/model_card.md`](docs/model_card.md).

## Author and further reading

Author: [Shubham Anandani](https://www.imshubh.com). The articles below informed design and evaluation choices; they do not report results from this repository.

- [Quantization tradeoffs](https://www.imshubh.com/articles/intel/quantization-tradeoffs): informs treating model optimization as an explicit measured choice.
- [The conversion gap](https://www.imshubh.com/articles/intel/conversion-gap): informs checking embedding parity between execution backends.
- [Benchmarking inference](https://www.imshubh.com/articles/intel/benchmarking-inference): informs warm-up, repeated latency measurements, and throughput reporting.
- [CPU-only edge AI](https://www.imshubh.com/articles/intel/cpu-only-edge-ai): informs the local CPU inference path.
- [Production inference packaging](https://www.imshubh.com/articles/intel/production-inference-packaging): informs pinned dependencies and the clean installation workflow.

## Citation

See [`CITATION.cff`](CITATION.cff) for the paper citation and repository metadata.
