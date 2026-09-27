import os

import pytest

from aceu_evidence_explorer.embedder import OpenVINOEmbedder


@pytest.mark.skipif(not os.getenv("RUN_OPENVINO_INTEGRATION"), reason="Optional real inference integration test; set RUN_OPENVINO_INTEGRATION=1 to run.")
def test_openvino_embedding_inference_runs():
    embedder = OpenVINOEmbedder()
    embeddings = embedder.encode(["public interest engineering", "community mentoring and systems design"])
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
