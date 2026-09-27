from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_and_has_selection_widgets():
    script = '''
import streamlit as st
from aceu_evidence_explorer import app as app_module

class FakeEmbedder:
    def __init__(self, *args, **kwargs):
        self.model_name = "fake-openvino"
        self.device = "cpu"
        self.backend_name = "openvino"

    @property
    def info(self):
        return {
            "model_id": "fake-openvino",
            "backend": "openvino",
            "device": "cpu",
            "platform": "Darwin",
            "machine": "x86_64",
        }

    def similarity(self, query_text: str, candidate_texts: list[str]):
        if not candidate_texts:
            return []
        return [0.82, 0.74, 0.68, 0.59][: len(candidate_texts)]

app_module.OpenVINOEmbedder = FakeEmbedder
app_module.main()
'''
    at = AppTest.from_string(script, default_timeout=30)
    at.run(timeout=30)

    assert at.title[0].value == "ACE-U Evidence Explorer"
    assert len(at.selectbox) >= 2
    assert "Relevant artifacts" in [item.value for item in at.subheader]
    assert not at.exception
