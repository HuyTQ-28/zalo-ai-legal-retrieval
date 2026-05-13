"""
Entry point — Gradio demo cho hệ thống truy vấn tài liệu pháp lý.
Chạy: python app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import gradio as gr

from ui.search_tab import build_search_tab
from ui.compare_tab import build_compare_tab

CSS = """
.gradio-container { max-width: 1200px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Zalo AI Legal Retrieval", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# Zalo AI Legal Text Retrieval
Hệ thống tìm kiếm tài liệu pháp lý tiếng Việt — BM25+ & TF-IDF
        """
    )

    with gr.Tabs():
        with gr.Tab("Tìm kiếm"):
            build_search_tab()

        with gr.Tab("So sánh phương pháp"):
            build_compare_tab()


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
