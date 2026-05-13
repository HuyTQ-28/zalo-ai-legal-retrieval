"""Tab 1 — Tìm kiếm tài liệu pháp lý."""

import gradio as gr


def _format_results(results: list[dict]) -> str:
    if not results:
        return "<p style='color:gray'>Không tìm thấy kết quả.</p>"

    html = ""
    for i, r in enumerate(results, 1):
        score_color = "#2e7d32" if r["score"] > 0.3 else "#f57c00" if r["score"] > 0.1 else "#757575"
        title = r["title"] or "(Không có tiêu đề)"
        html += f"""
<div style="border:1px solid #e0e0e0; border-radius:8px; padding:14px; margin-bottom:10px; background:#fafafa;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
    <span style="font-weight:600; font-size:15px;">#{i} — {title}</span>
    <span style="background:{score_color}; color:white; padding:2px 8px; border-radius:12px; font-size:12px;">
      score: {r['score']:.4f}
    </span>
  </div>
  <div style="font-size:12px; color:#888; margin-bottom:6px;">ID: {r['doc_id']}</div>
  <div style="font-size:14px; color:#333; line-height:1.6;">{r['snippet']}</div>
</div>
"""
    return html


METHOD_MAP = {
    "BM25+ (Query Expansion)": "bm25_qe",
    "BM25+": "bm25",
    "TF-IDF": "tfidf",
}


def do_search(query: str, method: str, top_k: int) -> str:
    from api.search import search

    method_key = METHOD_MAP.get(method, "bm25_qe")
    try:
        results = search(query, method=method_key, top_k=int(top_k))
    except FileNotFoundError as e:
        return f"<p style='color:red'><b>Lỗi:</b> {e}</p>"
    except Exception as e:
        return f"<p style='color:red'><b>Lỗi không xác định:</b> {e}</p>"

    return _format_results(results)


def build_search_tab() -> None:
    gr.Markdown("## Tìm kiếm tài liệu pháp lý")
    gr.Markdown("Nhập câu hỏi hoặc từ khóa liên quan đến pháp luật Việt Nam.")

    with gr.Row():
        with gr.Column(scale=5):
            query_input = gr.Textbox(
                label="Câu truy vấn",
                placeholder="VD: Điều kiện đăng ký kinh doanh hộ cá thể là gì?",
                lines=2,
            )
        with gr.Column(scale=1):
            method_radio = gr.Radio(
                choices=["BM25+ (Query Expansion)", "BM25+", "TF-IDF"],
                value="BM25+ (Query Expansion)",
                label="Phương pháp",
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=50, value=10, step=1,
                label="Top-K kết quả",
            )

    search_btn = gr.Button("Tìm kiếm", variant="primary")

    results_html = gr.HTML(label="Kết quả")

    search_btn.click(
        fn=do_search,
        inputs=[query_input, method_radio, top_k_slider],
        outputs=results_html,
    )
    query_input.submit(
        fn=do_search,
        inputs=[query_input, method_radio, top_k_slider],
        outputs=results_html,
    )
