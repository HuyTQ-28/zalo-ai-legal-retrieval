"""Tab 2 — So sánh kết quả 3 phương pháp cùng một truy vấn."""

import gradio as gr

METHOD_MAP = {
    "BM25+ (Query Expansion)": "bm25_qe",
    "BM25+": "bm25",
    "TF-IDF": "tfidf",
}
ALL_METHODS = list(METHOD_MAP.keys())


def _result_column(results: list[dict], other_ids: set[str]) -> str:
    if not results:
        return "<p style='color:gray'>Không có kết quả.</p>"

    html = ""
    for i, r in enumerate(results, 1):
        shared = r["doc_id"] in other_ids
        border_color = "#1565c0" if shared else "#e0e0e0"
        badge = (
            "<span style='background:#1565c0;color:white;padding:1px 6px;"
            "border-radius:10px;font-size:11px;margin-left:6px;'>chung</span>"
            if shared else ""
        )
        title = r["title"] or "(Không có tiêu đề)"
        html += f"""
<div style="border:1px solid {border_color}; border-radius:8px; padding:10px;
            margin-bottom:8px; background:{'#e8f0fe' if shared else '#fafafa'};">
  <div style="font-weight:600; font-size:13px; margin-bottom:4px;">
    #{i} {title}{badge}
  </div>
  <div style="font-size:11px; color:#888;">
    ID: {r['doc_id']} &nbsp;|&nbsp; score: {r['score']:.4f}
  </div>
  <div style="font-size:13px; color:#444; margin-top:4px; line-height:1.5;">
    {r['snippet'][:200]}{'...' if len(r['snippet']) > 200 else ''}
  </div>
</div>
"""
    return html


def do_compare(query: str, top_k: int, methods: list[str]) -> tuple[str, str, str, str]:
    from api.search import search

    if not query.strip():
        return "", "", "", ""

    if not methods:
        return "<p style='color:orange'>Vui lòng chọn ít nhất một phương pháp.</p>", "", "", ""

    results_by_method: dict[str, list[dict]] = {}
    try:
        for m in methods:
            results_by_method[m] = search(query, method=METHOD_MAP[m], top_k=int(top_k))
    except FileNotFoundError as e:
        err = f"<p style='color:red'><b>Lỗi:</b> {e}</p>"
        return err, err, err, ""
    except Exception as e:
        err = f"<p style='color:red'><b>Lỗi:</b> {e}</p>"
        return err, err, err, ""

    # IDs của tất cả các phương pháp được chọn
    ids_per_method = {m: {r["doc_id"] for r in res} for m, res in results_by_method.items()}

    # "chung" = xuất hiện ở ít nhất 2 phương pháp được chọn
    from collections import Counter
    id_count: Counter = Counter()
    for ids in ids_per_method.values():
        for doc_id in ids:
            id_count[doc_id] += 1
    shared_ids = {doc_id for doc_id, cnt in id_count.items() if cnt >= 2}

    # Build overlap summary
    overlap_parts = []
    all_selected_ids = set().union(*ids_per_method.values()) if ids_per_method else set()
    for m in methods:
        only = ids_per_method[m] - (all_selected_ids - ids_per_method[m])
        overlap_parts.append(f"{m} riêng: {len(only)}")
    overlap_info = (
        f"<div style='text-align:center; padding:8px; background:#e8f5e9; "
        f"border-radius:8px; font-size:14px;'>"
        f"Tài liệu <b>chung</b> ≥2 phương pháp (highlight xanh): <b>{len(shared_ids)}</b> / {top_k}"
        f" &nbsp;|&nbsp; " + " &nbsp;|&nbsp; ".join(overlap_parts) +
        f"</div>"
    )

    # Render 3 cột (trống nếu method không được chọn)
    cols = []
    for m in ALL_METHODS:
        if m in results_by_method:
            cols.append(_result_column(results_by_method[m], shared_ids))
        else:
            cols.append("<p style='color:#bbb;font-style:italic;'>Không được chọn</p>")

    return cols[0], cols[1], cols[2], overlap_info


def build_compare_tab() -> None:
    gr.Markdown("## So sánh các phương pháp truy vấn")
    gr.Markdown(
        "Nhập cùng một truy vấn để thấy sự khác biệt giữa các phương pháp. "
        "Tài liệu xuất hiện ở **ít nhất 2** phương pháp được đánh dấu màu xanh."
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="Câu truy vấn",
            placeholder="VD: Mức phạt vi phạm hành chính trong lĩnh vực giao thông",
            lines=2,
            scale=4,
        )
        top_k_slider = gr.Slider(
            minimum=1, maximum=30, value=10, step=1,
            label="Top-K",
            scale=1,
        )
        method_check = gr.CheckboxGroup(
            choices=ALL_METHODS,
            value=ALL_METHODS,
            label="Phương pháp so sánh",
            scale=2,
        )

    compare_btn = gr.Button("So sánh", variant="primary")
    overlap_html = gr.HTML()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### BM25+ (Query Expansion)")
            bm25qe_html = gr.HTML()
        with gr.Column():
            gr.Markdown("### BM25+")
            bm25_html = gr.HTML()
        with gr.Column():
            gr.Markdown("### TF-IDF")
            tfidf_html = gr.HTML()

    compare_btn.click(
        fn=do_compare,
        inputs=[query_input, top_k_slider, method_check],
        outputs=[bm25qe_html, bm25_html, tfidf_html, overlap_html],
    )
    query_input.submit(
        fn=do_compare,
        inputs=[query_input, top_k_slider, method_check],
        outputs=[bm25qe_html, bm25_html, tfidf_html, overlap_html],
    )
