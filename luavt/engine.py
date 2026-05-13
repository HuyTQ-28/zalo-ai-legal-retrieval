from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="GreenNode/zalo-ai-legal-text-retrieval-vn",
    repo_type="dataset",
    local_dir="./data"
)
