from huggingface_hub import snapshot_download
import sys

repo_id = "Rostlab/prot_t5_xl_uniref50"
local_dir = "deepStabP/models/Rostlab/prot_t5_xl_uniref50"

try:
    print("Starting snapshot_download for:", repo_id)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("snapshot_download complete")
except Exception as e:
    print("snapshot_download failed:", e)
    sys.exit(2)
