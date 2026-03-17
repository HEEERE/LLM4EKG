import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import NLI_CONFIG  # noqa: E402


def main() -> int:
    model_name = (NLI_CONFIG or {}).get("model_name") or "microsoft/deberta-v3-large-mnli"
    cache_dir = (NLI_CONFIG or {}).get("cache_dir")
    local_dir = (NLI_CONFIG or {}).get("local_dir")
    device = int((NLI_CONFIG or {}).get("device", -1))

    if cache_dir:
        os.makedirs(str(cache_dir), exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HF_HUB_CACHE"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

    try:
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )
    except Exception as e:
        raise RuntimeError("缺少依赖 transformers（以及 torch / sentencepiece）。请先安装再运行。") from e

    if local_dir:
        os.makedirs(str(local_dir), exist_ok=True)
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )

    load_ref = str(local_dir) if local_dir else model_name

    AutoTokenizer.from_pretrained(load_ref)
    AutoModelForSequenceClassification.from_pretrained(load_ref)

    clf = pipeline(
        "text-classification",
        model=load_ref,
        tokenizer=load_ref,
        device=device,
        truncation=True,
    )

    out = clf({"text": "A cat sits on the mat.", "text_pair": "An animal is on the mat."})
    print("model_name:", model_name)
    print("cache_dir:", cache_dir or "(default)")
    print("local_dir:", local_dir or "(none)")
    print("sample_output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
