import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for path in (str(ROOT_DIR), str(CURRENT_DIR), str(ROOT_DIR / "V3")):
    if path not in sys.path:
        sys.path.append(path)

from utils import discover_binaries
from utils import read_json, write_pickle

DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODEL_PATH = os.getenv("TRACE_BASE_ENCODER_PATH", "")

FEATURE_SUFFIX = "_feature.json"
SLICE_FEATURE_SUFFIXES = ("_inline_slice_feature.json",)
SLICE_FEATURE_SUFFIX = SLICE_FEATURE_SUFFIXES[0]
OUTPUT_SUFFIX = "_feature_global_embeddings.pkl"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


class GlobalEncoder:
    def __init__(self, model_path: str, gpu_id: int, max_seq_length: int, batch_size: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda(f"cuda:{gpu_id}")
        self.model = torch.nn.DataParallel(self.model, device_ids=[gpu_id])
        self.gpu_id = gpu_id
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def encode(self, texts, desc: str = "global"):
        device = torch.device(f"cuda:{self.gpu_id}")
        out = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc=desc):
            batch = texts[i:i + self.batch_size]
            res = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output = self.model(res["input_ids"].long(), attention_mask=res["attention_mask"].int())
                emb = mean_pooling(output, res["attention_mask"].int())
                emb = F.normalize(emb, p=2, dim=1)
                out.extend(emb.detach().cpu().numpy().tolist())
        return out


def _pick_source_feature(binary_path: str):
    primary = f"{binary_path}{FEATURE_SUFFIX}"
    if os.path.exists(primary):
        return primary, FEATURE_SUFFIX
    for suf in SLICE_FEATURE_SUFFIXES:
        fallback = f"{binary_path}{suf}"
        if os.path.exists(fallback):
            return fallback, suf
    return None, None


def extract_global_text(entry: dict) -> str:
    """Return the raw IDA ``pseudocode`` for a feature entry.

    Strict single-field selection — no ``modified_code`` / ``lines`` fallback.
    Returns "" when the field is missing or empty; callers treat that as a
    hard failure for the whole binary (so the pool never mixes sources).
    """
    if not isinstance(entry, dict):
        return ""
    ps = entry.get("pseudocode")
    if isinstance(ps, str) and ps:
        return ps
    return ""


def build_embeddings(
    binary_path: str,
    model_path: str,
    gpu_id: int,
    max_seq_length: int,
    batch_size: int,
    force: bool,
    output_suffix: str = OUTPUT_SUFFIX,
):
    output_path = f"{binary_path}{output_suffix}"
    if os.path.exists(output_path):
        if not force:
            return output_path
        os.remove(output_path)
    source_path, source_suffix = _pick_source_feature(binary_path)
    if source_path is None:
        print(
            f"[global-emb] ERROR {os.path.basename(binary_path)}: neither "
            f"{FEATURE_SUFFIX} nor {SLICE_FEATURE_SUFFIXES[0]} found — skipping "
            f"this binary (pipeline continues).",
            flush=True,
        )
        return None

    feature_dict = read_json(source_path)
    fvas = list(feature_dict.keys())
    texts = [extract_global_text(feature_dict[fva]) for fva in fvas]
    empty_idx = [i for i, t in enumerate(texts) if not t]
    if empty_idx:
        sample = [fvas[i] for i in empty_idx[:5]]
        print(
            f"[global-emb] FAIL {os.path.basename(binary_path)} source={source_suffix}: "
            f"{len(empty_idx)}/{len(texts)} funcs have no pseudocode — skipping this "
            f"binary to keep the encoding source consistent (pipeline continues). "
            f"First missing fvas: {sample}",
            flush=True,
        )
        return None

    encoder = GlobalEncoder(model_path, gpu_id, max_seq_length, batch_size)
    basename = os.path.basename(binary_path)
    embeddings = encoder.encode(texts, desc=f"global:{basename}")

    result = {fva: {"global": emb} for fva, emb in zip(fvas, embeddings)}
    write_pickle(result, output_path)
    print(
        f"[global-emb] done {basename} source={source_suffix} funcs={len(fvas)} out={output_path}",
        flush=True,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract whole-function global embeddings from *_feature.json (fallback: "
        "*_inline_slice_feature.json) using a single encoder (default /model/).",
    )
    parser.add_argument("-t", "--target", required=True, help="Binary file or directory.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-suffix",
        default=OUTPUT_SUFFIX,
        help=(
            f"Filename suffix for the output pickle (default: {OUTPUT_SUFFIX}). "
            "Override to keep multiple encoder caches side by side, e.g. "
            "'_feature_global_embeddings_ft.pkl' for the fine-tuned model."
        ),
    )
    args = parser.parse_args()

    output_suffix = args.output_suffix

    binaries = discover_binaries(args.target)
    missing_src = [b for b in binaries if _pick_source_feature(b)[0] is None]
    for b in missing_src:
        print(
            f"[global-emb] ERROR {os.path.basename(b)}: neither {FEATURE_SUFFIX} nor "
            f"{SLICE_FEATURE_SUFFIXES[0]} found — skipping this binary.",
            flush=True,
        )
    binaries = [b for b in binaries if _pick_source_feature(b)[0] is not None]
    if not args.force:
        binaries = [b for b in binaries if not os.path.exists(f"{b}{output_suffix}")]
    if not binaries:
        print("All global embedding files already exist. Nothing to do.")
        return

    print(
        f"[global-emb] start total={len(binaries)} model={args.model_path} gpu={args.gpu_id} "
        f"workers={args.workers} max_len={args.max_length} batch={args.batch_size} "
        f"output_suffix={output_suffix}",
        flush=True,
    )
    if args.workers <= 1:
        for binary in binaries:
            build_embeddings(
                binary,
                args.model_path,
                args.gpu_id,
                args.max_length,
                args.batch_size,
                args.force,
                output_suffix,
            )
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                build_embeddings,
                binary,
                args.model_path,
                args.gpu_id,
                args.max_length,
                args.batch_size,
                args.force,
                output_suffix,
            )
            for binary in binaries
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
