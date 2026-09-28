"""Download the released weights / token datasets into the layout the code expects.

    python scripts/download_pretrained.py tokenizers                 # 4 tokenizers      -> $PHAEDRA_OUTPUT_ROOT/tokenizers/
    python scripts/download_pretrained.py operators                  # 21 operators      -> $PHAEDRA_OUTPUT_ROOT/operators/
    python scripts/download_pretrained.py operators --models phaedra_38m_kh fno_38m_kh
    python scripts/download_pretrained.py mae                        # 8 MAE models      -> $PHAEDRA_OUTPUT_ROOT/mae/
    python scripts/download_pretrained.py tokens --tokens phaedra    # token datasets    -> $PHAEDRA_DATA_ROOT/tokens/
    python scripts/download_pretrained.py eval phaedra_38m_kh        # everything needed to evaluate one model
    python scripts/download_pretrained.py all                        # everything (~50 GB)

The Hub layout mirrors $PHAEDRA_OUTPUT_ROOT / $PHAEDRA_DATA_ROOT, so after the
download `python -m evaluation.eval_downstream --models ...` works unchanged.
The source fields are not re-hosted: build them with scripts/prepare_poseidon_data.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hub.utils.hf import DATA_REPO, MODEL_REPO, REVISION  # noqa: E402

PROBLEMS = {"kh": "KelvinHelmholtz", "rc": "RiemannCurved", "rkh": "RiemannKelvinHelmholtz"}
FAMILIES = {  # operator stem -> (tokenizer dir, data pattern needed for evaluation)
    "phaedra_38m": ("phaedra_4x4", "tokens/phaedra/CEU2D_{P}Tokens.nc"),
    "fsq_38m": ("fsq", "tokens/fsq/CEU2D_{P}Tokens.nc"),
    "vqvae2_38m": ("vqvae2", "tokens/vqvae2/CEU2D_{P}Tokens.nc"),
    "continuous_38m": ("continuous", "latents/continuous/CEU2D_{P}Latents.nc"),
    "fno_38m": (None, None), "cno_38m": (None, None), "vit_38m": (None, None),
}
OPERATORS = [f"{s}_{d}" for s in FAMILIES for d in PROBLEMS]
TOKENIZERS = ["phaedra_4x4", "fsq", "vqvae2", "continuous"]
TOKEN_SETS = {"phaedra": "tokens/phaedra/*", "fsq": "tokens/fsq/*", "vqvae2": "tokens/vqvae2/*",
              "continuous": "latents/continuous/*"}


def _root(var: str) -> Path:
    v = os.environ.get(var)
    if not v:
        sys.exit(f"{var} is not set -- `source env.sh` first (see env.example.sh)")
    return Path(v)


def _get(repo: str, repo_type: str, patterns: list[str], local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    patterns = sorted(set(patterns))
    print(f"[download] {repo} ({repo_type}) -> {local_dir}\n           " + "\n           ".join(patterns), flush=True)
    snapshot_download(repo_id=repo, repo_type=repo_type, revision=REVISION, local_dir=str(local_dir),
                      allow_patterns=patterns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["tokenizers", "operators", "mae", "tokens", "eval", "all"])
    ap.add_argument("models", nargs="*", help="for `eval`: operator ids, e.g. phaedra_38m_kh")
    ap.add_argument("--models", dest="models_opt", nargs="*", default=None, help="operator ids (default: all 21)")
    ap.add_argument("--tokens", nargs="*", default=list(TOKEN_SETS), choices=list(TOKEN_SETS))
    args = ap.parse_args()
    out_root, data_root = _root("PHAEDRA_OUTPUT_ROOT"), None
    model_pats, data_pats = [], []
    if args.what in ("tokenizers", "all"):
        model_pats += [f"tokenizers/{t}/*" for t in TOKENIZERS]
    if args.what in ("operators", "all"):
        model_pats += [f"operators/{m}/*" for m in (args.models_opt or OPERATORS)]
    if args.what in ("mae", "all"):
        model_pats += ["mae/*"]
    if args.what in ("tokens", "all"):
        data_pats += [TOKEN_SETS[t] for t in args.tokens]
    if args.what == "eval":
        for m in args.models or args.models_opt or []:
            stem, ds = m.rsplit("_", 1)
            if stem not in FAMILIES or ds not in PROBLEMS:
                sys.exit(f"unknown operator id {m}; choose from {OPERATORS}")
            tok, data = FAMILIES[stem]
            model_pats.append(f"operators/{m}/*")
            if tok:
                model_pats.append(f"tokenizers/{tok}/*")
            if data:  # physical-space models (FNO/CNO/ViT) only need the source fields
                data_pats.append(data.format(P=PROBLEMS[ds]))
    if model_pats:
        _get(MODEL_REPO, "model", model_pats, out_root)
    if data_pats:
        data_root = _root("PHAEDRA_DATA_ROOT")
        _get(DATA_REPO, "dataset", data_pats, data_root)
    if args.what in ("eval", "all", "tokens"):
        fields = (data_root or _root("PHAEDRA_DATA_ROOT")) / "fields"
        if not any(fields.glob("CEU_2D_*LowRes.nc")):
            print("\n[download] note: the source fields are not re-hosted. Build them from the public "
                  "Poseidon datasets:\n    python scripts/prepare_poseidon_data.py --datasets CE-KH CE-CRP CE-RPUI --download")


if __name__ == "__main__":
    main()
