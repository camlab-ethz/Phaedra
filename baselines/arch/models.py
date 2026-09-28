"""Model builders used by the VQ-VAE-2 token baseline (dual-decoder, FA2 blocks)."""
from .two_stage_seq2seq import TwoStageOutput, TwoStageSeq2SeqOperator  # noqa: F401
from .dual_decoder import (  # noqa: F401
    DualDecoderConfig,
    DualDecoderSeq2SeqOperator,
    build_dual_parallel,
    build_dual_sequential,
)

BUILDERS = {
    "dual_sequential": build_dual_sequential,
    "dual_parallel": build_dual_parallel,
}
