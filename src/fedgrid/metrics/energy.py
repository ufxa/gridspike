"""Energy model: per-operation costs and the SNN-vs-ANN energy ratio.

Constants are the community-standard 45 nm CMOS figures from M. Horowitz,
"1.1 Computing's energy problem (and what we can do about it)", IEEE ISSCC 2014,
DOI 10.1109/ISSCC.2014.6757323 (fp32 MAC 4.6 pJ = 3.7 pJ multiply + 0.9 pJ add;
one SynOp = one accumulate = 0.9 pJ; INT8 entries from the same table). Two
caveats the manuscript must carry (SOTA report §iii): the figures cover
arithmetic only — memory access and peripherals are excluded — and the result
is an *estimate via SynOps accounting*, not a hardware power measurement.
"""

from __future__ import annotations

__all__ = ["E_AC_INT8_PJ", "E_AC_PJ", "E_MAC_INT8_PJ", "E_MAC_PJ", "energy_uj"]

E_MAC_PJ: float = 4.6  # 32-bit float MAC, 45 nm (Horowitz ISSCC'14)
E_AC_PJ: float = 0.9  # 32-bit float add — one SynOp
E_MAC_INT8_PJ: float = 0.2  # 8-bit int multiply (INT8 table rows)
E_AC_INT8_PJ: float = 0.03  # 8-bit int add

_PJ_TO_UJ = 1e-6


def energy_uj(synops: float, macs: float) -> dict[str, float]:
    """``{'snn_uJ', 'ann_uJ', 'ratio'}`` from accumulated SynOps and analytic MACs.

    SNN energy = synops × E_AC (each synaptic event is one accumulate); ANN
    energy = macs × E_MAC. ``ratio`` = snn_uJ / ann_uJ — the paper's headline
    number (< 1 means the SNN is cheaper). fp32 constants; the INT8 table is
    composed by the caller from the exported ``*_INT8_PJ`` constants.
    """
    if synops < 0:
        raise ValueError(f"synops must be non-negative, got {synops}")
    if macs <= 0:
        raise ValueError(f"macs must be positive, got {macs}")
    snn = synops * E_AC_PJ * _PJ_TO_UJ
    ann = macs * E_MAC_PJ * _PJ_TO_UJ
    return {"snn_uJ": snn, "ann_uJ": ann, "ratio": snn / ann}
