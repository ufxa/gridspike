"""Fig. 1 (overview) of GridSpike as a diagrams.net diagram: six numbered stages with a photo and icons, the
FedAvg/FedProx baseline as an upper branch that skips the defense, and a strip of headline results, each with its
evidence tag.
Numbers: scripts/numeros_fig1.csv (source and evidence label per value; fails if one is missing).
Photo and thumbnails: scripts/figura1_insumos.py.
Icons and photo: figures/fig1_parts/ (origin and licence in figures/fig1_parts/fontes/CREDITOS.md).
Output: figures/fig1_overview.drawio; open it in diagrams.net (or its desktop app) and export the PNG.
Usage (from paper/): python3 -I scripts/figura1_drawio.py
"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drawio_lib import CINZA, FONTE, INK, Diagrama, clarear, numeros  # noqa: E402

N = numeros("scripts/numeros_fig1.csv", exigir=[
    "nFeeders", "nClients", "nWindowsTotal", "normalShare", "nRuns", "nMsgs", "f1CnnChrono", "f1SsmChrono",
    "f1SnnChrono", "energyPct", "geoAloneAdaptive", "pipeGeoAdaptive", "ldpTauMax", "p99Gpu", "p99Share",
    "latPbft", "nValidators", "fTol", "tauRef", "nByz"])
P = Path("figures/fig1_parts")
GRAY = "#6B6B6B"   # um só cinza para todas as etapas (faixa do cabeçalho clareada a partir dele)
d = Diagrama(largura=1800, altura=920)


def svg(x, y, w, h, nome):
    """Ícone SVG embutido (vetorial) como data URI."""
    b64 = base64.b64encode((P / "icones" / nome).read_bytes()).decode()
    d.caixa(x, y, w, h, f"shape=image;html=1;imageAspect=1;aspect=fixed;image=data:image/svg+xml,{b64};")


def linha(x0, y0, x1, y1, largura=3):
    """Segmento sem ponta (a biblioteca só desenha setas)."""
    i = d._id()
    d.cells.append(f'<mxCell id="{i}" style="endArrow=none;html=1;strokeColor={INK};strokeWidth={largura};" edge="1" '
                   f'parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{x0}" y="{y0}" as="sourcePoint"/>'
                   f'<mxPoint x="{x1}" y="{y1}" as="targetPoint"/></mxGeometry></mxCell>')


# ------------------------------------------------------------------ geometria
# As pontas das setas ficam a 3 px das bordas: no editor embutido, uma aresta solta que começa ou termina exatamente
# sobre a borda de uma caixa não é desenhada na exportação.
# Dois ramos, bem separados: em cima a linha de base FedAvg/FedProx (7), que vai direto do treinamento local à
# avaliação; embaixo o ramo com defesa (3-5), agrupado numa moldura própria. O modelo global volta por baixo.
FX, FY, FW, FH = 10, 10, 1780, 712
BOX_Y, BOX_H = 26, 80                   # ramo de cima: linha de base (7)
CY, CH, CW, GAP = 146, 430, 262, 34     # cartões
XS = [29 + k * (CW + GAP) for k in range(6)]
BAND_Y, BAND_H = CY + 78, 150           # faixa da foto / dos ícones
TXT_Y = CY + 246
DEF_Y0, DEF_Y1 = CY - 14, CY + CH + 40  # moldura do ramo com defesa
Y_RET = DEF_Y1 + 30                     # retorno do modelo global

d.moldura(FX, FY, FW, FH, "Simulated OpenDSS benchmark · event awareness only: the classifier takes no part in trip decisions")

CARDS = [
    (1, "Substation telemetry", GRAY,
     [f"{N['nFeeders']} OpenDSS feeders · 60 s", "11 × 64 windows · 8 classes", f"{N['nWindowsTotal']} windows",
      f"{N['normalShare']}% normal"], ["photo illustrative · data simulated"]),
    (2, "Local training", GRAY,
     [f"{N['nClients']} substation clients", "CNN-LSTM · SNN · Mamba", "non-IID: feeder × Dirichlet",
      f"{N['nByz']} of {N['nClients']} Byzantine"], ["± local DP before submit"]),
    (3, "Fidelity scoring", GRAY,
     ["z = 1 − cos(Δᵢ, median Δ)", f"admit if z ≤ τ (τ = {N['tauRef']})", "scale-free: blind to norm"],
     ["attacker aims inside the cone"]),
    (4, "Simulated PBFT", GRAY,
     [f"M = {N['nValidators']}, f = {N['fTol']}", f"{N['nMsgs']} messages per round",
      f"{N['latPbft']} ms (simulated WAN)"], ["lower bound · fail-closed"]),
    (5, "Robust aggregation", GRAY,
     ["Krum | geometric median", "on the admitted updates", "SHA-256 hash chain"], ["audit record, not persisted"]),
    (6, "Evaluation", GRAY,
     ["chronological split", f"{N['nRuns']} seeded runs", "pre-registered H1–H5", "latency measured"],
     ["energy estimated (SynOps)"]),
]
for (num, titulo, cor, linhas, notas), x in zip(CARDS, XS):
    d.cartao(x, CY, CW, CH, num, titulo, cor=cor, linhas=linhas, notas=notas, y_texto=TXT_Y)

# imagens das etapas
d.imagem(XS[0] + 11, BAND_Y, CW - 22, (CW - 22) * 500 / 800, str(P / "substation_photo.png"), borda=True)
ICONES = {1: ["tabler-cpu.svg", "carbon-machine-learning-model.svg"], 2: ["tabler-filter-check.svg"],
          3: ["tabler-topology-full.svg"], 4: ["carbon-data-collection.svg"],
          5: ["tabler-zoom-check.svg", "carbon-chart-evaluation.svg"]}
for k, nomes in ICONES.items():
    lado = 112 if len(nomes) == 1 else 96
    total = lado * len(nomes) + 20 * (len(nomes) - 1)
    x0 = XS[k] + (CW - total) / 2
    for j, nome in enumerate(nomes):
        svg(x0 + j * (lado + 20), BAND_Y + (BAND_H - lado) / 2, lado, lado, nome)

# setas entre etapas
MID = BAND_Y + BAND_H // 2
for a, b in zip(XS[:-1], XS[1:]):
    d.seta([(a + CW + 3, MID), (b - 3, MID)])

c2, c5, c6 = XS[1] + CW // 2, XS[4] + CW // 2, XS[5] + CW // 2
bx0, bx1 = XS[2], XS[4] + CW

# ramo com defesa (3-5): moldura tracejada própria
d.caixa(bx0 - 12, DEF_Y0, bx1 - bx0 + 24, DEF_Y1 - DEF_Y0, f"rounded=1;arcSize=3;html=1;fillColor=none;"
        f"strokeColor={CINZA};strokeWidth=1.5;dashed=1;dashPattern=8 5;")
d.texto(bx0, DEF_Y1 - 34, bx1 - bx0, 26, "defended branch: admission → agreement → robust aggregation (fail-closed)",
        tam=16, cor=CINZA, italico=True)

# ramo de cima (7): linha de base FedAvg/FedProx, que pula as etapas 3-5
d.caixa(bx0, BOX_Y, bx1 - bx0, BOX_H, f"rounded=1;arcSize=12;html=1;fillColor={clarear('#6B6B6B', 0.9)};"
                                       f"strokeColor={INK};strokeWidth=1.5;")
d.caixa(bx0 + 14, BOX_Y + 20, 40, 40, f"ellipse;html=1;fillColor={INK};strokeColor=none;fontColor=#FFFFFF;{FONTE}"
                                       "fontSize=20;fontStyle=1;", "7")
svg(bx0 + 66, BOX_Y + 10, 60, 60, "tabler-topology-star-3.svg")
d.texto(bx0 + 140, BOX_Y + 8, bx1 - bx0 - 160, 32, "Baseline branch: FedAvg | FedProx", tam=20, negrito=True,
        alinhar="left")
d.texto(bx0 + 140, BOX_Y + 40, bx1 - bx0 - 160, 30, "no gate, no agreement: plain averaging of all updates",
        tam=17, alinhar="left")
ymid = BOX_Y + BOX_H // 2
linha(c2, CY - 3, c2, ymid)
d.seta([(c2, ymid), (bx0 - 3, ymid)])
linha(bx1 + 3, ymid, c6, ymid)
d.seta([(c6, ymid), (c6, CY - 3)])

# o modelo global volta da agregação para os clientes, por baixo (ou fica parado se a rodada não confirma)
linha(c5, CY + CH + 3, c5, Y_RET)
linha(c5, Y_RET, c2, Y_RET)
d.seta([(c2, Y_RET), (c2, CY + CH + 3)])
d.texto(c2 + 20, Y_RET + 4, c5 - c2 - 40, 22, "global model, or hold", tam=15, cor=CINZA, italico=True)

# ------------------------------------------------------------------ resultados com selo de evidência
OUT = [
    (f"Dense models tie: macro-F1 {N['f1CnnChrono']} vs {N['f1SsmChrono']}", "measured · chronological"),
    (f"Spiking at {N['energyPct']}% of the energy, macro-F1 {N['f1SnnChrono']}", "energy estimated"),
    (f"Adaptive attack: geo-median alone {N['geoAloneAdaptive']} vs pipeline {N['pipeGeoAdaptive']}",
     "measured · chronological"),
    (f"Local DP + gate: every client rejected at τ ≤ {N['ldpTauMax']}", "measured · random split"),
    (f"Only CNN-LSTM fits 4 ms: GPU p99 {N['p99Gpu']} ms ({N['p99Share']}%)", "measured · one machine"),
]
BW, BGAP = 329, 24
BY, BH = FY + FH + 36, 150
for k, (t, selo) in enumerate(OUT):
    x = 29 + k * (BW + BGAP)
    d.seta([(x + BW // 2, FY + FH + 3), (x + BW // 2, BY - 3)], largura=2.5)
    d.resultado(x, BY, BW, BH, t, selo, cor="#3A3A3A")

print(d.salvar("figures/fig1_overview.drawio"))
