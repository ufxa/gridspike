"""Miniaturas dos cartões da Fig. 1 (visão geral), recortadas das figuras do próprio artigo.
Entrada: paper/figures/*.pdf (rasterizados a 300 dpi por pdftoppm). Saída: paper/figures/fig1_parts/*_thumb.png.
Nenhum dado novo: cada miniatura é um painel de uma figura que já está no manuscrito ou no suplementar.
Uso (a partir de paper/): python3 -I scripts/figura1_insumos.py
"""
import subprocess
from pathlib import Path

from PIL import Image

FIG, OUT = Path("figures"), Path("figures/fig1_parts")
OUT.mkdir(parents=True, exist_ok=True)
# figura -> recorte (esquerda, topo, direita, base) em fração da página rasterizada
RECORTES = {
    "fig_example_windows": (0.0, 0.0, 1.0, 0.215),  # NORMAL, LG
    "fig_label_histogram": (0.0, 0.0, 0.429, 0.50), # painel alpha = 0.1
    "fig_magnitude":       (0.0, 0.0, 1.0, 0.225),  # painel CNN-LSTM
    "fig_fidelity_dist":   (0.0, 0.0, 1.0, 1.0),    # figura inteira
    "fig_attack_heatmap":  (0.0, 0.0, 1.0, 0.345),  # painel CNN-LSTM (split cronológico)
}
for nome, (l, t, r, b) in RECORTES.items():
    png = OUT / f"{nome}.png"
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(FIG / f"{nome}.pdf"), str(png.with_suffix(""))],
                   check=True)
    im = Image.open(png)
    w, h = im.size
    im.crop((int(l * w), int(t * h), int(r * w), int(b * h))).save(OUT / f"{nome}_thumb.png", optimize=True)
    png.unlink()
    print(OUT / f"{nome}_thumb.png")

# Foto do cartão 1 (CC0, ver figures/fig1_parts/fontes/CREDITOS.md): recorte central em 1.6:1, 800 px de largura.
foto = Image.open(OUT / "fontes" / "substation_cc0.jpg").convert("RGB")
w, h = foto.size
alvo_h = int(w / 1.6)
topo = (h - alvo_h) // 2
foto.crop((0, topo, w, topo + alvo_h)).resize((800, 500), Image.LANCZOS).save(OUT / "substation_photo.png", optimize=True)
print(OUT / "substation_photo.png")
