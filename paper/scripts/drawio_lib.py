#!/usr/bin/env python3
"""Monta diagramas do diagrams.net (draw.io) por código, para figuras de visão geral (Fig. 1 / "capa").

A figura fica editável no diagrams.net, mas é REGERADA por script: os números vêm de um CSV de macros
(e.g. scripts/numeros_fig1.csv) e as imagens dos blocos (miniaturas das figuras do próprio artigo, mapas de dados
públicos, logotipos com licença conferida) entram embutidas no arquivo.

Uso típico:
    from drawio_lib import Diagrama, numeros
    N = numeros("scripts/numeros_fig1.csv", exigir=["nClients", "f1CnnChrono"])
    d = Diagrama(largura=1800, altura=780)
    d.cartao(30, 55, 300, 410, 1, "Data", cor="#6B6B6B", linhas=[f"{N['nClients']} clients"])
    d.imagem(40, 130, 280, 180, "paper/figures/fig1_parts/map.png")
    d.seta([(330, 225), (360, 225)])
    d.salvar("paper/figures/fig1_overview.drawio")
Depois abra o .drawio no diagrams.net (ou no aplicativo desktop) e exporte o PNG.
Só biblioteca padrão.
"""
from __future__ import annotations

import base64
import csv
from pathlib import Path
from xml.sax.saxutils import escape

INK, CINZA, NAVY = "#1A1A1A", "#6B6B6B", "#1F2F4A"
FONTE = "fontFamily=Helvetica;"


def numeros(caminho: str, exigir=()):
    """Lê o CSV macro→valor gerado pelo script de números; falha se faltar alguma macro exigida."""
    n = {r["macro"]: r["valor"] for r in csv.DictReader(open(caminho, encoding="utf-8"))}
    falta = [m for m in exigir if m not in n]
    if falta:
        raise SystemExit(f"macros ausentes em {caminho}: {falta}")
    return n


def clarear(cor: str, f: float = 0.85) -> str:
    r, g, b = (int(cor[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


class Diagrama:
    def __init__(self, largura=1800, altura=780):
        self.w, self.h, self.cells, self._n = largura, altura, [], 1

    def _id(self):
        self._n += 1
        return f"c{self._n}"

    def caixa(self, x, y, w, h, estilo, valor=""):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{escape(valor, {chr(34): "&quot;"})}" style="{estilo}" vertex="1" '
                          f'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def texto(self, x, y, w, h, valor, tam=16, cor=INK, negrito=False, italico=False, alinhar="center"):
        fs = (1 if negrito else 0) + (2 if italico else 0)
        return self.caixa(x, y, w, h, f"text;html=1;{FONTE}fontSize={tam};fontColor={cor};fontStyle={fs};align={alinhar};"
                                       f"verticalAlign=middle;whiteSpace=wrap;strokeColor=none;fillColor=none;", valor)

    def seta(self, pontos, tracejada=False, largura=3):
        i = self._id()
        (x0, y0), (x1, y1) = pontos[0], pontos[-1]
        meio = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pontos[1:-1])
        st = f"endArrow=block;endFill=1;html=1;strokeColor={INK};strokeWidth={largura};rounded=0;" + ("dashed=1;" if tracejada else "")
        self.cells.append(f'<mxCell id="{i}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                          f'<mxPoint x="{x0}" y="{y0}" as="sourcePoint"/><mxPoint x="{x1}" y="{y1}" as="targetPoint"/>'
                          + (f'<Array as="points">{meio}</Array>' if meio else "") + "</mxGeometry></mxCell>")

    def imagem(self, x, y, w, h, arquivo, borda=False):
        """Embute um PNG (data URI): miniatura de figura do artigo, mapa de dados públicos, logotipo licenciado."""
        b64 = base64.b64encode(Path(arquivo).read_bytes()).decode()
        st = f"shape=image;html=1;imageAspect=1;aspect=fixed;verticalAlign=top;image=data:image/png,{b64};" + \
             (f"imageBorder={CINZA};" if borda else "")
        return self.caixa(x, y, w, h, st)

    def moldura(self, x, y, w, h, rotulo=""):
        self.caixa(x, y, w, h, f"rounded=1;arcSize=3;html=1;fillColor=none;strokeColor={CINZA};strokeWidth=1.5;dashed=1;dashPattern=8 5;")
        if rotulo:
            self.texto(x + 20, y + h - 40, w - 40, 30, rotulo, tam=16, cor=CINZA, italico=True, alinhar="left")

    def cartao(self, x, y, w, h, num, titulo, cor="#0072B2", linhas=(), notas=(), y_texto=None):
        """Cartão numerado (círculo preto + título em faixa colorida), estilo das figuras de visão geral do grupo."""
        self.caixa(x, y, w, h, f"rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor={INK};strokeWidth=1.5;")
        self.caixa(x + 1, y + 1, w - 2, 62, f"rounded=1;arcSize=10;html=1;fillColor={clarear(cor)};strokeColor=none;")
        self.caixa(x + 14, y + 12, 40, 40, f"ellipse;html=1;fillColor={INK};strokeColor=none;fontColor=#FFFFFF;{FONTE}fontSize=20;fontStyle=1;", str(num))
        self.texto(x + 60, y + 8, w - 66, 48, titulo, tam=21, negrito=True, alinhar="left")
        self.caixa(x + 1, y + 62, w - 2, 3, f"html=1;fillColor={cor};strokeColor=none;")
        yy = y_texto if y_texto is not None else y + 80
        for ln in linhas:
            if ln == "---":
                self.caixa(x + 24, yy + 4, w - 48, 1, "html=1;fillColor=#C8C8C8;strokeColor=none;")
                yy += 12
                continue
            self.texto(x + 8, yy, w - 16, 32, ln, tam=18)
            yy += 34
        for nt in notas:
            self.texto(x + 8, yy + 2, w - 16, 30, nt, tam=15.5, cor=CINZA, italico=True)
            yy += 31

    def resultado(self, x, y, w, h, texto, selo, cor=NAVY):
        """Barra escura de resultado com o selo de evidência (measured / estimated / provisional / simulated)."""
        self.caixa(x, y, w, h, f"rounded=1;arcSize=8;html=1;fillColor={cor};strokeColor=none;")
        self.texto(x + 12, y + 8, w - 24, h - 60, texto, tam=21, cor="#FFFFFF", negrito=True)
        pw = min(w - 30, 28 + 9.4 * len(selo))
        self.caixa(x + (w - pw) / 2, y + h - 48, pw, 36,
                   f"rounded=1;arcSize=50;html=1;fillColor=#FFFFFF;strokeColor=none;{FONTE}fontSize=16;fontColor={INK};", selo)

    def salvar(self, caminho):
        xml = ('<mxfile host="app.diagrams.net"><diagram id="fig" name="Fig"><mxGraphModel grid="0" page="0" '
               f'pageWidth="{self.w}" pageHeight="{self.h}" background="#FFFFFF" math="0" shadow="0"><root>'
               '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self.cells) + "</root></mxGraphModel></diagram></mxfile>")
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        Path(caminho).write_text(xml, encoding="utf-8")
        return caminho
