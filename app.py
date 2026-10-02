import io
import json
import os
import re
from pathlib import Path

import mapclassify
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ADIS comunal", layout="wide")

SIN_FILTRO = "(sin filtro)"
COL_SEL = "Personas con las características consultadas"
COL_TOTAL = "Total de personas con RSH"
COL_TOT = "Total"
COL_SI = "Sin información"
NOTA = (
    "*A partir de agosto de 2025 se cuenta con una nueva distribución de UVs para la comuna de La Serena. "
    "Tenga precaución a la hora de comparar resultados desagregados por UV de antes y después de esa fecha."
)
NOTA_RESERVA = (
    "Las celdas con reserva estadística se muestran como «1 a 9». Las cifras que suman categorías con reserva "
    "se muestran como valor aproximado (punto medio). En gráficos y mapas se usa el valor central."
)
AYUDA = """
**Cómo usar**
1. Sube la base de datos (Excel) en el recuadro «Base de datos».
2. Elige la vista: **Tablas**, **Gráficos** o **Mapas**.
3. Opcional: elige una *variable de filtro* y sus categorías, y un *tipo de apertura*.
4. Usa **Descargar** para llevarte la tabla a Excel.

**Qué significan las cifras**
- **1 a 9**: ADIS reserva las cifras bajas por secreto estadístico. En gráficos se dibujan con el valor central (5) y en mapas con gris.
- **~430**: ADIS entrega la cifra aproximada, no exacta.
- **Cifras sin «a»**: son valores publicados por ADIS. Si suman categorías con reserva, se muestran como valor aproximado.
- **Sin información**: registros sin dato en esa variable.

**Números y porcentajes**
- **% de la fila**: sobre el total de cada unidad vecinal.
- **% de la columna**: cuánto aporta cada unidad vecinal al total comunal.

**Cruces**
Solo existen los cruces descargados de ADIS (o su inverso). Si eliges uno que no existe, la app lo avisa. El detalle está en «Cruces disponibles», al final de la página.
"""
CLASIFICACIONES = ["Quiebres naturales", "Cuantiles", "Intervalos iguales"]
PALETAS_MAPA = {
    "Azules": ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
    "Amarillo a violeta": ["#fdf6c0", "#b5e3b8", "#6fc6c9", "#5aa0d6", "#6b6fc2"],
    "Verdes": ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"],
    "Naranjos": ["#feedde", "#fdbe85", "#fd8d3c", "#e6550d", "#a63603"],
    "Rojos": ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"],
    "Violetas": ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"],
}
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
COLUMNAS_BASE = ["filtro_variable", "filtro_categoria", "apertura_variable", "apertura_categoria",
                 "unidad_vecinal", "personas", "estado"]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 1rem; max-width: 100%;}
    html, body, [class*="st-"] {font-size: 0.92rem;}
    .st-key-panel_izq [data-testid="stVerticalBlock"], .st-key-panel_der [data-testid="stVerticalBlock"] {gap: 0.55rem;}
    .st-key-panel_izq label p {font-size: 0.82rem; margin-bottom: 0;}
    .st-key-panel_izq hr {margin: 0.4rem 0;}
    .st-key-panel_izq [data-baseweb="select"] > div {min-height: 2.2rem;}
    .st-key-panel_izq button, .st-key-panel_der button {min-height: 2.2rem;}
    .st-key-panel_der [data-testid="stElementToolbar"] {display:none;}
    .st-key-panel_der h1, .st-key-panel_der h2, .st-key-panel_der h3 {padding: 0.2rem 0;}
    .st-key-panel_izq button {padding: 0.25rem 0.35rem;}
    .st-key-panel_izq {background:#f3f5f8; border-top:6px solid #5b7fa8; padding:10px 12px 12px 12px;}
    .st-key-panel_der {background:#f3f5f8; border-top:6px solid #1f3a5f; padding:10px 16px 8px 16px;}
    button[kind="primary"] {background:#1f4e8c; border-color:#1f4e8c; color:#fff;}
    button[kind="primary"]:hover {background:#173c6c; border-color:#173c6c; color:#fff;}
    button[kind="secondary"] {border-color:#1f4e8c; color:#1f4e8c; background:#fff;}
    button[kind="secondary"]:hover {border-color:#173c6c; color:#173c6c; background:#e8eef6;}
    .resumen {margin: 2px 0 4px 0; line-height: 2;}
    .resumen-et {color:#4a5a70; font-size:0.82rem; margin-right:6px;}
    .caja {display:inline-block; background:#1f4e8c; color:#fff; border-radius:5px; padding:1px 9px; margin:0 5px 0 0; font-size:0.8rem;}
    .nota {color:#4a5a70; font-size:0.78rem; line-height:1.3; margin-top:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- cartografía
@st.cache_data
def cargar_geo():
    ruta = Path(__file__).parent / "uv_la_serena.geojson"
    gj = json.loads(ruta.read_text(encoding="utf-8"))
    filas = []
    for f in gj["features"]:
        xs, ys = zip(*f["geometry"]["coordinates"][0])
        p = dict(f["properties"])
        p.update(w=min(xs), e=max(xs), s=min(ys), n=max(ys))
        filas.append(p)
    return gj, pd.DataFrame(filas).set_index("codigo")


def orden_uv(uv: str):
    m = re.match(r"(\d+)(.*)", str(uv))
    return (int(m.group(1)), m.group(2)) if m else (10**9, str(uv))


# ---------------------------------------------------------------- base de datos
def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega límites inferior/superior: exacto = valor; reserva (1 a 9) = 1 y 9."""
    df = df.copy()
    df["unidad_vecinal"] = df["unidad_vecinal"].astype(str).str.strip().str.upper()
    for c in ["filtro_variable", "filtro_categoria", "apertura_variable", "apertura_categoria"]:
        df[c] = df[c].astype(str).str.strip()
    reserva = df["estado"].isin(["1 a 9", "oculto"]) | df["personas"].isna()
    df["apertura_variable"] = df["apertura_variable"].replace({"Zona urbana/rural": "Zona"})
    df["lo"] = np.where(reserva, 1.0, df["personas"])
    df["hi"] = np.where(reserva, 9.0, df["personas"])
    df["derivado"] = False
    df["ap_flag"] = (df["estado"] == "aproximado").astype(float)
    return agregar_invertidos(df)


def agregar_invertidos(df: pd.DataFrame) -> pd.DataFrame:
    """Un cruce «A filtrado × B» contiene los mismos datos que «B filtrado × A»: se agrega el inverso.
    No se calcula nada: son las mismas cifras de ADIS reubicadas. El total sale de la tabla simple de B."""
    existentes = set(map(tuple, df[["filtro_variable", "apertura_variable"]].drop_duplicates().values))
    src = df[(df["filtro_variable"] != SIN_FILTRO) & ~df["apertura_categoria"].isin([COL_TOT, COL_SI])]
    inv = src.rename(columns={
        "filtro_variable": "apertura_variable", "apertura_variable": "filtro_variable",
        "filtro_categoria": "apertura_categoria", "apertura_categoria": "filtro_categoria"})
    inv = inv[[(f, a) not in existentes for f, a in zip(inv["filtro_variable"], inv["apertura_variable"])]]
    if inv.empty:
        return df
    # fila «Total»: personas con la característica del nuevo filtro (tabla simple de esa variable)
    simple = df[(df["filtro_variable"] == SIN_FILTRO) & ~df["apertura_categoria"].isin([COL_TOT, COL_SI])]
    simple = simple.drop(columns=["filtro_variable", "filtro_categoria"]).rename(
        columns={"apertura_variable": "filtro_variable", "apertura_categoria": "filtro_categoria"})
    simple = simple[["filtro_variable", "filtro_categoria", "unidad_vecinal", "personas", "estado", "lo", "hi", "ap_flag"]]
    claves = inv[["filtro_variable", "filtro_categoria", "apertura_variable"]].drop_duplicates()
    tot = claves.merge(simple, on=["filtro_variable", "filtro_categoria"])
    tot["apertura_categoria"] = COL_TOT
    inv = inv.copy()
    inv["derivado"] = True
    tot["derivado"] = True
    return pd.concat([df, inv.reindex(columns=df.columns), tot.reindex(columns=df.columns)], ignore_index=True)


def leer_corte(archivo) -> str | None:
    """Lee el corte del RSH (p. ej. «Agosto 2026») desde la hoja «leeme» y lo devuelve como «agosto del 2026»."""
    try:
        if hasattr(archivo, "seek"):
            archivo.seek(0)
        x = pd.read_excel(archivo, sheet_name="leeme", header=None)
        fila = x[x.iloc[:, 0].astype(str).str.lower().str.contains("corte")]
        m = re.search(r"([A-Za-zÁÉÍÓÚáéíóúñÑ]+)\s+(\d{4})", str(fila.iloc[0, 1]))
        return f"{m.group(1).lower()} del {m.group(2)}" if m else None
    except Exception:
        return None
    finally:
        if hasattr(archivo, "seek"):
            archivo.seek(0)


def leer_base(archivo) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(archivo, sheet_name="datos")
    except Exception:
        st.error("No se pudo leer el archivo. Debe tener una hoja llamada «datos».")
        return None
    faltan = [c for c in COLUMNAS_BASE if c not in df.columns]
    if faltan:
        st.error(f"A la hoja «datos» le faltan columnas: {', '.join(faltan)}")
        return None
    return preparar(df)


@st.cache_data
def datos_demo() -> pd.DataFrame:
    """Base ficticia con la misma estructura que la base real."""
    _, info = cargar_geo()
    rng = np.random.default_rng(7)
    cats = {
        "Tramo CSE": ["Tramo 40", "Tramo 50", "Tramo 60", "Tramo 70", "Tramo 80", "Tramo 90", "Tramo 100"],
        "Sexo": ["Hombre", "Mujer"],
        "Nacionalidad": ["Chilena", "Extranjera"],
    }
    pares = [(SIN_FILTRO, "-", a) for a in cats] + [("Sexo", c, "Tramo CSE") for c in cats["Sexo"]] + [
        ("Nacionalidad", c, "Tramo CSE") for c in cats["Nacionalidad"]
    ] + [("Tramo CSE", c, "Sexo") for c in cats["Tramo CSE"]]
    pesos = {"Tramo CSE": [6, 5, 4, 3, 2, 1, 1], "Sexo": [1, 1.1], "Nacionalidad": [12, 1]}
    filas = []
    for uv in sorted(info.index, key=orden_uv):
        total = int(rng.integers(60, 700) if info.loc[uv, "area"] == "Rural" else rng.integers(800, 8000))
        for fv, fc, ap in pares:
            fp = (np.array(pesos[fv][cats[fv].index(fc)] / sum(pesos[fv])) if fv != SIN_FILTRO else 1.0)
            p = np.array(pesos[ap], float) / sum(pesos[ap])
            n = rng.multinomial(int(total * fp), p)
            for cat, v in zip(cats[ap], n):
                filas.append((fv, fc if fv != SIN_FILTRO else "Todas las personas", ap, cat, uv, float(v),
                              "1 a 9" if 0 < v < 10 else "exacto"))
            filas.append((fv, fc if fv != SIN_FILTRO else "Todas las personas", ap, COL_TOT, uv,
                          float(n.sum()), "exacto"))
    out = pd.DataFrame(filas, columns=COLUMNAS_BASE)
    out.loc[out["estado"] == "1 a 9", "personas"] = np.nan
    return out


def a_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return buf.getvalue()


@st.cache_data
def estructura(df: pd.DataFrame):
    """Variables de filtro, categorías y aperturas disponibles según los cruces cargados."""
    pares = df[["filtro_variable", "apertura_variable"]].drop_duplicates()
    aperturas = {}
    for fv, ap in pares.itertuples(index=False):
        aperturas.setdefault(fv, []).append(ap)
    cat_filtro, cat_ap = {}, {}
    for fv, g in df.groupby("filtro_variable", sort=False):
        cat_filtro[fv] = list(pd.unique(g["filtro_categoria"]))
    for av, g in df.groupby("apertura_variable", sort=False):
        c = [x for x in pd.unique(g["apertura_categoria"]) if x not in (COL_TOT, COL_SI)]
        cat_ap[av] = c
    # una variable que solo existe como apertura de la comuna completa también sirve de filtro de sí misma
    return aperturas, cat_filtro, cat_ap


# ---------------------------------------------------------------- cálculo
def calcular(df, fvar, fcats, ap, porcentaje, base_pct="fila"):
    """Devuelve (lo, hi) con filas = UV + S.I. + Total y columnas = categorías.
    lo.attrs["aprox"] indica las celdas que ADIS publicó con «~».
    base_pct: «fila» (sobre el total de cada UV) o «columna» (sobre el total de la columna)."""
    uvs = sorted([u for u in df["unidad_vecinal"].unique() if u not in ("S.I.", "TOTAL")], key=orden_uv)
    filas = uvs + [u for u in ["S.I."] if u in set(df["unidad_vecinal"])] + ["TOTAL"]

    base = df[(df["filtro_variable"] == SIN_FILTRO) & (df["apertura_categoria"] == COL_TOT)]
    total_rsh = base.groupby("unidad_vecinal")["lo"].first().reindex(filas)
    flag_rsh = base.groupby("unidad_vecinal")["ap_flag"].first().reindex(filas).fillna(0) > 0

    if fvar == SIN_FILTRO:
        sub = df[df["filtro_variable"] == SIN_FILTRO]
    else:
        sub = df[(df["filtro_variable"] == fvar) & (df["filtro_categoria"].isin(fcats))]

    if ap is None:
        ap0 = sub["apertura_variable"].iloc[0]
        s = sub[(sub["apertura_variable"] == ap0) & (sub["apertura_categoria"] == COL_TOT)]
        g = s.groupby("unidad_vecinal")[["lo", "hi", "ap_flag"]].sum().reindex(filas)
        lo = pd.DataFrame({COL_SEL: g["lo"], COL_TOTAL: total_rsh})
        hi = pd.DataFrame({COL_SEL: g["hi"], COL_TOTAL: total_rsh})
        flag = pd.DataFrame({COL_SEL: g["ap_flag"].fillna(0) > 0, COL_TOTAL: flag_rsh})
        den = total_rsh
    else:
        s = sub[sub["apertura_variable"] == ap]
        orden = [c for c in pd.unique(s["apertura_categoria"]) if c != COL_TOT]
        orden = [c for c in orden if c != COL_SI] + [c for c in orden if c == COL_SI] + [COL_TOT]
        g = s.groupby(["unidad_vecinal", "apertura_categoria"])[["lo", "hi", "ap_flag"]].sum()
        lo = g["lo"].unstack().reindex(index=filas, columns=orden)
        hi = g["hi"].unstack().reindex(index=filas, columns=orden)
        flag = g["ap_flag"].unstack().reindex(index=filas, columns=orden).fillna(0) > 0
        den = lo[COL_TOT]

    reserva = (hi > lo) & (lo < 10)  # cifras bajas sin valor exacto (antes de pasar a %)

    if porcentaje and base_pct == "columna":
        for c in lo.columns:
            if c == COL_TOTAL:
                continue
            d = lo.at["TOTAL", c] if "TOTAL" in lo.index and pd.notna(lo.at["TOTAL", c]) and lo.at["TOTAL", c] > 0 \
                else lo.drop(index="TOTAL", errors="ignore")[c].sum()
            lo[c], hi[c] = lo[c] / d * 100, hi[c] / d * 100
    elif porcentaje:
        d = den.replace(0, np.nan)
        for c in [c for c in lo.columns if c not in (COL_TOTAL, COL_TOT)]:
            lo[c], hi[c] = lo[c] / d * 100, hi[c] / d * 100
        if ap is not None:
            lo[COL_TOT], hi[COL_TOT] = 100.0, 100.0
    lo.index = [("Total" if i == "TOTAL" else i) for i in lo.index]
    hi.index = lo.index
    flag.index = lo.index
    reserva.index = lo.index
    lo.index.name = hi.index.name = "Unidades Vecinales"
    lo.attrs["aprox"] = flag
    lo.attrs["reserva"] = reserva
    return lo, hi


def f_num(v):
    return f"{v:,.0f}".replace(",", ".")


def f_pct(v):
    return f"{v:.1f}".replace(".", ",") + " %"


def texto_tabla(lo, hi, porcentaje):
    flag = lo.attrs.get("aprox")
    reserva = lo.attrs.get("reserva")
    out = pd.DataFrame(index=lo.index, columns=lo.columns, dtype=object)
    for c in lo.columns:
        f = f_num if c == COL_TOTAL else (f_pct if porcentaje else f_num)
        for i in lo.index:
            a, b = lo.at[i, c], hi.at[i, c]
            if pd.isna(a):
                out.at[i, c] = ""
                continue
            if abs(a - b) < 1e-9 or c == COL_TOTAL:
                t = f(a)
            else:
                es_res = bool(reserva.at[i, c]) if reserva is not None else a < 10
                if not es_res:
                    t = f_pct((a + b) / 2) if porcentaje else f_num((a + b) / 2)
                elif porcentaje:
                    t = ("< 0,1 %" if round(b, 1) == 0 else f_pct(b)) if round(a, 1) == round(b, 1) \
                        else f"{a:.1f} a {b:.1f} %".replace(".", ",")
                else:
                    t = f"{f_num(a)} a {f_num(b)}"
            aprox = flag is not None and bool(flag.at[i, c])
            es_rango = " a " in t
            out.at[i, c] = ("~" + t) if (aprox and not es_rango) else t
    out = out.rename(columns={COL_SI: COL_SI + "*"})
    out.columns.name = None
    cols = [c for c in out.columns if c not in (COL_TOT, COL_TOTAL)] + [c for c in out.columns if c in (COL_TOT, COL_TOTAL)]
    return out[cols]


# ---------------------------------------------------------------- gráficos y mapa
PALETAS = {
    "Colores": px.colors.qualitative.Set2,
    "Vivos": px.colors.qualitative.Bold,
    "Suaves": px.colors.qualitative.Pastel,
}
ORDENES = ["Unidad vecinal", "Valor (mayor a menor)", "Valor (menor a mayor)"]


def grafico(valores, apertura, porcentaje, color, paleta, orden, apilado=True):
    unidad = "%" if porcentaje else "personas"
    v = valores.copy()
    if orden != "Unidad vecinal":
        v = v.loc[v.sum(axis=1).sort_values(ascending=(orden == "Valor (menor a mayor)")).index]
    fmt_y = ",.1f" if porcentaje else ",.0f"
    fig = go.Figure()
    for i, col in enumerate(v.columns):
        nombre = "Total" if apertura is None else col
        fig.add_trace(
            go.Bar(
                x=v.index.astype(str),
                y=v[col],
                name=nombre,
                marker_color=color if apertura is None else PALETAS[paleta][i % len(PALETAS[paleta])],
                hovertemplate=f"UV %{{x}}<br>{nombre}: %{{y:{fmt_y}}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack" if apilado else "group",
        showlegend=True,
        height=620,
        margin=dict(l=10, r=10, t=20, b=60),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        xaxis=dict(type="category", tickangle=-60, rangeslider=dict(visible=True, thickness=0.07)),
        yaxis=dict(title=unidad, gridcolor="#ddd"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.28, title_text=""),
    )
    return fig


def calcular_bins(valores, metodo: str, k: int = 5):
    """Límites superiores de cada clase, calculados con los valores exactos (sin reserva)."""
    v = np.asarray(pd.Series(valores).dropna(), dtype=float)
    if v.size == 0:
        return np.array([0.0])
    k = max(1, min(k, len(np.unique(v))))
    if k == 1:
        return np.array([v.max()])
    if metodo == "Quiebres naturales":
        cl = mapclassify.NaturalBreaks(v, k=k)
    elif metodo == "Cuantiles":
        cl = mapclassify.Quantiles(v, k=k)
    else:
        cl = mapclassify.EqualInterval(v, k=k)
    return np.unique(np.asarray(cl.bins, dtype=float))


def zoom_para(w, s, e, n, px_w=900, px_h=640):
    span_lon = max(e - w, 1e-6)
    span_lat = max(n - s, 1e-6) / np.cos(np.radians((n + s) / 2))
    z_w = np.log2(px_w * 360 / (512 * span_lon))
    z_h = np.log2(px_h * 360 / (512 * span_lat))
    return float(min(z_w, z_h)) - 0.15


COLOR_RESERVA = "#cfd4da"


def colores_bins(bins, paleta_mapa):
    paleta = PALETAS_MAPA[paleta_mapa]
    return {c: paleta[i] for c, i in enumerate(np.linspace(0, 4, len(bins)).round().astype(int))}


def leyenda_bins(bins, porcentaje, paleta_mapa, con_reserva):
    fmt = f_pct if porcentaje else f_num
    unidad = 0.1 if porcentaje else 1
    col = colores_bins(bins, paleta_mapa)
    out = [(col[c], f"{fmt(0 if c == 0 else bins[c - 1] + unidad)} - {fmt(bins[c])}") for c in range(len(bins))]
    if con_reserva:
        out.append((COLOR_RESERVA, "Reserva estadística (cifra baja)"))
    return out


def texto_reserva(a, b):
    return f"{f_num(a)} a {f_num(b)}"


def mapa_tematico(serie, lo_n, hi_n, totales, porcentaje, bins, base, rotulos, zona, paleta_mapa, nombre, etq_total, altura=640):
    """serie: valor central (números o %). lo_n/hi_n: límites en número de personas.
    Las UV con reserva (cifra baja, sin valor exacto) se pintan en gris y se rotulan «1 a 9»."""
    gj, info = cargar_geo()
    idx = [c for c in serie.index if c in info.index and pd.notna(serie[c])]
    serie, lo_n, hi_n = serie[idx], lo_n[idx], hi_n[idx]
    fmt = f_pct if porcentaje else f_num
    reserva = (hi_n > lo_n) & (lo_n < 10)
    exactas = serie[~reserva]
    clase = pd.Series(np.searchsorted(bins, exactas.to_numpy(dtype=float), side="left").clip(0, len(bins) - 1), index=exactas.index)
    usadas = sorted(clase.unique())
    colores = colores_bins(bins, paleta_mapa)

    def texto(c):
        return texto_reserva(lo_n[c], hi_n[c]) if reserva[c] else fmt(serie[c])

    def hover(c):
        if reserva[c]:
            cuerpo = f"{nombre}: {texto_reserva(lo_n[c], hi_n[c])} personas"
        elif porcentaje:
            cuerpo = f"{nombre}: {fmt(serie[c])}"
        else:
            tot = totales.get(c, np.nan)
            cuerpo = f"{nombre}: {fmt(serie[c])} personas" + (f" ({serie[c] / tot * 100:.1f} %)".replace(".", ",") if tot and tot > 0 else "")
        return f"<b>UV {c} · {info.loc[c, 'nombre']}</b><br>{cuerpo}<br>{etq_total}: {f_num(totales.get(c, 0))}"

    fig = go.Figure()

    def capa(codigos, color):
        fig.add_trace(go.Choroplethmap(
            geojson=gj, featureidkey="id", locations=codigos, z=[1] * len(codigos), zmin=0, zmax=1,
            colorscale=[[0, color], [1, color]], showscale=False,
            marker=dict(line=dict(width=1, color="#444"), opacity=0.82 if base != "Sin mapa base" else 1),
            customdata=[[hover(c)] for c in codigos], hovertemplate="%{customdata[0]}<extra></extra>"))

    for c in usadas:
        capa(list(clase[clase == c].index), colores[c])
    if reserva.any():
        capa(list(reserva[reserva].index), COLOR_RESERVA)
    if rotulos:
        cods = list(serie.index)
        fig.add_trace(go.Scattermap(
            lon=info.loc[cods, "lon"], lat=info.loc[cods, "lat"], mode="text",
            text=[f"UV {c}<br>{texto(c)}" for c in cods], textfont=dict(size=10, color="#111"),
            hoverinfo="skip", showlegend=False))

    sub = info if zona == "Toda la comuna" else info[info["area"] == "Urbana"]
    w, e, s, n = sub["w"].min(), sub["e"].max(), sub["s"].min(), sub["n"].max()
    mapa = dict(center=dict(lon=(w + e) / 2, lat=(s + n) / 2), zoom=zoom_para(w, s, e, n, px_h=altura))
    if base == "Calles":
        mapa["style"] = "open-street-map"
    elif base == "Satélite":
        mapa["style"] = "white-bg"
        mapa["layers"] = [dict(below="traces", sourcetype="raster", source=[ESRI],
                               sourceattribution="Imágenes © Esri, Maxar, Earthstar Geographics")]
    else:
        mapa["style"] = "white-bg"
    fig.update_layout(map=mapa, margin=dict(l=0, r=0, t=0, b=0), height=altura, showlegend=False, uirevision=zona)
    return fig


def leyenda_html(titulo, subtitulo, clases):
    filas = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0">'
        f'<span style="width:20px;height:14px;background:{c};border:1px solid #666;display:inline-block"></span>'
        f"<span>{l}</span></div>"
        for c, l in clases
    )
    return (
        '<div style="background:#fff;border:1px solid #ccc;padding:10px 12px;font-size:13px">'
        f"<b>{titulo}</b><div style='color:#4a5a70;font-size:12px'>{subtitulo}</div>"
        f"<div style='margin-top:6px'>{filas}</div></div>"
    )


# ---------------------------------------------------------------- estado y callbacks
ss = st.session_state
ss.setdefault("vista", "Tablas")
ss.setdefault("editar", False)
ss.setdefault("_fvar", SIN_FILTRO)
ss.setdefault("_fcats", [])
if os.environ.get("ADIS_DEMO_FILTRO") and "demo_aplicado" not in ss:  # solo para pruebas
    fv, fc = os.environ["ADIS_DEMO_FILTRO"].split("=")
    ss["_fvar"], ss["_fcats"] = fv, fc.split("|")
    ss["demo_aplicado"] = True


def ir(vista):
    ss.vista = vista


def editar_consulta():
    ss.editar = True


def cerrar_consulta():
    ss.editar = False


def nueva_consulta():
    ss[f"w_cats_{ss['_fvar']}"] = []
    ss["_fvar"], ss["_fcats"] = SIN_FILTRO, []
    ss["w_fvar"] = SIN_FILTRO


def restaurar():
    ss.modo = "Números"
    ss.apertura_et = "Sin apertura"
    ss.tipo_clas = "Quiebres naturales"


def cambia_categorias():
    ss["_fcats"] = list(ss.get(f"w_cats_{ss['_fvar']}", []))


def cambia_variable():
    ss["_fvar"] = ss["w_fvar"]
    ss["_fcats"] = []


def matriz_cruces():
    filas = [SIN_FILTRO] + [v for v in TODAS if v in CAT_FILTRO]
    tabla = pd.DataFrame({
        "Filtro": ["Sin filtro" if f == SIN_FILTRO else f for f in filas],
        "Se puede abrir por": [", ".join(sorted(a for a in APERTURAS.get(f, []) if a != f)) or "—" for f in filas],
    })
    with st.expander("Cruces disponibles"):
        st.caption("Cada fila indica con qué variables se puede abrir ese filtro. Lo que no aparece no está disponible con la base actual.")
        st.dataframe(tabla, hide_index=True, width="stretch")


# ---------------------------------------------------------------- interfaz
izq, der = st.columns([1.4, 3], gap="medium")

with izq:
    with st.container(key="panel_izq"):
        z_botones = st.container()
        z_controles = st.container()
        z_acciones = st.container()
        z_carga = st.container()
        z_ayuda = st.container()

ruta_local = os.environ.get("ADIS_BASE_LOCAL")  # solo para pruebas
with z_carga:
    with st.expander("Base de datos", expanded=True):
        archivo = st.file_uploader("Subir base (Excel)", type=["xlsx"])
        df = None
        if archivo is not None:
            df = leer_base(archivo)
            ss["corte"] = leer_corte(archivo)
        elif ruta_local and Path(ruta_local).exists():
            df = leer_base(ruta_local)
            ss["corte"] = leer_corte(ruta_local)
if df is None:
    with der:
        st.markdown(
            '<div style="background:#f3f5f8;border-top:6px solid #1f3a5f;padding:28px 24px;margin-top:8px">'
            '<h3 style="margin:0 0 8px 0;color:#1f3a5f">Sube la base de datos para comenzar</h3>'
            '<p style="margin:0;color:#4a5a70">Usa el botón <b>Upload</b> del recuadro «Base de datos» '
            '(panel izquierdo) y selecciona el archivo Excel de la base ADIS. '
            'Al cargarla se mostrarán las tablas, gráficos y mapas por unidad vecinal.</p></div>',
            unsafe_allow_html=True,
        )
    st.stop()

APERTURAS, CAT_FILTRO, CAT_AP = estructura(df)
with z_ayuda:
    with st.expander("Ayuda"):
        st.markdown(AYUDA)
TODAS = sorted({v for v in df["filtro_variable"].unique() if v != SIN_FILTRO} | set(df["apertura_variable"].unique()))
fvar = ss["_fvar"] if ss["_fvar"] in APERTURAS else SIN_FILTRO
fcats = [c for c in ss["_fcats"] if c in CAT_FILTRO.get(fvar, [])]
if not fcats:
    fvar = SIN_FILTRO
ap_validas = [a for a in APERTURAS[fvar] if a in CAT_AP]
if fvar != SIN_FILTRO and fvar in ap_validas:
    ap_validas.remove(fvar)  # abrir por la misma variable del filtro no aporta
ap_todas = [v for v in TODAS if v != fvar]
if ss.get("apertura_et") not in ["Sin apertura"] + ap_todas:
    ss["apertura_et"] = "Sin apertura"

with z_botones:
    botones = st.columns(3)
    for caja, (nombre, icono) in zip(
        botones, [("Tablas", ":material/table_rows:"), ("Gráficos", ":material/bar_chart:"), ("Mapas", ":material/map:")]
    ):
        caja.button(
            nombre, icon=icono, key=f"btn_{nombre}", width="stretch",
            type="primary" if ss.vista == nombre else "secondary", on_click=ir, args=(nombre,),
        )
    st.divider()

with z_controles:
    if ss.vista == "Mapas":
        st.selectbox("Tipo de mapa", ["Mapa temático por UV"], key="tipo_mapa")
        st.selectbox("Tipo de clasificación", CLASIFICACIONES, key="tipo_clas")
        st.selectbox("Paleta de colores", list(PALETAS_MAPA), key="paleta_mapa")
    color, paleta, orden = "#1f4e8c", "Colores", ORDENES[0]
    if ss.vista == "Gráficos":
        c_col, c_ord = st.columns(2)
        with c_col.popover("Colorear", icon=":material/format_color_fill:", width="stretch"):
            color = st.color_picker("Color de las barras (sin apertura)", "#1f4e8c", key="color_barras")
            paleta = st.selectbox("Paleta (con apertura)", list(PALETAS), key="paleta")
        with c_ord.popover("Ordenar", icon=":material/sort_by_alpha:", width="stretch"):
            orden = st.radio("Ordenar por", ORDENES, key="orden_grafico")
    apilado = True
    if ss.vista == "Gráficos":
        apilado = st.radio("Tipo de gráfico", ["Barras apiladas", "Barras agrupadas"], key="tipo_grafico", horizontal=True) == "Barras apiladas"
    c_modo, c_ap = st.columns(2)
    modo = c_modo.selectbox("Números o Porcentajes", ["Números", "% de la fila", "% de la columna"], key="modo")
    apertura_et = c_ap.selectbox(
        "Tipo de Apertura", ["Sin apertura"] + ap_todas, key="apertura_et",
        format_func=lambda x: x if (x == "Sin apertura" or x in ap_validas) else f"{x} (sin datos)")
    apertura = None if apertura_et == "Sin apertura" else apertura_et
    cruce_ok = apertura is None or apertura in ap_validas
    categoria = None
    if ss.vista == "Mapas":
        if apertura and cruce_ok:
            categoria = st.selectbox("Categoría a mapear", ["Todas las categorías"] + CAT_AP[apertura],
                                     index=1, key=f"cat_{apertura}")
        base = st.selectbox("Mapa base", ["Calles", "Satélite", "Sin mapa base"], key="base_mapa")
        zona = st.radio("Zona", ["Toda la comuna", "Área urbana"], key="zona", horizontal=True)
        rotulos = st.checkbox("Mostrar rótulos", value=True, key="rotulos")

porcentaje = modo != "Números"
base_pct = "columna" if modo == "% de la columna" else "fila"
lo, hi = calcular(df, fvar, fcats, apertura, porcentaje, base_pct) if cruce_ok else (None, None)

with z_acciones:
    if cruce_ok:
        st.download_button(
            "Descargar", a_excel(texto_tabla(lo, hi, porcentaje).reset_index()), file_name="consulta_adis_comunal.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/download:", type="primary", width="stretch",
        )
    else:
        st.button("Descargar", icon=":material/download:", disabled=True, width="stretch", key="btn_desc_off")

descripcion = (", ".join(fcats) if fvar != SIN_FILTRO else "Todas las personas") + " en La Serena por unidades vecinales"
descripcion += f", según {apertura}." if apertura else "."

with der:
    with st.container(key="panel_der"):
        titulo_slot = st.container()
        with st.container():
            a, c_n, c_r = st.columns([2.2, 1, 1], vertical_alignment="bottom")
            opciones = [SIN_FILTRO] + [v for v in TODAS if v in CAT_FILTRO]
            a.selectbox("Variable de filtro", opciones, index=opciones.index(ss["_fvar"]) if ss["_fvar"] in opciones else 0,
                        key="w_fvar", on_change=cambia_variable,
                        format_func=lambda x: "Sin filtro" if x == SIN_FILTRO else x)
            c_n.button("Nueva consulta", key="btn_nueva", width="stretch", on_click=nueva_consulta)
            c_r.button("Restaurar", icon=":material/refresh:", key="btn_restaurar", on_click=restaurar, width="stretch")
            if ss["_fvar"] != SIN_FILTRO:
                st.pills("Categorías (puedes elegir varias)", CAT_FILTRO[ss["_fvar"]], selection_mode="multi",
                         default=ss["_fcats"], key=f"w_cats_{ss['_fvar']}", on_change=cambia_categorias)
                if ss["_fvar"] == "Cuidados" and len(ss["_fcats"]) > 1:
                    st.warning(
                        "Una persona puede ser cuidadora y, a la vez, requerir cuidados. ADIS la cuenta una sola vez; "
                        "esta suma la cuenta en cada categoría, por lo que el resultado puede estar sobrestimado.",
                        icon=":material/warning:")

        fvar = ss["_fvar"]
        fcats = [c for c in ss["_fcats"] if c in CAT_FILTRO.get(fvar, [])]
        if not fcats:
            fvar = SIN_FILTRO
        cruce_ok = apertura is None or apertura in [a for a in APERTURAS[fvar] if a in CAT_AP and a != fvar]
        if cruce_ok:
            lo, hi = calcular(df, fvar, fcats, apertura, porcentaje, base_pct)
        descripcion = (", ".join(fcats) if fvar != SIN_FILTRO else "Todas las personas") + " en La Serena por unidades vecinales"
        descripcion += f", según {apertura}." if apertura else "."

        fecha_txt = f" a {ss.get('corte')}" if ss.get("corte") else ""
        titulo_slot.markdown(
            f"**Personas presentes en el Registro Social de Hogares{fecha_txt} con las siguientes características:**"
        )
        cajas = ["Personas en el RSH"]
        if fvar != SIN_FILTRO:
            cajas.append(f"{fvar}: " + " o ".join(fcats))
        if apertura:
            cajas.append(f"Apertura: {apertura}")
        cajas += ["La Serena por unidades vecinales", {"Números": "Números", "% de la fila": "% del total de la fila",
                                                       "% de la columna": "% del total de la columna"}[modo]]
        titulo_slot.markdown(
            '<div class="resumen"><span class="resumen-et">Ver estadísticas de:</span>'
            + "".join(f'<span class="caja">{c}</span>' for c in cajas) + "</div>",
            unsafe_allow_html=True,
        )

        if porcentaje:
            if base_pct == "columna":
                st.caption("Porcentaje sobre el total de la columna: cuánto aporta cada unidad vecinal al total comunal.")
            else:
                st.caption(
                    "Porcentaje sobre el total de personas con RSH de cada unidad vecinal."
                    if apertura is None else "Porcentaje sobre el total de personas de la consulta en cada unidad vecinal."
                )

        if not cruce_ok:
            filtro_txt = "Sin filtro" if fvar == SIN_FILTRO else fvar
            st.markdown(
                '<div style="background:#fff;border:1px solid #c9d3e0;border-left:6px solid #1f4e8c;padding:16px 18px;margin:10px 0">'
                '<b>Este cruce no está disponible con los datos actuales.</b><br>'
                f'<span style="color:#4a5a70">«{filtro_txt}» × «{apertura}» no existe en la base cargada. '
                'Elige otra apertura u otro filtro; el detalle de lo disponible está en «Cruces disponibles», más abajo.</span></div>',
                unsafe_allow_html=True)
            matriz_cruces()
            st.stop()

        mid = (lo + hi) / 2
        excluir = [c for c in mid.columns if c in (COL_TOTAL, COL_TOT, COL_SI)]
        valores = mid.drop(index=[i for i in ["Total", "S.I."] if i in mid.index], columns=excluir)

        if ss.vista == "Tablas":
            st.dataframe(texto_tabla(lo, hi, porcentaje).reset_index(), hide_index=True, width="stretch", height=430)
        elif ss.vista == "Gráficos":
            st.plotly_chart(
                grafico(valores, apertura, porcentaje, color, paleta, orden, apilado), width="stretch",
                config={"displaylogo": False, "toImageButtonOptions": {"filename": "grafico_adis_comunal", "scale": 2}})
        else:
            lo_n, hi_n = (lo, hi) if not porcentaje else calcular(df, fvar, fcats, apertura, False)
            quitar = [i for i in ["Total", "S.I."] if i in lo_n.index]
            n_lo, n_hi = lo_n.drop(index=quitar, columns=excluir), hi_n.drop(index=quitar, columns=excluir)
            totales = (lo_n[COL_TOTAL] if COL_TOTAL in lo_n.columns else lo_n[COL_TOT]).drop(quitar).fillna(0)
            etq_total = "Total personas con RSH" if apertura is None else "Total de la consulta en la UV"
            if categoria is None:
                cats_mapa = [valores.columns[0]]
            elif categoria == "Todas las categorías":
                cats_mapa = list(valores.columns)
            else:
                cats_mapa = [categoria]
            ref = []
            for c in cats_mapa:
                res = (n_hi[c] > n_lo[c]) & (n_lo[c] < 10)
                ref.append(valores[c][~res])
            bins = calcular_bins(pd.concat(ref), ss.tipo_clas)
            sub_t = "Porcentaje sobre el total" if porcentaje else "Número de personas"
            hay_res = any(bool(((n_hi[c] > n_lo[c]) & (n_lo[c] < 10)).any()) for c in cats_mapa)
            leyenda = leyenda_bins(bins, porcentaje, ss.paleta_mapa, hay_res)
            if len(cats_mapa) == 1:
                c = cats_mapa[0]
                nombre = c if categoria else COL_SEL
                fig = mapa_tematico(valores[c], n_lo[c], n_hi[c], totales, porcentaje, bins, base, rotulos,
                                    zona, ss.paleta_mapa, nombre, etq_total)
                m1, m2 = st.columns([4, 1])
                m1.plotly_chart(fig, width="stretch")
                m2.markdown(leyenda_html(nombre, sub_t, leyenda), unsafe_allow_html=True)
            else:
                m1, m2 = st.columns([4, 1])
                with m1:
                    for k in range(0, len(cats_mapa), 3):
                        cols3 = st.columns(3)
                        for caja, c in zip(cols3, cats_mapa[k:k + 3]):
                            fig = mapa_tematico(valores[c], n_lo[c], n_hi[c], totales, porcentaje, bins, base,
                                                False, zona, ss.paleta_mapa, c, etq_total, altura=330)
                            caja.markdown(f"**{c}**")
                            caja.plotly_chart(fig, width="stretch", key=f"m_{c}")
                m2.markdown(leyenda_html("Todas las categorías", sub_t + " (misma escala en todos)", leyenda), unsafe_allow_html=True)
            st.caption("Los límites de las unidades vecinales son referenciales.")

        if fvar != SIN_FILTRO:
            fsub = df[(df["filtro_variable"] == fvar) & df["filtro_categoria"].isin(fcats) & (df["apertura_variable"] == (apertura or df["apertura_variable"].iloc[0]))]
            if apertura and fsub["derivado"].any():
                st.caption("Este cruce se obtuvo invirtiendo uno descargado de ADIS (mismas cifras); no incluye la columna «Sin información».")
        st.markdown(f'<div class="nota">{NOTA}<br>{NOTA_RESERVA}</div>', unsafe_allow_html=True)
        matriz_cruces()
