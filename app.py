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
    "Las celdas con reserva estadística se muestran como «1 a 9». En gráficos y mapas se representan con 5."
)
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
    df["lo"] = np.where(reserva, 1.0, df["personas"])
    df["hi"] = np.where(reserva, 9.0, df["personas"])
    return df


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
def calcular(df, fvar, fcats, ap, porcentaje):
    """Devuelve (lo, hi) con filas = UV + S.I. + Total y columnas = categorías."""
    uvs = sorted([u for u in df["unidad_vecinal"].unique() if u not in ("S.I.", "TOTAL")], key=orden_uv)
    filas = uvs + [u for u in ["S.I."] if u in set(df["unidad_vecinal"])] + ["TOTAL"]

    base = df[(df["filtro_variable"] == SIN_FILTRO) & (df["apertura_categoria"] == COL_TOT)]
    total_rsh = base.groupby("unidad_vecinal")["lo"].first().reindex(filas)

    if fvar == SIN_FILTRO:
        sub = df[df["filtro_variable"] == SIN_FILTRO]
    else:
        sub = df[(df["filtro_variable"] == fvar) & (df["filtro_categoria"].isin(fcats))]

    if ap is None:
        ap0 = sub["apertura_variable"].iloc[0]
        s = sub[(sub["apertura_variable"] == ap0) & (sub["apertura_categoria"] == COL_TOT)]
        g = s.groupby("unidad_vecinal")[["lo", "hi"]].sum().reindex(filas)
        lo = pd.DataFrame({COL_SEL: g["lo"], COL_TOTAL: total_rsh})
        hi = pd.DataFrame({COL_SEL: g["hi"], COL_TOTAL: total_rsh})
        den = total_rsh
    else:
        s = sub[sub["apertura_variable"] == ap]
        orden = [c for c in pd.unique(s["apertura_categoria"]) if c != COL_TOT]
        orden = [c for c in orden if c != COL_SI] + [c for c in orden if c == COL_SI] + [COL_TOT]
        g = s.groupby(["unidad_vecinal", "apertura_categoria"])[["lo", "hi"]].sum()
        lo = g["lo"].unstack().reindex(index=filas, columns=orden)
        hi = g["hi"].unstack().reindex(index=filas, columns=orden)
        den = lo[COL_TOT]

    if porcentaje:
        d = den.replace(0, np.nan)
        cols = [c for c in lo.columns if c not in (COL_TOTAL,)]
        for c in cols:
            if c in (COL_TOT,):
                continue
            lo[c], hi[c] = lo[c] / d * 100, hi[c] / d * 100
        if ap is not None:
            lo[COL_TOT], hi[COL_TOT] = 100.0, 100.0
        else:
            pass
    lo.index = [("Total" if i == "TOTAL" else i) for i in lo.index]
    hi.index = lo.index
    lo.index.name = hi.index.name = "Unidades Vecinales"
    return lo, hi


def f_num(v):
    return f"{v:,.0f}".replace(",", ".")


def f_pct(v):
    return f"{v:.1f}".replace(".", ",") + " %"


def texto_tabla(lo, hi, porcentaje):
    f = f_pct if porcentaje else f_num
    out = pd.DataFrame(index=lo.index, columns=lo.columns, dtype=object)
    for c in lo.columns:
        for i in lo.index:
            a, b = lo.at[i, c], hi.at[i, c]
            if pd.isna(a):
                out.at[i, c] = ""
            elif abs(a - b) < 1e-9 or (c == COL_TOTAL):
                out.at[i, c] = f(a)
            elif porcentaje:
                out.at[i, c] = f"{a:.1f} a {b:.1f} %".replace(".", ",")
            else:
                out.at[i, c] = f"{f_num(a)} a {f_num(b)}"
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


def grafico(valores, apertura, porcentaje, color, paleta, orden):
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
        barmode="stack" if not porcentaje else "group",
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


def clasificar(serie: pd.Series, metodo: str, k: int = 5) -> pd.Series:
    v = serie.astype(float)
    k = max(1, min(k, v.nunique()))
    if k == 1:
        return pd.Series(0, index=v.index)
    if metodo == "Quiebres naturales":
        cl = mapclassify.NaturalBreaks(v.to_numpy(), k=k)
    elif metodo == "Cuantiles":
        cl = mapclassify.Quantiles(v.to_numpy(), k=k)
    else:
        cl = mapclassify.EqualInterval(v.to_numpy(), k=k)
    return pd.Series(cl.yb, index=v.index).rank(method="dense").astype(int) - 1


def zoom_para(w, s, e, n, px_w=900, px_h=640):
    span_lon = max(e - w, 1e-6)
    span_lat = max(n - s, 1e-6) / np.cos(np.radians((n + s) / 2))
    z_w = np.log2(px_w * 360 / (512 * span_lon))
    z_h = np.log2(px_h * 360 / (512 * span_lat))
    return float(min(z_w, z_h)) - 0.15


def mapa_tematico(serie, totales, porcentaje, metodo, base, rotulos, zona, paleta_mapa):
    gj, info = cargar_geo()
    serie = serie[serie.index.isin(info.index)].dropna()
    fmt = f_pct if porcentaje else f_num
    clases = clasificar(serie, metodo)
    n_clases = int(clases.max()) + 1
    colores = [PALETAS_MAPA[paleta_mapa][i] for i in np.linspace(0, 4, n_clases).round().astype(int)]

    fig = go.Figure()
    leyenda = []
    for i in range(n_clases):
        codigos = list(clases[clases == i].index)
        vals = serie[codigos]
        etiqueta = f"[{fmt(vals.min())} - {fmt(vals.max())}]"
        leyenda.append((colores[i], etiqueta))
        fig.add_trace(
            go.Choroplethmap(
                geojson=gj,
                featureidkey="id",
                locations=codigos,
                z=[1] * len(codigos),
                zmin=0,
                zmax=1,
                colorscale=[[0, colores[i]], [1, colores[i]]],
                showscale=False,
                marker=dict(line=dict(width=1, color="#444"), opacity=0.82 if base != "Sin mapa base" else 1),
                customdata=[[c, info.loc[c, "nombre"], fmt(serie[c]), f_num(totales[c])] for c in codigos],
                hovertemplate=(
                    "<b>UV %{customdata[0]} · %{customdata[1]}</b><br>"
                    "Valor: %{customdata[2]}<br>Total personas con RSH: %{customdata[3]}<extra></extra>"
                ),
                name=etiqueta,
            )
        )
    if rotulos:
        cods = list(serie.index)
        fig.add_trace(
            go.Scattermap(
                lon=info.loc[cods, "lon"],
                lat=info.loc[cods, "lat"],
                mode="text",
                text=[f"UV {c}<br>{fmt(serie[c])}" for c in cods],
                textfont=dict(size=10, color="#111"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    sub = info if zona == "Toda la comuna" else info[info["area"] == "Urbana"]
    w, e, s, n = sub["w"].min(), sub["e"].max(), sub["s"].min(), sub["n"].max()
    mapa = dict(center=dict(lon=(w + e) / 2, lat=(s + n) / 2), zoom=zoom_para(w, s, e, n))
    if base == "Calles":
        mapa["style"] = "open-street-map"
    elif base == "Satélite":
        mapa["style"] = "white-bg"
        mapa["layers"] = [dict(below="traces", sourcetype="raster", source=[ESRI],
                               sourceattribution="Imágenes © Esri, Maxar, Earthstar Geographics")]
    else:
        mapa["style"] = "white-bg"
    fig.update_layout(map=mapa, margin=dict(l=0, r=0, t=0, b=0), height=640, showlegend=False, uirevision=zona)
    return fig, leyenda


def leyenda_html(titulo, clases):
    filas = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0">'
        f'<span style="width:20px;height:14px;background:{c};border:1px solid #666;display:inline-block"></span>'
        f"<span>{l}</span></div>"
        for c, l in clases
    )
    return (
        '<div style="background:#fff;border:1px solid #ccc;padding:10px 12px;font-size:13px">'
        f"<b>{titulo}</b><div style='margin-top:6px'>{filas}</div></div>"
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


# ---------------------------------------------------------------- interfaz
izq, der = st.columns([1.4, 3], gap="medium")

with izq:
    with st.container(key="panel_izq"):
        z_botones = st.container()
        z_controles = st.container()
        z_acciones = st.container()
        z_carga = st.container()

with z_carga:
    with st.expander("Base de datos"):
        archivo = st.file_uploader("Subir base (Excel)", type=["xlsx"])
        df = None
        ruta_local = os.environ.get("ADIS_BASE_LOCAL")
        if archivo is not None:
            df = leer_base(archivo)
        elif ruta_local and Path(ruta_local).exists():
            df = leer_base(ruta_local)
            st.caption("Base cargada desde el equipo.")
        else:
            df = preparar(datos_demo())
            st.caption("Datos de demostración ficticios.")
            st.download_button(
                "Descargar base de ejemplo",
                a_excel(datos_demo()),
                file_name="base_ejemplo_adis_comunal.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
if df is None:
    st.stop()

APERTURAS, CAT_FILTRO, CAT_AP = estructura(df)
fvar = ss["_fvar"] if ss["_fvar"] in APERTURAS else SIN_FILTRO
fcats = [c for c in ss["_fcats"] if c in CAT_FILTRO.get(fvar, [])]
if not fcats:
    fvar = SIN_FILTRO
ap_validas = [a for a in APERTURAS[fvar] if a in CAT_AP]
if fvar != SIN_FILTRO and fvar in ap_validas:
    ap_validas.remove(fvar)  # abrir por la misma variable del filtro no aporta
if ss.get("apertura_et") not in ["Sin apertura"] + ap_validas:
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
    c_modo, c_ap = st.columns(2)
    modo = c_modo.selectbox("Números o Porcentajes", ["Números", "Porcentajes"], key="modo")
    apertura_et = c_ap.selectbox("Tipo de Apertura", ["Sin apertura"] + ap_validas, key="apertura_et")
    apertura = None if apertura_et == "Sin apertura" else apertura_et
    categoria = None
    if ss.vista == "Mapas":
        if apertura:
            categoria = st.selectbox("Categoría a mapear", CAT_AP[apertura], key=f"cat_{apertura}")
        base = st.selectbox("Mapa base", ["Calles", "Satélite", "Sin mapa base"], key="base_mapa")
        zona = st.radio("Zona", ["Toda la comuna", "Área urbana"], key="zona", horizontal=True)
        rotulos = st.checkbox("Mostrar rótulos", value=True, key="rotulos")

porcentaje = modo == "Porcentajes"
lo, hi = calcular(df, fvar, fcats, apertura, porcentaje)

with z_acciones:
    st.download_button(
        "Descargar", a_excel(texto_tabla(lo, hi, porcentaje).reset_index()), file_name="consulta_adis_comunal.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:", type="primary", width="stretch",
    )

descripcion = (", ".join(fcats) if fvar != SIN_FILTRO else "Todas las personas") + " en La Serena por unidades vecinales"
descripcion += f", según {apertura}." if apertura else "."

with der:
    with st.container(key="panel_der"):
        titulo_slot = st.container()
        with st.container():
            a, c_n, c_r = st.columns([2.2, 1, 1], vertical_alignment="bottom")
            opciones = [SIN_FILTRO] + [v for v in APERTURAS if v != SIN_FILTRO]
            a.selectbox("Variable de filtro", opciones, index=opciones.index(ss["_fvar"]) if ss["_fvar"] in opciones else 0,
                        key="w_fvar", on_change=cambia_variable,
                        format_func=lambda x: "Sin filtro" if x == SIN_FILTRO else x)
            c_n.button("Nueva consulta", key="btn_nueva", width="stretch", on_click=nueva_consulta)
            c_r.button("Restaurar", icon=":material/refresh:", key="btn_restaurar", on_click=restaurar, width="stretch")
            if ss["_fvar"] != SIN_FILTRO:
                st.pills("Categorías (puedes elegir varias)", CAT_FILTRO[ss["_fvar"]], selection_mode="multi",
                         default=ss["_fcats"], key=f"w_cats_{ss['_fvar']}", on_change=cambia_categorias)

        fvar = ss["_fvar"]
        fcats = [c for c in ss["_fcats"] if c in CAT_FILTRO.get(fvar, [])]
        if not fcats:
            fvar = SIN_FILTRO
        lo, hi = calcular(df, fvar, fcats, apertura, porcentaje) if (apertura is None or apertura in APERTURAS[fvar]) else (lo, hi)
        descripcion = (", ".join(fcats) if fvar != SIN_FILTRO else "Todas las personas") + " en La Serena por unidades vecinales"
        descripcion += f", según {apertura}." if apertura else "."

        titulo_slot.markdown(
            "**Personas presentes en el Registro Social de Hogares a agosto del 2026 con las siguientes características:**"
        )
        titulo_slot.markdown(f"*{descripcion}*")

        if porcentaje:
            st.caption(
                "Porcentaje sobre el total de personas con RSH de cada unidad vecinal."
                if apertura is None else "Porcentaje sobre el total de personas de la consulta en cada unidad vecinal."
            )

        mid = (lo + hi) / 2
        excluir = [c for c in mid.columns if c in (COL_TOTAL, COL_TOT, COL_SI)]
        valores = mid.drop(index=[i for i in ["Total", "S.I."] if i in mid.index], columns=excluir)

        if ss.vista == "Tablas":
            st.dataframe(texto_tabla(lo, hi, porcentaje).reset_index(), hide_index=True, width="stretch", height=430)
        elif ss.vista == "Gráficos":
            st.plotly_chart(grafico(valores, apertura, porcentaje, color, paleta, orden), width="stretch")
        else:
            serie = valores[categoria] if categoria else valores.iloc[:, 0]
            totales = (lo[COL_TOTAL] if COL_TOTAL in lo.columns else lo[COL_TOT]).drop(
                [i for i in ["Total", "S.I."] if i in lo.index])
            fig, leyenda = mapa_tematico(serie, totales.fillna(0), porcentaje, ss.tipo_clas, base, rotulos, zona, ss.paleta_mapa)
            m1, m2 = st.columns([4, 1])
            m1.plotly_chart(fig, width="stretch")
            m2.markdown(leyenda_html(categoria or COL_SEL, leyenda), unsafe_allow_html=True)
            st.caption("Los límites de las unidades vecinales son referenciales.")

        st.markdown(f'<div class="nota">{NOTA}<br>{NOTA_RESERVA}</div>', unsafe_allow_html=True)
