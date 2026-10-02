import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ADIS comunal (prototipo)", layout="centered")

COL_UV = "unidad_vecinal"
COL_N = "personas"

# Etiqueta que ve el usuario -> nombre de columna en el Excel
VARIABLES = {
    "Tramo CSE": "tramo_cse",
    "Sexo": "sexo",
    "Nacionalidad": "nacionalidad",
    "Nivel de dependencia": "dependencia",
}

CATEGORIAS = {
    "tramo_cse": ["Tramo 40", "Tramo 50", "Tramo 60", "Tramo 70", "Tramo 80", "Tramo 90", "Tramo 100"],
    "sexo": ["Hombre", "Mujer"],
    "nacionalidad": ["Chilena", "Extranjera"],
    "dependencia": ["Dependencia severa", "Dependencia moderada", "Sin dependencia moderada ni severa"],
}


# ---------------------------------------------------------------- datos
@st.cache_data
def datos_demo() -> pd.DataFrame:
    """Base ficticia: una fila por unidad vecinal x combinación de categorías."""
    rng = np.random.default_rng(42)
    uvs = [str(i) for i in range(1, 41)] + [f"{i}R" for i in range(10, 20)]
    cols = list(CATEGORIAS)
    combos = pd.MultiIndex.from_product([CATEGORIAS[c] for c in cols], names=cols).to_frame(index=False)
    filas = []
    for uv in uvs:
        rural = uv.endswith("R")
        total = int(rng.integers(60, 700) if rural else rng.integers(800, 8000))
        probs = np.ones(len(combos))
        for c, base in zip(cols, [[6, 5, 4, 3, 2, 1, 1], [1, 1.1], [12, 1], [1, 3, 30]]):
            p = rng.dirichlet(np.array(base) * 3)
            probs = probs * combos[c].map(dict(zip(CATEGORIAS[c], p))).to_numpy()
        probs = probs / probs.sum()
        d = combos.copy()
        d[COL_UV] = uv
        d[COL_N] = rng.multinomial(total, probs)
        filas.append(d)
    return pd.concat(filas, ignore_index=True)[[COL_UV] + cols + [COL_N]]


def leer_excel(archivo) -> pd.DataFrame | None:
    df = pd.read_excel(archivo, dtype={COL_UV: str})
    df.columns = [str(c).strip().lower() for c in df.columns]
    faltan = [c for c in [COL_UV, COL_N] if c not in df.columns]
    if faltan:
        st.error(f"Al archivo le faltan columnas: {', '.join(faltan)}")
        return None
    df[COL_UV] = df[COL_UV].astype(str).str.strip()
    return df


def a_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return buf.getvalue()


def orden_uv(uv: str):
    m = re.match(r"(\d+)(.*)", uv)
    return (int(m.group(1)), m.group(2)) if m else (10**9, uv)


# ---------------------------------------------------------------- cálculo
def calcular(df, filtros, apertura, porcentaje):
    uvs = sorted(df[COL_UV].unique(), key=orden_uv)
    total_rsh = df.groupby(COL_UV)[COL_N].sum().reindex(uvs)

    f = df
    for col, vals in filtros.items():
        if vals:
            f = f[f[col].isin(vals)]

    if apertura is None:
        sel = f.groupby(COL_UV)[COL_N].sum().reindex(uvs, fill_value=0)
        out = pd.DataFrame({"Personas con las características consultadas": sel})
    else:
        out = (
            f.pivot_table(index=COL_UV, columns=apertura, values=COL_N, aggfunc="sum", fill_value=0)
            .reindex(index=uvs, fill_value=0)
            .reindex(columns=CATEGORIAS.get(apertura, None), fill_value=0)
        )
        out.columns.name = None

    out.index.name = "Unidad vecinal"
    out["Total de personas con RSH"] = total_rsh
    valores = out.drop(columns="Total de personas con RSH")

    if porcentaje:
        valores = valores.div(out["Total de personas con RSH"], axis=0).mul(100).round(1)

    tabla = pd.concat([valores, out[["Total de personas con RSH"]]], axis=1)

    # fila de totales
    suma = out.sum()
    if porcentaje:
        fila = (suma.drop("Total de personas con RSH") / suma["Total de personas con RSH"] * 100).round(1)
    else:
        fila = suma.drop("Total de personas con RSH")
    fila["Total de personas con RSH"] = suma["Total de personas con RSH"]
    tabla.loc["Total"] = fila
    return tabla, valores


def descripcion(filtros):
    partes = [", ".join(v) for v in filtros.values() if v]
    return ", ".join(partes) if partes else "sin filtros"


# ---------------------------------------------------------------- gráficos
def grafico(valores, apertura, porcentaje):
    unidad = "%" if porcentaje else "personas"
    top = valores.assign(_t=valores.sum(axis=1)).nlargest(15, "_t").drop(columns="_t")
    top = top.iloc[::-1]
    if apertura is None:
        fig = px.bar(top, x=top.columns[0], y=top.index.astype(str), orientation="h")
        fig.update_layout(showlegend=False)
    else:
        fig = px.bar(top, x=top.columns, y=top.index.astype(str), orientation="h", barmode="stack")
        fig.update_layout(legend_title_text="", legend_orientation="h", legend_y=-0.25)
    fig.update_layout(
        xaxis_title=unidad,
        yaxis_title="Unidad vecinal",
        yaxis_type="category",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        title="15 unidades vecinales con mayor valor",
    )
    return fig


def mapa_calor(valores, apertura, porcentaje):
    sufijo = "%" if porcentaje else ""
    if apertura is None:
        s = valores.iloc[:, 0]
        ncol = 8
        nfil = int(np.ceil(len(s) / ncol))
        z = np.full(nfil * ncol, np.nan)
        z[: len(s)] = s.to_numpy()
        txt = np.full(nfil * ncol, "", dtype=object)
        txt[: len(s)] = [f"UV {u}<br>{v:,.0f}{sufijo}".replace(",", ".") for u, v in s.items()]
        fig = go.Figure(
            go.Heatmap(
                z=z.reshape(nfil, ncol),
                text=txt.reshape(nfil, ncol),
                texttemplate="%{text}",
                textfont_size=10,
                colorscale="YlOrRd",
                xgap=2,
                ygap=2,
                showscale=True,
            )
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, autorange="reversed")
        fig.update_layout(height=max(300, 70 * nfil), margin=dict(l=5, r=5, t=30, b=5))
    else:
        fig = go.Figure(
            go.Heatmap(
                z=valores.to_numpy(),
                x=list(valores.columns),
                y=[str(i) for i in valores.index],
                colorscale="YlOrRd",
                xgap=1,
                ygap=1,
                colorbar_title=sufijo or "personas",
            )
        )
        fig.update_yaxes(type="category", autorange="reversed", title="Unidad vecinal")
        fig.update_xaxes(side="top")
        fig.update_layout(height=max(350, 16 * len(valores) + 120), margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ---------------------------------------------------------------- interfaz
st.title("ADIS comunal")
st.caption("Prototipo · Registro Social de Hogares por unidad vecinal")

archivo = st.file_uploader("Subir base (Excel)", type=["xlsx"])
df = leer_excel(archivo) if archivo else None
if archivo is None:
    df = datos_demo()
    st.warning("Estás viendo datos de demostración ficticios.")
    st.download_button(
        "Descargar plantilla de ejemplo",
        a_excel(df),
        file_name="plantilla_adis_comunal.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if df is None:
    st.stop()

variables = {k: v for k, v in VARIABLES.items() if v in df.columns}

with st.expander("Filtros", expanded=True):
    filtros = {}
    for etiqueta, col in variables.items():
        opciones = CATEGORIAS.get(col) or sorted(df[col].dropna().unique())
        filtros[col] = st.multiselect(etiqueta, opciones, placeholder="Todas")

c1, c2 = st.columns(2)
modo = c1.radio("Mostrar", ["Números", "Porcentajes"], horizontal=True)
apertura_et = c2.selectbox("Tipo de apertura", ["Sin apertura"] + list(variables))
apertura = None if apertura_et == "Sin apertura" else variables[apertura_et]
porcentaje = modo == "Porcentajes"

tabla, valores = calcular(df, filtros, apertura, porcentaje)

st.markdown(f"**Personas presentes en el RSH con las siguientes características:** {descripcion(filtros)}")
if porcentaje:
    st.caption("Porcentaje sobre el total de personas con RSH de cada unidad vecinal.")

tab_t, tab_g, tab_m = st.tabs(["Tablas", "Gráficos", "Mapa de calor"])
with tab_t:
    st.dataframe(tabla, width="stretch", height=420)
    st.download_button(
        "Descargar tabla",
        a_excel(tabla.reset_index()),
        file_name="consulta_adis_comunal.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with tab_g:
    st.plotly_chart(grafico(valores, apertura, porcentaje), width="stretch")
with tab_m:
    st.plotly_chart(mapa_calor(valores, apertura, porcentaje), width="stretch")
    st.caption("Mapa de calor por unidad vecinal. Para un mapa geográfico se necesita la cartografía de las unidades vecinales.")
