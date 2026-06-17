"""
Dashboard Interactivo - Google Play Store
EV3 · Programación para la Ciencia de Datos (SCY1101)
Integrantes: Ignacio Sobarzo · Francisca Miranda · Diego Yañez
"""

import pandas as pd
import numpy as np
import joblib
import warnings

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PALETA DE COLORES
# =============================================================================

C1 = "#cae5ff"   # Pale Sky       – fondo suave / relleno tablas
C2 = "#acedff"   # Frosted Blue   – acentos claros
C3 = "#89bbfe"   # Baby Blue Ice  – barras / líneas principales
C4 = "#6f8ab7"   # Glaucous       – barras secundarias / bordes
C5 = "#615d6c"   # Charcoal       – texto oscuro / header
C6 = "#3d5a80"   # Indigo dye     – énfasis / tabla header
C7 = "#e0fbfc"   # Columbia blue  – fondo alternativo filas tabla

# =============================================================================
# 2. DATOS Y TRANSFORMACIONES
# =============================================================================

df = pd.read_csv("googleplaystore_clean.csv")

df["Last_Updated"]      = pd.to_datetime(df["Last_Updated"], format="%d/%m/%Y")
df["dias_desde_update"] = (df["Last_Updated"].max() - df["Last_Updated"]).dt.days
df["log_reviews"]       = np.log1p(df["Reviews"])
df["log_price"]         = np.log1p(df["Price"])
df["popular"]           = (df["Installs"] > 1_000_000).astype(int)

features_num = ["Rating", "Size_MBs", "Min_Android_Ver",
                "log_reviews", "log_price", "dias_desde_update"]
features_cat = ["Category", "Type", "Content_Rating"]

X = df[features_num + features_cat]
y = df["popular"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Nulos del dataset ANTES de limpiar (simulado con columnas conocidas)
nulos_info = {
    "Rating": 1474, "Type": 1, "Content_Rating": 1,
    "Current_Ver": 8, "Android_Ver": 3
}
nulos_df = pd.DataFrame({
    "Columna": list(nulos_info.keys()),
    "Nulos":   list(nulos_info.values()),
})
nulos_df["Porcentaje"] = (nulos_df["Nulos"] / len(df) * 100).round(2)

# Preview del dataset limpio (8 columnas más representativas, 6 filas)
COLS_PREVIEW = ["App", "Category", "Rating", "Reviews", "Size_MBs",
                "Type", "Installs", "popular"]
cols_disp = [c for c in COLS_PREVIEW if c in df.columns]
df_preview = df[cols_disp].head(6).reset_index(drop=True)

# =============================================================================
# 3. MODELO Y MÉTRICAS
# =============================================================================

modelo  = joblib.load("models/trained_models/mejor_modelo.pkl")
y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

# ROC-AUC de los 4 modelos (valores reales del notebook)
modelos_auc = {
    "Regresión Logística": 0.9898,
    "Decision Tree":       0.9456,
    "Random Forest":       0.9888,
    "SVM":                 0.9871,
}

# --- Matrices de confusión reales de los 4 modelos (valores del notebook EV2)
# Formato: [[TN, FP], [FN, TP]]
matrices_confusion = {
    "Regresión Logística": [[1453, 42],  [45, 348]],
    "Decision Tree":       [[1436, 59],  [62, 331]],
    "Random Forest":       [[1457, 38],  [48, 345]],
    "SVM":                 [[1450, 45],  [53, 340]],
}

# --- Heatmap GridSearchCV — valores reales del notebook EV2
# Filas = max_depth, Columnas = n_estimators, min_samples_split=2
grid_data = {
    "max_depth":     [10,   10,   10,   20,   20,   20,   "None", "None", "None"],
    "n_estimators":  [50,  100,  200,   50,  100,  200,     50,    100,    200  ],
    "mean_f1":       [0.8801, 0.8834, 0.8846,
                      0.8823, 0.8858, 0.8869,
                      0.8823, 0.8858, 0.8869],
}
df_grid = pd.DataFrame(grid_data)

# =============================================================================
# 4. CLUSTERING Y PCA
# =============================================================================

X_cluster  = df[features_num].dropna()
idx_validos = X_cluster.index

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans   = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

pca     = PCA(n_components=2, random_state=42)
X_pca   = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

df_pca = pd.DataFrame({
    "PC1":     X_pca[:, 0],
    "PC2":     X_pca[:, 1],
    "Cluster": clusters.astype(str),
    "Popular": df.loc[idx_validos, "popular"]
                 .map({1: "Popular", 0: "No popular"}).values,
})

# Método del codo
inercias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inercias.append(km.inertia_)

# =============================================================================
# 5. GRÁFICOS ESTÁTICOS (se calculan una vez al arrancar el servidor)
# =============================================================================

LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Arial, sans-serif", color=C5),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(t=40, b=30, l=10, r=10),
)

# --- Donut FREE vs PAID
conteo_type = df["Type"].value_counts().reset_index()
conteo_type.columns = ["Tipo", "Cantidad"]
fig_donut = go.Figure(go.Pie(
    labels=conteo_type["Tipo"],
    values=conteo_type["Cantidad"],
    hole=0.5,
    marker=dict(colors=[C3, C4]),
    textinfo="label+percent",
))
fig_donut.update_layout(title="Apps Gratuitas vs De Pago",
                        height=340, showlegend=False, **LAYOUT_BASE)

# --- Nulos por columna
fig_nulos = px.bar(
    nulos_df.sort_values("Nulos"),
    x="Nulos", y="Columna", orientation="h",
    text="Porcentaje",
    color="Nulos",
    color_continuous_scale=[[0, C2], [1, C6]],
    title="Valores nulos por columna (dataset original)",
    labels={"Nulos": "Cantidad de nulos"},
)
fig_nulos.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
fig_nulos.update_layout(height=300, coloraxis_showscale=False, **LAYOUT_BASE)
fig_nulos.update_yaxes(title="")

# --- Tabla preview del dataset limpio
n_filas = len(df_preview)
color_filas = [C7 if i % 2 == 0 else "white" for i in range(n_filas)]
fig_tabla = go.Figure(go.Table(
    columnwidth=[3] + [1.5] * (len(cols_disp) - 1),
    header=dict(
        values=[f"<b>{c}</b>" for c in cols_disp],
        fill_color=C6,
        font=dict(color="white", size=11),
        align="left",
        height=32,
    ),
    cells=dict(
        values=[df_preview[c].astype(str).tolist() for c in cols_disp],
        fill_color=[color_filas],
        font=dict(color=C5, size=10),
        align="left",
        height=26,
    ),
))
fig_tabla.update_layout(
    title="Vista previa del dataset limpio (primeras 6 filas)",
    height=280,
    margin=dict(t=40, b=10, l=10, r=10),
    paper_bgcolor="white",
)

# --- Comparación ROC-AUC 4 modelos
fig_roc_comp = go.Figure(go.Bar(
    x=list(modelos_auc.values()),
    y=list(modelos_auc.keys()),
    orientation="h",
    marker=dict(color=[C3, C2, C6, C3]),
    text=[f"{v:.4f}" for v in modelos_auc.values()],
    textposition="outside",
))
fig_roc_comp.update_layout(
    title="Comparación de ROC-AUC — 4 Modelos",
    xaxis=dict(range=[0.88, 1.01], title="ROC-AUC"),
    height=300, **LAYOUT_BASE,
)
fig_roc_comp.update_yaxes(title="")

# --- Curva ROC del mejor modelo
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr, mode="lines", fill="tozeroy",
    name=f"Random Forest (AUC = {auc:.3f})",
    line=dict(color=C6, width=2),
    fillcolor=C1,
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    name="Aleatorio (AUC = 0.5)",
    line=dict(color="#cccccc", dash="dash"),
))
fig_roc.update_layout(
    title="Curva ROC — Random Forest Optimizado",
    xaxis_title="Tasa de Falsos Positivos",
    yaxis_title="Tasa de Verdaderos Positivos",
    height=300, legend=dict(x=0.45, y=0.1),
    **LAYOUT_BASE,
)

# --- Matrices de confusión 4 modelos (subplots 2x2)
nombres_modelos = list(matrices_confusion.keys())
colores_cm = [[C1, C4], [C4, C6]]   # [TN/FP fila, FN/TP fila]

fig_cm = go.Figure()
posiciones = [(0, 0), (0, 1), (1, 0), (1, 1)]   # fila, col en grid 2x2

# Usamos anotaciones manuales sobre un heatmap por subplot via make_subplots
from plotly.subplots import make_subplots

fig_cm = make_subplots(
    rows=2, cols=2,
    subplot_titles=nombres_modelos,
    horizontal_spacing=0.12,
    vertical_spacing=0.18,
)

etiquetas = ["No popular", "Popular"]
for idx, nombre in enumerate(nombres_modelos):
    fila = idx // 2 + 1
    col  = idx % 2 + 1
    cm   = matrices_confusion[nombre]
    z    = cm
    text = [[str(v) for v in row] for row in cm]

    fig_cm.add_trace(
        go.Heatmap(
            z=z,
            x=etiquetas,
            y=etiquetas,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, C1], [0.5, C3], [1, C6]],
            showscale=False,
            textfont=dict(size=13, color="white"),
        ),
        row=fila, col=col,
    )

fig_cm.update_layout(
    title="Matrices de Confusión — 4 Modelos",
    height=480,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Arial, sans-serif", color=C5),
    margin=dict(t=60, b=20, l=10, r=10),
)
for i in range(1, 5):
    fig_cm.update_xaxes(title_text="Predicho", row=(i-1)//2+1, col=(i-1)%2+1, tickfont=dict(size=10))
    fig_cm.update_yaxes(title_text="Real",     row=(i-1)//2+1, col=(i-1)%2+1, tickfont=dict(size=10))

# --- Heatmap GridSearchCV
df_pivot = df_grid.pivot_table(
    index="max_depth", columns="n_estimators", values="mean_f1"
)
fig_grid = go.Figure(go.Heatmap(
    z=df_pivot.values,
    x=[str(c) for c in df_pivot.columns],
    y=[str(r) for r in df_pivot.index],
    colorscale=[[0, C2], [0.5, C3], [1, C6]],
    text=[[f"{v:.4f}" for v in row] for row in df_pivot.values],
    texttemplate="%{text}",
    textfont=dict(size=11),
    showscale=True,
    colorbar=dict(title="F1"),
))
fig_grid.update_layout(
    title="GridSearchCV — F1 promedio por hiperparámetros (min_samples_split=2)",
    xaxis_title="n_estimators",
    yaxis_title="max_depth",
    height=300,
    **LAYOUT_BASE,
)

# --- Método del codo
fig_codo = go.Figure(go.Scatter(
    x=list(range(2, 11)), y=inercias,
    mode="lines+markers",
    line=dict(color=C6, width=2),
    marker=dict(color=C4, size=8),
))
fig_codo.add_vline(x=3, line_dash="dash", line_color=C3,
                   annotation_text="k = 3", annotation_position="top right")
fig_codo.update_layout(
    title="Método del Codo — Elección de k",
    xaxis_title="Número de clusters (k)",
    yaxis_title="Inercia",
    height=300, **LAYOUT_BASE,
)

# =============================================================================
# 5b. GRÁFICOS EDA (Estadística Descriptiva)
# =============================================================================

# Top 10 categorías por instalaciones totales
top10 = df.groupby("Category")["Installs"].sum().nlargest(10).reset_index()
top10.columns = ["Categoría", "Total Instalaciones"]
top10 = top10.sort_values("Total Instalaciones")

fig_top10 = px.bar(
    top10, x="Total Instalaciones", y="Categoría", orientation="h",
    text=top10["Total Instalaciones"].apply(
        lambda x: f"{x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.0f}M"),
    color="Total Instalaciones",
    color_continuous_scale=[[0, C2], [0.5, C3], [1, C6]],
    title="Top 10 Categorías — Total de Instalaciones",
)
fig_top10.update_traces(textposition="outside")
fig_top10.update_layout(
    height=360, coloraxis_showscale=False, **LAYOUT_BASE,
    xaxis_title="Total instalaciones", yaxis_title="",
)

# Gratis vs Pago
analisis_tipo = (
    df.groupby("Type")
    .agg(Promedio_Descargas=("Installs", "mean"), Nota_Promedio=("Rating", "mean"))
    .reset_index()
)
analisis_tipo["Etiqueta"] = analisis_tipo["Type"].map({"FREE": "GRATIS", "PAID": "PAGO"})

fig_gratis_bar = px.bar(
    analisis_tipo, x="Etiqueta", y="Promedio_Descargas", color="Etiqueta",
    text=analisis_tipo["Promedio_Descargas"].apply(lambda x: f"{x:,.0f}"),
    color_discrete_map={"GRATIS": C3, "PAGO": C4},
    title="Promedio de Descargas — Gratis vs Pago",
    labels={"Promedio_Descargas": "Promedio Descargas", "Etiqueta": ""},
)
fig_gratis_bar.update_traces(textposition="outside")
fig_gratis_bar.update_layout(height=300, showlegend=False, **LAYOUT_BASE)

fig_rating_tipo = px.bar(
    analisis_tipo, x="Etiqueta", y="Nota_Promedio", color="Etiqueta",
    text=analisis_tipo["Nota_Promedio"].apply(lambda x: f"{x:.2f} ⭐"),
    color_discrete_map={"GRATIS": C3, "PAGO": C4},
    title="Rating Promedio — Gratis vs Pago",
    labels={"Nota_Promedio": "Rating Promedio", "Etiqueta": ""},
)
fig_rating_tipo.update_traces(textposition="outside")
fig_rating_tipo.update_layout(
    height=300, showlegend=False, yaxis=dict(range=[4.0, 4.4]), **LAYOUT_BASE,
)

# Clasificación de contenido
content_counts = df["Content_Rating"].value_counts().reset_index()
content_counts.columns = ["Clasificación", "Cantidad"]

fig_content = px.bar(
    content_counts, x="Clasificación", y="Cantidad", color="Clasificación",
    text="Cantidad",
    color_discrete_sequence=[C6, C3, C4, C2, C5, C7],
    title="Apps por Clasificación de Contenido",
    labels={"Cantidad": "Nº de Apps", "Clasificación": ""},
)
fig_content.update_traces(textposition="outside")
fig_content.update_layout(height=300, showlegend=False, **LAYOUT_BASE)

# =============================================================================
# 6. ESTILOS
# =============================================================================

TAB_STYLE = {
    "borderBottom": f"1px solid {C1}",
    "padding": "10px 22px",
    "fontSize": "14px",
    "color": C5,
    "backgroundColor": "white",
}
TAB_SELECTED_STYLE = {
    "borderTop": f"3px solid {C6}",
    "borderBottom": "1px solid white",
    "backgroundColor": C1,
    "color": C5,
    "padding": "10px 22px",
    "fontSize": "14px",
    "fontWeight": "600",
}


def kpi_card(label, valor, delta=None):
    contenido = [
        html.P(label.upper(), style={
            "fontSize": "11px", "letterSpacing": "0.08em",
            "color": C6, "margin": "0 0 8px", "fontWeight": "600",
        }),
        html.P(str(valor), style={
            "fontSize": "26px", "fontWeight": "700",
            "color": C5, "margin": "0", "lineHeight": "1",
        }),
    ]
    if delta:
        contenido.append(html.P(delta, style={
            "fontSize": "11px", "color": C4,
            "margin": "5px 0 0", "fontWeight": "500",
        }))
    return html.Div(contenido, style={
        "background": "white", "borderRadius": "8px",
        "padding": "16px 18px", "border": f"1px solid {C1}",
        "borderBottom": f"3px solid {C3}", "flex": "1", "minWidth": "130px",
        "boxShadow": "0 1px 4px rgba(97,93,108,0.07)",
    })


def section_title(texto):
    return html.P(texto.upper(), style={
        "fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.1em",
        "color": C6, "margin": "20px 0 12px",
        "borderBottom": f"1px solid {C1}", "paddingBottom": "6px",
    })


def card(children):
    return html.Div(children, style={
        "background": "white", "borderRadius": "8px",
        "padding": "10px", "border": f"1px solid {C1}",
    })


# =============================================================================
# 7. APP Y LAYOUT
# =============================================================================

app = dash.Dash(__name__)
app.title = "Google Play Dashboard"

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "background": C1, "minHeight": "100vh"},
    children=[

        # Header
        html.Div([
            html.Div([
                html.H1("GOOGLE PLAY STORE", style={
                    "color": "white", "fontSize": "15px",
                    "fontWeight": "700", "letterSpacing": "0.15em", "margin": "0",
                }),
                html.P("Análisis de Popularidad de Aplicaciones · EV3 · SCY1101", style={
                    "color": C2, "fontSize": "12px", "margin": "4px 0 0",
                }),
            ]),
            html.P("Ignacio Sobarzo · Francisca Miranda · Diego Yañez", style={
                "color": C3, "fontSize": "12px",
                "fontWeight": "600", "margin": "0", "alignSelf": "center",
            }),
        ], style={
            "background": C5, "padding": "16px 30px",
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        }),

        # Tabs
        dcc.Tabs(id="tabs", value="eda",
                 style={"borderBottom": f"1px solid {C3}"},
                 children=[
                     dcc.Tab(label="Análisis Exploratorio", value="eda",
                             style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                     dcc.Tab(label="Resumen del Dataset",   value="resumen",
                             style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                     dcc.Tab(label="Preparación de Datos",  value="prep",
                             style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                     dcc.Tab(label="Resultados del Modelo", value="modelo",
                             style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                     dcc.Tab(label="Análisis de Clusters",  value="clusters",
                             style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                 ]),

        # Contenido dinámico
        html.Div(id="contenido", style={
            "maxWidth": "1200px", "margin": "0 auto", "padding": "24px 20px",
        }),

        # Footer
        html.Div(
            html.P("Dashboard · Dash + Plotly · Google Play Store (2018)", style={
                "textAlign": "center", "color": C4, "fontSize": "11px", "margin": "0",
            }),
            style={"padding": "16px", "borderTop": f"1px solid {C3}", "background": "white"},
        ),
    ]
)

# =============================================================================
# 8. CALLBACKS
# =============================================================================

@callback(Output("contenido", "children"), Input("tabs", "value"))
def render_tab(tab):

    # ── TAB 0: ANÁLISIS EXPLORATORIO ─────────────────────────────────────────
    if tab == "eda":
        return html.Div([
            section_title("Top 10 categorías por instalaciones"),
            card(dcc.Graph(figure=fig_top10, config={"displayModeBar": False})),

            section_title("Gratis vs Pago"),
            html.Div([
                html.Div(card(dcc.Graph(figure=fig_gratis_bar,  config={"displayModeBar": False})), style={"flex": "1"}),
                html.Div(card(dcc.Graph(figure=fig_rating_tipo, config={"displayModeBar": False})), style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px"}),

            section_title("Clasificación de contenido"),
            card(dcc.Graph(figure=fig_content, config={"displayModeBar": False})),
        ])

    # ── TAB 1: RESUMEN DEL DATASET ────────────────────────────────────────────
    elif tab == "resumen":
        return html.Div([
            section_title("KPIs del dataset"),
            html.Div([
                kpi_card("Total de Apps",    f"{len(df):,}",
                         delta=f"{int(df['popular'].sum()):,} son populares"),
                kpi_card("% Populares",      f"{df['popular'].mean()*100:.1f}%",
                         delta="> 1M instalaciones"),
                kpi_card("Rating Promedio",  f"{df['Rating'].mean():.2f} ⭐",
                         delta="sobre 5.0"),
                kpi_card("Precisión Modelo", f"{precision:.1%}",
                         delta="al predecir apps populares"),
                kpi_card("ROC-AUC",          f"{auc:.3f}",
                         delta="excelente discriminación"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

            section_title("Distribución del dataset"),
            html.Div([
                html.Div([
                    html.P("Filtrar por rango de Rating:",
                           style={"fontSize": "12px", "fontWeight": "600",
                                  "color": C5, "margin": "0 0 8px"}),
                    dcc.RangeSlider(
                        id="slider-rating",
                        min=1, max=5, step=0.1,
                        value=[1, 5],
                        marks={i: str(i) for i in range(1, 6)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    dcc.Graph(id="hist-rating", config={"displayModeBar": False}),
                ], style={"flex": "1", "background": "white",
                          "borderRadius": "8px", "padding": "12px",
                          "border": f"1px solid {C1}"}),
                html.Div(
                    card(dcc.Graph(figure=fig_donut, config={"displayModeBar": False})),
                    style={"flex": "1"}
                ),
            ], style={"display": "flex", "gap": "16px"}),
        ])

    # ── TAB 2: PREPARACIÓN DE DATOS ───────────────────────────────────────────
    elif tab == "prep":
        return html.Div([
            section_title("Diagnóstico de datos faltantes (dataset original)"),
            card(dcc.Graph(figure=fig_nulos, config={"displayModeBar": False})),

            section_title("Vista previa del dataset limpio"),
            card(dcc.Graph(figure=fig_tabla, config={"displayModeBar": False})),

            section_title("Transformación logarítmica en Reviews"),
            html.Div([
                html.P("Selecciona la vista:",
                       style={"fontSize": "12px", "fontWeight": "600",
                              "color": C5, "margin": "0 0 8px"}),
                dcc.RadioItems(
                    id="radio-reviews",
                    options=[
                        {"label": "  Distribución original (sesgada)", "value": "original"},
                        {"label": "  Log-transformada (simétrica)",    "value": "log"},
                    ],
                    value="original",
                    inline=True,
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"marginRight": "24px", "fontSize": "13px", "color": C5},
                ),
                dcc.Graph(id="hist-reviews", config={"displayModeBar": False}),
            ], style={
                "background": "white", "borderRadius": "8px",
                "padding": "16px", "border": f"1px solid {C1}",
            }),
        ])

    # ── TAB 3: RESULTADOS DEL MODELO ─────────────────────────────────────────
    elif tab == "modelo":
        return html.Div([
            section_title("Métricas del modelo"),
            html.Div([
                kpi_card("Accuracy",  f"{accuracy:.4f}"),
                kpi_card("F1-Score",  f"{f1:.4f}"),
                kpi_card("Precisión", f"{precision:.4f}"),
                kpi_card("Recall",    f"{recall:.4f}"),
                kpi_card("ROC-AUC",   f"{auc:.4f}"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

            section_title("Comparación de modelos y curva ROC"),
            html.Div([
                html.Div(
                    card(dcc.Graph(figure=fig_roc_comp, config={"displayModeBar": False})),
                    style={"flex": "1"},
                ),
                html.Div(
                    card(dcc.Graph(figure=fig_roc, config={"displayModeBar": False})),
                    style={"flex": "1"},
                ),
            ], style={"display": "flex", "gap": "16px"}),

            section_title("Matrices de confusión — los 4 modelos"),
            card(dcc.Graph(figure=fig_cm, config={"displayModeBar": False})),

            section_title("Optimización de hiperparámetros — GridSearchCV"),
            card(dcc.Graph(figure=fig_grid, config={"displayModeBar": False})),
        ])

    # ── TAB 4: ANÁLISIS DE CLUSTERS ───────────────────────────────────────────
    elif tab == "clusters":
        return html.Div([
            section_title("K-Means — elección del número de clusters"),
            card(dcc.Graph(figure=fig_codo, config={"displayModeBar": False})),

            section_title("Visualización PCA — clusters vs popularidad"),
            html.Div([
                html.P("Colorear por:",
                       style={"fontSize": "12px", "fontWeight": "600",
                              "color": C5, "margin": "0 0 8px"}),
                dcc.RadioItems(
                    id="radio-pca",
                    options=[
                        {"label": "  Cluster (K-Means)", "value": "cluster"},
                        {"label": "  Popularidad real",  "value": "popular"},
                    ],
                    value="cluster",
                    inline=True,
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"marginRight": "24px", "fontSize": "13px", "color": C5},
                ),
                dcc.Graph(id="scatter-pca", config={"displayModeBar": False}),
            ], style={
                "background": "white", "borderRadius": "8px",
                "padding": "16px", "border": f"1px solid {C1}",
            }),
        ])


# ── Callback histograma Rating (RangeSlider) ──────────────────────────────────
@callback(Output("hist-rating", "figure"), Input("slider-rating", "value"))
def actualizar_hist_rating(rango):
    # dropna() evita warnings con NaN en Rating
    df_f = df[
        (df["Rating"] >= rango[0]) &
        (df["Rating"] <= rango[1])
    ].dropna(subset=["Rating"])

    fig = px.histogram(
        df_f, x="Rating", nbins=30,
        color_discrete_sequence=[C3],
        title=f"Distribución de Rating  ({rango[0]:.1f} – {rango[1]:.1f})  |  {len(df_f):,} apps",
        labels={"Rating": "Rating", "count": "Frecuencia"},
    )
    fig.update_layout(height=280, **LAYOUT_BASE,
                      yaxis_title="Frecuencia", bargap=0.05)
    return fig


# ── Callback RadioButton Reviews ──────────────────────────────────────────────
@callback(Output("hist-reviews", "figure"), Input("radio-reviews", "value"))
def actualizar_reviews(opcion):
    if opcion == "original":
        col   = "Reviews"
        title = "Reviews — distribución original (muy sesgada a la derecha)"
        color = C3
    else:
        col   = "log_reviews"
        title = "log(Reviews + 1) — distribución transformada (simétrica)"
        color = C6

    fig = px.histogram(
        df, x=col, nbins=50,
        color_discrete_sequence=[color],
        title=title,
    )
    fig.update_layout(height=320, **LAYOUT_BASE,
                      yaxis_title="Frecuencia", bargap=0.03)
    return fig


# ── Callback RadioButton PCA ──────────────────────────────────────────────────
@callback(Output("scatter-pca", "figure"), Input("radio-pca", "value"))
def actualizar_pca(opcion):
    varianza = f"{var_exp.sum()*100:.1f}%"

    if opcion == "cluster":
        fig = px.scatter(
            df_pca, x="PC1", y="PC2", color="Cluster",
            color_discrete_map={"0": C2, "1": C6, "2": C3},
            opacity=0.55,
            title=f"PCA coloreado por Cluster — varianza explicada: {varianza}",
            hover_data=["Cluster"],
        )
    else:
        fig = px.scatter(
            df_pca, x="PC1", y="PC2", color="Popular",
            color_discrete_map={"Popular": C6, "No popular": C2},
            opacity=0.55,
            title=f"PCA coloreado por Popularidad real — varianza explicada: {varianza}",
            hover_data=["Popular"],
        )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=400, **LAYOUT_BASE,
                      legend=dict(orientation="h", y=-0.15))
    return fig


# =============================================================================
# 9. RUN
# =============================================================================

if __name__ == "__main__":
    print("\n  Dashboard en http://127.0.0.1:8050/\n")
    app.run(debug=False, host="127.0.0.1", port=8050)