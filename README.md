# Ciencia-_de_datos_google_playstore

##  Contenido

- [Descripción](#descripción)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Conclusiones](#conclusiones)
- [Limitaciones](#limitaciones)

---

##  Descripción

Este proyecto implementa un **análisis exhaustivo** del dataset de Google Play Store con el objetivo de:

1. **Explorar y entender** los datos de 9,438 aplicaciones
2. **Construir modelos de clasificación** para predecir popularidad
3. **Aplicar técnicas de clustering** para descubrir patrones no supervisados
4. **Visualizar resultados** mediante un dashboard interactivo

### Objetivo Principal
Predecir si una app será **popular** (>1,000,000 instalaciones) usando:
- **Modelos supervisados**: Regresión Logística, Decision Tree, Random Forest, SVM
- **Aprendizaje no supervisado**: K-Means + PCA
- **Optimización**: GridSearchCV y RandomizedSearchCV

---

##  Estructura del Proyecto

```
proyecto_googleplay/
├── proyecto_googleplay_EV2_final.ipynb    # Notebook principal (3 partes)
├── app_dashboard.py                       # Dashboard interactivo (Dash + Plotly)
├── googleplaystore_clean.csv              # Dataset (9,438 apps)
├── models/
│   └── trained_models/
│       └── mejor_modelo.pkl               # Modelo entrenado (Random Forest)
└── README.md                              # Este archivo
```

### Partes del Notebook

| Parte | Contenido |
|-------|-----------|
| **Parte 1** | Exploración (EDA), Preprocesamiento, Ingeniería de Features |
| **Parte 2** | Modelado Supervisado, Evaluación, Optimización de Hiperparámetros |
| **Parte 3** | Aprendizaje No Supervisado (K-Means + PCA), Conclusiones |

---

##  Instalación

### Requisitos Previos
- Python 3.10 o superior
- pip o conda

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/proyecto-googleplay.git
cd proyecto-googleplay
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```


##  Uso

### Opción 1: Ejecutar el Notebook Completo
```bash
jupyter notebook proyecto_googleplay_EV2_final.ipynb
```

### Opción 2: Ejecutar el Dashboard Interactivo
```bash
python app_dashboard.py
```

Luego abre tu navegador en: **http://127.0.0.1:8050/**

El dashboard incluye:
- 📈 **Vista Gerencial**: KPIs y métricas principales
- 🔬 **Vista Técnica**: Matriz de confusión, Curva ROC, Métricas detalladas
- 🔍 **Exploración**: Análisis del dataset, distribuciones, correlaciones

---

##  Resultados

### Métricas de Evaluación

| Modelo | Accuracy | F1 Score | ROC-AUC |
|--------|----------|----------|---------|
| **Random Forest**  | **95.44%** | **0.8892** | **0.9888** |
| Regresión Logística | 95.39% | 0.8889 | 0.9898 |
| SVM | 94.81% | 0.8740 | 0.9881 |
| Decision Tree | 93.59% | 0.8455 | 0.9014 |

###  Mejor Modelo: Random Forest Optimizado
- **Accuracy**: 95.44%
- **F1 Score**: 0.8892 (balance perfecto entre precisión y recall)
- **ROC-AUC**: 0.9888 (casi perfecto)
- **Parámetros óptimos**: 200 estimadores, max_depth=None, min_samples_split=2

### Validación Cruzada (5-Fold)
- Regresión Logística: F1 = 0.8667 ± 0.0151 (muy estable)
- Random Forest: F1 = 0.8618 ± 0.0148 (muy estable)

### Matriz de Confusión (Test Set)
```
                Predicción
                No Pop   Pop
Real    No Pop   1452    43   (97% precision en negs)
        Pop      45      348  (88% precision en pos)
```

### Análisis de Características (Importancia)

Las variables más predictivas son:
1. **log_reviews**: Número de reseñas (transformado logarítmico)
2. **dias_desde_update**: Días desde última actualización
3. **Size_MBs**: Tamaño de la aplicación
4. **log_price**: Precio (transformado logarítmico)

---

## Key Findings

### Datos Exploratorios
- **9,438 apps** analizadas del dataset de 2018
- **20.8%** de las apps son consideradas populares (>1M instalaciones)
- **92.7%** de las apps son gratuitas (modelo dominante en Play Store)
- **Categorías principales**: FAMILY (1,849), GAME (1,139), TOOLS (749)
- **Rating promedio**: 4.19 / 5.0 (mayoría entre 4.1 - 4.5)

### Clustering (K-Means, k=3)
K-Means descubrió 3 grupos naturales **sin usar la etiqueta de popularidad**:
- **Cluster 1** (3,926 apps): Apps muy populares (10.6M reviews, actualizadas recientemente)
- **Cluster 0** (3,882 apps): Apps moderadas (3.5K reviews, rating bajo)
- **Cluster 2** (1,630 apps): Apps antiguas/abandonadas (800+ días sin update)

### PCA (Reducción de Dimensionalidad)
- Los **2 primeros componentes** explican **47.2%** de la varianza
- PC1 está dominado por **log_reviews** e **Installs** (popularidad)
- Los clusters de K-Means se alinean parcialmente con popularidad real ✓

---

## Conclusiones

###  Lo que Logramos

1. **Modelo robusto**: Random Forest logra 95.44% de accuracy con excelente generalización
2. **Características significativas**: `log_reviews`, tamaño, y actualización son clave
3. **Estructura en los datos**: K-Means encontró patrones naturales relacionados con popularidad
4. **Viabilidad**: El problema es clasificable con estructura clara (ROC-AUC > 0.98)

###  Recomendaciones

- **Para startups**: Enfoque en categorías FAMILY, GAME, TOOLS (más volumen)
- **Para optimizar**: Mantener actualizaciones frecuentes y acumular reseñas
- **Para predicciones**: El modelo funciona bien para apps ya publicadas (requiere Reviews)

---

## Limitaciones

1. **Datos antiguos**: Dataset es de 2018. El mercado de apps cambió significativamente desde entonces.

2. **Data Leakage Conceptual**: `log_reviews` es altamente predictiva pero no está disponible para apps nuevas al momento del lanzamiento.
   - **Solución**: Para predecir éxito pre-lanzamiento, excluir `Reviews` de las features.

3. **Desbalance de clases**: Solo 20.8% de apps son populares, pero el modelo maneja bien el desequilibrio.

4. **Información limitada**: El dataset no incluye datos de marketing, presupuesto de desarrollo, o equipo.

5. **Sesgo temporal**: Patrones de 2018 (era pre-COVID, diferentes tendencias de apps)

---

##  Dependencias

```python
pandas==3.0.2              # Manipulación de datos
numpy==2.4.4               # Computación numérica
scikit-learn==1.8.0        # Machine Learning
matplotlib==3.10.8         # Visualizaciones estáticas
seaborn==0.13.2            # Visualizaciones estadísticas
plotly==6.8.0              # Gráficos interactivos
dash==4.2.0                # Dashboard web
joblib==1.5.3              # Serialización de modelos
scipy==1.17.1              # Funciones científicas
```

---

## 📚 Referencia de Algoritmos

### Modelos Supervisados
- **Regresión Logística**: Clasificación lineal, interpretable
- **Decision Tree**: Reglas de decisión, fácil de explicar
- **Random Forest**: Ensemble de árboles, robusto a overfitting ⭐
- **SVM**: Margen máximo entre clases, efectivo en altas dimensiones

### Optimización
- **GridSearchCV**: Búsqueda exhaustiva de hiperparámetros (27 combinaciones)
- **RandomizedSearchCV**: Búsqueda aleatoria (20 iteraciones)
- **Validación Cruzada (5-Fold)**: Evaluación robusta sin data leakage

### Técnicas No Supervisadas
- **K-Means**: Agrupamiento por proximidad euclidiana
- **Silhouette Score**: Validación de calidad de clusters
- **PCA**: Reducción de dimensionalidad preservando varianza

---

## 🔄 Pipeline de Datos

```
Raw Data (googleplaystore_clean.csv)
    ↓
[Carga e Inspección]
    ↓
[EDA - Análisis Exploratorio]
    ↓
[Feature Engineering]
    ├─ Transformación log (Reviews, Price)
    ├─ Feature temporal (días desde update)
    └─ Escalado y One-Hot Encoding
    ↓
[Train-Test Split] (80-20, stratificado)
    ↓
[Preprocesamiento]
    ├─ Numéricas: Imputer (mediana) → StandardScaler
    └─ Categóricas: Imputer (moda) → OneHotEncoder
    ↓
[Entrenamiento de Modelos]
    ├─ Regresión Logística
    ├─ Decision Tree
    ├─ Random Forest
    └─ SVM
    ↓
[Evaluación y Comparación]
    ├─ Accuracy, Precision, Recall, F1
    ├─ Confusion Matrix
    └─ ROC-AUC Curve
    ↓
[Optimización (GridSearchCV/RandomizedSearchCV)]
    ↓
[Aprendizaje No Supervisado]
    ├─ K-Means Clustering
    └─ PCA Visualization
    ↓
[Dashboard Interactivo]
```



## 🙏 Agradecimientos

- Dataset: [Google Play Store Apps](https://www.kaggle.com/datasets/lava18/google-play-store-apps)
- Herramientas: scikit-learn, Plotly, Dash, pandas
- Técnicas: Hands-on Machine Learning, scikit-learn documentation


---

## 📈 Roadmap Futuro

- [ ] Actualizar a dataset 2024
- [ ] Implementar Deep Learning (Neural Networks)
- [ ] API REST para predicciones en tiempo real
- [ ] Análisis de sentimiento de reviews
- [ ] Predicción de trending apps
- [ ] Deployment en Heroku/AWS
- [ ] Interfaz mobile-friendly

---

**Última actualización**: 17 de junio de 2026
