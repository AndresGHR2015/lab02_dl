# Laboratorio 02: Clasificación de Deterioro Cognitivo mediante Redes Neuronales Superficiales

**Curso:** Deep Learning  
**Institución:** Universidad Católica del Norte  
**Objetivo:** Implementar y evaluar una red neuronal superficial (Shallow Neural Network) con dropout para la clasificación multilabel del deterioro cognitivo, utilizando validación cruzada anidada y cuantificación de incertidumbre predictiva mediante Monte Carlo Dropout.

---

## Descripción del Laboratorio

Este laboratorio aborda el problema de clasificación del deterioro cognitivo a partir de datos tabulares de orientación cognitiva. El dataset contiene 15 atributos de orientación (temporal, espacial y personal) y seis variables objetivo basadas en la escala GDS (Global Deterioration Scale): `GDS`, `GDS_R1`, `GDS_R2`, `GDS_R3`, `GDS_R4` y `GDS_R5`.

### Problema

Cada variable objetivo es de naturaleza multiclase y es convertida a una representación **multilabel** mediante codificación One-Hot, lo que permite entrenar un clasificador binario por clase simultáneamente mediante `BCEWithLogitsLoss`. El desbalance de clases es mitigado mediante pesos positivos calculados dinámicamente en función de la distribución de cada split de entrenamiento.

### Pipeline General

```
Datos crudos (.sav)
       │
       ▼
  DataPreprocessor
  ├── Imputación de valores faltantes (mediana/moda)
  ├── Escalado estándar (StandardScaler)
  └── Codificación One-Hot del target
       │
       ▼
  NestedCrossValidator (5-fold externo × 3-fold interno)
  ├── Selección automática de hiperparámetros (F1-micro)
  └── Evaluación de generalización por fold externo
       │
       ▼
  Entrenamiento del modelo final (dataset completo)
       │
       ▼
  MonteCarloDropoutEstimator
  └── 50 forward passes estocásticos → media y desviación estándar
       │
       ▼
   Artefactos de salida:
   ├── data/processed/features_<TARGET>.csv
   ├── data/processed/target_<TARGET>.csv
   ├── models/best_model_<TARGET>.pth
   └── results/metrics_summary.json
```

---

## Estructura de Directorios

Parte de la estructura es generada post-ejecucion.

```
lab02_dl/
├── data/
│   ├── raw/                        # Dataset original en formato .sav (SPSS)
│   │   └── 15 atributos R0-R5.sav
│   └── processed/                  # Datasets preprocesados generados por el pipeline
│       ├── features_GDS.csv
│       ├── target_GDS.csv
│       ├── features_GDS_R1.csv
│       ├── target_GDS_R1.csv
│       ├── features_GDS_R2.csv
│       ├── target_GDS_R2.csv
│       ├── features_GDS_R3.csv
│       ├── target_GDS_R3.csv
│       ├── features_GDS_R4.csv
│       ├── target_GDS_R4.csv
│       ├── features_GDS_R5.csv
│       └── target_GDS_R5.csv
│
├── models/                         # Pesos de los modelos entrenados (.pth)
│   ├── best_model_GDS.pth
│   ├── best_model_GDS_R1.pth
│   ├── best_model_GDS_R2.pth
│   ├── best_model_GDS_R3.pth
│   ├── best_model_GDS_R4.pth
│   └── best_model_GDS_R5.pth
│
├── notebooks/
│   └── exploracion.ipynb           # Análisis exploratorio de datos (EDA)
│
├── results/
│   └── metrics_summary.json        # Reporte consolidado de métricas por variable objetivo
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Dataset PyTorch personalizado (CognitiveDataset)
│   ├── evaluation.py               # Validación cruzada anidada y cálculo de métricas
│   ├── main.py                     # Punto de entrada del pipeline de entrenamiento
│   ├── models.py                   # Definición de la arquitectura (ShallowNeuralNetwork)
│   ├── preprocessing.py            # Preprocesamiento e imputación (DataPreprocessor)
│   └── uncertainty.py              # Estimación de incertidumbre (MonteCarloDropoutEstimator)
│
├── requirements.txt                # Dependencias del proyecto
├── .gitignore
└── README.md
```

---

## Descripción de Módulos

| Módulo | Clase Principal | Responsabilidad |
|---|---|---|
| `data_loader.py` | `CognitiveDataset` | Encapsula features y targets en tensores `float32` compatibles con PyTorch `DataLoader`. |
| `preprocessing.py` | `DataPreprocessor` | Imputa valores faltantes, escala features y codifica el target en representación multilabel. |
| `models.py` | `ShallowNeuralNetwork` | Define la arquitectura `Input → Linear → ReLU → Dropout → Linear (logits)`. |
| `evaluation.py` | `NestedCrossValidator`, `MetricsCalculator` | Ejecuta validación cruzada anidada con estratificación multilabel y calcula Hamming Loss, F1-micro, F1-macro, Precision, Recall y Exact Match. |
| `uncertainty.py` | `MonteCarloDropoutEstimator` | Realiza `N` forward passes con Dropout activo para estimar la distribución predictiva posterior (media y desviación estándar). |
| `main.py` | — | Orquesta el pipeline completo: carga, preprocesamiento, validación cruzada, entrenamiento final y persistencia de artefactos. |

---

## Arquitectura del Modelo

La arquitectura implementada es una **red neuronal superficial** (una sola capa oculta) diseñada para clasificación multilabel:

```
Input (15 features)
    │
    ▼
Linear(input_dim → hidden_dim)
    │
    ▼
ReLU
    │
    ▼
Dropout(p=dropout_rate)
    │
    ▼
Linear(hidden_dim → output_dim)
    │
    ▼
Logits (sin activación de salida)
```

La función de pérdida utilizada es `BCEWithLogitsLoss` con pesos positivos ajustados por clase para manejar el desbalance. La activación sigmoid se aplica únicamente durante inferencia.

### Espacio de Hiperparámetros Evaluado

| Configuración | `hidden_dim` | `dropout_rate` | `lr` | `batch_size` | `epochs` |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 32 | 0.10 | 0.010 | 32 | 150 |
| 2 | 64 | 0.20 | 0.005 | 64 | 150 |
| 3 | 128 | 0.20 | 0.001 | 128 | 150 |

---

## Resultados

Las métricas reportadas corresponden al promedio de los 5 folds externos de la validación cruzada anidada (estratificación multilabel).

| Variable Objetivo | Hamming Loss ↓ | Precision Micro ↑ | Recall Micro ↑ | F1 Micro ↑ | F1 Macro ↑ | Exact Match ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `GDS` | 0.2013 | 0.3968 | 0.7864 | 0.5274 | 0.4001 | 0.0778 |
| `GDS_R1` | 0.1412 | 0.7675 | 0.8267 | 0.7957 | 0.5943 | 0.7507 |
| `GDS_R2` | 0.2192 | 0.6381 | 0.7918 | 0.7066 | 0.6662 | 0.5693 |
| `GDS_R3` | 0.1693 | 0.8311 | 0.8302 | 0.8306 | 0.7449 | 0.8258 |
| `GDS_R4` | 0.2553 | 0.6008 | 0.6989 | 0.6461 | 0.5653 | 0.5219 |
| `GDS_R5` | 0.2696 | 0.5706 | 0.7739 | 0.6568 | 0.5900 | 0.4138 |

> Los resultados completos se encuentran en `results/metrics_summary.json`.

---

## Requisitos Previos

- **Python** `>= 3.11`
- **Conda** (recomendado para gestión del entorno)
- Acceso al archivo de datos: `data/raw/15 atributos R0-R5.sav`

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/AndresGHR2015/lab02_dl.git
cd lab02_dl
```

### 2. Crear y activar el entorno Conda

Se recomienda utilizar un entorno aislado para evitar conflictos de dependencias.

```bash
conda create -n lab02_dl python=3.11 -y
conda activate lab02_dl
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

Las dependencias principales son:

| Paquete | Versión mínima | Propósito |
|---|---|---|
| `torch` | `>= 2.0.0` | Framework de deep learning (PyTorch) |
| `pandas` | `>= 2.0.0` | Manipulación de datos tabulares |
| `scikit-learn` | `>= 1.3.0` | Preprocesamiento, métricas y splits |
| `numpy` | `>= 1.26.0` | Operaciones numéricas |
| `iterative-stratification` | `>= 0.1.9` | Estratificación multilabel para KFold |

> **Nota:** Para ejecutar el notebook `exploracion.ipynb` se requiere adicionalmente `jupyter` y `matplotlib`:
> ```bash
> pip install jupyter matplotlib seaborn
> ```

---

## Ejecución

### Pipeline Principal de Entrenamiento

El script `main.py` ejecuta el pipeline completo para las seis variables objetivo de forma secuencial. Debe ejecutarse desde la raíz del repositorio.

```bash
python -m src.main
```

**Salidas generadas:**

- `data/processed/features_<TARGET>.csv` — Features escaladas e imputadas para cada variable objetivo.
- `data/processed/target_<TARGET>.csv` — Target codificado en representación multilabel (One-Hot) por variable objetivo.
- `models/best_model_<TARGET>.pth` — Pesos del modelo final por cada variable objetivo.
- `results/metrics_summary.json` — Reporte consolidado de métricas de validación cruzada.

> **Importante:** El script espera el archivo de datos en la ruta relativa `data/raw/15 atributos R0-R5.sav`. Asegúrese de que el archivo exista antes de ejecutar.

### Notebook de Exploración de Datos

Para ejecutar el análisis exploratorio de datos:

```bash
jupyter notebook notebooks/exploracion.ipynb
```

O bien, desde JupyterLab:

```bash
jupyter lab notebooks/exploracion.ipynb
```

---

## Flujo de Datos Detallado

```
data/raw/15 atributos R0-R5.sav
        │
        │  pd.read_spss()
        ▼
   DataFrame crudo (N filas × 16+ columnas)
        │
        │  DataPreprocessor.fit_transform(target_column)
        ▼
   ┌─────────────────────────┐
   │  features: (N, 15)      │  ← StandardScaler + SimpleImputer
   │  target:   (N, K)       │  ← OneHotEncoder (K clases del target)
   └─────────────────────────┘
        │
        │  NestedCrossValidator.execute()
        ▼
   MultilabelStratifiedKFold (outer=5, inner=3)
   ├── Inner loop: selección de hiperparámetros (F1-micro)
   └── Outer loop: evaluación de generalización
        │
        ▼
   Métricas promediadas → all_metrics[target]
        │
        ▼
   Entrenamiento final (dataset completo, best_params)
        │
        ▼
   MonteCarloDropoutEstimator (N=50 iteraciones)
   └── mean_probs, std_probs (muestra representativa)
        │
        ▼
   Persistencia:
   ├── data/processed/features_<TARGET>.csv
   ├── data/processed/target_<TARGET>.csv
   ├── models/best_model_<TARGET>.pth
   └── results/metrics_summary.json
```

---

## Reproducibilidad

La semilla aleatoria `random_state=42` se establece en las instancias de `MultilabelStratifiedKFold` para garantizar la reproducibilidad de los splits. Los modelos son entrenados con optimizador Adam y se recomienda fijar también la semilla de PyTorch para reproducibilidad completa:

```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
```

---

## Estructura del Dataset

El archivo `15 atributos R0-R5.sav` contiene variables de orientación cognitiva recopiladas en formato SPSS. Las columnas utilizadas como **features** son:

| Columna | Tipo | Descripción |
|---|---|---|
| `Día` | Binaria | Día del mes (orientación temporal) |
| `Mes` | Binaria | Mes del año (orientación temporal) |
| `Año` | Binaria | Año (orientación temporal) |
| `Estación` | Binaria | Estación del año (orientación temporal) |
| `País` | Binaria | País de residencia (orientación espacial) |
| `Ciudad` | Binaria | Ciudad de residencia (orientación espacial) |
| `CalleLugar` | Binaria | Calle o lugar (orientación espacial) |
| `NumeroPiso` | Binaria | Número o piso (orientación espacial) |
| `Miguel2` | Binaria | Ítem de orientación personal |
| `González2` | Binaria | Ítem de orientación personal |
| `Avenida2` | Binaria | Ítem de orientación personal |
| `Imperial2` | Binaria | Ítem de orientación personal |
| `A682` | Binaria | Ítem de orientación personal |
| `Caldera2` | Binaria | Ítem de orientación personal |
| `Copiapo2` | Binaria | Ítem de orientación personal |

Las **variables objetivo** (`GDS`, `GDS_R1`, ..., `GDS_R5`) representan la clasificación de deterioro cognitivo en distintas revisiones de la evaluación clínica.
