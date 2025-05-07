
# TFM-Forex

**TFM-Forex** es un pipeline de aprendizaje automático diseñado para analizar y predecir tendencias del mercado de divisas (Forex), con un enfoque particular en el par EUR/USD. Este proyecto integra componentes de preprocesamiento, entrenamiento, validación y despliegue de modelos para ofrecer un flujo de trabajo completo de análisis Forex.

## Tabla de Contenidos

- [Resumen](#resumen)
- [Características](#características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Primeros Pasos](#primeros-pasos)
  - [Requisitos Previos](#requisitos-previos)
  - [Instalación](#instalación)
  - [Uso](#uso)
- [Plan de Desarrollo](#plan-de-desarrollo)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)
- [Agradecimientos](#agradecimientos)

## Resumen

Este proyecto busca proporcionar una base robusta para la predicción de divisas utilizando técnicas de machine learning. Incluye desde la recolección y transformación de datos hasta el desarrollo, validación y despliegue de modelos.

## Características

- **Preprocesamiento**: Escalado y división de datos históricos de divisas.
- **Desarrollo de Modelos**: Implementación de modelos predictivos para tendencias de Forex.
- **Evaluación**: Métricas de rendimiento y validación.
- **Despliegue**: Scripts para servir el modelo entrenado.
- **MLOps**: Integración con MLflow para gestión de experimentos y modelos.

## Estructura del Proyecto

```
TFM-Forex/
├── Analysis.ipynb
├── data/
│   ├── EURUSD60.csv
│   ├── scaler.pkl
│   ├── scalers/
│   ├── test_scaled.csv
│   └── train_scaled.csv
├── mlops_pipeline/
│   ├── Evaluation.ipynb
│   ├── log_recommendator.py
│   ├── mlflow.db
│   ├── model_utils.py
│   ├── models/
│   ├── recommendation_model.py
│   └── serve_model.sh
├── requirements.txt
└── README.md
```

## Primeros Pasos

### Requisitos Previos

- Python 3.7 o superior
- pip
- Herramienta de entornos virtuales (opcional)

### Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/Marconator/TFM-Forex.git
cd TFM-Forex
```

2. Crear entorno virtual (opcional):

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Arrancar servidor MLflow:

```bash
mlflow server --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

5. Comenzar entrenamiento:

```bash
py ./mlops_pipeline/train.py
```

### Uso

- Ejecuta `Analysis.ipynb` para exploración y preparación de datos.
- Corre `Evaluation.ipynb` dentro de `mlops_pipeline/` para validar modelos.
- Despliega el modelo con `serve_model.sh`.

### Alternativas

- De forma alternativa se puede acceder al Google Collab original del proyecto en este enlace https://colab.research.google.com/drive/1FSqN-hwPUiqgl3diei5AySxyx2cn8a9f

## Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo [LICENSE](LICENSE).
