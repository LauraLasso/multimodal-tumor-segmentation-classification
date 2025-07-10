# Proyecto de Segmentación Cerebral con MONAI, UNETR, SegResNet y SAM2

Bienvenido al repositorio de segmentación y análisis de imágenes médicas cerebrales. Este proyecto integra arquitecturas avanzadas de segmentación (**SegResNet**, **UNETR**, **U-Net 3D**, **ResNet-UNet**, **SAM2** y variantes) usando **MONAI** y **PyTorch**, junto con utilidades para análisis de datos, entrenamiento, visualización y evaluación.

---

## Índice

- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Descripción de Carpetas y Módulos](#descripción-de-carpetas-y-módulos)
- [Uso Rápido](#uso-rápido)
- [Entrenamiento de Modelos](#entrenamiento-de-modelos)
- [Evaluación y Visualización](#evaluación-y-visualización)
- [Referencias y Créditos](#referencias-y-créditos)
- [Contacto](#contacto)
---

## Requisitos

- Python 3.8+
- PyTorch >= 1.10
- MONAI >= 1.0
- nibabel
- scikit-image
- segmentation-models-pytorch
- torchmetrics
- tqdm
- matplotlib, seaborn
- pandas, numpy
- ultralytics (para SAM2)
- cv2 (OpenCV)
- scikit-learn

**Instalación de requisitos:**
pip install -r requirements.txt


---

## Descripción de Carpetas y Módulos

### `common/`
- **config.py**: Configuración global y funciones de semilla.
- **dataset.py**: Dataset BraTS y utilidades de carga de datos.

### `models/monai_segmentation/`
- **segresnet/** y **unetr/**: Implementaciones, entrenamiento, inferencia y visualización de SegResNet y UNETR (MONAI).

### `models/unet_3d/`
- Implementación modular de U-Net 3D para segmentación volumétrica.

### `models/resnet34_unet/` y `resnet101_unet/`
- Modelos U-Net con encoder ResNet34 y ResNet101, y scripts de entrenamiento.

### `models/sam2/`
- **ultralytics/**: Uso y predicción con SAM2 de Ultralytics.
- **meta/**: Dataset, entrenamiento y utilidades para SAM2 de Meta.
- **medsam2/**: Entrenamiento, pérdidas, validación y análisis para MedSAM2.

### `regression_analysis/`
- **hyperparameter_optimization/**: Búsqueda de hiperparámetros, configuraciones y utilidades.
- **evaluation/**: Métricas, evaluación y visualización de resultados de regresión.
- **feature_processing/3d_ae/**: Limpieza de características latentes con autoencoders.

### `visualizations/`
- Visualizaciones generales, análisis de datasets, valores faltantes, 3D y comparativas.

### `notebooks/`
- Notebooks organizados por flujo de trabajo: exploración de datos, entrenamiento, análisis y visualización de resultados.

---

## Uso Rápido

### 1. Configuración de entorno

Asegúrate de tener los datos en la ruta indicada en `common/config.py`. Modifica las rutas si es necesario.

### 2. Entrenamiento de un modelo (ejemplo SegResNet)

from common.config import GlobalConfig, seed_everything
from common.dataset import BratsDataset
from models.monai_segmentation.segresnet.model import get_segresnet_model
from models.monai_segmentation.segresnet.train import train_segresnet

seed_everything(GlobalConfig.seed)

Carga de datos y DataLoader aquí...
model = get_segresnet_model(device="cuda")
train_segresnet(model, train_loader, val_loader, device="cuda", max_epochs=300)


### 3. Inferencia

from models.monai_segmentation.segresnet.inference import run_inference

run_inference(model, val_loader, output_dir="./outputs", device="cuda")


### 4. Visualización

from models.monai_segmentation.segresnet.visualization import visualizar_pred_vs_gt

visualizar_pred_vs_gt(paciente_id, input_image, gt_dir, slice_index=60)


---

## Entrenamiento de Modelos

- Todos los scripts de entrenamiento están en la carpeta de cada modelo.
- Ajusta hiperparámetros y rutas en los archivos de configuración o directamente en los scripts.
- Usa los notebooks para pruebas rápidas y análisis exploratorio.

---

## Evaluación y Visualización

- **Métricas de segmentación:** Dice, IoU, métricas por clase.
- **Métricas de regresión:** MAE, MSE, R², etc.
- **Visualizaciones:** cortes axiales/coronales/sagitales, segmentación 3D, gráficos de métricas y comparativas.

---

## Referencias y Créditos

- [MONAI: Medical Open Network for AI](https://monai.io/)
- [BraTS: Multimodal Brain Tumor Segmentation Challenge](https://www.med.upenn.edu/cbica/brats2020/)
- [Ultralytics SAM2](https://github.com/ultralytics/ultralytics)
- [Meta SAM2](https://github.com/facebookresearch/segment-anything)
- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

