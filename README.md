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

## Estructura del Proyecto

- **multimodal-tumor-segmentation-classification/**
    - **data/**
        - BCBM-RadioGenomics_Images_Masks_Dec2024/
        - brats20-dataset-training-validation/
        - split_regression/
        - train_data.csv
        - df_with_voxel_stats_and_latent.csv
    - **models/**
        - LViT/
        - SegResNet/
        - ae_3d/
        - brats20logs/
        - monai_models/
            - SegResNet_Monai/
            - unetr_Monai/
        - sam2/
            - core/
            - medsam2/
            - meta/
            - ultralytics/
        - unet_3d/
        - unimodal_nets/
    - **regression_analysis/**
        - evaluation/
        - feature_processing/
            - ae_3d/
        - hyperparameter_optimization/
        - __init__.py
    - **visualizations/**
    - **notebooks/**
        - regression_analysis/
        - evaluation/
        - feature_processing/
        - hyperparameter_optimization/
    - .gitignore
    - .gitattributes
    - README.md



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

### `data/`
- Datos brutos, procesados y archivos de entrenamiento y validación, incluyendo datasets de BraTS y BCBM-RadioGenomics.

### `models/`
- **LViT/**: Implementación y utilidades para redes multimodales LViT.
- **SegResNet/**: Segmentación con arquitectura SegResNet.
- **ae_3d/**: Autoencoder 3D para extracción de características latentes.
- **monai_models/**: Modelos de segmentación con MONAI, incluyendo subcarpetas para SegResNet y UNETR.
- **sam2/**: Arquitectura SAM2 y variantes:
  - **core/**: Código base y utilidades compartidas de SAM2.
  - **medsam2/**: Entrenamiento, pérdidas, validación y análisis para MedSAM2.
  - **meta/**: Dataset, entrenamiento y utilidades para la variante Meta de SAM2.
  - **ultralytics/**: Uso y predicción con la variante Ultralytics de SAM2.
- **unet_3d/**: Implementación modular de U-Net 3D para segmentación volumétrica.
- **unimodal_nets/**: Modelos unimodales para experimentos comparativos.

### `regression_analysis/`
- **evaluation/**: Métricas, evaluación y visualización de resultados de regresión.
- **feature_processing/ae_3d/**: Limpieza de características latentes con autoencoders.
- **hyperparameter_optimization/**: Búsqueda de hiperparámetros, configuraciones y utilidades.

### `visualizations/`
- Scripts y utilidades para visualización de resultados, análisis de datasets, valores faltantes y comparativas.

### `notebooks/`
- Notebooks organizados por funcionalidad: exploración de datos, entrenamiento, análisis de regresión, evaluación y visualización.

---

## Uso Rápido

### Entrenamiento y Evaluación con 3D U-Net

Esta sección demuestra el flujo completo de trabajo con la arquitectura 3D U-Net, desde el entrenamiento hasta la visualización de resultados.

#### 1. Entrenamiento del Modelo

Configuración e inicialización del modelo 3D U-Net para segmentación volumétrica de tumores cerebrales:

from models.unet_3d.model import build_3dunet
from models.unet_3d.train import train_model
from models.unet_3d.data_generator import BraTSDataGenerator

Configuración de generadores de datos
train_generator = BraTSDataGenerator(
train_files,
batch_size=2,
augmentation=True
)
val_generator = BraTSDataGenerator(
val_files,
batch_size=2,
augmentation=False
)

Construcción de la arquitectura del modelo
model = build_3dunet(
input_shape=(128, 128, 96, 4), # (H, W, D, Canales)
num_classes=3,
dropout_rate=0.2
)

Proceso de entrenamiento
training_history = train_model(
model=model,
train_generator=train_generator,
val_generator=val_generator,
epochs=100,
learning_rate=1e-4,
checkpoint_path="checkpoints/unet3d_best_model.pth"
)


#### 2. Inferencia y Predicción

Carga del modelo entrenado y generación de segmentaciones para nuevos volúmenes:

import torch
from models.unet_3d.inference import predict_volume

Carga del modelo pre-entrenado
checkpoint = torch.load("checkpoints/unet3d_best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

Predicción sobre volumen de entrada
with torch.no_grad():
predicted_segmentation = predict_volume(
model=model,
input_volume=input_volume,
device='cuda',
batch_size=4
)


#### 3. Análisis y Visualización de Resultados

Evaluación cuantitativa y visualización comparativa de las segmentaciones predichas:

from models.unet_3d.visualization import plot_segmentation_results
from models.unet_3d.metrics import calculate_dice_score, calculate_hausdorff_distance

Cálculo de métricas de evaluación
dice_scores = calculate_dice_score(predicted_segmentation, ground_truth_mask)
hausdorff_dist = calculate_hausdorff_distance(predicted_segmentation, ground_truth_mask)

Visualización comparativa multi-planar
plot_segmentation_results(
input_volume=input_volume,
ground_truth=ground_truth_mask,
prediction=predicted_segmentation,
slice_index=64,
save_path="results/segmentation_comparison.png"
)

print(f"Dice Score - WT: {dice_scores['wt']:.4f}, TC: {dice_scores['tc']:.4f}, ET: {dice_scores['et']:.4f}")


#### 4. Entrenamiento de Modelos MONAI (SegResNet/UNETR)

Ejemplo de entrenamiento utilizando arquitecturas avanzadas de MONAI:

from models.monai_models.SegResNet_Monai.model import get_segresnet_model
from models.monai_models.SegResNet_Monai.train import train_segresnet
from common.config import GlobalConfig, seed_everything
from common.dataset import BratsDataset

Configuración de reproducibilidad
seed_everything(GlobalConfig.seed)

Inicialización del modelo SegResNet
model = get_segresnet_model(
device="cuda",
in_channels=4,
out_channels=3,
dropout_prob=0.2
)

Proceso de entrenamiento con validación
training_metrics = train_segresnet(
model=model,
train_loader=train_loader,
val_loader=val_loader,
device="cuda",
max_epochs=300,
learning_rate=1e-4,
save_path="checkpoints/segresnet_model.pth"
)

---

## Entrenamiento de Modelos

- Todos los scripts de entrenamiento y validación están en la carpeta de cada modelo.
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
- [LViT: Multimodal Vision Transformer for Brain Tumor Segmentation](https://github.com/tu-berlin-mannheim/LViT)
