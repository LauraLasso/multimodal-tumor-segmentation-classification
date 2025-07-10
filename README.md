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

proyecto_segmentacion/
├── data/
│ ├── BCBM-RadioGenomics_Images_Masks_Dec2024/
│ ├── brats20-dataset-training-validation/
│ ├── split_regression/
│ ├── train_data.csv
│ └── df_with_voxel_stats_and_latent.csv
├── models/
│ ├── LViT/
│ ├── SegResNet/
│ ├── ae_3d/
│ ├── brats20logs/
│ ├── monai_models/
│ │ ├── SegResNet_Monai/
│ │ └── unetr_Monai/
│ ├── sam2/
│ │ ├── core/
│ │ ├── medsam2/
│ │ ├── meta/
│ │ └── ultralytics/
│ ├── unet_3d/
│ └── unimodal_nets/
├── regression_analysis/
│ ├── evaluation/
│ ├── feature_processing/
│ │ └── ae_3d/
│ ├── hyperparameter_optimization/
│ └── init.py
├── visualizations/
├── notebooks/
│ ├── regression_analysis/
│ ├── evaluation/
│ ├── feature_processing/
│ └── hyperparameter_optimization/
├── .gitignore
├── .gitattributes
└── README.md


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

### Ejemplo de uso rápido: entrenamiento y predicción con 3D U-Net

#### 1. Entrenamiento de 3D U-Net

from models.unet_3d.model import build_3dunet
from models.unet_3d.train import train_model
from models.unet_3d.data_generator import BraTSDataGenerator

Inicializa el generador de datos
train_generator = BraTSDataGenerator(train_files, batch_size=2)
val_generator = BraTSDataGenerator(val_files, batch_size=2)

Construye el modelo
model = build_3dunet(input_shape=(128, 128, 96, 4))

Entrena el modelo
history = train_model(
model,
train_generator,
val_generator,
epochs=100,
checkpoint_path="unet3d_best_model.pth"
)


#### 2. Inferencia con 3D U-Net

from models.unet_3d.inference import predict_volume

Carga el modelo entrenado
model.load_state_dict(torch.load("unet3d_best_model.pth"))
model.eval()

Realiza la predicción sobre un volumen
pred_mask = predict_volume(model, input_volume)


#### 3. Visualización de resultados

from models.unet_3d.visualization import plot_segmentation_results

plot_segmentation_results(input_volume, ground_truth_mask, pred_mask, slice_index=64)

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
