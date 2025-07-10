from models.sam2.core.build_sam import build_sam2
from models.sam2.core.sam2_image_predictor import SAM2ImagePredictor

def load_pretrained_sam2(
    checkpoint_path="sam2_hiera_small.pt",
    config_path="./configs/sam2/sam2_hiera_s.yaml",
    device="cuda"
):
    """
    Carga el modelo SAM2 preentrenado y devuelve el predictor listo para inferencia.

    Args:
        checkpoint_path (str): Ruta al checkpoint preentrenado de SAM2.
        config_path (str): Ruta al archivo de configuración YAML del modelo.
        device (str): Dispositivo de ejecución, por defecto "cuda".

    Returns:
        predictor (SAM2ImagePredictor): Predictor SAM2 listo para usar.
    """
    # Construir el modelo SAM2
    sam2_model = build_sam2(config_path, checkpoint_path, device=device)
    # Inicializar el predictor
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor