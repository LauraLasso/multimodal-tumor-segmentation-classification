import torch
from models.sam2.core.build_sam import build_sam2
from models.sam2.core.sam2_image_predictor import SAM2ImagePredictor

def initialize_sam2_model(
    sam_config='sam2_hiera_t',
    sam_ckpt='MedSAM2_pretrain.pth',
    device="cuda"
):
    """
    Inicializa el modelo SAM2 con configuración y checkpoint especificados.

    Args:
        sam_config (str): Nombre de la configuración del modelo SAM2.
        sam_ckpt (str): Ruta al checkpoint preentrenado.
        device (str): Dispositivo de ejecución ('cuda' o 'cpu').

    Returns:
        net: Modelo SAM2 cargado y listo para usar.
    """
    # Configurar autocast y TF32 si es compatible
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net = build_sam2(sam_config, sam_ckpt, device=device)
    return net
