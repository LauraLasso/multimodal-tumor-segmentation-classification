import numpy as np
import plotly.graph_objects as go
from skimage import measure
from image_processing import load_nifti

def plot_segmentation_3D_with_shadows(seg_path, threshold=0.5):
    """
    Crear visualización 3D con sombras usando Mesh3d
    """
    # Cargar la segmentación
    seg = load_nifti(seg_path)
    
    # Crear máscara binaria
    binary_mask = (seg > threshold).astype(np.uint8)
    
    # Aplicar marching cubes para generar superficie 3D
    try:
        vertices, faces, normals, values = measure.marching_cubes(
            binary_mask, 
            level=0.5,
            spacing=(1.0, 1.0, 1.0)
        )
        
        # Crear mesh 3D
        fig = go.Figure(data=[go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1], 
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='green',
            opacity=0.7,
            lighting=dict(
                ambient=0.3,
                diffuse=0.8,
                specular=0.2,
                roughness=0.1,
                fresnel=0.02
            ),
            lightposition=dict(
                x=100,
                y=200,
                z=0
            )
        )])
        
        # Configurar layout para mejor visualización
        fig.update_layout(
            title="Segmentación 3D con Sombras",
            scene=dict(
                xaxis_title="Eje X",
                yaxis_title="Eje Y",
                zaxis_title="Eje Z",
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        fig.show()
        
    except Exception as e:
        print(f"Error generando superficie 3D: {e}")
        print("Intentando con visualización de puntos mejorada...")
        plot_segmentation_3D_points_improved(seg_path, threshold)

def plot_segmentation_3D_points_improved(seg_path, threshold=0.5):
    """
    Versión mejorada con puntos si marching cubes falla
    """
    seg = load_nifti(seg_path)
    
    # Obtener coordenadas
    x, y, z = np.where(seg > threshold)
    
    # Crear colores basados en intensidad para dar profundidad
    colors = seg[seg > threshold]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            opacity=0.6,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Intensidad")
        )
    )])
    
    fig.update_layout(
        title="Segmentación 3D - Puntos con Intensidad",
        scene=dict(
            xaxis_title="Eje X",
            yaxis_title="Eje Y", 
            zaxis_title="Eje Z",
            aspectmode='data'
        )
    )
    
    fig.show()
