# -*- coding: utf-8 -*-
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

class SimpleLogAnalyzer:
    def __init__(self, log_path: str):
        """
        Inicializar el analizador simple de logs
        
        Args:
            log_path: Ruta al archivo de log
        """
        self.log_path = Path(log_path)
        self.train_metrics = {
            'loss': [],
            'iou': [],
            'dice': []
        }
        self.val_metrics = {
            'loss': [],
            'iou': [],
            'dice': []
        }
        self.df = None
        
        # Patrón regex modificado para extraer valores promedio (Avg)
        self.pattern = r"Loss:.*?\(Avg\s+([\d.]+)\).*?IoU:.*?\(Avg\s+([\d.]+)\).*?Dice:.*?\(Avg\s+([\d.]+)\)"
    
    def read_log_file(self) -> List[str]:
        """
        Leer el archivo de log
        
        Returns:
            Lista de líneas del archivo
        """
        if not self.log_path.exists():
            raise FileNotFoundError(f"Archivo de log no encontrado: {self.log_path}")
        
        with open(self.log_path, "r") as file:
            return file.readlines()
    
    def extract_metrics(self, lines: List[str]) -> None:
        """
        Extraer métricas promedio de las líneas del log
        
        Args:
            lines: Lista de líneas del archivo de log
        """
        # Reinicializar métricas
        self.train_metrics = {'loss': [], 'iou': [], 'dice': []}
        self.val_metrics = {'loss': [], 'iou': [], 'dice': []}
        
        for line in lines:
            if "[Train]" in line:
                match = re.search(self.pattern, line)
                if match:
                    self.train_metrics['loss'].append(float(match.group(1)))
                    self.train_metrics['iou'].append(float(match.group(2)))
                    self.train_metrics['dice'].append(float(match.group(3)))
            
            elif "[Val]" in line:
                match = re.search(self.pattern, line)
                if match:
                    self.val_metrics['loss'].append(float(match.group(1)))
                    self.val_metrics['iou'].append(float(match.group(2)))
                    self.val_metrics['dice'].append(float(match.group(3)))
    
    def average_by_epoch(self, metric_list: List[float], steps_per_epoch: int) -> List[float]:
        """
        Agrupar métricas por época calculando el promedio
        
        Args:
            metric_list: Lista de valores de la métrica
            steps_per_epoch: Número de pasos por época
            
        Returns:
            Lista de promedios por época
        """
        return [np.mean(metric_list[i:i+steps_per_epoch]) 
                for i in range(0, len(metric_list), steps_per_epoch)]
    
    def process_metrics(self, train_steps_per_epoch: int = 12, 
                       val_steps_per_epoch: int = 3) -> Dict[str, List[float]]:
        """
        Procesar métricas agrupándolas por época
        
        Args:
            train_steps_per_epoch: Pasos de entrenamiento por época
            val_steps_per_epoch: Pasos de validación por época
            
        Returns:
            Diccionario con métricas procesadas
        """
        # Agrupar por época
        processed_metrics = {
            'train_loss': self.average_by_epoch(self.train_metrics['loss'], train_steps_per_epoch),
            'val_loss': self.average_by_epoch(self.val_metrics['loss'], val_steps_per_epoch),
            'train_iou': self.average_by_epoch(self.train_metrics['iou'], train_steps_per_epoch),
            'val_iou': self.average_by_epoch(self.val_metrics['iou'], val_steps_per_epoch),
            'train_dice': self.average_by_epoch(self.train_metrics['dice'], train_steps_per_epoch),
            'val_dice': self.average_by_epoch(self.val_metrics['dice'], val_steps_per_epoch)
        }
        
        # Asegurar que todas las listas tengan la misma longitud mínima
        min_len = min(len(values) for values in processed_metrics.values())
        
        # Recortar todas las listas a la longitud mínima
        for key in processed_metrics:
            processed_metrics[key] = processed_metrics[key][:min_len]
        
        return processed_metrics
    
    def create_dataframe(self, train_steps_per_epoch: int = 12, 
                        val_steps_per_epoch: int = 3) -> pd.DataFrame:
        """
        Crear DataFrame con las métricas procesadas
        
        Args:
            train_steps_per_epoch: Pasos de entrenamiento por época
            val_steps_per_epoch: Pasos de validación por época
            
        Returns:
            DataFrame con métricas por época
        """
        # Leer archivo si no se ha hecho
        if not self.train_metrics['loss'] and not self.val_metrics['loss']:
            lines = self.read_log_file()
            self.extract_metrics(lines)
        
        # Procesar métricas
        processed = self.process_metrics(train_steps_per_epoch, val_steps_per_epoch)
        
        # Crear DataFrame
        self.df = pd.DataFrame({
            "Epoch": list(range(1, len(processed['train_loss']) + 1)),
            "Train Loss": processed['train_loss'],
            "Val Loss": processed['val_loss'],
            "Train IoU": processed['train_iou'],
            "Val IoU": processed['val_iou'],
            "Train Dice": processed['train_dice'],
            "Val Dice": processed['val_dice']
        })
        
        return self.df
    
    def plot_metrics(self, figsize: Tuple[int, int] = (8, 10), 
                colors: List[str] = ['deepskyblue', 'crimson'],
                show_grid: bool = False,
                save_path: Optional[str] = None) -> plt.Figure:
        """
        Crear gráficos de las métricas (formato vertical como la imagen)
        """
        if self.df is None:
            raise ValueError("Debe crear el DataFrame primero usando create_dataframe()")
        
        with plt.style.context("default"):
            fig, axes = plt.subplots(3, 1, figsize=figsize)
            
            # 1. LOSS
            axes[0].plot(self.df.index, self.df["Val Loss"], c=colors[0], label="Val")
            axes[0].plot(self.df.index, self.df["Train Loss"], c=colors[1], label="Train")
            axes[0].set_title(
                f"Loss\nTrain: {self.df['Train Loss'].iloc[-1]:.4f} | Val: {self.df['Val Loss'].iloc[-1]:.4f}")
            axes[0].legend(loc="upper right")
            if show_grid:
                axes[0].grid(True)
            
            # 2. Dice
            axes[1].plot(self.df.index, self.df["Val Dice"], c=colors[0], label="Val")
            axes[1].plot(self.df.index, self.df["Train Dice"], c=colors[1], label="Train")
            axes[1].set_title(
                f"Dice Score\nTrain: {self.df['Train Dice'].iloc[-1]:.4f} | Val: {self.df['Val Dice'].iloc[-1]:.4f}")
            axes[1].legend(loc="upper right")
            if show_grid:
                axes[1].grid(True)
            
            # 3. IoU
            axes[2].plot(self.df.index, self.df["Val IoU"], c=colors[0], label="Val")
            axes[2].plot(self.df.index, self.df["Train IoU"], c=colors[1], label="Train")
            axes[2].set_title(
                f"IoU\nTrain: {self.df['Train IoU'].iloc[-1]:.4f} | Val: {self.df['Val IoU'].iloc[-1]:.4f}")
            axes[2].legend(loc="upper right")
            if show_grid:
                axes[2].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig

    
    def get_metrics_summary(self) -> Dict[str, float]:
        """
        Obtener resumen de métricas finales
        
        Returns:
            Diccionario con métricas finales
        """
        if self.df is None:
            raise ValueError("Debe crear el DataFrame primero usando create_dataframe()")
        
        return {
            'final_train_loss': self.df["Train Loss"].iloc[-1],
            'final_val_loss': self.df["Val Loss"].iloc[-1],
            'final_train_iou': self.df["Train IoU"].iloc[-1],
            'final_val_iou': self.df["Val IoU"].iloc[-1],
            'final_train_dice': self.df["Train Dice"].iloc[-1],
            'final_val_dice': self.df["Val Dice"].iloc[-1],
            'best_val_loss': self.df["Val Loss"].min(),
            'best_val_iou': self.df["Val IoU"].max(),
            'best_val_dice': self.df["Val Dice"].max()
        }
    
    def export_data(self, output_path: str, format: str = 'excel') -> None:
        """
        Exportar DataFrame a archivo
        
        Args:
            output_path: Ruta del archivo de salida
            format: Formato ('excel' o 'csv')
        """
        if self.df is None:
            raise ValueError("Debe crear el DataFrame primero usando create_dataframe()")
        
        output_path = Path(output_path)
        
        if format.lower() == 'excel':
            self.df.to_excel(output_path, index=False)
        elif format.lower() == 'csv':
            self.df.to_csv(output_path, index=False)
        else:
            raise ValueError("Formato no soportado. Use 'excel' o 'csv'")
        
        print(f"Datos exportados a: {output_path}")
    
    def analyze_complete_workflow(self, train_steps_per_epoch: int = 12, 
                                 val_steps_per_epoch: int = 3,
                                 show_plots: bool = True,
                                 save_plot_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Ejecutar el flujo completo de análisis
        
        Args:
            train_steps_per_epoch: Pasos de entrenamiento por época
            val_steps_per_epoch: Pasos de validación por época
            show_plots: Si mostrar los gráficos
            save_plot_path: Ruta para guardar los gráficos
            
        Returns:
            Tuple con DataFrame y resumen de métricas
        """
        # Crear DataFrame
        df = self.create_dataframe(train_steps_per_epoch, val_steps_per_epoch)
        
        # Mostrar DataFrame
        print("DataFrame creado:")
        print(df.head())
        
        # Crear gráficos
        fig = self.plot_metrics(save_path=save_plot_path)
        if show_plots:
            plt.show()
        
        # Obtener resumen
        summary = self.get_metrics_summary()
        
        return df, summary
