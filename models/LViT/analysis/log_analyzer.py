# -*- coding: utf-8 -*-
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List

class TrainingLogAnalyzer:
    def __init__(self, log_path: str, debug: bool = False):
        """
        Inicializar el analizador de logs de entrenamiento
        
        Args:
            log_path: Ruta al archivo de log
            debug: Si True, muestra información de depuración
        """
        self.log_path = Path(log_path)
        self.debug = debug
        self.train_metrics = {}
        self.val_metrics = {}
        self.combined_df = None
    
    def debug_print(self, message: str):
        """Imprimir mensaje de debug si está habilitado"""
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def parse_log_file_flexible(self) -> Tuple[Dict, Dict]:
        """
        Parsing flexible que maneja ambos formatos de log
        """
        if not self.log_path.exists():
            raise FileNotFoundError(f"Archivo de log no encontrado: {self.log_path}")
        
        train_metrics = {}
        val_metrics = {}
        
        with open(self.log_path, "r") as file:
            lines = file.readlines()
        
        if self.debug:
            self.debug_print(f"Procesando {len(lines)} líneas del archivo")
        
        # Recorrer el log con parsing flexible
        for line in lines:
            # TRAIN
            if "[Train]" in line and re.search(r"\[\d+/\d+\]", line):
                # Intentar FORMATO 1: Con "Total Loss" (segundo código)
                match = re.search(
                    r"Epoch:\s*\[(\d+)\]\[(\d+)/\d+\]\s+Loss:[\d.]+\s+\(Avg\s+([\d.]+)\)\s+Regression Loss:[\d.]+\s+\(Avg\s+([\d.]+)\)\s+Total Loss:[\d.]+\s+\(Avg\s+([\d.]+)\)\s+IoU:[\d.]+\s+\(Avg\s+([\d.]+)\)\s+Dice:[\d.]+\s+\(Avg\s+([\d.]+)\)",
                    line
                )
                if match:
                    epoch = int(match.group(1))
                    step = int(match.group(2))
                    if epoch not in train_metrics or step > train_metrics[epoch].get("step", 0):
                        train_metrics[epoch] = {
                            "step": step,
                            "Loss": float(match.group(3)),
                            "RegLoss": float(match.group(4)),
                            "IoU": float(match.group(6)),  # Nota: saltamos el grupo 5 (Total Loss)
                            "Dice": float(match.group(7)),
                        }
                        if self.debug:
                            self.debug_print(f"TRAIN Formato 1 - Época {epoch}, Step {step}: Loss={match.group(3)}")
                else:
                    # Intentar FORMATO 2: Sin "Total Loss" (código original)
                    match = re.search(
                        r"Epoch:\s*\[(\d+)\]\[(\d+)/\d+\]\s+Loss:[\d.]+\s+\(Avg\s+([\d.]+)\).*?Regression Loss:[\d.]+\(Avg\s+([\d.]+)\).*?IoU:[\d.]+\s+\(Avg\s+([\d.]+)\).*?Dice:[\d.]+\s+\(Avg\s+([\d.]+)\)",
                        line
                    )
                    if match:
                        epoch = int(match.group(1))
                        step = int(match.group(2))
                        if epoch not in train_metrics or step > train_metrics[epoch].get("step", 0):
                            train_metrics[epoch] = {
                                "step": step,
                                "Loss": float(match.group(3)),
                                "RegLoss": float(match.group(4)),
                                "IoU": float(match.group(5)),
                                "Dice": float(match.group(6)),
                            }
                            if self.debug:
                                self.debug_print(f"TRAIN Formato 2 - Época {epoch}, Step {step}: Loss={match.group(3)}")

            # VAL
            elif "[Val]" in line and re.search(r"\[\d+/\d+\]", line):
                # Intentar FORMATO 1: Con "Total Loss" (segundo código)
                match = re.search(
                    r"Epoch:\s*\[(\d+)\]\[(\d+)/\d+\]\s+Loss:[\d.]+\s+\(Avg\s+([\d.]+)\)\s+Regression Loss:[\d.]+\s+\(Avg\s+([\d.]+)\)\s+Total Loss:[\d.]+\s+\(Avg\s+([\d.]+)\)\s+IoU:[\d.]+\s+\(Avg\s+([\d.]+)\)\s+Dice:[\d.]+\s+\(Avg\s+([\d.]+)\)",
                    line
                )
                if match:
                    epoch = int(match.group(1))
                    step = int(match.group(2))
                    if epoch not in val_metrics or step > val_metrics[epoch].get("step", 0):
                        val_metrics[epoch] = {
                            "step": step,
                            "Loss": float(match.group(3)),
                            "RegLoss": float(match.group(4)),
                            "IoU": float(match.group(6)),  # Nota: saltamos el grupo 5 (Total Loss)
                            "Dice": float(match.group(7)),
                        }
                        if self.debug:
                            self.debug_print(f"VAL Formato 1 - Época {epoch}, Step {step}: Loss={match.group(3)}")
                else:
                    # Intentar FORMATO 2: Sin "Total Loss" (código original)
                    match = re.search(
                        r"Epoch:\s*\[(\d+)\]\[(\d+)/\d+\]\s+Loss:[\d.]+\s+\(Avg\s+([\d.]+)\).*?Regression Loss:[\d.]+\(Avg\s+([\d.]+)\).*?IoU:[\d.]+\s+\(Avg\s+([\d.]+)\).*?Dice:[\d.]+\s+\(Avg\s+([\d.]+)\)",
                        line
                    )
                    if match:
                        epoch = int(match.group(1))
                        step = int(match.group(2))
                        if epoch not in val_metrics or step > val_metrics[epoch].get("step", 0):
                            val_metrics[epoch] = {
                                "step": step,
                                "Loss": float(match.group(3)),
                                "RegLoss": float(match.group(4)),
                                "IoU": float(match.group(5)),
                                "Dice": float(match.group(6)),
                            }
                            if self.debug:
                                self.debug_print(f"VAL Formato 2 - Época {epoch}, Step {step}: Loss={match.group(3)}")
        
        if self.debug:
            self.debug_print(f"Métricas extraídas - Train: {len(train_metrics)}, Val: {len(val_metrics)}")
        
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        
        return train_metrics, val_metrics

    
    def parse_log_file(self, train_steps_per_epoch: int = 12, val_steps_per_epoch: int = 3) -> Tuple[Dict, Dict]:
        """
        Método principal de parsing simplificado
        """
        return self.parse_log_file_flexible()
    
    def create_dataframes(self, train_steps_per_epoch: int = 12, val_steps_per_epoch: int = 3) -> pd.DataFrame:
        """
        Crear DataFrames exactamente como el código que funciona
        """
        if not self.train_metrics and not self.val_metrics:
            self.parse_log_file(train_steps_per_epoch, val_steps_per_epoch)
        
        if not self.train_metrics and not self.val_metrics:
            print("ERROR: No se pudieron extraer métricas del log")
            # CORRECCIÓN PRINCIPAL: Asegurar que combined_df nunca sea None
            self.combined_df = pd.DataFrame()
            return self.combined_df
        
        # Convertir a DataFrames - EXACTAMENTE como el código que funciona
        train_df = pd.DataFrame.from_dict(self.train_metrics, orient='index')
        val_df = pd.DataFrame.from_dict(self.val_metrics, orient='index')
        
        # Eliminar columna "step" - EXACTAMENTE como el código que funciona
        if not train_df.empty and "step" in train_df.columns:
            train_df = train_df.drop(columns=["step"])
        if not val_df.empty and "step" in val_df.columns:
            val_df = val_df.drop(columns=["step"])
        
        train_df.index.name = "Epoch"
        val_df.index.name = "Epoch"
        
        # Unificar para graficar juntos - EXACTAMENTE como el código que funciona
        df = train_df.join(val_df, lsuffix="_Train", rsuffix="_Val")
        self.combined_df = df
        
        if self.debug:
            print(f"Métricas extraídas: {len(train_df)} épocas de train, {len(val_df)} épocas de val")
            print("Columnas del DataFrame:", df.columns.tolist())
        
        return df
    
    def display_dataframe(self):
        """
        Mostrar el DataFrame con manejo correcto de None
        """
        if self.combined_df is None:
            self.create_dataframes()
        
        # CORRECCIÓN PRINCIPAL: Verificar que combined_df no sea None
        if self.combined_df is not None:
            if not self.combined_df.empty:
                try:
                    from IPython.display import display
                    display(self.combined_df)
                except ImportError:
                    print(self.combined_df)
            else:
                print("No hay datos para mostrar - DataFrame vacío")
        else:
            print("Error: No se pudo crear el DataFrame")
    
    def plot_metrics(self, figsize: Tuple[int, int] = (18, 6), 
                style: str = "simple",  # "modern" o "simple"
                save_path: Optional[str] = None) -> plt.Figure:
        """
        Crear gráficos exactamente como el código que funciona
        """
        if self.combined_df is None:
            self.create_dataframes()
        
        if self.combined_df is None or self.combined_df.empty:
            print("No hay datos para graficar")
            return None
        
        # DEFINIR COLORES PERSONALIZADOS
        train_color = 'deepskyblue'
        val_color = 'crimson'
        
        # Plot con colores personalizados
        plt.figure(figsize=figsize)

        plt.subplot(2, 2, 1)
        plt.plot(self.combined_df.index, self.combined_df["Loss_Train"], 
                color=train_color, label="Train")
        plt.plot(self.combined_df.index, self.combined_df["Loss_Val"], 
                color=val_color, label="Val")
        plt.title("Loss por Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(self.combined_df.index, self.combined_df["RegLoss_Train"], 
                color=train_color, label="Train")
        plt.plot(self.combined_df.index, self.combined_df["RegLoss_Val"], 
                color=val_color, label="Val")
        plt.title("Regression Loss por Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Regression Loss")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(self.combined_df.index, self.combined_df["IoU_Train"], 
                color=train_color, label="Train")
        plt.plot(self.combined_df.index, self.combined_df["IoU_Val"], 
                color=val_color, label="Val")
        plt.title("IoU por Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(self.combined_df.index, self.combined_df["Dice_Train"], 
                color=train_color, label="Train")
        plt.plot(self.combined_df.index, self.combined_df["Dice_Val"], 
                color=val_color, label="Val")
        plt.title("Dice Score")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return plt.gcf()

    
    def get_metrics_summary(self) -> Dict[str, float]:
        """
        Obtener resumen de métricas finales
        """
        if self.combined_df is None:
            self.create_dataframes()
        
        if self.combined_df is None or self.combined_df.empty:
            return {}
        
        return {
            'final_train_loss': self.combined_df["Loss_Train"].iloc[-1],
            'final_val_loss': self.combined_df["Loss_Val"].iloc[-1],
            'final_train_regloss': self.combined_df["RegLoss_Train"].iloc[-1],
            'final_val_regloss': self.combined_df["RegLoss_Val"].iloc[-1],
            'final_train_iou': self.combined_df["IoU_Train"].iloc[-1],
            'final_val_iou': self.combined_df["IoU_Val"].iloc[-1],
            'final_train_dice': self.combined_df["Dice_Train"].iloc[-1],
            'final_val_dice': self.combined_df["Dice_Val"].iloc[-1],
            'best_val_loss': self.combined_df["Loss_Val"].min(),
            'best_val_regloss': self.combined_df["RegLoss_Val"].min(),
            'best_val_iou': self.combined_df["IoU_Val"].max(),
            'best_val_dice': self.combined_df["Dice_Val"].max()
        }
    
    def export_data(self, output_path: str, format: str = 'excel') -> None:
        """
        Exportar DataFrame a archivo
        """
        if self.combined_df is None:
            self.create_dataframes()
        
        if self.combined_df is None or self.combined_df.empty:
            print("No hay datos para exportar")
            return
        
        output_path = Path(output_path)
        
        if format.lower() == 'excel':
            self.combined_df.to_excel(output_path)
        elif format.lower() == 'csv':
            self.combined_df.to_csv(output_path)
        else:
            raise ValueError("Formato no soportado. Use 'excel' o 'csv'")
        
        print(f"Datos exportados a: {output_path}")
