import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_tensorboard_metrics(events_path):
    ea = event_accumulator.EventAccumulator(events_path)
    ea.Reload()
    metrics = ['Train/Loss', 'Val/Loss', 'Train/IOU', 'Val/IOU', 'Train/DICE', 'Val/DICE']
    dfs = []
    for metric in metrics:
        if metric in ea.Tags()['scalars']:
            events = ea.Scalars(metric)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            df_temp = pd.DataFrame({'step': steps, metric: values})
            dfs.append(df_temp)
    df = dfs[0]
    for df_temp in dfs[1:]:
        df = pd.merge(df, df_temp, on='step', how='outer')
    df = df.sort_values(by='step').reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle("Métricas de entrenamiento y validación")
    def plot_if_exists(ax, df, col, label, style='-'):
        if col in df.columns:
            ax.plot(df['step'], df[col], style, label=label)
    plot_if_exists(axes[0], df, 'Train/Loss', 'Train Loss')
    plot_if_exists(axes[0], df, 'Val/Loss', 'Val Loss')
    axes[0].set_title("Loss por Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    plot_if_exists(axes[1], df, 'Train/IOU', 'Train IoU')
    plot_if_exists(axes[1], df, 'Val/IOU', 'Val IoU')
    axes[1].set_title("IoU por Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].legend()
    axes[1].grid(True)
    plot_if_exists(axes[2], df, 'Train/DICE', 'Train Dice')
    plot_if_exists(axes[2], df, 'Val/DICE', 'Val Dice')
    axes[2].set_title("Dice por Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
