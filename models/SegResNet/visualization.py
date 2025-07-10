import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_history(log_path='training.log'):
    """Plot training history from CSV log"""
    history = pd.read_csv(log_path, sep=',', engine='python')
    
    hist = history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    epoch = range(len(acc))
    loss = hist['loss']
    val_loss = hist['val_loss']
    train_dice1 = hist['dice_class1']
    val_dice = hist['val_dice_class1']
    
    with plt.style.context("default"):
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        colors = ['deepskyblue', 'crimson']
        
        # Loss
        axes[0].plot(epoch, val_loss, c=colors[0], label="Val")
        axes[0].plot(epoch, loss, c=colors[1], label="Train")
        axes[0].set_title(f"Loss\nTrain: {loss.iloc[-1]:.4f} | Val: {val_loss.iloc[-1]:.4f}")
        axes[0].legend(loc="upper right")
        
        # Accuracy
        axes[1].plot(epoch, val_acc, c=colors[0], label="Val")
        axes[1].plot(epoch, acc, c=colors[1], label="Train")
        axes[1].set_title(f"Accuracy\nTrain: {acc.iloc[-1]:.4f} | Val: {val_acc.iloc[-1]:.4f}")
        axes[1].legend(loc="upper right")
        
        # Dice Coefficient
        axes[2].plot(epoch, val_dice, c=colors[0], label="Val")
        axes[2].plot(epoch, train_dice1, c=colors[1], label="Train")
        axes[2].set_title(f"Dice Coefficient\nTrain: {train_dice1.iloc[-1]:.4f} | Val: {val_dice.iloc[-1]:.4f}")
        axes[2].legend(loc="upper right")
        
        plt.tight_layout()
        plt.show()

def plot_dice_by_class(log_path='training.log'):
    """Plot dice scores by class"""
    df = pd.read_csv(log_path)
    
    class_names = ["Core", "Edema", "EnhancingTumour"]
    train_cols = ["dice_class1", "dice_class2", "dice_class3"]
    val_cols = ["val_dice_class1", "val_dice_class2", "val_dice_class3"]
    
    with plt.style.context("default"):
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        colors = ['deepskyblue', 'crimson']
        
        for i in range(3):
            axes[i].plot(df["epoch"], df[val_cols[i]], c=colors[0], label="Val")
            axes[i].plot(df["epoch"], df[train_cols[i]], c=colors[1], label="Train")
            
            train_final = df[train_cols[i]].iloc[-1]
            val_final = df[val_cols[i]].iloc[-1]
            axes[i].set_title(f"Dice Score - {class_names[i]}\nTrain: {train_final:.4f} | Val: {val_final:.4f}")
            
            axes[i].legend(loc="upper right")
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()

def plot_validation_metrics(model, val_generator):
    """Plot validation metrics with custom styling"""
    import tensorflow as tf
    
    def dice_score(y_true, y_pred, smooth=1e-6):
        intersection = np.sum(y_true * y_pred, axis=(0, 1, 2))
        union = np.sum(y_true, axis=(0, 1, 2)) + np.sum(y_pred, axis=(0, 1, 2))
        return (2. * intersection + smooth) / (union + smooth)
    
    def iou_score(y_true, y_pred, smooth=1e-6):
        intersection = np.sum(y_true * y_pred, axis=(0, 1, 2))
        union = np.sum(y_true + y_pred, axis=(0, 1, 2)) - intersection
        return (intersection + smooth) / (union + smooth)
    
    dice_scores = []
    iou_scores = []
    
    for i in range(len(val_generator)):
        x_val, y_val = val_generator[i]
        y_pred = model.predict(x_val)
        
        y_val_argmax = np.argmax(y_val, axis=-1)
        y_pred_argmax = np.argmax(y_pred, axis=-1)
        
        for b in range(x_val.shape[0]):
            y_true_b = tf.one_hot(y_val_argmax[b], depth=4).numpy()
            y_pred_b = tf.one_hot(y_pred_argmax[b], depth=4).numpy()
            
            dice_scores.append(dice_score(y_true_b, y_pred_b))
            iou_scores.append(iou_score(y_true_b, y_pred_b))
    
    dice_scores = np.mean(dice_scores, axis=0)
    iou_scores = np.mean(iou_scores, axis=0)
    
    labels = ['WT dice', 'WT jaccard', 'TC dice', 'TC jaccard', 'ET dice', 'ET jaccard']
    values = [
        dice_scores[1], iou_scores[1],
        dice_scores[2], iou_scores[2],
        dice_scores[3], iou_scores[3],
    ]
    
    colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
    palette = sns.color_palette(colors)
    
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = sns.barplot(x=labels, y=values, palette=palette, ax=ax)
    
    ax.set_ylim(0, 1)
    ax.set_title("Dice and Jaccard Coefficients from Validation", fontsize=14)
    ax.set_xticklabels(labels, fontsize=11, rotation=30)
    
    for i, p in enumerate(bars.patches):
        height = p.get_height()
        ax.annotate(f'{height*100:.1f}%', (p.get_x() + p.get_width() / 2. - 0.1, height + 0.02),
                    fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("metrics_custom_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
