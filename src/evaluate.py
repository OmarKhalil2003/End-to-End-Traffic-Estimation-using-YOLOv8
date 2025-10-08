import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
from src.config import Config

def plot_learning_curve(df, train_col, val_col, title):
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x='epoch', y=train_col, label='Train Loss', color='#141140', linestyle='-', linewidth=2)
    sns.lineplot(data=df, x='epoch', y=val_col, label='Validation Loss', color='orangered', linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, results_csv_path, confusion_matrix_path):
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()
    
    plot_learning_curve(df, 'train/box_loss', 'val/box_loss', 'Box Loss Learning Curve')
    plot_learning_curve(df, 'train/cls_loss', 'val/cls_loss', 'Classification Loss Learning Curve')
    plot_learning_curve(df, 'train/dfl_loss', 'val/dfl_loss', 'Distribution Focal Loss Learning Curve')
    
    cm_img = cv2.imread(confusion_matrix_path)
    cm_img = cv2.cvtColor(cm_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10), dpi=120)
    plt.imshow(cm_img)
    plt.axis('off')
    plt.show()
    
    metrics = model.val(split='val')
    metrics_df = pd.DataFrame.from_dict(metrics.results_dict, orient='index', columns=['Metric Value'])
    return metrics_df.round(3)