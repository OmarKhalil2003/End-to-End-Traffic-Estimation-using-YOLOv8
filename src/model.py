from ultralytics import YOLO
from src.config import Config

def load_model(model_path=Config.MODEL_PATH):
    try:
        return YOLO(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
    
def train_model(model, yaml_path=Config.YAML_FILE):
    results = model.train(
        data=yaml_path,
        epochs=25,
        imgsz=Config.IMG_SIZE,
        device=0,
        patience=10,
        batch=32,
        optimizer='Adam',
        lr0=0.0001,
        lrf=0.1,
        dropout=0.1,
        seed=0
    )
    return results

def predict(model, source, imgsz=Config.IMG_SIZE, conf=Config.CONF_THRESHOLD):
    return model.predict(source=source, imgsz=imgsz, conf=conf)