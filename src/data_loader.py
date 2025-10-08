import os
import cv2
from PIL import Image
import yaml
from src.config import Config

def load_yaml(yaml_path=Config.YAML_FILE):
    try:
        with open(yaml_path, 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise ValueError(f"YAML file not found: {yaml_path}")

def get_image_sizes(image_dir):
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    image_sizes = set()
    num_images = 0
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            num_images += 1
            img_path = os.path.join(image_dir, filename)
            try:
                with Image.open(img_path) as img:
                    image_sizes.add(img.size)
            except Exception as e:
                print(f"Warning: Could not read image {img_path}: {e}")
    return num_images, image_sizes

def validate_dataset():
    try:
        train_num, train_sizes = get_image_sizes(Config.TRAIN_IMAGES)
        valid_num, valid_sizes = get_image_sizes(Config.VALID_IMAGES)
        return {
            "train_images": train_num,
            "valid_images": valid_num,
            "train_size": train_sizes.pop() if len(train_sizes) == 1 else "Varying",
            "valid_size": valid_sizes.pop() if len(valid_sizes) == 1 else "Varying"
        }
    except ValueError as e:
        raise ValueError(f"Dataset validation failed: {e}")

def load_video(video_path=Config.VIDEO_PATH):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    return frame if ret else None