import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.config import Config
from src.traffic_analyzer import get_traffic_intensity

def annotate_frame(frame, results, left_count, right_count):
    processed_frame = results[0].plot(line_width=1)
    processed_frame[:Config.VERTICES_LEFT[0][1], :] = frame[:Config.VERTICES_LEFT[0][1], :].copy()
    processed_frame[Config.VERTICES_RIGHT[2][1]:, :] = frame[Config.VERTICES_RIGHT[2][1]:, :].copy()
    
    cv2.polylines(processed_frame, [np.array(Config.VERTICES_LEFT, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(processed_frame, [np.array(Config.VERTICES_RIGHT, np.int32)], True, (255, 0, 0), 2)
    
    left_intensity = get_traffic_intensity(left_count)
    cv2.rectangle(
        processed_frame,
        (Config.TEXT_POS_LEFT[0]-10, Config.TEXT_POS_LEFT[1]-25),
        (Config.TEXT_POS_LEFT[0]+460, Config.TEXT_POS_LEFT[1]+10),
        Config.BG_COLOR, -1
    )
    cv2.putText(
        processed_frame,
        f'Vehicles in Left Lane: {left_count}',
        Config.TEXT_POS_LEFT,
        Config.FONT,
        Config.FONT_SCALE,
        Config.FONT_COLOR,
        2,
        cv2.LINE_AA
    )
    cv2.rectangle(
        processed_frame,
        (Config.INTENSITY_POS_LEFT[0]-10, Config.INTENSITY_POS_LEFT[1]-25),
        (Config.INTENSITY_POS_LEFT[0]+460, Config.INTENSITY_POS_LEFT[1]+10),
        Config.BG_COLOR, -1
    )
    cv2.putText(
        processed_frame,
        f'Traffic Intensity: {left_intensity}',
        Config.INTENSITY_POS_LEFT,
        Config.FONT,
        Config.FONT_SCALE,
        Config.FONT_COLOR,
        2,
        cv2.LINE_AA
    )

    right_intensity = get_traffic_intensity(right_count)
    cv2.rectangle(
        processed_frame,
        (Config.TEXT_POS_RIGHT[0]-10, Config.TEXT_POS_RIGHT[1]-25),
        (Config.TEXT_POS_RIGHT[0]+460, Config.TEXT_POS_RIGHT[1]+10),
        Config.BG_COLOR, -1
    )
    cv2.putText(
        processed_frame,
        f'Vehicles in Right Lane: {right_count}',
        Config.TEXT_POS_RIGHT,
        Config.FONT,
        Config.FONT_SCALE,
        Config.FONT_COLOR,
        2,
        cv2.LINE_AA
    )
    cv2.rectangle(
        processed_frame,
        (Config.INTENSITY_POS_RIGHT[0]-10, Config.INTENSITY_POS_RIGHT[1]-25),
        (Config.INTENSITY_POS_RIGHT[0]+460, Config.INTENSITY_POS_RIGHT[1]+10),
        Config.BG_COLOR, -1
    )
    cv2.putText(
        processed_frame,
        f'Traffic Intensity: {right_intensity}',
        Config.INTENSITY_POS_RIGHT,
        Config.FONT,
        Config.FONT_SCALE,
        Config.FONT_COLOR,
        2,
        cv2.LINE_AA
    )
    return processed_frame

def save_video(frames, output_path=Config.OUTPUT_VIDEO, fps=Config.FPS):
    if not frames:
        return
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def plot_image(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 15))
    plt.imshow(image_rgb)
    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.show()

def plot_validation_samples(model, image_dir=Config.VALID_IMAGES, num_samples=9):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    selected_images = [image_files[i] for i in range(0, len(image_files), len(image_files) // num_samples)]
    fig, axes = plt.subplots(3, 3, figsize=(20, 21))
    fig.suptitle('Validation Set Inferences', fontsize=24)
    for i, ax in enumerate(axes.flatten()):
        img_path = os.path.join(image_dir, selected_images[i])
        results = model.predict(img_path, imgsz=Config.IMG_SIZE, conf=Config.CONF_THRESHOLD)
        annotated = results[0].plot(line_width=1)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ax.imshow(annotated_rgb)
        ax.axis('off')
    plt.tight_layout()
    plt.show()