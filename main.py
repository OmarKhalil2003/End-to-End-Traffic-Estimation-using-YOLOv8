import argparse
from src.config import Config
from src.data_loader import load_video, get_frame, validate_dataset
from src.model import load_model, predict
from src.traffic_analyzer import count_vehicles_in_lanes, get_traffic_intensity
from src.visualizer import annotate_frame, save_video, plot_image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def process_video(model, video_path=Config.VIDEO_PATH, output_path=Config.OUTPUT_VIDEO):
    cap = load_video(video_path)
    frames = []
    
    while True:
        frame = get_frame(cap)
        if frame is None:
            break
        detection_frame = frame.copy()
        detection_frame[:Config.VERTICES_LEFT[0][1], :] = 0
        detection_frame[Config.VERTICES_RIGHT[2][1]:, :] = 0
        results = predict(model, detection_frame)
        left_count, right_count = count_vehicles_in_lanes(results)
        annotated_frame = annotate_frame(frame, results, left_count, right_count)
        frames.append(annotated_frame)
    
    save_video(frames, output_path)
    cap.release()

def process_image(model, image_path):
    results = predict(model, image_path)
    annotated_image = results[0].plot(line_width=2)
    plot_image(annotated_image, 'Detected Objects in Sample Image')

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Traffic Estimation")
    parser.add_argument('--video', default=Config.VIDEO_PATH, help="Path to input video")
    parser.add_argument('--image', help="Path to input image")
    parser.add_argument('--output', default=Config.OUTPUT_VIDEO, help="Path to output video")
    args = parser.parse_args()
    
    model = load_model("yolov8n.pt")  
    if args.image:
        process_image(model, args.image)
    else:
        process_video(model, args.video, args.output)

if __name__ == "__main__":
    main()