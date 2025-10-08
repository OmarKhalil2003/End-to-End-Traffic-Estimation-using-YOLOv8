import os

class Config:
    MODEL_PATH = "yolov8n.pt"
    DATASET_PATH = "D:\certificates\Side Projects\Computer Vision\(End-to-End) Real-Time Traffic Estimation\data\Vehicle_Detection_Image_Dataset"
    YAML_FILE = os.path.join(DATASET_PATH, "data.yaml")
    TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images")
    VALID_IMAGES = os.path.join(DATASET_PATH, "valid", "images")
    OUTPUT_DIR = "D:\certificates\Side Projects\Computer Vision\(End-to-End) Real-Time Traffic Estimation\output"
    VIDEO_PATH = os.path.join(DATASET_PATH, "sample_video.mp4")
    OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "traffic_density_analysis.mp4")
    IMG_SIZE = 640
    CONF_THRESHOLD = 0.4
    HEAVY_TRAFFIC_THRESHOLD = 10
    LANE_THRESHOLD = 609
    VERTICES_LEFT = [(465, 350), (609, 350), (510, 630), (2, 630)]
    VERTICES_RIGHT = [(678, 350), (815, 350), (1203, 630), (743, 630)]
    TEXT_POS_LEFT = (10, 50)
    TEXT_POS_RIGHT = (820, 50)
    INTENSITY_POS_LEFT = (10, 100)
    INTENSITY_POS_RIGHT = (820, 100)
    FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1
    FONT_COLOR = (255, 255, 255)
    BG_COLOR = (0, 0, 255)
    FPS = 20