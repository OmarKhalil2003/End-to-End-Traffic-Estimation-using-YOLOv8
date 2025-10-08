from src.config import Config

def count_vehicles_in_lanes(results):
    bounding_boxes = results[0].boxes.xyxy
    left_count = sum(1 for box in bounding_boxes if box[0] < Config.LANE_THRESHOLD)
    right_count = sum(1 for box in bounding_boxes if box[0] >= Config.LANE_THRESHOLD)
    return left_count, right_count

def get_traffic_intensity(count):
    return "Heavy" if count > Config.HEAVY_TRAFFIC_THRESHOLD else "Smooth"