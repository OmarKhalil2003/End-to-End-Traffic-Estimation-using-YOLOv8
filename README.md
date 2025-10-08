
# Real-Time Traffic Density Estimation using YOLOv8

This project implements an **End-to-End Computer Vision pipeline** for real-time **vehicle detection and traffic density estimation** using **YOLOv8**.  
It processes traffic videos or images to detect vehicles, count them per lane, and visualize **traffic intensity** (“Smooth” or “Heavy”) dynamically.

---

## Project Overview

The system:
- Uses **YOLOv8** for vehicle detection.
- Counts vehicles in left and right lanes.
- Analyzes lane-based **traffic intensity**.
- Displays annotated output frames and saves the processed video.
- Includes dataset validation, evaluation visualization, and model training utilities.

---

## Project Structure

```

(End-to-End) Real-Time Traffic Estimation/
│
├── src/
│   ├── config.py              # Configuration paths and parameters
│   ├── data_loader.py         # Data loading and validation utilities
│   ├── model.py               # Model loading, training, and inference
│   ├── evaluate.py            # Performance visualization and metrics
│   ├── traffic_analyzer.py    # Vehicle counting and traffic intensity logic
│   └── visualizer.py          # Frame annotation and visualization functions
│
├── main.py                    # Main entry point (run video/image inference)
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

````

---

## Setup Instructions

### 1️⃣ Clone or Download the Repository
```bash
git clone https://github.com/OmarKhalil2003/End-to-End-Traffic-Estimation-using-YOLOv8
cd traffic-density-estimation
````

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  How to Run

### Process a Video

```bash
python main.py --video "data/sample_video.mp4"
```

### Process a Single Image

```bash
python main.py --image "data/sample_image.jpg"
```

Output video will be saved in:

```
output/traffic_density_analysis.mp4
```

---

##  Configuration

All paths and parameters are defined in `src/config.py`, including:

* `MODEL_PATH`: YOLOv8 model weights (`yolov8n.pt` by default)
* `DATASET_PATH`: Dataset directory path
* `LANE_THRESHOLD`: x-coordinate dividing left and right lanes
* `HEAVY_TRAFFIC_THRESHOLD`: Count threshold for heavy traffic
* `IMG_SIZE`, `CONF_THRESHOLD`, etc.

You can easily adjust them as needed.

---

## Evaluation & Visualization

* **Training and validation loss curves** are plotted via `evaluate.py`.
* **Confusion matrix** and **validation samples** visualization are included.
* **Dataset integrity check** via `data_loader.validate_dataset()`.

---

## Key Libraries Used

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* OpenCV
* NumPy
* Matplotlib
* Pillow
* Seaborn
* Pandas
* PyYAML

---

## Example Output

| Input Video     | Output (Annotated)                                               |
| --------------- | ---------------------------------------------------------------- |
| Traffic footage | Detected vehicles with left/right lane counts & intensity labels |

---

##  Author

**Omar Khalil**
MSc Student @ AASTMT
Connect on [LinkedIn](https://www.linkedin.com/in/omar-khalil-10om01) or check more projects!

---

## License

This project is for **educational and research purposes**.
All dataset and model weights belong to their respective owners.

---
