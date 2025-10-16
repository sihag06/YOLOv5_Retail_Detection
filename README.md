# YOLOv5 Retail Item Detection System

## 🎯 Project Overview
AI-powered retail inventory management system using YOLOv5 for automated product detection and counting across 73 retail categories.

## 📊 Performance Metrics
- **mAP@0.5:** 80.9%
- **Precision:** 48.9%
- **Recall:** 87.4%
- **Inference Time:** 0.1-0.15s per frame
- **Classes:** 73 retail product categories

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv yolov5-env
source yolov5-env/bin/activate  # On Windows: yolov5-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Detection
```bash
# Detect objects in images
python detect.py --weights weights/yolov5s.pt --source input_ouput_images/images --conf 0.3

# Detect single image
python detect.py --weights weights/yolov5s.pt --source your_image.jpg --conf 0.3
```

### 3. Train Model
```bash
# Train on custom dataset
python train.py --weights weights/yolov5s.pt --data data.yaml --epochs 100 --batch-size 16
```

### 4. Run Web Application
```bash
# Launch Streamlit app
streamlit run retail_inventory_app.py
```

### 5. Evaluate Model
```bash
# Test model performance
python test.py --weights weights/best.pt --data data.yaml --verbose
```

## 📁 Project Structure
```
├── data_preprocessing/     # Data loading and preprocessing utilities
├── models/                 # YOLOv5 model definitions
├── weights/               # Trained model weights
├── input_ouput_images/    # Input images and output results
├── detect.py              # Object detection script
├── train.py               # Model training script
├── test.py                # Model evaluation script
├── retail_inventory_app.py # Streamlit web application
├── data.yaml              # Dataset configuration
└── requirements.txt       # Python dependencies
```

## 🔧 Technical Stack
- **Deep Learning:** PyTorch 2.8.0, YOLOv5
- **Computer Vision:** OpenCV
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pandas
- **Visualization:** Plotly, Matplotlib

## 📈 Features
- Real-time object detection
- Multi-class classification (73 categories)
- Video processing capabilities
- Duplicate detection algorithm
- Interactive web interface
- Automated inventory reports
- CSV export functionality

## 🎯 Use Cases
- Retail inventory management
- Automated product counting
- Store surveillance analysis
- Inventory auditing
- Supply chain optimization

## 📞 Support
For questions or issues, please refer to the YOLOv5 documentation or create an issue in the project repository.
