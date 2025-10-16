import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import yaml
import tempfile
import os
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import zipfile
import io
from PIL import Image
import time

# Import YOLOv5 utilities
from data_preprocessing.datasets import LoadImages
from data_preprocessing.utils import non_max_suppression, scale_coords, plot_one_box
from models.yolo import Model

class RetailInventoryCounter:
    def __init__(self):
        self.model = None
        self.names = []
        self.colors = []
        self.device = 'cpu'
        self.load_model()
    
    def load_model(self):
        """Load the trained YOLOv5 model and class names"""
        try:
            # Load model
            weights_path = 'weights/yolov5s.pt'  # Your trained model
            self.model = torch.load(weights_path, map_location=self.device, weights_only=False)['model']
            self.model.to(self.device)
            self.model.eval()
            
            # Load custom class names from data.yaml
            with open('data.yaml', 'r') as f:
                data = yaml.safe_load(f)
            self.names = data['names']
            
            # Generate colors for each class
            self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
            
            st.success(f"✅ Model loaded successfully! {len(self.names)} retail classes available.")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def detect_objects_in_frame(self, frame, conf_threshold=0.5):
        """Detect objects in a single frame"""
        if self.model is None:
            return []
        
        # Preprocess frame
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_threshold, 0.45, classes=None, agnostic=False)
        
        detections = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to frame size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    if conf >= conf_threshold:
                        class_name = self.names[int(cls)]
                        detections.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': [int(x) for x in xyxy]
                        })
        
        return detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No intersection
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def process_video(self, video_path, conf_threshold=0.5, frame_skip=10, duplicate_threshold=0.3):
        """Process video and count inventory with duplicate detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Could not open video file")
            return {}
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        st.info(f"📹 Video Info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        # Process frames with duplicate detection
        inventory_counter = Counter()
        processed_frames = 0
        frame_count = 0
        
        # Track detected objects to prevent duplicates
        detected_objects = {}  # {class_name: [list of bbox positions]}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create temporary directory for annotated frames
        temp_dir = tempfile.mkdtemp()
        annotated_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing
            if frame_count % frame_skip == 0:
                # Detect objects in frame
                detections = self.detect_objects_in_frame(frame, conf_threshold)
                
                # Process detections with duplicate checking
                frame_detections = []
                for detection in detections:
                    class_name = detection['class']
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    
                    # Check if this is a duplicate of an existing detection
                    is_duplicate = False
                    if class_name in detected_objects:
                        for existing_bbox in detected_objects[class_name]:
                            if self._calculate_iou(bbox, existing_bbox) > duplicate_threshold:
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        # New unique detection
                        inventory_counter[class_name] += 1
                        detected_objects[class_name] = detected_objects.get(class_name, []) + [bbox]
                        frame_detections.append(detection)
                        print(f"✅ New unique detection: {class_name} (confidence: {confidence:.2f})")
                    else:
                        print(f"🔄 Duplicate detected: {class_name} (confidence: {confidence:.2f})")
                
                print(f"Frame {frame_count}: {len(frame_detections)} unique detections, total unique items: {sum(inventory_counter.values())}")
                
                # Draw bounding boxes and labels (only for unique detections)
                annotated_frame = frame.copy()
                for detection in frame_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    label = f"{detection['class']} {detection['confidence']:.2f}"
                    color = self.colors[self.names.index(detection['class']) % len(self.colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Save annotated frame
                frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, annotated_frame)
                annotated_frames.append(frame_path)
                
                processed_frames += 1
            
            frame_count += 1
            
            # Update progress
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} ({processed_frames} processed)")
        
        cap.release()
        progress_bar.progress(1.0)
        status_text.text("✅ Video processing completed!")
        
        # Debug: Print final results
        final_inventory = dict(inventory_counter)
        print(f"Final inventory: {final_inventory}")
        print(f"Total unique items detected: {sum(final_inventory.values())}")
        
        return final_inventory, annotated_frames, temp_dir
    
    def generate_inventory_report(self, inventory_dict):
        """Generate a comprehensive inventory report"""
        if not inventory_dict:
            return "No items detected in the video."
        
        # Sort by count (descending)
        sorted_items = sorted(inventory_dict.items(), key=lambda x: x[1], reverse=True)
        
        report = f"""
# 🏪 Retail Inventory Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Inventory Summary
"""
        
        total_items = sum(inventory_dict.values())
        report += f"**Total Items Detected:** {total_items}\n"
        report += f"**Unique Product Types:** {len(inventory_dict)}\n\n"
        
        # Detailed breakdown
        report += "## 📦 Product Breakdown\n\n"
        report += "| Product | Quantity | Percentage |\n"
        report += "|---------|----------|------------|\n"
        
        for product, count in sorted_items:
            percentage = (count / total_items) * 100
            report += f"| {product} | {count} | {percentage:.1f}% |\n"
        
        return report

def main():
    st.set_page_config(
        page_title="🏪 Retail Inventory Counter",
        page_icon="🏪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏪 Retail Inventory Counter")
    st.markdown("**AI-Powered Video Inventory Analysis for Retail Stores**")
    
    # Initialize the inventory counter
    if 'inventory_counter' not in st.session_state:
        st.session_state.inventory_counter = RetailInventoryCounter()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model status
    if st.session_state.inventory_counter.model is not None:
        st.sidebar.success("✅ Model Ready")
        st.sidebar.info(f"📊 {len(st.session_state.inventory_counter.names)} Classes Available")
    else:
        st.sidebar.error("❌ Model Not Loaded")
        st.stop()
    
    # Configuration options
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher values = more confident detections only"
    )
    
    frame_skip = st.sidebar.slider(
        "Frame Skip",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
        help="Process every Nth frame (higher = faster processing)"
    )
    
    duplicate_threshold = st.sidebar.slider(
        "Duplicate Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="IoU threshold for considering detections as duplicates (higher = stricter)"
    )
    
    # Main content area
    tab1, tab2 = st.tabs(["📹 Video Upload", "📊 Results"])
    
    with tab1:
        st.header("📹 Upload Your Shop Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video of your shop to analyze inventory"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            st.success("✅ Video uploaded successfully!")
            
            # Video preview
            st.subheader("🎬 Video Preview")
            st.video(video_path)
            
            # Process button
            if st.button("🚀 Process Video", type="primary"):
                with st.spinner("Processing video... This may take a few minutes."):
                    inventory_dict, annotated_frames, temp_dir = st.session_state.inventory_counter.process_video(
                        video_path, conf_threshold, frame_skip, duplicate_threshold
                    )
                
                # Store results in session state
                st.session_state.inventory_dict = inventory_dict
                st.session_state.annotated_frames = annotated_frames
                st.session_state.temp_dir = temp_dir
                
                st.success("✅ Processing completed!")
                
                # Clean up
                os.unlink(video_path)
    
    with tab2:
        st.header("📊 Inventory Results")
        
        # Check if we have results
        if 'inventory_dict' in st.session_state and st.session_state.inventory_dict:
            inventory_dict = st.session_state.inventory_dict
            
            # Debug: Show what we have
            st.write("Debug - Inventory dict:", inventory_dict)
            st.write("Debug - Dict length:", len(inventory_dict))
            
            if inventory_dict:  # Check if dict is not empty
                # Summary cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Items", sum(inventory_dict.values()))
                
                with col2:
                    st.metric("Product Types", len(inventory_dict))
                
                with col3:
                    if inventory_dict:
                        most_common = max(inventory_dict.items(), key=lambda x: x[1])
                        st.metric("Most Common", f"{most_common[0]} ({most_common[1]})")
                
                with col4:
                    if inventory_dict:
                        avg_count = sum(inventory_dict.values()) / len(inventory_dict)
                        st.metric("Avg per Type", f"{avg_count:.1f}")
                
                # Detailed results
                st.subheader("📦 Detailed Inventory")
                
                # Create DataFrame for better display
                df = pd.DataFrame(list(inventory_dict.items()), columns=['Product', 'Quantity'])
                df = df.sort_values('Quantity', ascending=False)
                df['Percentage'] = (df['Quantity'] / df['Quantity'].sum() * 100).round(1)
                
                st.dataframe(df, width='stretch')
                
                # Generate and display report
                st.subheader("📋 Inventory Report")
                report = st.session_state.inventory_counter.generate_inventory_report(inventory_dict)
                st.markdown(report)
                
                # Download options
                st.subheader("💾 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv,
                        file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Download Report
                    st.download_button(
                        label="📋 Download Report",
                        data=report,
                        file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col3:
                    # Download Annotated Frames
                    if 'annotated_frames' in st.session_state and st.session_state.annotated_frames:
                        # Create ZIP file
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            for i, frame_path in enumerate(st.session_state.annotated_frames[:50]):  # Limit to 50 frames
                                if os.path.exists(frame_path):
                                    zip_file.write(frame_path, f"annotated_frame_{i:06d}.jpg")
                        
                        st.download_button(
                            label="🖼️ Download Frames",
                            data=zip_buffer.getvalue(),
                            file_name=f"annotated_frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
            else:
                st.warning("⚠️ No items detected in the video. Try lowering the confidence threshold.")
        
        else:
            st.info("👆 Upload and process a video to see results here.")
    
    
    # Footer
    st.markdown("---")
    st.markdown("**🏪 Retail Inventory Counter** - Powered by YOLOv5 AI")
    st.markdown("Upload your shop video to get an automated inventory count!")

if __name__ == "__main__":
    main()
