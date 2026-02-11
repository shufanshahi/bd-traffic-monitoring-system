import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import tempfile
import subprocess
import os

# Page configuration
st.set_page_config(page_title="BD Traffic Monitoring System", page_icon="🚗", layout="wide")

# Title
st.title("🚗 Bangladesh Traffic Monitoring System")
st.markdown("---")

# Load YOLO model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best.pt")
    model = YOLO(model_path)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Sidebar for settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)

# File uploader
st.header("📁 Upload Image or Video")
uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split("/")[0]
    
    if file_type == "image":
        # Process Image
        image = Image.open(uploaded_file)
        
        # Run inference
        with st.spinner("🔄 Running detection..."):
            results = model.predict(
                source=image,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Get annotated image
            annotated_image = results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Side-by-side: Input left, Output right
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.subheader("📸 Input Image")
            st.image(image, caption="Original Image", use_column_width=True)
        
        with right_col:
            st.subheader("🎯 Detection Results")
            st.image(annotated_image_rgb, caption="Detected Objects", use_column_width=True)
        
        # Count detections by class
        boxes = results[0].boxes
        class_names = results[0].names
        
        if len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            class_counts = {}
            
            for class_id in class_ids:
                class_name = class_names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Display class counts
            st.subheader("📊 Detection Statistics")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Total Objects Detected", len(boxes))
                st.metric("Unique Classes", len(class_counts))
            
            with col2:
                # Create dataframe for class counts
                df_counts = pd.DataFrame(
                    list(class_counts.items()),
                    columns=["Class", "Count"]
                ).sort_values("Count", ascending=False)
                
                st.dataframe(df_counts, use_container_width=True)
                
                # Create bar chart
                st.bar_chart(df_counts.set_index("Class"))
        else:
            st.warning("⚠️ No objects detected. Try adjusting the confidence threshold.")
    
    elif file_type == "video":
        # Process Video
        st.subheader("🎥 Video Processing")
        
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Show original video on the left
        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("📥 Original Video")
            st.video(tfile.name)
        
        # Process video button
        if st.button("🚀 Process Video"):
            with st.spinner("🔄 Processing video... This may take a while."):
                # Open video
                cap = cv2.VideoCapture(tfile.name)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Create temp output video (mp4v codec)
                raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))
                
                # Track all detections
                all_class_counts = {}
                progress_bar = st.progress(0)
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run inference
                    results = model.predict(
                        source=frame,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )
                    
                    # Get annotated frame
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                    
                    # Count detections
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        class_names = results[0].names
                        class_ids = boxes.cls.cpu().numpy().astype(int)
                        
                        for class_id in class_ids:
                            class_name = class_names[class_id]
                            all_class_counts[class_name] = all_class_counts.get(class_name, 0) + 1
                    
                    # Update progress
                    frame_count += 1
                    progress_bar.progress(frame_count / total_frames)
                
                cap.release()
                out.release()
                progress_bar.empty()
                
                # Re-encode to H.264 MP4 for browser playback
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                subprocess.run(
                    ["ffmpeg", "-y", "-i", raw_output_path,
                     "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", output_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                os.unlink(raw_output_path)
                
                # Display processed video on the right
                st.success("✅ Video processing complete!")
                with right_col:
                    st.subheader("🎯 Processed Video")
                    with open(output_path, "rb") as f:
                        video_bytes = f.read()
                    st.video(video_bytes)
                
                # Display statistics
                st.subheader("📊 Video Detection Statistics")
                
                if all_class_counts:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Total Detections", sum(all_class_counts.values()))
                        st.metric("Unique Classes", len(all_class_counts))
                        st.metric("Total Frames", total_frames)
                    
                    with col2:
                        # Create dataframe for class counts
                        df_counts = pd.DataFrame(
                            list(all_class_counts.items()),
                            columns=["Class", "Count"]
                        ).sort_values("Count", ascending=False)
                        
                        st.dataframe(df_counts, use_container_width=True)
                        
                        # Create bar chart
                        st.bar_chart(df_counts.set_index("Class"))
                else:
                    st.warning("⚠️ No objects detected in the video.")
                
                # Cleanup
                os.unlink(tfile.name)
                os.unlink(output_path)
else:
    st.info("👆 Please upload an image or video file to start detection.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Bangladesh Traffic Monitoring")