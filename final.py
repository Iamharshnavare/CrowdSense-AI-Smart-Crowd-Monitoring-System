import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                           QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
                           QComboBox, QGroupBox, QSlider, QInputDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from ultralytics import YOLO

class CrowdMonitoringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Model configuration - directly using the specific model path
        self.model = YOLO('C:\Coding\codes\WEBDEV\BASICS\\VireoV1.pt')
        self.conf_threshold = 0.5
        
        # Video source variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Current state tracking
        self.is_webcam_active = False
        self.current_video_path = None
        self.current_image_path = None
        self.inference_running = False
        
        # Crowd limit tracking
        self.expected_count = 0
        self.has_expected_count = False
        self.last_alert_status = "normal"
        self.current_detected_count = 0
        
        # Initialize UI
        self.initUI()
        
    def initUI(self):
        # Main layout (horizontal to accommodate right panel)
        main_layout = QHBoxLayout()
        
        # Left side layout (for existing controls and display)
        left_layout = QVBoxLayout()
        
        # Top controls
        top_controls = QHBoxLayout()
        
        # Model group
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        
        # Model info - now just showing the loaded model without selection option
        model_info_layout = QHBoxLayout()
        self.model_path_label = QLabel("Model: VireoV1.pt")
        model_info_layout.addWidget(self.model_path_label)
        model_layout.addLayout(model_info_layout)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)  # Default 0.5
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        self.conf_value_label = QLabel("0.50")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        model_layout.addLayout(conf_layout)
        
        # Expected count settings
        count_layout = QHBoxLayout()
        count_label = QLabel("Expected Count:")
        self.expected_count_label = QLabel("Not set")
        self.set_expected_btn = QPushButton("Set Expected Count")
        self.set_expected_btn.clicked.connect(self.set_expected_count)
        count_layout.addWidget(count_label)
        count_layout.addWidget(self.expected_count_label)
        count_layout.addWidget(self.set_expected_btn)
        model_layout.addLayout(count_layout)
        
        model_group.setLayout(model_layout)
        top_controls.addWidget(model_group)
        
        # Input source group
        source_group = QGroupBox("Input Source")
        source_layout = QVBoxLayout()
        
        # Source selection buttons
        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.clicked.connect(self.toggle_webcam)
        source_layout.addWidget(self.webcam_btn)
        
        self.image_btn = QPushButton("Load Image")
        self.image_btn.clicked.connect(self.load_image)
        source_layout.addWidget(self.image_btn)
        
        self.video_btn = QPushButton("Load Video")
        self.video_btn.clicked.connect(self.load_video)
        source_layout.addWidget(self.video_btn)
        
        source_group.setLayout(source_layout)
        top_controls.addWidget(source_group)
        
        left_layout.addLayout(top_controls)
        
        # Display area
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(960, 720)  # Smaller to accommodate right panel
        self.display_label.setMaximumSize(960, 720)  
        self.display_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.display_label.setText("Select an input source to begin")
        left_layout.addWidget(self.display_label)
        
        # Status bar
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)
        
        # Create right panel for crowd metrics
        right_panel = QVBoxLayout()
        
        # Create a stats group box
        stats_group = QGroupBox("Crowd Metrics")
        stats_group.setMinimumWidth(300)
        stats_layout = QVBoxLayout()
        
        # Current count display
        self.count_display = QLabel("Count: 0")
        self.count_display.setAlignment(Qt.AlignCenter)
        self.count_display.setFont(QFont("Arial", 24, QFont.Bold))
        self.count_display.setStyleSheet("color: #0066cc;")
        stats_layout.addWidget(self.count_display)
        
        # Limit display
        self.limit_display = QLabel("Limit: Not set")
        self.limit_display.setAlignment(Qt.AlignCenter)
        self.limit_display.setFont(QFont("Arial", 18))
        stats_layout.addWidget(self.limit_display)
        
        # Spacer
        stats_layout.addSpacing(20)
        
        # Status display
        self.status_display = QLabel("Status: Normal")
        self.status_display.setAlignment(Qt.AlignCenter)
        self.status_display.setFont(QFont("Arial", 16, QFont.Bold))
        self.status_display.setStyleSheet("background-color: #e0f0e0; border-radius: 10px; padding: 10px;")
        stats_layout.addWidget(self.status_display)
        
        # Alert display
        self.alert_display = QLabel("")
        self.alert_display.setAlignment(Qt.AlignCenter)
        self.alert_display.setFont(QFont("Arial", 14, QFont.Bold))
        self.alert_display.setWordWrap(True)
        stats_layout.addWidget(self.alert_display)
        
        # Add stretch to push everything to the top
        stats_layout.addStretch()
        
        # Info section
        info_label = QLabel("Monitoring Information")
        info_label.setFont(QFont("Arial", 12, QFont.Bold))
        stats_layout.addWidget(info_label)
        
        self.info_display = QLabel("No active monitoring")
        self.info_display.setWordWrap(True)
        stats_layout.addWidget(self.info_display)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        # Add layouts to main layout
        main_layout.addLayout(left_layout, 3)  # 3:1 ratio
        main_layout.addLayout(right_panel, 1)
        
        # Set main widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Window settings
        self.setWindowTitle("V I R E ◎")
        self.setGeometry(100, 100, 1280, 800)
        self.setWindowIcon(QIcon("C:\Coding\codes\WEBDEV\BASICS\logo_demo2.png"))
        self.show()
    
    def set_expected_count(self):
        count, ok = QInputDialog.getInt(
            self, "Set Expected Count", 
            "Enter expected number of people:", 
            value=self.expected_count if self.has_expected_count else 10, 
            min=1, max=1000
        )
        
        if ok:
            self.expected_count = count
            self.has_expected_count = True
            self.expected_count_label.setText(str(count))
            self.limit_display.setText(f"Limit: {count}")
            self.status_label.setText(f"Expected count set to {count}")
    
    def update_conf_threshold(self):
        self.conf_threshold = self.conf_slider.value() / 100
        self.conf_value_label.setText(f"{self.conf_threshold:.2f}")
    
    def toggle_webcam(self):
        if self.is_webcam_active:
            self.stop_inference()
            self.webcam_btn.setText("Start Webcam")
            self.is_webcam_active = False
        else:
            # Prompt for expected count if not set
            if not self.has_expected_count:
                self.set_expected_count()
                if not self.has_expected_count:  # User canceled
                    return
                    
            self.stop_inference()  # Stop any previous inference
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Could not open webcam")
                
                self.timer.start(30)  # 30ms refresh (~33 fps)
                self.webcam_btn.setText("Stop Webcam")
                self.is_webcam_active = True
                self.inference_running = True
                self.status_label.setText("Running inference on webcam")
                self.info_display.setText("Monitoring: Live webcam feed")
            except Exception as e:
                self.status_label.setText(f"Error: {str(e)}")
    
    def load_image(self):
        self.stop_inference()
        
        # Prompt for expected count if not set
        if not self.has_expected_count:
            self.set_expected_count()
            if not self.has_expected_count:  # User canceled
                return
                
        image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if image_path:
            self.current_image_path = image_path
            file_name = image_path.split('/')[-1].split('\\')[-1]
            self.info_display.setText(f"Monitoring: Image file\n{file_name}")
            self.process_image(image_path)
    
    def load_video(self):
        self.stop_inference()
        
        # Prompt for expected count if not set
        if not self.has_expected_count:
            self.set_expected_count()
            if not self.has_expected_count:  # User canceled
                return
                
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)")
        if video_path:
            try:
                self.cap = cv2.VideoCapture(video_path)
                if not self.cap.isOpened():
                    raise Exception("Could not open video file")
                
                self.current_video_path = video_path
                file_name = video_path.split('/')[-1].split('\\')[-1]
                self.info_display.setText(f"Monitoring: Video file\n{file_name}")
                self.timer.start(30)  # 30ms refresh (~33 fps)
                self.inference_running = True
                self.status_label.setText(f"Running inference on: {file_name}")
            except Exception as e:
                self.status_label.setText(f"Error: {str(e)}")
    
    def stop_inference(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.inference_running = False
        self.current_video_path = None
        self.is_webcam_active = False
        self.webcam_btn.setText("Start Webcam")
        self.info_display.setText("No active monitoring")
    
    def process_image(self, image_path):
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not read image")
            
            # Run inference
            results = self.model(image, conf=self.conf_threshold)[0]
            
            # Get detection count
            detected_count = len(results.boxes)
            self.current_detected_count = detected_count
            
            # Check if count exceeds threshold
            alert_status = self.check_crowd_threshold(detected_count)
            
            # Draw detections (without text overlays)
            annotated_img = self.draw_detections(image, results, alert_status)
            
            # Display results
            self.display_image(annotated_img)
            
            # Update the right panel
            self.update_status_panel(detected_count, alert_status)
            
            # Update status bar
            if alert_status == "alert":
                self.status_label.setText(f"ALERT! Detected {detected_count} people (exceeds limit of {self.expected_count})")
            elif alert_status == "warning":
                self.status_label.setText(f"WARNING! Detected {detected_count} people (≥90% of limit {self.expected_count})")
            else:
                self.status_label.setText(f"Detected {detected_count} people in image")
            
        except Exception as e:
            self.status_label.setText(f"Error processing image: {str(e)}")
    
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.stop_inference()
            self.status_label.setText("Error: Video source disconnected")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            # Video ended - restart it for looping if it's a file (not webcam)
            if self.current_video_path and not self.is_webcam_active:
                self.cap.release()  # Release current capture
                self.cap = cv2.VideoCapture(self.current_video_path)  # Reopen the video
                ret, frame = self.cap.read()  # Try to read first frame
                
                if not ret:  # If still can't read, there's a problem with the file
                    self.stop_inference()
                    self.status_label.setText("Error: Could not loop video")
                    return
                
                file_name = self.current_video_path.split('/')[-1].split('\\')[-1]
                self.status_label.setText(f"Looping video: {file_name}")
            else:  # Webcam error
                self.stop_inference()
                self.status_label.setText("Error: Could not read frame from webcam")
                return
        
        # Run inference
        results = self.model(frame, conf=self.conf_threshold)[0]
        
        # Get detection count
        detected_count = len(results.boxes)
        self.current_detected_count = detected_count
        
        # Check if count exceeds threshold
        alert_status = self.check_crowd_threshold(detected_count)
        
        # Draw detections (without text overlays)
        annotated_frame = self.draw_detections(frame, results, alert_status)
        
        # Display results
        self.display_image(annotated_frame)
        
        # Update the right panel
        self.update_status_panel(detected_count, alert_status)
        
        # Update status bar with count
        source_type = "webcam" if self.is_webcam_active else "video"
        if alert_status == "alert":
            self.status_label.setText(f"ALERT! Detected {detected_count} people on {source_type} (exceeds limit of {self.expected_count})")
        elif alert_status == "warning":
            self.status_label.setText(f"WARNING! Detected {detected_count} people on {source_type} (≥90% of limit {self.expected_count})")
        else:
            self.status_label.setText(f"Detected {detected_count} people on {source_type}")
            
        # Remember last alert status for future reference if needed
        self.last_alert_status = alert_status
    
    def update_status_panel(self, count, alert_status):
        """Update the right panel with current count and status information"""
        # Update count display
        self.count_display.setText(f"Count: {count}")
        
        # Update status display based on alert status
        if alert_status == "normal":
            self.status_display.setText("Status: Normal")
            self.status_display.setStyleSheet("background-color: #e0f0e0; color: #008800; border-radius: 10px; padding: 10px;")
            self.alert_display.setText("")
        elif alert_status == "warning":
            self.status_display.setText("Status: Warning")
            self.status_display.setStyleSheet("background-color: #fff0d0; color: #bb7700; border-radius: 10px; padding: 10px;")
            self.alert_display.setText("APPROACHING LIMIT!")
            self.alert_display.setStyleSheet("color: #bb7700; font-weight: bold;")
        elif alert_status == "alert":
            self.status_display.setText("Status: Alert")  
            self.status_display.setStyleSheet("background-color: #ffe0e0; color: #cc0000; border-radius: 10px; padding: 10px;")
            self.alert_display.setText("CROWD LIMIT EXCEEDED!")
            self.alert_display.setStyleSheet("color: #cc0000; font-weight: bold;")
    
    def check_crowd_threshold(self, detected_count):
        """Check if the detected count exceeds expected thresholds"""
        if not self.has_expected_count or self.expected_count <= 0:
            return "normal"
            
        if detected_count > self.expected_count:
            return "alert"
        elif detected_count >= 0.9 * self.expected_count:
            return "warning"
        else:
            return "normal"
    
    def draw_detections(self, image, results, alert_status="normal"):
        annotated_img = image.copy()
        
        # Process each detection
        for box in results.boxes:
            # Get coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw bounding box - color changes based on alert status
            box_color = (0, 255, 0)  # Default green
            if alert_status == "warning":
                box_color = (0, 165, 255)  # Orange
            elif alert_status == "alert":
                box_color = (0, 0, 255)  # Red
                
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label with confidence
            label = f"Person: {conf:.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # No longer adding count and warning text overlays to the image
        # These are now shown in the right panel
        
        return annotated_img
    
    def display_image(self, image):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        display_width = self.display_label.width()
        display_height = self.display_label.height()
        scaled_pixmap = pixmap.scaled(display_width, display_height, Qt.KeepAspectRatio)
        
        # Set to display
        self.display_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        self.stop_inference()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CrowdMonitoringApp()
    sys.exit(app.exec_())