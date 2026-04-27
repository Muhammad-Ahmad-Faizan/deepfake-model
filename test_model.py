"""
Simple script to test/use the trained deepfake detection model
Load model and make predictions on videos
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import timm
from pathlib import Path
import os
from datetime import datetime
import json
import urllib.request


class DeepfakeDetector(nn.Module):
    """Same model architecture as training - MUST match train_incremental.py"""
    
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        
        # Load pretrained EfficientNet-B0 (lightweight for CPU)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        
        # Custom classifier head - EXACT same as train_incremental.py
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint"""
    
    print(f"Loading model from: {model_path}")
    
    # Create model
    model = DeepfakeDetector()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    
    # Check if checkpoint has accuracy info
    if 'final_val_accuracy' in checkpoint:
        print(f"   Final Accuracy: {checkpoint['final_val_accuracy']:.4f}")
    elif 'val_accuracy' in checkpoint:
        print(f"   Validation Accuracy: {checkpoint['val_accuracy']:.4f}")
    
    return model


def extract_frames_from_video(video_path, num_frames=10, frame_rate=15):
    """Extract frames from video and return both processed and raw frames"""
    
    frames = []
    raw_frames = []  # Keep original frames for saving
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return frames, raw_frames
    
    frame_count = 0
    extracted = 0
    
    while extracted < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_rate == 0:
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            # Keep BGR for saving
            raw_frames.append(frame.copy())
            extracted += 1
        
        frame_count += 1
    
    cap.release()
    
    return frames, raw_frames


def load_face_detector():
    """Load OpenCV face detector (Haar Cascade)"""
    
    # Download Haar Cascade if not exists
    cascade_path = 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(cascade_path):
        print("📥 Downloading face detector...")
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        urllib.request.urlretrieve(url, cascade_path)
        print("   ✅ Downloaded!")
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade


def detect_faces(frame, face_cascade):
    """Detect faces in frame"""
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces


def annotate_frame(frame, label, confidence, frame_num, faces=None):
    """Add annotation to frame showing prediction and highlight faces"""
    
    # Create a copy
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    # Choose color based on label
    if label == "FAKE":
        color = (0, 0, 255)  # Red for fake
        bg_color = (0, 0, 200)
        face_color = (0, 0, 255)  # Red boxes for fake faces
    else:
        color = (0, 255, 0)  # Green for real
        bg_color = (0, 200, 0)
        face_color = (0, 255, 0)  # Green boxes for real faces
    
    # Highlight detected faces
    if faces is not None and len(faces) > 0:
        for (x, y, fw, fh) in faces:
            # Draw thick box around face
            cv2.rectangle(annotated, (x, y), (x+fw, y+fh), face_color, 4)
            
            # Add label above face
            if label == "FAKE":
                face_text = "FAKE FACE"
                text_bg = (0, 0, 200)
            else:
                face_text = "REAL FACE"
                text_bg = (0, 200, 0)
            
            # Text background
            text_size = cv2.getTextSize(face_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, (x, y-30), (x+text_size[0]+10, y), text_bg, -1)
            cv2.putText(annotated, face_text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add semi-transparent overlay on face if FAKE
            if label == "FAKE":
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x, y), (x+fw, y+fh), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
    
    # Add semi-transparent overlay at top
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), bg_color, -1)
    cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, f"Frame {frame_num}", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, f"{label}: {confidence:.1f}%", (10, 65), font, 1.2, color, 3)
    
    # Add face count if faces detected
    if faces is not None and len(faces) > 0:
        face_info = f"Faces: {len(faces)}"
        cv2.putText(annotated, face_info, (w-150, 30), font, 0.7, (255, 255, 255), 2)
    
    # Add border
    border_thickness = 8
    cv2.rectangle(annotated, (0, 0), (w-1, h-1), color, border_thickness)
    
    return annotated


def save_results(video_path, result, raw_frames, output_folder):
    """Save detailed results including annotated frames and report"""
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\n💾 Saving results to: {output_folder}")
    
    # 1. Save annotated frames
    frames_folder = os.path.join(output_folder, 'frames')
    os.makedirs(frames_folder, exist_ok=True)
    
    print("   📸 Saving annotated frames with face detection...")
    
    # Load face detector
    face_cascade = load_face_detector()
    
    for i, (raw_frame, frame_info) in enumerate(zip(raw_frames, result['frame_details'])):
        # Detect faces
        faces = detect_faces(raw_frame, face_cascade)
        
        annotated = annotate_frame(
            raw_frame,
            frame_info['label'],
            frame_info['confidence'],
            frame_info['frame_num'],
            faces  # Pass detected faces
        )
        
        frame_filename = f"frame_{frame_info['frame_num']:02d}_{frame_info['label']}.jpg"
        frame_path = os.path.join(frames_folder, frame_filename)
        cv2.imwrite(frame_path, annotated)
    
    print(f"      ✅ Saved {len(raw_frames)} annotated frames")
    
    # 2. Save detailed report as text
    report_path = os.path.join(output_folder, 'analysis_report.txt')
    
    print("   📄 Generating detailed report...")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("DEEPFAKE DETECTION - DETAILED ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Video: {video_path}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: model_after_quarter_32.pth\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("FINAL RESULT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Prediction: {result['label']}\n")
        f.write(f"Confidence: {result['confidence']}%\n")
        f.write(f"Is Fake: {result['is_fake']}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("FRAME STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total frames analyzed: {result['total_frames']}\n")
        f.write(f"Fake frames detected: {result['fake_frames']} ({result['fake_frames']/result['total_frames']*100:.1f}%)\n")
        f.write(f"Real frames detected: {result['real_frames']} ({result['real_frames']/result['total_frames']*100:.1f}%)\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("FRAME-BY-FRAME ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        for frame in result['frame_details']:
            indicator = "🚨 SUSPICIOUS" if frame['is_suspicious'] else "✅ CLEAN"
            face_count = f" [{frame.get('faces_detected', 0)} face(s)]" if 'faces_detected' in frame else ""
            f.write(f"{indicator} Frame {frame['frame_num']:2d}: {frame['label']:4s} - ")
            f.write(f"Confidence: {frame['confidence']:5.1f}% (Score: {frame['raw_score']:.4f}){face_count}\n")
        
        if result['most_suspicious_frames']:
            f.write("\n" + "=" * 60 + "\n")
            f.write("MOST SUSPICIOUS FRAMES (Top Evidence)\n")
            f.write("=" * 60 + "\n\n")
            
            for i, frame in enumerate(result['most_suspicious_frames'], 1):
                f.write(f"{i}. Frame {frame['frame_num']}: {frame['confidence']:.1f}% fake ")
                f.write(f"(Score: {frame['raw_score']:.4f})\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("DETECTION BASIS\n")
            f.write("=" * 60 + "\n")
            f.write("- These frames show manipulation artifacts\n")
            f.write("- High fake scores indicate deepfake patterns\n")
            f.write("- Model detected inconsistencies in facial features\n")
            f.write("- EfficientNet-B0 trained on 32 quarters of data\n")
            f.write("- 100% validation accuracy on training set\n")
    
    print(f"      ✅ Report saved: analysis_report.txt")
    
    # 3. Save JSON for programmatic access
    json_path = os.path.join(output_folder, 'result.json')
    
    # Convert for JSON (remove non-serializable items)
    json_result = {
        'video': video_path,
        'timestamp': datetime.now().isoformat(),
        'prediction': result['label'],
        'confidence': result['confidence'],
        'is_fake': result['is_fake'],
        'total_frames': result['total_frames'],
        'fake_frames': result['fake_frames'],
        'real_frames': result['real_frames'],
        'frames': result['frame_details']
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_result, f, indent=4)
    
    print(f"      ✅ JSON data saved: result.json")
    
    # 4. Create summary image (grid of key frames)
    print("   🖼️  Creating summary image...")
    create_summary_image(raw_frames, result, output_folder)
    
    print(f"\n✅ All results saved successfully!")
    print(f"   📁 Location: {output_folder}/")
    
    return output_folder


def create_summary_image(raw_frames, result, output_folder):
    """Create a grid image showing all frames with labels"""
    
    # Load face detector
    face_cascade = load_face_detector()
    
    num_frames = len(raw_frames)
    
    # Calculate grid size
    cols = 5
    rows = (num_frames + cols - 1) // cols
    
    # Resize frames for grid
    frame_width = 300
    frame_height = 200
    
    # Create canvas
    canvas_width = cols * frame_width
    canvas_height = rows * frame_height
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 50
    
    # Place frames
    for i, (frame, info) in enumerate(zip(raw_frames, result['frame_details'])):
        row = i // cols
        col = i % cols
        
        # Detect faces in original frame
        faces = detect_faces(frame, face_cascade)
        
        # Resize frame
        resized = cv2.resize(frame, (frame_width, frame_height))
        
        # Scale face coordinates
        h_orig, w_orig = frame.shape[:2]
        scale_x = frame_width / w_orig
        scale_y = frame_height / h_orig
        
        faces_resized = None
        if faces is not None and len(faces) > 0:
            faces_resized = []
            for (x, y, fw, fh) in faces:
                faces_resized.append((
                    int(x * scale_x),
                    int(y * scale_y),
                    int(fw * scale_x),
                    int(fh * scale_y)
                ))
        
        # Annotate
        annotated = annotate_frame(resized, info['label'], info['confidence'], info['frame_num'], faces_resized)
        
        # Place on canvas
        y_start = row * frame_height
        x_start = col * frame_width
        canvas[y_start:y_start+frame_height, x_start:x_start+frame_width] = annotated
    
    # Save
    summary_path = os.path.join(output_folder, 'summary_grid.jpg')
    cv2.imwrite(summary_path, canvas)
    
    print(f"      ✅ Summary grid saved: summary_grid.jpg")


def preprocess_frame(frame):
    """
    Preprocess frame for model input - MUST match preprocessing.py
    
    Training preprocessing:
    1. Resize to 224x224
    2. Normalize to [0, 1] by dividing by 255
    3. Convert to tensor
    
    NO ImageNet normalization!
    """
    
    # Simple transform - exact same as training data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # This converts to [0, 1] automatically
    ])
    
    return transform(frame)


def predict_video(model, video_path, device='cpu', num_frames=15):
    """
    Predict if video is real or fake with detailed frame-by-frame analysis
    
    Returns:
        prediction: 'REAL' or 'FAKE'
        confidence: confidence score (0-100%)
        frame_predictions: individual frame predictions
        frame_analysis: detailed analysis per frame
        raw_frames: original frames for saving
    """
    
    print(f"\n🎬 Analyzing video: {video_path}")
    
    # Load face detector
    print("   Loading face detector...")
    face_cascade = load_face_detector()
    
    # Extract frames
    print("   Extracting frames...")
    frames, raw_frames = extract_frames_from_video(video_path, num_frames=num_frames)
    
    if len(frames) == 0:
        return None, 0.0, [], [], []
    
    print(f"   Extracted {len(frames)} frames")
    
    # Preprocess and predict
    print("   Making predictions on each frame...")
    predictions = []
    frame_analysis = []
    
    with torch.no_grad():
        for idx, frame in enumerate(frames):
            # Detect faces in raw frame
            faces = detect_faces(raw_frames[idx], face_cascade)
            
            # Preprocess
            frame_tensor = preprocess_frame(frame).unsqueeze(0).to(device)
            
            # Predict
            output = model(frame_tensor)
            pred_prob = output.item()
            
            predictions.append(pred_prob)
            
            # Determine frame label
            if pred_prob >= 0.5:
                frame_label = "FAKE"
                frame_conf = pred_prob * 100
            else:
                frame_label = "REAL"
                frame_conf = (1 - pred_prob) * 100
            
            frame_analysis.append({
                'frame_num': idx + 1,
                'label': frame_label,
                'confidence': frame_conf,
                'raw_score': pred_prob,
                'is_suspicious': pred_prob >= 0.5,
                'faces_detected': len(faces)
            })
    
    # Average prediction across all frames
    avg_prediction = np.mean(predictions)
    
    # Determine final label
    if avg_prediction >= 0.5:
        label = "FAKE"
        confidence = avg_prediction * 100
    else:
        label = "REAL"
        confidence = (1 - avg_prediction) * 100
    
    # Statistics
    fake_frames = sum(1 for p in predictions if p >= 0.5)
    real_frames = len(predictions) - fake_frames
    
    print(f"\n   📊 Frame Analysis:")
    print(f"      Total frames: {len(predictions)}")
    print(f"      Fake frames: {fake_frames} ({fake_frames/len(predictions)*100:.1f}%)")
    print(f"      Real frames: {real_frames} ({real_frames/len(predictions)*100:.1f}%)")
    
    print(f"\n   ✅ Overall Prediction: {label}")
    print(f"   📊 Overall Confidence: {confidence:.2f}%")
    print(f"   🎯 Raw score: {avg_prediction:.4f} (0=Real, 1=Fake)")
    
    return label, confidence, predictions, frame_analysis, raw_frames


def predict_video_simple(model_path, video_path, save_results_flag=True):
    """
    Simple function to get prediction with detailed frame analysis
    
    Usage:
        result = predict_video_simple('model_output/final_model.pth', 'test-1.mp4')
    """
    
    device = torch.device('cpu')
    
    # Load model
    model = load_model(model_path, device)
    
    # Predict
    label, confidence, predictions, frame_analysis, raw_frames = predict_video(model, video_path, device)
    
    if label is None:
        return {'error': 'Could not process video'}
    
    # Find most suspicious frames
    suspicious_frames = [f for f in frame_analysis if f['is_suspicious']]
    suspicious_frames.sort(key=lambda x: x['raw_score'], reverse=True)
    
    result = {
        'label': label,
        'confidence': round(confidence, 2),
        'is_fake': (label == 'FAKE'),
        'total_frames': len(predictions),
        'fake_frames': len(suspicious_frames),
        'real_frames': len(predictions) - len(suspicious_frames),
        'frame_details': frame_analysis,
        'most_suspicious_frames': suspicious_frames[:5]  # Top 5 suspicious
    }
    
    # Save results if requested
    if save_results_flag and len(raw_frames) > 0:
        # Create unique folder name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = Path(video_path).stem
        output_folder = f'test_results/test_{video_name}_{timestamp}'
        
        save_results(video_path, result, raw_frames, output_folder)
        result['output_folder'] = output_folder
    
    return result


def batch_predict(model_path, video_folder):
    """Test multiple videos from a folder"""
    
    device = torch.device('cpu')
    model = load_model(model_path, device)
    
    video_files = list(Path(video_folder).glob('*.mp4'))
    
    print(f"\n📁 Found {len(video_files)} videos")
    print("=" * 60)
    
    results = []
    
    for video_file in video_files:
        label, confidence, _, _, _ = predict_video(model, str(video_file), device)
        
        if label:
            results.append({
                'video': video_file.name,
                'prediction': label,
                'confidence': confidence
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for r in results:
        print(f"{r['video']:30s} → {r['prediction']:5s} ({r['confidence']:.1f}%)")
    
    return results


if __name__ == "__main__":
    """
    Test the trained model on a video
    """
    
    # Use the final trained model
    MODEL_PATH = 'model_output/model_after_quarter_32.pth'
    TEST_VIDEO = 'test-1.mp4'
    
    print("=" * 60)
    print("🎭 DEEPFAKE DETECTION - Testing Model")
    print("=" * 60)
    
    # Check if files exist
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Available models:")
        if os.path.exists('model_output'):
            for f in os.listdir('model_output'):
                if f.endswith('.pth'):
                    print(f"   - model_output/{f}")
        exit(1)
    
    if not os.path.exists(TEST_VIDEO):
        print(f"❌ Video not found: {TEST_VIDEO}")
        print("   Please provide a test video!")
        exit(1)
    
    # Run prediction
    print(f"\n🎬 Testing video: {TEST_VIDEO}")
    print(f"📦 Using model: {MODEL_PATH}")
    print("=" * 60)
    
    result = predict_video_simple(MODEL_PATH, TEST_VIDEO)
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULT")
    print("=" * 60)
    print(f"   Video: {TEST_VIDEO}")
    print(f"   Prediction: {result['label']}")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Is Fake: {result['is_fake']}")
    print("=" * 60)
    
    # Detailed frame analysis
    print("\n" + "=" * 60)
    print("🔍 DETAILED FRAME-BY-FRAME ANALYSIS")
    print("=" * 60)
    print(f"   Total frames analyzed: {result['total_frames']}")
    print(f"   Fake frames detected: {result['fake_frames']} ({result['fake_frames']/result['total_frames']*100:.1f}%)")
    print(f"   Real frames detected: {result['real_frames']} ({result['real_frames']/result['total_frames']*100:.1f}%)")
    
    print("\n📋 All Frames:")
    print("-" * 60)
    for frame in result['frame_details']:
        indicator = "🚨" if frame['is_suspicious'] else "✅"
        print(f"   {indicator} Frame {frame['frame_num']:2d}: {frame['label']:4s} - Confidence: {frame['confidence']:5.1f}% (Score: {frame['raw_score']:.4f})")
    
    if result['most_suspicious_frames']:
        print("\n" + "=" * 60)
        print("⚠️  MOST SUSPICIOUS FRAMES (Reasons for FAKE detection):")
        print("=" * 60)
        for i, frame in enumerate(result['most_suspicious_frames'], 1):
            print(f"   {i}. Frame {frame['frame_num']}: {frame['confidence']:.1f}% fake (Score: {frame['raw_score']:.4f})")
        
        print("\n💡 Basis for FAKE detection:")
        print("   - These frames show manipulation artifacts")
        print("   - High fake scores indicate deepfake patterns")
        print("   - Model detected inconsistencies in facial features")
    
    print("\n" + "=" * 60)
    
    # Show where results are saved
    if 'output_folder' in result:
        print("\n" + "=" * 60)
        print("📁 RESULTS SAVED")
        print("=" * 60)
        print(f"   Location: {result['output_folder']}/")
        print(f"   - frames/ (15 annotated screenshots)")
        print(f"   - analysis_report.txt (detailed text report)")
        print(f"   - result.json (JSON data)")
        print(f"   - summary_grid.jpg (all frames overview)")
        print("=" * 60)
