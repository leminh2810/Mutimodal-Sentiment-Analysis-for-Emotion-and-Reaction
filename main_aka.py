import os
import gradio as gr
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import librosa
from PIL import Image
import cv2
from model import generate_model
from opts import parse_opts
import transforms
from facenet_pytorch import MTCNN
import tempfile
import os
import subprocess
import matplotlib.pyplot as plt

class InferenceConfig:
    def __init__(self):
        self.model = 'multimodalcnn'
        self.fusion = 'ia'  # Choose from 'ia', 'it', 'lt' based on your trained model
        self.n_classes = 8  # RAVDESS has 8 emotion classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pretrain_path = 'None'  # Not needed for inference
        self.sample_duration = 15  # Must match training config
        self.video_norm_value = 255
        self.num_heads = 1  # Must match training config
        
# Keep all original helper functions
def extract_audio_from_video(video_path, sr=22050, target_time=3.6):
    """Extract audio from video file and preprocess it to match expected input format"""
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    
    # Extract audio using ffmpeg
    try:
        subprocess.run(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', temp_audio_path, '-y'], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error extracting audio with ffmpeg: {e}")
        # Cleanup
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return np.zeros(int(sr * target_time)), sr
    
    # Load the extracted audio
    try:
        y, sr = librosa.load(temp_audio_path, sr=sr)
        
        # Cleanup
        os.unlink(temp_audio_path)
        
        # Match the expected length (same as in extract_audios.py)
        target_length = int(sr * target_time)
        if len(y) < target_length:
            y = np.array(list(y) + [0 for i in range(target_length - len(y))])
        else:
            remain = len(y) - target_length
            y = y[remain//2:-(remain - remain//2)]
        
        return y, sr
        
    except Exception as e:
        print(f"Error loading extracted audio: {e}")
        # Cleanup
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return np.zeros(int(sr * target_time)), sr

def get_mfccs(y, sr):
    """Extract MFCC features from audio"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

def load_video(video_path, sample_size=224, sample_duration=15, device='cpu'):
    """Load and preprocess video frames with face detection like extract_faces.py"""
    # For .npy files (preprocessed face crops)
    if video_path.endswith('.npy'):
        video = np.load(video_path)
        video_data = []
        for i in range(np.shape(video)[0]):
            video_data.append(Image.fromarray(video[i,:,:,:]))
        return video_data
    
    # For video files, perform face detection and preprocessing
    mtcnn = MTCNN(image_size=sample_size, device=device)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Calculate number of frames and save length
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_length = 3.6  # seconds, same as in extract_faces.py
    
    # Function to select evenly distributed frames
    select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    
    # Adjust if video is shorter than desired length
    if save_length * fps > frame_count:
        skip_begin = 0
    else:
        skip_begin = int((frame_count - (save_length * fps)) // 2)
    
    # Skip frames at the beginning if needed
    for i in range(skip_begin):
        ret, _ = cap.read()
        if not ret:
            break
    
    # Calculate which frames to select
    framen = min(int(save_length * fps), frame_count - skip_begin)
    frames_to_select = select_distributed(sample_duration, framen)
    
    # Process selected frames
    video_data = []
    frame_ctr = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_ctr not in frames_to_select:
            frame_ctr += 1
            continue
        else:
            frames_to_select.remove(frame_ctr)
            frame_ctr += 1
        
        try:
            # Convert for MTCNN (expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = torch.from_numpy(frame_rgb).to(device)
            
            # Detect face
            boxes, _ = mtcnn.detect(frame_rgb)
            if boxes is not None and len(boxes) > 0:
                # Use first detected face
                bbox = boxes[0]
                bbox = [int(round(x)) for x in bbox]
                x1, y1, x2, y2 = bbox
                
                # Crop face
                face = frame[y1:y2, x1:x2, :]
                face = cv2.resize(face, (sample_size, sample_size))
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                video_data.append(Image.fromarray(face_rgb))
            else:
                # If no face detected, add blank frame
                blank = np.zeros((sample_size, sample_size, 3), dtype=np.uint8)
                video_data.append(Image.fromarray(blank))
        except Exception as e:
            print(f"Error processing frame: {e}")
            blank = np.zeros((sample_size, sample_size, 3), dtype=np.uint8)
            video_data.append(Image.fromarray(blank))
    
    cap.release()
    
    # Pad with blank frames if needed
    while len(video_data) < sample_duration:
        blank = np.zeros((sample_size, sample_size, 3), dtype=np.uint8)
        video_data.append(Image.fromarray(blank))
    
    # Ensure exactly sample_duration frames
    if len(video_data) > sample_duration:
        video_data = video_data[:sample_duration]
    
    return video_data

def predict_emotion(model, video_path, config):
    """Make prediction using the trained model, extracting audio from the video"""
    # Extract and preprocess audio from video
    y, sr = extract_audio_from_video(video_path)
    audio_features = get_mfccs(y, sr)
    audio_features = torch.tensor(audio_features).unsqueeze(0).float()
    
    # Load and preprocess video
    video_transform = transforms.Compose([
        transforms.ToTensor(config.video_norm_value)
    ])
    video_data = load_video(video_path, sample_duration=config.sample_duration, device=config.device)
    
    if len(video_data) != config.sample_duration:
        print(f"Warning: Expected {config.sample_duration} frames, got {len(video_data)}")
    
    video_data = [video_transform(img) for img in video_data]
    video_data = torch.stack(video_data, 0).permute(1, 0, 2, 3).unsqueeze(0)
    
    # Format video data to match model expectations
    video_data = video_data.permute(0, 2, 1, 3, 4)
    video_data = video_data.reshape(video_data.shape[0]*video_data.shape[1], 
                                     video_data.shape[2], 
                                     video_data.shape[3], 
                                     video_data.shape[4])
    
    # Move data to device
    audio_features = audio_features.to(config.device)
    video_data = video_data.to(config.device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(audio_features, video_data)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Map prediction to emotion label
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    return emotions[predicted_class], probabilities.cpu().numpy()[0]

MODEL = None
CONFIG = None

def initialize_model():
    global MODEL, CONFIG
    
    # Parse configuration
    CONFIG = InferenceConfig()
    
    # Load the model
    opt = parse_opts()
    opt.model = 'multimodalcnn'
    opt.fusion = 'ia'  # Ensure fusion type matches the saved model
    opt.n_classes = 8
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.pretrain_path = 'None'
    opt.sample_duration = 15
    opt.video_norm_value = 255
    opt.num_heads = 1
    
    try:
        model, _ = generate_model(opt)
        
        # Load checkpoint
        checkpoint = torch.load('results/RAVDESS_multimodalcnn_15_best0.pth', 
                               map_location=torch.device(opt.device))
        
        # Handle DataParallel prefix if needed
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Load with strict=False to skip mismatched parameters
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded successfully")
        MODEL = model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_emotion_chart(probabilities):
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(emotions, probabilities * 100, color='skyblue')
    
    max_idx = np.argmax(probabilities)
    bars[max_idx].set_color('navy')
    
    ax.set_ylim(0, 100)
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Probability (%)')
    ax.set_title(f'Detected Emotion: {emotions[max_idx].upper()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# Gradio interface function
def process_video(video_file):
    global MODEL, CONFIG
    
    if MODEL is None:
        initialize_model()
        if MODEL is None:
            return "Error: Model could not be loaded", None, None
    
    try:
        # Get temporary file path for the uploaded video
        temp_path = video_file
        
        # Process the video and get prediction
        emotion, probabilities = predict_emotion(MODEL, temp_path, CONFIG)
        
        # Create probability dictionary for display
        emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        prob_dict = {emotion: f"{prob*100:.2f}%" for emotion, prob in zip(emotions, probabilities)}
        
        # Create emotion chart
        fig = create_emotion_chart(probabilities)
        
        # Get top 3 emotions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_emotions = [emotions[i] for i in top_indices]
        top_probs = [probabilities[i] for i in top_indices]
        
        result_text = f"Predicted emotion: {emotion.upper()}\n\n"
        result_text += "Top 3 emotions:\n"
        for emo, prob in zip(top_emotions, top_probs):
            result_text += f"- {emo}: {prob*100:.2f}%\n"
        
        # Extract a frame with the detected face for display
        cap = cv2.VideoCapture(temp_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Try to detect a face in the first frame
            mtcnn = MTCNN(image_size=224, device=CONFIG.device)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb_tensor = torch.from_numpy(frame_rgb).to(CONFIG.device)
            
            boxes, _ = mtcnn.detect(frame_rgb_tensor)
            if boxes is not None and len(boxes) > 0:
                # Use first detected face
                bbox = boxes[0]
                bbox = [int(round(x)) for x in bbox]
                x1, y1, x2, y2 = bbox
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            # Add emotion text
            cv2.putText(frame, f"Emotion: {emotion.upper()}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            return result_text, prob_dict, fig, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return result_text, prob_dict, fig, None
        
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        print(error_msg)
        return error_msg, None, None, None

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Multimodal Emotion Recognition") as interface:
        gr.Markdown("# Multimodal Emotion Recognition")
        gr.Markdown("Upload a video to analyze the emotion of the person in it. The system uses both facial expressions and voice to make predictions.")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video")
                submit_btn = gr.Button("Analyze Emotion")
                
                with gr.Accordion("How It Works", open=False):
                    gr.Markdown("""
                    This system uses a multimodal approach that combines:
                    1. **Visual analysis**: Face detection and facial expression recognition
                    2. **Audio analysis**: Voice tone and speech emotion recognition
                    
                    The system classifies emotions into 8 categories: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.
                    """)
            
            with gr.Column(scale=2):
                with gr.Tab("Results"):
                    with gr.Row():
                        with gr.Column():
                            prediction_output = gr.Textbox(label="Prediction Results", lines=6)
                            probability_output = gr.JSON(label="All Emotion Probabilities")
                        
                        with gr.Column():
                            face_output = gr.Image(label="Detected Face", type="numpy")
                    
                    plot_output = gr.Plot(label="Emotion Probability Distribution")
                
                with gr.Tab("Processing Status"):
                    status_output = gr.Textbox(label="Status", value="Waiting for video upload...")
        
        submit_btn.click(
            fn=lambda: "Processing video (extracting frames and audio)...",
            inputs=None,
            outputs=status_output
        ).then(
            fn=process_video,
            inputs=video_input,
            outputs=[prediction_output, probability_output, plot_output, face_output]
        ).then(
            fn=lambda: "Analysis complete!",
            inputs=None,
            outputs=status_output
        )
        
        gr.Markdown("## Tips for Best Results")
        gr.Markdown("""
        - Ensure good lighting for clear facial visibility
        - Make sure audio is clear and the person's voice is audible
        - Try to have only one person in the frame
        - Videos of 3-5 seconds work best
        - Look directly at the camera when recording
        """)
    
    initialize_model()
    
    interface.launch(share=True)

if __name__ == "__main__":
    create_interface()