# Deepfake Video Detector

A robust deepfake detection system that uses a hybrid CNN-LSTM-ResNet model to identify manipulated videos with high accuracy. This project provides an intuitive web interface built with Streamlit for easy video analysis.

## Project Overview

This deepfake detection system leverages advanced machine learning techniques to analyze video content and determine whether it contains authentic or synthetically generated (deepfake) content. The model combines the power of Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and ResNet architecture to achieve accurate frame-by-frame analysis of video sequences.

## Features

- **Video Upload**: Upload video files directly through the web interface for analysis
- **YouTube Link Analysis**: Analyze videos directly from YouTube URLs using pytube integration
- **Frame-by-Frame Detection**: Comprehensive analysis of individual video frames to detect deepfake content
- **Graphical Output**: Interactive visualizations showing detection results over time
- **Real-time Processing**: Live processing with progress indicators and confidence scores
- **Streamlit UI**: User-friendly web interface for easy interaction and visualization
- **Video Stamping**: Option to generate stamped videos with detection results overlaid

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gitsofaryan/DF-Detection.git
cd DF-Detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Application

1. **Upload a Video**: Use the file uploader to select a video file from your local machine
2. **Analyze YouTube Videos**: Paste a YouTube URL to download and analyze videos directly
3. **View Results**: The application will display:
   - Frame-by-frame detection results
   - Confidence scores and probabilities
   - Interactive plots showing detection over time
   - Overall video classification (Real/Fake)

## Model Architecture

The deepfake detection system uses a hybrid architecture combining:

- **ResNet50**: Pre-trained convolutional neural network for feature extraction from individual frames
- **LSTM Networks**: Sequential processing to capture temporal dependencies between frames
- **CNN Layers**: Additional convolutional layers for spatial feature learning
- **Dense Layers**: Fully connected layers for final classification

This architecture enables the model to understand both spatial features within individual frames and temporal patterns across frame sequences, making it highly effective at detecting sophisticated deepfake content.

## Technology Stack

- **Frontend**: Streamlit for web interface
- **Machine Learning**: TensorFlow/Keras for model implementation
- **Computer Vision**: OpenCV for video processing and frame extraction
- **Data Processing**: NumPy for numerical operations
- **Visualization**: Matplotlib and Plotly for interactive charts and graphs
- **Video Handling**: pytube for YouTube video processing
- **Model Architecture**: ResNet50, LSTM, CNN hybrid approach

## Dataset

Training data reference: [Deepfake Detection Dataset](https://drive.google.com/open?id=10NGF38RgF8FZneKOuCOdRIsPzpC7_WDd)

## Requirements

See `requirements.txt` for complete list of dependencies. Main requirements include:
- Python 3.7+
- TensorFlow 2.x
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Plotly
- pytube

## License

This project is licensed under the terms specified in the LICENSE file.
