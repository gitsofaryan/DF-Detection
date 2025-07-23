# Deepfake Video Detection

A sophisticated deepfake detection system that leverages state-of-the-art machine learning techniques to identify artificially generated or manipulated videos. Built with a hybrid CNN-LSTM architecture enhanced with ResNet50 for superior accuracy and reliability.

## 🎯 Purpose

This application helps identify deepfake videos - AI-generated content where a person's appearance or actions are digitally altered to create convincing but fake footage. Our tool provides:

- **Real-time video analysis** for uploaded files and YouTube URLs
- **Visual analytics** with graphical reports and detection confidence scores
- **Frame-by-frame analysis** to pinpoint manipulation locations
- **Comprehensive reporting** with downloadable results

## ✨ Key Features

### 🔍 **Advanced Detection**
- Hybrid CNN-LSTM model with ResNet50 backbone
- Achieves >85% accuracy on challenging datasets like Celeb-DF
- Analyzes temporal patterns across video sequences
- Detects subtle inconsistencies invisible to human eyes

### 🖥️ **User-Friendly Interface**
- Built with Streamlit for intuitive web interface
- Support for multiple video formats (MP4, AVI, MOV)
- YouTube URL integration for direct analysis
- Real-time processing with progress indicators

### 📊 **Comprehensive Analytics**
- Interactive charts showing fake probability per frame
- Distribution analysis of real vs. fake content
- Speedometer visualization for overall confidence
- Anomaly detection with detailed explanations

### 📋 **Detailed Reporting**
- Downloadable analysis reports
- Frame-by-frame breakdown
- Technical explanations for classification decisions
- Visual evidence highlighting

## 🏗️ Technology Stack

### **Machine Learning**
- **ResNet50**: Pre-trained convolutional neural network for feature extraction
- **LSTM**: Long Short-Term Memory network for temporal sequence analysis
- **TensorFlow/Keras**: Deep learning framework
- **MTCNN**: Face detection and alignment

### **Frontend & Visualization**
- **Streamlit**: Web application framework
- **Plotly**: Interactive data visualization
- **Matplotlib**: Static plotting
- **OpenCV**: Computer vision and video processing

### **Additional Libraries**
- **NumPy**: Numerical computations
- **PyTube**: YouTube video downloading
- **Python**: Core programming language

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU support optional but recommended for faster processing

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/gitsofaryan/DF-Detection.git
   cd DF-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the pre-trained model**
   ```bash
   # The model file 'deepfake_detection_model_lstm.h5' should be placed in the root directory
   # Download from: [Add your model download link here]
   ```

### Alternative Installation with Virtual Environment

```bash
# Create virtual environment
python -m venv deepfake_env

# Activate virtual environment
# On Windows:
deepfake_env\Scripts\activate
# On macOS/Linux:
source deepfake_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:8501`
   - The application interface will load automatically

### Using the Detection System

1. **Upload Video or Enter URL**
   - Go to the "Upload & Detect" tab
   - Choose a video file (MP4 format recommended)
   - OR enter a YouTube URL

2. **Run Analysis**
   - Click the "Detect" button
   - Wait for processing to complete
   - View real-time results with confidence scores

3. **Review Results**
   - Navigate to "Results" tab for detailed analytics
   - Check "Anomalies" tab for specific manipulation indicators
   - Download comprehensive reports from "Report" tab

### Supported Input Formats
- **Video Files**: MP4, AVI, MOV
- **URLs**: YouTube links (public videos only)
- **Resolution**: Automatic scaling to 128x128 for processing
- **Duration**: Optimized for videos up to 10 minutes

## 📊 Model Performance

### Architecture Details
- **Input**: Sequences of 10 frames (128x128 pixels)
- **Feature Extraction**: ResNet50 (pre-trained on ImageNet)
- **Temporal Analysis**: LSTM with 64 hidden units
- **Output**: Binary classification (Real/Fake) with confidence score

### Performance Metrics
- **Training Accuracy**: ~92%
- **Validation Accuracy**: ~85%
- **Dataset**: Celeb-DF, YouTube-Real, and custom synthetic data
- **Processing Speed**: ~2-5 seconds per video segment

### Detection Capabilities
- High-quality deepfakes from advanced generators
- Face-swap manipulations
- Lip-sync inconsistencies
- Temporal artifacts and unnatural movements

## 🔧 Configuration

### Model Threshold
The default detection threshold is set to 0.25 (25% fake probability). You can adjust this in `app.py`:

```python
# Line 75 in app.py
final_label = 'FAKE' if mean_fake_probability > 0.25 else 'REAL'
```

### Video Processing Settings
- **Frame Sequence Length**: 10 frames per analysis window
- **Resolution**: 128x128 pixels (automatically resized)
- **Frame Rate**: 30 FPS for output videos

## 🤝 Contributing

We welcome contributions to improve the detection accuracy and user experience:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚨 Disclaimer

This tool is designed for educational and research purposes. While we strive for high accuracy, no deepfake detection system is 100% reliable. Always verify important content through multiple sources and methods.

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check the documentation and FAQ sections
- Contact the development team

## 🙏 Acknowledgments

- **Dataset**: Celeb-DF dataset contributors
- **Models**: TensorFlow and Keras communities
- **UI Framework**: Streamlit development team
- **Research**: Based on latest deepfake detection research papers

---

**⚠️ Important Note**: Ensure you have the required model file (`deepfake_detection_model_lstm.h5`) in the root directory before running the application. The model can be trained using the provided `model.py` script or downloaded from the releases section.
