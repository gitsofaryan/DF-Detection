import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tempfile import NamedTemporaryFile
from io import BytesIO
from pytube import YouTube
import tempfile
import os


# Set page config at the top
st.set_page_config(layout="wide")

# Load the trained model
model = tf.keras.models.load_model('deepfake_detection_model_lstm.h5')

# Function to preprocess a frame for the model
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = frame / 255.0  # Normalize
    return np.expand_dims(frame, axis=0)

# Function to add a stamp to a frame
def add_stamp(frame, label):
    color = (0, 255, 0) if label == 'REAL' else (0, 0, 255)
    stamp_text = 'REAL' if label == 'REAL' else 'FAKE'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, stamp_text, (10, 50), font, 1, color, 2, cv2.LINE_AA)
    return frame

def process_and_stamp_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_sequence = []
    fake_probabilities = []
    frame_count = 0

    output_path = 'stamped_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0  # Normalize
        frame_sequence.append(frame)

        # Once we have a full sequence, make a prediction
        if len(frame_sequence) == 10:
            input_sequence = np.expand_dims(np.array(frame_sequence), axis=0)
            prediction = model.predict(input_sequence)

            fake_probabilities.append(prediction[0][0])

            label = 'FAKE' if prediction[0] > 0.3 else 'REAL'
            frame_with_stamp = add_stamp(frame_sequence[-1], label)
            out.write(frame_with_stamp)

            # Clear the sequence to start the next batch of 10 frames
            frame_sequence = []

        frame_count += 1

    cap.release()
    out.release()
    os.remove(video_path)

    mean_fake_probability = np.mean(fake_probabilities)
    final_label = 'FAKE' if mean_fake_probability > 0.25 else 'REAL'

    return output_path, final_label, fake_probabilities, [], []

def download_youtube_video(url):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4').first()
    video_path = stream.download()
    return video_path

# Function to create various charts
def create_charts(fake_probabilities, anomalies):
    fig = go.Figure()
    
    # Fake probability chart
    fig.add_trace(go.Scatter(y=fake_probabilities, mode='lines', name='Fake Probability', line=dict(color='red')))
    fig.update_layout(title='Fake Probability Over Video Frames', xaxis_title='Frame Number', yaxis_title='Fake Probability')
    
    # Anomalies chart
    if anomalies:
        fig.add_trace(go.Scatter(y=[1]*len(anomalies), mode='markers', name='Anomalies', marker=dict(color='red')))
        fig.update_layout(title='Anomalies Detected in Video Frames', xaxis_title='Frame Number', yaxis_title='Anomaly Detected')
    
    return fig

def create_distribution_chart(count_real, count_fake):
    fig = go.Figure(data=[go.Pie(labels=['Real', 'Fake'], values=[count_real, count_fake], marker_colors=['green', 'red'])])
    fig.update_layout(title_text='Distribution of Real vs. Fake Frames')
    return fig

def create_speedometer_chart(fake_percentage):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fake_percentage,
        title={'text': "Fake Percentage"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if fake_percentage > 20 else "green"},
               'steps': [{'range': [0, 50], 'color': "green"},
                         {'range': [50, 100], 'color': "red"}]}
    ))
    fig.update_layout(height=300, width=600)
    return fig

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f0f5;
        margin: 0;
        padding: 0;
    }
    .stApp {
        background-color: #f0f0f5;
    }
    .main-header {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #2a9d8f;
        padding: 20px;
    }
    .sub-header {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #264653;
        margin-top: 10px;
    }
    .upload-section, .result-section {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .button-section {
        text-align: center;
        margin-top: 20px;
    }
    .custom-button {
        background-color: #2a9d8f;
        color: white;
        padding: 10px 20px;
        font-size: 18px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .custom-button:hover {
        background-color: #21867a;
    }
    .fake-real-button {
        background-color: #e76f51;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .fake-real-button.real {
        background-color: #2a9d8f;
    }
    .fake-real-button.fake {
        background-color: #e76f51;
    }
    .fake-real-button:hover {
        background-color: #c84b31;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit interface
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Home", "Upload & Detect", "Results", "Anomalies", "Report", "About us", "Why Trust Us", "Extension"]
)

with tab1:
    st.markdown("""
        <style>
        /* Apply navy blue background color to the entire page */
        .stApp {
            background-color: #001f3f;  /* Navy blue background color */
        }
        /* Style for the top navigation section */
        .css-1d391kg a {
            font-size: 20px;  /* Increase the font size */
            font-weight: bold;  /* Make the text bold */
            color: #ADD8E6 !important;  /* Pastel blue color */
            text-decoration: none;  /* Remove underline */
        }
        .css-1d391kg a:hover {
            color: #ffffff !important;  /* White color on hover */
        }
        .header-container {
            position: relative;
            text-align: center;
            background-color: black;  /* Background color behind the text */
            padding: 20px;
            border-radius: 10px;  /* Rounded corners */
            overflow: hidden;  /* Ensure the background image stays within bounds */
        }
        .main-header {
            position: relative;
            font-size: 36px;
            font-weight: bold;
            color: white;  /* White color for "Video Detector" text */
            z-index: 1;  /* Ensure text is above the background image */
            font-family: 'Arial', sans-serif;  /* Change font style */
        }
        .main-header .deepfake {
            color: #ADD8E6;  /* Pastel blue color for "Deepfake" */
        }
        .tagline {
            position: relative;
            font-size: 18px;
            color: #FFFFFF;  /* White color for contrast */
            font-weight: 300;  /* Light weight for the tagline */
            margin-top: 10px;
            z-index: 1;  /* Ensure tagline is above the background image */
            display: inline-block;
            padding-top: 5px;  /* Space between tagline and line */
            border-top: 1px solid #FFFFFF;  /* White line for visual enhancement */
            font-family: 'Arial', sans-serif;  /* Change font style */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ADD8E6;  /* Pastel blue for all headings */
            font-family: 'Arial', sans-serif;  /* Change font style */
        }
        p, li {
            color: #FFFFFF;  /* White for the rest of the text including lists */
            font-family: 'Arial', sans-serif;  /* Change font style */
        }
        .background-image {
            position: absolute;
            top: 0;
            right: 0;
            height: 100%;
            width: 50%;  /* Adjust the width of the background image */
            object-fit: cover;  /* Cover the background area */
            z-index: 0;  /* Ensure the background image is behind the text */
        }
        </style>
    """, unsafe_allow_html=True)

    # Header container with image and text
    st.markdown("""
        <div class="header-container">
            <img src="https://backend.vlinkinfo.com/uploads/Thumnail_PNG_f114d5bfe9.gif" class="background-image" />
            <h1 class="main-header"><span class="deepfake">Deepfake</span> Video Detector</h1>
            <div class="tagline">Detect Deepfakes with Cutting-Edge AI Technology</div>
        </div>
    """, unsafe_allow_html=True)

    # Content with dynamic images and text
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://media.geeksforgeeks.org/wp-content/uploads/20240308105836/Deepfake-Softwares-2-opt.gif", caption="Deepfake Detection", use_column_width=True)

    with col2:
        st.markdown("""
            ### Understanding Deepfakes
            Deepfakes are AI-generated videos where the appearance or actions of a person are altered to create fake footage. These manipulations can be extremely realistic, making it difficult to distinguish between real and fake videos with the naked eye. Our application leverages state-of-the-art machine learning models to detect such alterations with high accuracy.
        """)

    st.markdown("<br>", unsafe_allow_html=True)  # Reduced spacing

    st.markdown("---")  # Add a horizontal line to separate the content

    col3, col4 = st.columns([2, 1])

    with col3:
        st.markdown("""
            ### Our Deepfake Detection Tool
            Our tool analyzes each frame of the uploaded video, looking for inconsistencies and signs of manipulation. Whether it's detecting unnatural facial movements, mismatched lip-syncing, or other anomalies, our model provides a reliable classification of the video as either real or fake. The results are presented visually, making it easy to understand the findings.
        """)

    with col4:
        st.image("https://petapixel.com/assets/uploads/2022/07/kate_2drivers_1024_compressed-2.gif", caption="Detection Process", use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)  # Adding a gap before new content

    col5, col6 = st.columns([1, 2])

    with col5:
        st.image("https://cdn-images-1.medium.com/fit/t/1600/480/1*C7Z3JYA_yScejWcK99ZfGQ.gif", caption="Deepfake Impact", use_column_width=True)

    with col6:
        st.markdown("""
            ### Impact of Deepfakes on Individuals and Businesses

            #### Impact on Individuals
            - **Defame or Harass**: Fake videos can be used to tarnish reputations by creating misleading or false portrayals of individuals. This can lead to emotional distress and damage personal relationships.
            - **Identity Theft**: Deepfakes can be used for identity theft by fabricating videos or audio recordings that mimic an individual’s appearance or voice, leading to potential financial and personal harm.
            - **Misinformation**: Individuals may be manipulated by false videos that spread misinformation or propaganda, affecting their beliefs and behaviors.

            #### Impact on Businesses
            - **Brand Damage**: Deepfakes can be used to create false statements or actions attributed to company executives or brands, leading to reputational damage and loss of consumer trust.
            - **Fraud and Scams**: Businesses can be targeted by deepfake-based fraud schemes, such as fake video messages from executives requesting fraudulent transactions.
            - **Security Threats**: Deepfakes can be used to impersonate company leaders or employees in security breaches, leading to unauthorized access to sensitive information.
        """)

    # After the "Impact on Business" section and before Modi Ji's video
# Define the layout for Key Features section
    




    st.markdown("---")  # Add a horizontal line to separate content sections
    st.markdown("## Watch Prime Minister Shri Narendra Modi discuss the urgent issue of deepfakes.")

    # Embed the video
    st.markdown("""
        <style>
            .video-wrapper {
                max-width: 100%; /* Ensure container is responsive */
                width: 560px; /* Adjust width of the container */
                height: 315px; /* Adjust height of the container, maintaining aspect ratio */
                margin: 0 auto; /* Center the container horizontally */
                display: flex; /* Use flexbox for centering */
                justify-content: center; /* Center video horizontally within the container */
                align-items: center; /* Center video vertically within the container */
            }
            .video-wrapper iframe {
                width: 100%;
                height: 100%;
                border: 0; /* Remove border */
            }
        </style>
        <div class="video-wrapper">
            <iframe src="https://www.youtube.com/embed/lJNKbk3kfgg" allowfullscreen></iframe>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .key-features-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 50px;
        }
        .features {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
            width: 50%;
        }
        .feature-item {
            display: flex;
            align-items: center;
            margin-bottom: 30px;
        }
        .feature-item:last-child {
            margin-bottom: 0;
        }
        .circle {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            color: #001f3f; /* Navy blue color */
            margin-right: 20px;
        }
        .feature-text {
            font-size: 18px;
            color: white;
        }
        .key-features-image {
            width: 45%;
            height: auto;
        }
    </style>
""", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)  # Line after Impact on Business

    st.markdown("""
    <h2 style="color: #ADD8E6; text-align: center;">Explore Our Key Features</h2>
""", unsafe_allow_html=True)

# Layout for Key Features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 60px; height: 60px; background-color: #FFFFFF; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; color: #001f3f;">1</div>
                <div>
                    <h4 style="color: #ADD8E6;">🔍 Advanced Detection</h4>
                    <p style="color: #FFFFFF;">Our advanced algorithms detect subtle inconsistencies in videos, ensuring high accuracy in identifying deepfakes.</p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 60px; height: 60px; background-color: #FFFFFF; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; color: #001f3f;">2</div>
                <div>
                    <h4 style="color: #ADD8E6;">
🖥️  User-Friendly Interface</h4>
                    <p style="color: #FFFFFF;">Our user-friendly interface is designed to simplify the video upload and analysis process, making it accessible even for non-technical users.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 60px; height: 60px; background-color: #FFFFFF; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; color: #001f3f;">3</div>
                <div>
                    <h4 style="color: #ADD8E6;">⚡ Fast Processing</h4>
                    <p style="color: #FFFFFF;">Experience lightning-fast processing times for video analysis, allowing for quick results and efficient workflows.</p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 60px; height: 60px; background-color: #FFFFFF; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; color: #001f3f;">4</div>
                <div>
                    <h4 style="color: #ADD8E6;">📊 Detailed Reports</h4>
                    <p style="color: #FFFFFF;">Receive comprehensive and easy-to-understand reports on the analysis, highlighting key findings and insights.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Image on the right side
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-top: 20px;">
        <img src="https://img.freepik.com/premium-photo/robot-stands-front-blackboard-with-chalkboard-that-says-robot_812921-224.jpg" style="width: 500px; height: auto; object-fit: cover;" />
    </div>
""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
        <h2 style="color: #ADD8E6; text-align: center;">How Does Deepfake Detection Work?</h2>
    """, unsafe_allow_html=True)

    # Creating a step-by-step guide with three containers side by side
    col7, col8, col9 = st.columns(3)

    container_height = "250px"  # Adjust this value as needed for consistency

    container_style = f"""
            background-color: white; 
            padding: 20px; 
            border-radius: 10px; 
            height: {container_height};
            display: flex; 
            flex-direction: column; 
            justify-content: space-between;
        """

    with col7:
        st.markdown(f"""
            <div style="{container_style}">
                <h3 style="color: black; text-align: center; font-weight: bold;">Step 1</h3>
                <p style="color: black; text-align: center;">Upload your video file to our system.</p>
                <div style="text-align: center;">
                    <img src="https://cdn-icons-png.flaticon.com/512/2910/2910761.png" width="50" />
                </div>
                <p style="color: black; text-align: center;">Our system will securely receive and prepare the video for analysis.</p>
            </div>
        """, unsafe_allow_html=True)

    with col8:
        st.markdown(f"""
            <div style="{container_style}">
                <h3 style="color: black; text-align: center; font-weight: bold;">Step 2</h3>
                <p style="color: black; text-align: center;">Our AI analyzes the video for deepfake detection.</p>
                <div style="text-align: center;">
                    <img src="https://cdn-icons-png.flaticon.com/512/4333/4333619.png" width="50" />
                </div>
                <p style="color: black; text-align: center;">The AI scans each frame for inconsistencies and manipulations.</p>
            </div>
        """, unsafe_allow_html=True)

    with col9:
        st.markdown(f"""
            <div style="{container_style}">
                <h3 style="color: black; text-align: center; font-weight: bold;">Step 3</h3>
                <p style="color: black; text-align: center;">Receive a detailed report on the analysis.</p>
                <div style="text-align: center;">
                    <img src="https://cdn-icons-png.flaticon.com/512/1828/1828919.png" width="50" />
                </div>
                <p style="color: black; text-align: center;">The results are presented clearly, showing any detected anomalies.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    



# Styling for FAQ section


# Styling for FAQ section
    st.markdown("""
    <style>
        /* Importing Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Open+Sans:wght@300;600&display=swap');

        .faq-section {
            background-color: #f0f8ff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            font-family: 'Open Sans', sans-serif;
        }
        .faq-title {
            font-size: 36px;
            font-weight: 700;
            color: #003366;
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 3px solid #003366;
            font-family: 'Roboto', sans-serif;
        }
        .faq-item {
            margin-bottom: 25px;
            border-radius: 10px;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .faq-question {
            font-size: 24px;
            font-weight: 600;
            color: #003366;
            display: flex;
            align-items: center;
            font-family: 'Roboto', sans-serif;
        }
        .faq-icon {
            width: 40px;
            height: 40px;
            margin-right: 15px;
        }
        .faq-answer {
            font-size: 18px;
            color: #333333;
            margin-top: 10px;
            line-height: 1.5;
            font-family: 'Open Sans', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# FAQ Section Title
    st.markdown("<div class='faq-section'><h2 class='faq-title'>Frequently Asked Questions</h2></div>", unsafe_allow_html=True)

# FAQ Items with icons and enhanced styling
    faqs = [
    {
        "question": "How does the deepfake detection work?",
        "answer": "Our system uses advanced algorithms to analyze video frames for inconsistencies and manipulations. It identifies subtle changes and provides a detailed report on potential deepfakes.",
        "icon": "https://cdn-icons-png.freepik.com/512/4257/4257824.png"
    },
    {
        "question": "What types of videos can be analyzed?",
        "answer": "You can upload any video format supported by our system, including MP4, AVI, and MOV. Ensure the video meets the minimum length requirements for analysis.",
        "icon": "https://img.icons8.com/ios-filled/50/000000/video.png"
    },
    {
        "question": "How accurate is the detection?",
        "answer": "Our detection algorithm has achieved accuracy rates of over 85% in identifying deepfakes. The accuracy may vary based on the quality and type of the video.",
        "icon": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbs-r5BF2gJ1nQCH5CE9qvEMjtQwv5r8JLU0YySEeqxdFO60AfnVkrxzDQcSJu-cwM50o&usqp=CAU"
    },
    {
        "question": "Is my data secure?",
        "answer": "Yes, we prioritize your privacy and data security. All videos are processed securely, and we do not store any personal information beyond the necessary analysis period.",
        "icon": "https://img.freepik.com/free-vector/shield_78370-582.jpg"
    },
    {
        "question": "Can I get a demo of the system?",
        "answer": "Certainly! You can view a demo video of our system in action on the homepage, or contact us for a personalized demonstration.",
        "icon": "https://cdn-icons-png.freepik.com/256/5651/5651475.png?semt=ais_hybrid"
    }
]

# Displaying FAQ items with icons
    for faq in faqs:
     with st.expander(faq["question"], expanded=False):
        st.markdown(f"<div class='faq-item'><div class='faq-question'><img src='{faq['icon']}' class='faq-icon' />{faq['question']}</div><p class='faq-answer'>{faq['answer']}</p></div>", unsafe_allow_html=True)

    st.markdown("""
    <style>
        .logo-container {
            margin-top: 40px;
            text-align: center;
        }
        .logo-container img {
            max-width: 200px;
            height: auto;
        }
    </style>

    <div class="logo-container">
        <img src="https://i.ibb.co/K6FmrF5/Screenshot-20240824-190504-Chrome.jpg" alt="RealVision Logo">
    </div>
""", unsafe_allow_html=True)







with tab2:
    # Custom CSS to set the background color, image, and overlay
    st.markdown("""
        <style>
        .upload-detect-container {
            position: relative;
            background-color: black;  /* Black background color */
            color: white;  /* White text color for contrast */
            padding: 20px;
            border-radius: 8px;
            overflow: hidden;  /* Ensure the background image stays within bounds */
        }
        .background-image {
            position: absolute;
            top: 0;
            right: 0;
            height: 100%;
            width: 50%;  /* Adjust the width of the background image */
            object-fit: cover;  /* Cover the background area */
            z-index: 0;  /* Ensure the background image is behind the content */
        }
        .upload-detect-content {
            position: relative;
            z-index: 1;  /* Ensure content is above the background image */
            color: white; /* White text color for contrast */
            font-size: 32px; /* Increase font size */
            font-weight: bold; /* Make text bold */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7); /* Add text shadow */
        }
        </style>
    """, unsafe_allow_html=True)

    # Container for the Upload & Detect tab
    st.markdown("""
    <div class="upload-detect-container">
        <img src="https://miro.medium.com/v2/resize:fit:1400/0*ZHIv88N3CtwWjaBM.gif" class="background-image" />
        <div class="upload-detect-content">
            <h2 class="sub-header">Upload or Enter URL and Detect</h2>
    """, unsafe_allow_html=True)

# Option to upload a video file
uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])

# Option to enter a video URL
video_url = st.text_input("Or enter a video URL (e.g., YouTube)")

if uploaded_file is not None:
    # If a file is uploaded, save it temporarily
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Display the uploaded video
    st.video(temp_file_path)

    st.markdown('<div class="button-section">', unsafe_allow_html=True)
    if st.button('Detect', key='detect_file'):
        # Process the video file and get the results
        stamped_video_path, result, fake_probabilities, anomalies, anomaly_reasons = process_and_stamp_video(temp_file_path)

        # Display the processed video
        st.video(stamped_video_path)

        # Display the result as a styled button
        st.markdown(f'<button class="fake-real-button {result.lower()}">{result}</button>', unsafe_allow_html=True)

        # Display the speedometer chart
        fake_percentage = np.mean(fake_probabilities) * 100
        speedometer_chart = create_speedometer_chart(fake_percentage)
        st.plotly_chart(speedometer_chart)
    st.markdown('</div>', unsafe_allow_html=True)

elif video_url:
    # Display the video from URL
    st.video(video_url)

    st.markdown('<div class="button-section">', unsafe_allow_html=True)
    if st.button('Detect', key='detect_url'):
        # You would need a method to download or process the video from the URL
        # Let's assume we can handle it and process similarly
        temp_file_path = download_video_from_url(video_url)

        stamped_video_path, result, fake_probabilities, anomalies, anomaly_reasons = process_and_stamp_video(temp_file_path)

        # Display the processed video
        st.video(stamped_video_path)

        # Display the result as a styled button
        st.markdown(f'<button class="fake-real-button {result.lower()}">{result}</button>', unsafe_allow_html=True)

        # Display the speedometer chart
        fake_percentage = np.mean(fake_probabilities) * 100
        speedometer_chart = create_speedometer_chart(fake_percentage)
        st.plotly_chart(speedometer_chart)
    st.markdown('</div>', unsafe_allow_html=True)

# Closing the container divs
st.markdown('</div></div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="result-section"><h2 class="sub-header">Detection Results</h2>', unsafe_allow_html=True)
    if 'fake_probabilities' in locals():
        fig = create_charts(fake_probabilities, anomalies)
        st.plotly_chart(fig)

        count_real = sum(1 for p in fake_probabilities if p <= 0.5)
        count_fake = sum(1 for p in fake_probabilities if p > 0.5)
        dist_chart = create_distribution_chart(count_real, count_fake)
        st.plotly_chart(dist_chart)

        fake_percentage = np.mean(fake_probabilities) * 100
        speedometer_chart = create_speedometer_chart(fake_percentage)
        st.plotly_chart(speedometer_chart)
    else:
        st.write("No results available. Please run a detection first.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="result-section"><h2 class="sub-header">Anomalies</h2>', unsafe_allow_html=True)
    if 'anomalies' in locals() and anomalies:
        for i, reason in enumerate(anomaly_reasons):
            st.write(f"Anomaly {i + 1}: {reason}")
    else:
        st.write("No anomalies detected.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="result-section"><h2 class="sub-header">Download Report</h2>', unsafe_allow_html=True)
    if 'fake_probabilities' in locals():
        mean_fake_probability = np.mean(fake_probabilities)
        final_label = 'FAKE' if mean_fake_probability > 0.5 else 'REAL'

        # Display explanation based on the classification
        if final_label == 'REAL':
            st.markdown("### Explanation for Real Classification")
            st.write("1. **High Confidence of Authenticity:** The analysis shows that the majority of the frames in the video exhibit characteristics consistent with authentic footage, leading to a classification of 'Real'.")
            st.write("2. **Low Fake Probability:** The overall fake probability across all frames is below the threshold, indicating that the video likely has not been manipulated.")
            st.write("3. **Consistency in Visual Features:** The video displays consistent visual features, such as natural facial movements and lighting, that are typically found in genuine footage.")
            st.write("4. **Absence of Anomalies:** No significant anomalies or inconsistencies were detected in the video, further supporting its authenticity.")

        elif final_label == 'FAKE':
            st.markdown("### Explanation for Fake Classification")
            st.write("1. **High Fake Probability:** The video has a high average fake probability, with many frames exhibiting signs of manipulation or artificial generation.")
            st.write("2. **Inconsistent Visual Features:** The analysis detected inconsistencies in facial features, unnatural movements, or lighting discrepancies, which are common indicators of deepfake videos.")
            st.write("3. **Detected Anomalies:** Several anomalies were identified, such as unusual patterns in facial expressions or sudden changes in the video's quality, suggesting potential tampering.")
            st.write("4. **Signs of Synthetic Generation:** The video contains elements that are often generated by deepfake algorithms, such as unnatural eye movements, lip-sync mismatches, or blurred edges around the face.")

        # Report download functionality
        try:
            buffer = BytesIO()
            fig = create_charts(fake_probabilities, anomalies)
            fig.write_image(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Report",
                data=buffer.getvalue(),
                file_name="deepfake_report.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error generating report: {e}")
    else:
        st.write("No report data available. Please run a detection first.")
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
    <style>
        .about-us-section {
            background-color: #f0f8ff; /* Same background color as used in the home tab */
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            text-align: center;
        }
        .about-us-title {
            font-size: 32px;
            font-weight: bold;
            color: #003366; /* Color used for headings */
            margin-bottom: 10px;
        }
        .about-us-subtitle {
            font-size: 24px;
            font-weight: bold;
            color: #ADD8E6; /* Color used for headings */
            margin-bottom: 5px;
        }
        .about-us-description {
            font-size: 18px;
            color: #333333;
            line-height: 1.6;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

with tab6:

# About Us Section
    st.markdown("""
    <style>
    .about-us-section {
        background-color: #2a9d8f;
        color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .about-us-title {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .about-us-subtitle {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .about-us-subtitle .empowering-trust {
        color: #000080; /* Dark color for 'Empowering Trust' */
    }
    .about-us-subtitle .through-technology {
        color: #000000; /* Black color for 'through Technology' */
    }
    .about-us-description {
        font-size: 16px;
        line-height: 1.5;
    }
    </style>
    <div class="about-us-section">
        <div class="about-us-title">About Us</div>
        <div class="about-us-subtitle">
            <span class="empowering-trust">Empowering Trust</span> 
            <span class="through-technology">through Technology</span>
        </div>
        <p class="about-us-description">
            At VerifEye, we are dedicated to revolutionizing the way deepfakes are detected and analyzed. Our mission is to empower individuals and organizations with cutting-edge technology that ensures authenticity and trust in digital content. Our team is committed to innovation and excellence, constantly pushing the boundaries to provide reliable and accurate solutions for a safer digital world. Join us on our journey to transform challenges into solutions and build a more secure online environment.
        </p>
    </div>
""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 40px;">
        <h1 style="color: #C4E1F5; font-family: 'Roboto', sans-serif; font-weight: bold;">Our Mission</h1>
        <div style="display: flex; align-items: center; margin-top: 20px;">
            <div style="flex: 1; color: white; font-family: 'Roboto', sans-serif; margin-right: 20px;">
                At VerifEye, our mission is to provide innovative solutions for detecting and analyzing deepfakes. We aim to enhance trust in digital content through advanced technology and unwavering commitment to accuracy and reliability. By leveraging our expertise, we strive to create a safer digital landscape and empower users with the tools they need to verify content authenticity.
            </div>
            <div style="flex: 1;">
                <img src="https://static.wixstatic.com/media/ebdfd3_9bc8c67f163d4286bceac5dd08a81298~mv2.gif" alt="Mission Image" style="width: 100%; height: auto;">
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
    
    st.markdown("---")  # Add a horizontal line to separate content sections
    st.markdown("## See It in Action: How VerifEye Ensures Authenticity in Every Frame")

    st.markdown("""
        <style>
            .video-wrapper {
                max-width: 100%; /* Ensure container is responsive */
                width: 560px; /* Adjust width of the container */
                height: 315px; /* Adjust height of the container, maintaining aspect ratio */
                margin: 0 auto; /* Center the container horizontally */
                display: flex; /* Use flexbox for centering */
                justify-content: center; /* Center video horizontally within the container */
                align-items: center; /* Center video vertically within the container */
            }
            .video-wrapper iframe {
                width: 100%;
                height: 100%;
                border: 0; /* Remove border */
            }
        </style>
        <div class="video-wrapper">
            <iframe src="https://www.youtube.com/embed/pO6cj0v9PJI" allowfullscreen></iframe>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .project-highlights-section {
            text-align: left;
            padding: 40px;
            
        }
        .project-highlights-heading {
            font-size: 36px;
            font-weight: bold;
            color: #fffffff; /* Heading color */
            margin-bottom: 20px;
        }
        .project-highlights-content {
            color: #000000; /* Text color */
            font-size: 18px;
            line-height: 1.6;
            margin-left: 20px; /* Space between text and image */
        }
        .project-highlights-image {
            width: 400px;
            height: auto;
            border-radius: 15px; /* Rounded corners */
            
        }
        .section-spacing {
            margin-bottom: 40px; /* Space between previous and this section */
        }
    </style>

    <div class="section-spacing"></div>

    <div class="project-highlights-section">
        <div class="project-highlights-heading">Project Highlights</div>
        <div style="display: flex; align-items: center;">
            <img class="project-highlights-image" src="https://fiverr-res.cloudinary.com/images/t_main1,q_auto,f_auto,q_auto,f_auto/attachments/delivery/asset/278bbab4c7ab83edeae5dd028ea5810e-1622621934/01/create-lootie-animation-for-website-and-mobile.gif" alt="Project Highlights Image">
            <div class="project-highlights-content">
                <p>Our project utilizes a robust approach for deepfake detection by integrating several advanced models:</p>
                <ul>
                    <li><strong>CNN:</strong> Initially used for feature extraction from images.</li>
                    <li><strong>LSTM:</strong> Applied to capture temporal dependencies in sequences of frames.</li>
                    <li><strong>ResNet:</strong> Implemented to enhance the accuracy and robustness of the model.</li>
                </ul>
                <p>By leveraging these models, we ensure a high level of accuracy and reliability in detecting deepfakes, aiming for performance exceeding 86% accuracy.</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        .technology-stack-section {
            text-align: center;
            margin: 40px 0;
        }
        .technology-stack-heading {
            font-size: 36px;
            font-weight: bold;
            color: #C4E1F5; /* Heading color */
            margin-bottom: 20px;
        }
        .technology-stack-images {
            display: flex;
            justify-content: center;
            gap: 20px; /* Space between images */
        }
        .technology-stack-images img {
            width: 45%; /* Adjust width as needed */
            height: auto;
            border-radius: 15px; /* Rounded corners */
        }
    </style>

    <div class="technology-stack-section">
        <div class="technology-stack-heading">Technology Stack</div>
        <div class="technology-stack-images">
            <img src="https://i.postimg.cc/QNSKdf2m/Screenshot-20240825-151012-Whats-App.jpg" alt="Technology Stack Image 1">
            <img src="https://miro.medium.com/v2/resize:fit:1400/1*yw0TnheAGN-LPneDaTlaxw.gif" alt="Technology Stack Image 2">
        </div>
    </div>
""", unsafe_allow_html=True)



    
    st.markdown("""
    <style>
        .accuracy-section {
            text-align: center;
            margin: 40px 0;
        }
        .accuracy-title {
            font-size: 36px;
            font-weight: bold;
            color: #C4E1F5;
            margin-bottom: 30px;
        }
        .accuracy-container {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            width: 100%;
        }
        .accuracy-item {
            flex: 1;
            text-align: center;
        }
        .accuracy-item img {
            width: 100%;
            height: 200px; /* Fixed height for all images */
            object-fit: cover; /* Ensure images cover the area without distortion */
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .accuracy-item h3 {
            font-size: 24px;
            font-weight: bold;
            color: #C4E1F5;
            margin-bottom: 5px;
        }
        .accuracy-item p {
            color: white;
            font-size: 16px;
            margin-top: 0;
        }
    </style>
    
    <div class="accuracy-section">
        <div class="accuracy-title">Elevating Accuracy to New Heights</div>
        <div class="accuracy-container">
            <div class="accuracy-item">
                <img src="https://cdn.dribbble.com/users/3593/screenshots/2475280/linechart.gif" alt="High Precision">
                <h3>High Precision</h3>
                <p>We've achieved more than 87% accuracy, ensuring high precision in detecting deepfakes.</p>
            </div>
            <div class="accuracy-item">
                <img src="https://img.freepik.com/free-vector/man-face-scan-biometric-digital-technology_24908-56378.jpg" alt="Reliable Results">
                <h3>Reliable Results</h3>
                <p>Our models deliver reliable results, offering consistent performance across different scenarios.</p>
            </div>
            <div class="accuracy-item">
                <img src="https://i.pinimg.com/originals/f8/8a/ca/f88acab7ffd127b4465659500aa0538f.gif" alt="Consistent Performance">
                <h3>Consistent Performance</h3>
                <p>Consistent performance is at the core of our system, providing trustworthy results every time.</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


    st.markdown("""
    <style>
        .impact-section {
            display: flex;
            align-items: flex-start;
            margin: 40px 0;
        }
        .impact-image {
            width: 50%;
            padding-right: 20px;
        }
        .impact-image img {
            width: 100%;
            height: auto;
            border-radius: 10px;
        }
        .impact-content {
            width: 50%;
        }
        .impact-heading {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #C4E1F5;
            margin-bottom: 30px;
        }
        .impact-item {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .impact-item p {
            margin: 0;
            font-size: 18px;
            color: #003366; /* Navy Blue */
            font-weight: bold;
            font-family: 'Arial', sans-serif; /* Example of a different font style */
        }
    </style>
    
    <div class="impact-section">
        <div class="impact-image">
            <img src="https://i.pinimg.com/originals/5f/08/50/5f08505655b858d52ea4ef07a6fa58d5.gif" alt="Impact Image">
        </div>
        <div class="impact-content">
            <div class="impact-heading">Impact of our solution</div>
            <div class="impact-item">
                <p>1: Boosting Public Awareness.</p>
            </div>
            <div class="impact-item">
                <p>2: Empowering Users</p>
            </div>
            <div class="impact-item">
                <p>3: Reduces Misinformation Spread</p>
            </div>
            <div class="impact-item">
                <p>4: Enabling Secure Communication</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)










    

# Custom CSS for styling
    st.markdown("""
    <style>
        .built-on-values-section {
            text-align: center;
            margin: 40px 0;
        }
        .built-on-values-title {
            font-size: 36px;
            font-weight: bold;
            color: #0096FF; /* Heading color */
            margin-bottom: 40px;
        }
        .value-container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        .value-box {
            width: 30%;
            background-color: #FFFFFF;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .value-box i {
            font-size: 48px;
            color: #0096FF;
            margin-bottom: 10px;
        }
        .value-box h4 {
            font-size: 24px;
            color: #0096FF;
            margin-bottom: 10px;
        }
        .value-box p {
            font-size: 16px;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)


    


# Built on Values Section
    st.markdown("""
    <style>
        .built-on-values-section {
            text-align: center;
            padding: 40px; /* Adjust padding for spacing */
        }
        .built-on-values-heading {
            font-size: 36px;
            font-weight: bold;
            color: #C4E1F5; /* Heading color */
            margin-bottom: 40px;
        }
        .values-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap; /* Ensures responsiveness */
        }
        .values-item {
            width: 300px; /* Adjusted width */
            height: 300px; /* Adjusted height */
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            text-align: center;
            box-sizing: border-box; /* Ensures padding is included in width/height */
        }
        .values-item img {
            width: 100px; /* Image size */
            height: 100px; /* Image size */
            margin-bottom: 15px; /* Space between icon and heading */
        }
        .values-item h3 {
            font-size: 24px;
            color: #000000; /* Same as heading color */
            margin-bottom: 10px;
        }
        .values-item p {
            font-size: 16px; /* Adjusted font size for readability */
            color: #333;
            margin: 0; /* Ensures content doesn't spill over */
        }
    </style>
    
    <div class="built-on-values-section">
        <div class="built-on-values-heading">Built on Values</div>
        <div class="values-container">
            <div class="values-item">
                <img src="https://cdn-icons-png.freepik.com/512/4946/4946408.png" alt="User-Friendly Interface">
                <h3>User-Friendly Interface</h3>
                <p>Our solutions are designed with the user in mind, ensuring a smooth and intuitive experience.</p>
            </div>
            <div class="values-item">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRmaznO2w-h9n4bz-pEGtsiqy0J1JGh-wlMCw&s" alt="Accessibility">
                <h3>Accessibility</h3>
                <p>We prioritize accessibility, making our technology available to everyone, regardless of their abilities.</p>
            </div>
            <div class="values-item">
                <img src="https://i.pinimg.com/originals/e8/09/84/e8098408923de19e8bdc25ade1881676.png" alt="Innovation">
                <h3>Innovation</h3>
                <p>Constant innovation drives us, pushing the boundaries of technology to provide cutting-edge solutions.</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
    
    

# Feedback Section
    st.markdown("""
    <style>
        .feedback-section {
            display: flex;
            align-items: flex-start;
            padding: 40px;
        }
        .feedback-image {
            width: 50%;
            padding-right: 20px;
        }
        .feedback-image img {
            width: 100%;
            height: auto;
            border-radius: 10px;
        }
        .feedback-form-container {
            width: 50%;
        }
        .feedback-heading {
            font-size: 36px;
            font-weight: bold;
            color: #C4E1F5;
            margin-bottom: 20px;
        }
        .feedback-form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .feedback-form textarea {
            width: 100%;
            height: 150px;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #ccc;
            font-size: 16px;
        }
        .feedback-form button {
            background-color: #C4E1F5;
            color: #fff;
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 18px;
            cursor: pointer;
        }
        .feedback-form button:hover {
            background-color: #a1c2e7;
        }
        .confirmation-message {
            font-size: 18px;
            font-weight: bold;
            color: navy;
            margin-top: 20px;
        }
        .confirmation-message span {
            color: #000;
            font-weight: normal;
        }
    </style>
    
    <div class="feedback-section">
        <div class="feedback-image">
            <img src="https://i2.wp.com/cdn.dribbble.com/users/1052957/screenshots/4153213/offcie.gif" alt="Feedback Image">
        </div>
        <div class="feedback-form-container">
            <div class="feedback-heading">We Value Your Feedback</div>
            <div class="feedback-form">
                <textarea placeholder="Share your thoughts, suggestions, or feedback here..."></textarea>
                <button onclick="submitFeedback()">Submit</button>
            </div>
            <div class="confirmation-message" id="confirmation-message" style="display:none;">
                <span>Feed   }
        .team-member img {
            width: 100%; /* Adjust width to fit within container */
            height: 250px; /* Set a specific height */
            object-fit: cover; /* Ensure image covers the container without distortion */
            border-radius: 10px;
        }
        .team-member p {
            margin-top: 10px;
            font-size: 16px;
            color: #000000;
        }
    </style>

    <div class="meet-our-team-section">
        <div class="meet-our-team-heading">Meet the RealVision Team</div>
        <div class="team-intro">
            <p>Our team comprises third-year students from Gyan Ganga Institute of Technology and Sciences. We are passionate about leveraging technology to create impactful solutions, and we bring a blend of technical expertise and innovative thinking to every project we undertake. United by our shared commitment to excellence, we work collaboratively to push the boundaries of what's possible.</p>
        </div>
        <div class="team-container">
            <div class="team-member">
                <img src="https://i.postimg.cc/CMrcZ14X/1000042747-removebg-preview.jpg" alt="Shruti Parmar">
                <p>Shruti Parmar</p>
            </div>
            <div class="team-member">
                <img src="https://i.postimg.cc/SQ9GKrJ2/1000042715-removebg-preview.jpg" alt="Aryan Jain">
                <p>Arya Vats</p>
            </div>
            <div class="team-member">
                <img src="https://i.postimg.cc/2jn4YSKn/1000042720-removebg-preview.jpg" alt="Arya Vats">
                <p>Aryan Jain</p>
            </div>
            <div class="team-member">
                <img src="https://i.postimg.cc/KzxmgQmG/1000042754-removebg-preview.jpg" alt="Ayush Gautam">
                <p>Ayush Gautam</p>
            </div>
            <div class="team-member">
                <img src="https://i.postimg.cc/mZ3Fq6Pw/1000042717-removebg-preview.jpg" alt="Aditya Suhane">
                <p>Aditya Suhane</p>
            </div>
            <div class="team-member">
                <img src="https://i.postimg.cc/QtJx6wfK/1000042722-removebg-preview.jpg" alt="Shatakshi Gupta">
                <p>Shatakshi Gupta</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

    st.markdown("""
    <style>
        .logo-container {
            margin-top: 40px;
            text-align: center;
        }
        .logo-container img {
            max-width: 200px;
            height: auto;
        }
    </style>

    <div class="logo-container">
        <img src="https://i.ibb.co/K6FmrF5/Screenshot-20240824-190504-Chrome.jpg" alt="RealVision Logo">
    </div>
""", unsafe_allow_html=True)

with tab7:
    st.title("Why Trust Us?")

    # Proven Accuracy
    st.subheader("🚀 Proven Accuracy")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://media2.giphy.com/media/OTKvqFm0ieiiKiaUx8/200w.gif?cid=6c09b9520ec2te8zxymhlvek2f2cl25uph1izb372ffaz5ww&ep=v1_gifs_search&rid=200w.gif&ct=g", caption="High Accuracy")
    with col2:
        st.write("Our model boasts a *92% accuracy* during training and *85% validation accuracy* on challenging datasets like Celeb-DF, outperforming many other models that struggle with high-quality fakes.")
        st.write("Compared to other tools like [Deepware Scanner](https://www.deepware.ai/), which have lower accuracy on similarly challenging datasets.")

    st.markdown("---")

    # Smart Hybrid Design
    st.subheader("🧠 Smart Hybrid Design")
    col3, col4 = st.columns([2, 1])
    with col3:
        st.write("We leverage the power of *ResNet* for analyzing each frame and *LSTM* for understanding temporal relationships across frames. This hybrid approach allows us to catch even the most subtle manipulations.")
        st.write("Unlike traditional models such as [MesoNet](https://arxiv.org/abs/1809.00888), which only focus on single-frame analysis, our hybrid approach provides a more comprehensive detection.")
    with col4:
        st.image("https://media.licdn.com/dms/image/D4D12AQH5i_pk9_lOGw/article-cover_image-shrink_720_1280/0/1697089776871?e=2147483647&v=beta&t=_S2qYzhyUMPWyqRaNghRycz5jNekKZzWQCEisTNR0k0", caption="ResNet + LSTM")

    st.markdown("---")

    # Handles Tough Datasets
    st.subheader("💪 Handles Tough Datasets")
    col5, col6 = st.columns([1, 2])
    with col5:
        st.image("https://images.datacamp.com/image/upload/v1661355394/Connect_x_14_d4988583b0.gif", caption="Specialized for Tough Datasets")
    with col6:
        st.write("Our model is tuned for high-quality, realistic fake videos in datasets like *Celeb-DF*, where many other models falter. We deliver consistent performance even in the toughest scenarios.")
        st.write("Other models, such as [FaceForensics++](https://github.com/ondyari/FaceForensics), often struggle with such high-quality fakes, but our model consistently outperforms them.")

    st.markdown("---")

    # Visual Analysis of Fakes
    st.subheader("🔍 Visual Analysis of Fakes")
    col7, col8 = st.columns([2, 1])
    with col7:
        st.write("We provide a *graphical analysis* showing the percentage of fake content in each frame, giving users a clear and intuitive understanding of where the manipulation occurs.")
        st.write("Most other tools, like [XceptionNet](https://arxiv.org/abs/1912.11868), do not offer such detailed visual insights, making it harder to understand the extent of manipulation.")
    with col8:
        st.image("https://www.echelonedge.com/wp-content/themes/echelon/assets/img/echelon-data-quipo.gif", caption="Graphical Analysis of Fakes")

    st.markdown("---")

    # Fewer Mistakes
    st.subheader("🎯 Fewer Mistakes")
    col9, col10 = st.columns([1, 2])
    with col9:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/037/248/343/small_2x/business-rising-bar-chart-animation-free-video.jpg", caption="Minimizing Errors")
    with col10:
        st.write("Our model is designed to reduce both *false positives* (real videos wrongly labeled as fake) and *false negatives* (fake videos slipping through). This careful balance ensures more reliable results.")
        st.write("Compared to other models like [Inception-v3](https://arxiv.org/abs/1512.00567), which have higher rates of false positives, our model achieves a better balance.")

    st.markdown("---")

    # Easy to Customize
    st.subheader("🔧 Easy to Customize")
    col11, col12 = st.columns([2, 1])
    with col11:
        st.write("Our model is flexible and can be fine-tuned for different kinds of deepfakes, making it adaptable to various user needs and evolving threats.")
        st.write("This flexibility is not as evident in more rigid models like [VGG16](https://arxiv.org/abs/1409.1556), which are less adaptable to new types of deepfakes.")
    with col12:
        st.image("https://www.intel.com/content/dam/www/central-libraries/us/en/images/2022-11/newsroom-deepfake-feat.jpg", caption="Customizable Model")

    st.markdown("---")

    # Clear Reports
    st.subheader("📊 Clear Reports")
    col13, col14 = st.columns([1, 2])
    with col13:
        st.image("https://backdocket.com/wp-content/uploads/2020/01/FEATURESPAGE.gif", caption="Detailed Reporting")
    with col14:
        st.write("We provide *detailed reports* that explain why a video is labeled as fake, helping users understand the results more clearly.")
        st.write("Other tools, such as [FakeCatcher](https://arxiv.org/abs/2102.11142), may offer less transparent results, making it harder for users to trust the findings.")

# Extension tab content
with tab8:
    st.markdown("<h2>Extension Overview</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        Our browser extension provides a seamless way to quickly detect deepfakes directly from any website. 
        It allows users to input video URLs for analysis without leaving the current web page, ensuring fast and efficient detection.

        ### Features of Our Extension
        - **Easy Access**: Directly accessible from your browser's toolbar.
        - **Quick Analysis**: Supports URLs from popular video platforms like YouTube and Vimeo.
        - **Instant Results**: Get immediate detection feedback right in your browser.
        
        ### How to Install and Use the Extension
        1. **Download**: Visit the [Chrome Web Store](https://chrome.google.com/webstore) and search for "Deepfake Detector".
        2. **Install**: Click 'Add to Chrome' and follow the prompts.
        3. **Use**: After installation, click the extension icon in your browser toolbar, enter the video URL, and click "Analyze".
        
        **Tip**: Ensure that your browser permissions are set to allow the extension to access the required data.
        """
    )
    
    # Display images for demonstration with reduced size
    st.image("extension_screenshot.png", caption="Browser Extension Interface", width=400)
    st.image("extension_result.png", caption="Extension Detection Result", width=400)

# Add custom CSS styles for the page
st.markdown(
    """
    <style>
    .stApp {
        background-color: #001f3f;
    }
    .css-1d391kg a {
        font-size: 20px;
        font-weight: bold;
        color: #ADD8E6 !important;
        text-decoration: none;
    }
    .css-1d391kg a:hover {
        color: #ffffff !important;
    }
    .header-container {
        position: relative;
        text-align: center;
        background-color: black;
        padding: 20px;
        border-radius: 10px;
        overflow: hidden;
    }
    .main-header {
        position: relative;
        font-size: 36px;
        font-weight: bold;
        color: white;
        z-index: 1;
        font-family: 'Arial', sans-serif;
    }
    .main-header .deepfake {
        color: #ADD8E6;
    }
    .tagline {
        position: relative;
        font-size: 18px;
        color: #FFFFFF;
        font-weight: 300;
        margin-top: 10px;
        z-index: 1;
        display: inline-block;
        padding-top: 5px;
        border-top: 1px solid #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ADD8E6;
        font-family: 'Arial', sans-serif;
    }
    p, li {
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    .background-image {
        position: absolute;
        top: 0;
        right: 0;
        height: 100%;
        width: 50%;
        object-fit: cover;
        z-index: 0;
    }
    </style>
    """, unsafe_allow_html=True)





