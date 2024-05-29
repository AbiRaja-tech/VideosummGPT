import streamlit as st
from streamlit import session_state
import json
import re
import hashlib
import cv2
import os
import shutil
import google.generativeai as genai # type: ignore
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import subprocess
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory # type: ignore

load_dotenv()

DIRECTORY = "frames"
FRAME_PREFIX = "_frame"
VIDEO_PATH = "video.mp4"

session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0
    
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# def check_structural_similarity(image1, image2):
#     image1 = cv2.imread(image1)
#     image2 = cv2.imread(image2)
#     if image1.shape[-1] != 3:
#         image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
#     if image2.shape[-1] != 3:
#         image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
#     b1, g1, r1 = cv2.split(image1)
#     b2, g2, r2 = cv2.split(image1)
#     ssim_b, _ = ssim(b1, b2, full=True)
#     ssim_g, _ = ssim(g1, g2, full=True)
#     ssim_r, _ = ssim(r1, r2, full=True)
#     similarity_index = (ssim_b + ssim_g + ssim_r) / 3
#     return similarity_index


def create_frame_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)


def extract_frame_from_video(video_file_path):
    st.write(f"Extracting {video_file_path} at 1 frame per second. This might take a bit...")
    create_frame_output_dir(DIRECTORY)
    vidcap = cv2.VideoCapture(video_file_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps
    output_file_prefix = (
        os.path.basename(video_file_path).replace(".", "_").replace(" ", "_")
    )
    frame_count = 0
    previous_frame = None
    count = 0
    threshold = 0.9
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, "Extracting frames...")
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            break
        if int(count / fps) == frame_count:
            b, g, r = cv2.split(frame)
            if previous_frame is not None:
                ssim_b, _ = ssim(previous_frame[0], b, full=True)
                ssim_g, _ = ssim(previous_frame[1], g, full=True)
                ssim_r, _ = ssim(previous_frame[2], r, full=True)
                similarity_index = (ssim_b + ssim_g + ssim_r) / 3
                if similarity_index > threshold:
                    continue
            min = frame_count // 60
            sec = frame_count % 60
            time_string = f"{min:02d}_{sec:02d}"
            image_name = f"{output_file_prefix}{FRAME_PREFIX}{time_string}.jpg"
            output_filename = DIRECTORY + "//" + image_name
            cv2.imwrite(output_filename, frame)
            frame_count += 1
            prev_frame = output_filename
            progress_bar.progress(int(count / total_frames * 100), f"Extracting frames... {count}/{total_frames}")
            previous_frame = cv2.split(frame)
        count += 1
    vidcap.release()
    progress_bar.progress(100, f"Extracting frames... {total_frames}/{total_frames}")
    st.info(
        f"Completed video frame extraction!\n\n")
    
class File:
    def __init__(self, file_path: str, display_name: str = None):
        self.file_path = file_path
        if display_name:
            self.display_name = display_name
        self.timestamp = get_timestamp(file_path)

    def set_file_response(self, response):
        self.response = response

def extract_audio(video_file_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract audio from the video file
    output_audio_path = os.path.join(output_dir, "audio.wav")
    subprocess.run(["ffmpeg", "-y", "-i", video_file_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_audio_path], check=True)
    return output_audio_path

def get_timestamp(filename):
    """Extracts the frame count (as an integer) from a filename with the format
    'output_file_prefix_frame00:00.jpg'.
    """
    parts = filename.split(FRAME_PREFIX)
    if len(parts) != 2:
        return None  # Indicates the filename might be incorrectly formatted
    return parts[1].split(".")[0]


def process_files_in_directory(directory):
    files = os.listdir(directory)
    files = sorted(files)
    files_to_upload = []
    for file in files:
        files_to_upload.append(
            File(file_path=os.path.join(directory, file))
        )

    # audio_path = extract_audio(VIDEO_PATH, "audio")
    # files_to_upload.append(File(file_path=audio_path))
    
    uploaded_files = []
    st.write(f"Uploading {len(files_to_upload)} files. This might take a bit...")
    progress_bar = st.progress(0, "Uploading files...")

    for i, file in enumerate(files_to_upload):
        response = genai.upload_file(path=file.file_path)
        file.set_file_response(response)
        uploaded_files.append(file)
        progress_bar.progress(int((i + 1) / len(files_to_upload) * 100), f"Uploading files... {i + 1}/{len(files_to_upload)}")

    st.info(f"Completed file uploads!\n\nUploaded: {len(uploaded_files)} files")
    return uploaded_files


# def process_files_in_directory(directory):
#     files = os.listdir(directory)
#     files = sorted(files)
#     files_to_upload = []
#     for file in files:
#         files_to_upload.append(
#             File(file_path=os.path.join(directory, file))
#         )

#     audio_path = extract_audio(VIDEO_PATH, "audio")
#     files_to_upload.append(File(file_path=audio_path))
    
#     uploaded_files = []
#     print(
#         f"Uploading {len(files_to_upload)} files. This might take a bit..."
#     )

#     for file in files_to_upload:
#         response = genai.upload_file(path=file.file_path)
#         file.set_file_response(response)
#         uploaded_files.append(file)

#     print(f"Completed file uploads!\n\nUploaded: {len(uploaded_files)} files")
#     return uploaded_files


def generate_video_summary(uploaded_files, video_file_path):
    # Define safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
    prompt = "Please analyze the extracted frames from the video and summarize its content in detail. Analyze the images of the video and incorporate its insights into the summary. The video frames have been uploaded for analysis."
    # prompt = "Please analyze the extracted frames from the video and summarize its content in detail. Additionally, analyze the audio of the video and incorporate its insights into the summary. The video frames and audio have been uploaded for analysis."

    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    def make_request(prompt, files):
        request = [prompt]
        for file in files:
            try:
                file.timestamp = file.timestamp.replace("_", ":")
            except Exception as e:
                pass
            request.append(file.timestamp)
            request.append(file.response)
        return request

    request = make_request(prompt, uploaded_files)


    response = model.generate_content(
        request, request_options={"timeout": 600}, safety_settings=safety_settings
    )

    try:
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None
        
        
def delete_uploaded_files(uploaded_files):
    st.write(f"Deleting {len(uploaded_files)} images. This might take a bit...")
    progress_bar = st.progress(0, "Deleting uploaded files...")
    for i, file in enumerate(uploaded_files):
        genai.delete_file(file.response.name)
        progress_bar.progress(int((i + 1) / len(uploaded_files) * 100), f"Deleting uploaded files... {i + 1}/{len(uploaded_files)}")
    st.write(f"Completed deleting files!\n\nDeleted: {len(uploaded_files)} files")

# def delete_uploaded_files(uploaded_files):
#     print(f"Deleting {len(uploaded_files)} images. This might take a bit...")
#     for file in uploaded_files:
#         genai.delete_file(file.response.name)
#         print(f"Deleted {file.file_path} at URI {file.response.uri}")
#     print(f"Completed deleting files!\n\nDeleted: {len(uploaded_files)} files")

# def extract_frame_from_video(video_file_path):
#     print(
#         f"Extracting {video_file_path} at 1 frame per second. This might take a bit..."
#     )
#     create_frame_output_dir(DIRECTORY)
#     vidcap = cv2.VideoCapture(video_file_path)
#     fps = vidcap.get(cv2.CAP_PROP_FPS)
#     frame_duration = 1 / fps
#     output_file_prefix = (
#         os.path.basename(video_file_path).replace(".", "_").replace(" ", "_")
#     )
#     frame_count = 0
#     prev_frame = None
#     count = 0
#     while vidcap.isOpened():
#         success, frame = vidcap.read()
#         if not success:
#             break
#         if int(count / fps) == frame_count:
#             min = frame_count // 60
#             sec = frame_count % 60
#             time_string = f"{min:02d}_{sec:02d}"
#             image_name = f"{output_file_prefix}{FRAME_PREFIX}{time_string}.jpg"
#             output_filename = DIRECTORY + "//" + image_name
#             if prev_frame is not None:
#                 similarity = check_structural_similarity(
#                     prev_frame, output_filename)
#                 if similarity > 0.9:
#                     print(
#                         f"Skipping frame {frame_count} as it is similar to the previous frame."
#                     )
#                     frame_count += 1
#                     count += 1
#                     continue
#             cv2.imwrite(output_filename, frame)
#             frame_count += 1
#             prev_frame = output_filename
#         count += 1
#     vidcap.release()
#     print(
#         f"Completed video frame extraction!\n\nExtracted: {frame_count} frames")


def generate_summary():
    extract_frame_from_video(VIDEO_PATH)
    uploaded_files = process_files_in_directory(DIRECTORY)
    summary = generate_video_summary(uploaded_files, VIDEO_PATH)
    delete_uploaded_files(uploaded_files)
    return summary


    
def user_exists(email, json_file_path):
    # Function to check if user with the given email exists
    with open(json_file_path, "r") as file:
        users = json.load(file)
        for user in users["users"]:
            if user["email"] == email:
                return True
    return False


def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")
        if st.form_submit_button("Signup"):
            if not name:
                st.error("Name field cannot be empty.")
            elif not email:
                st.error("Email field cannot be empty.")
            elif not re.match(r"^[\w\.-]+@[\w\.-]+$", email):
                st.error("Invalid email format. Please enter a valid email address.")
            elif user_exists(email, json_file_path):
                st.error(
                    "User with this email already exists. Please choose a different email."
                )
            elif not age:
                st.error("Age field cannot be empty.")
            elif not password or len(password) < 6:  # Minimum password length of 6
                st.error("Password must be at least 6 characters long.")
            elif password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                user = create_account(
                    name, email, age, sex, password, json_file_path
                )
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Signup successful. You are now logged in!")
                    

def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)


        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
    
                return user
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None

def initialize_database(
    json_file_path="data.json"
):
    try:
        if not os.path.exists(json_file_path):
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)

        
    except Exception as e:
        print(f"Error initializing database: {e}")

def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        email = email.lower()
        password = hashlib.md5(password.encode()).hexdigest()
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
        }

        data["users"].append(user_info)

        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None

def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")
    password = hashlib.md5(password.encode()).hexdigest()
    username = username.lower()

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")


def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None

def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")
        
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
        


def main(json_file_path="data.json"):

    st.sidebar.title("Video Summarization App")
    page = st.sidebar.radio(
        "Go to",
        (
            "Signup/Login",
            "Dashboard",
            "Summarize Video",
        ),
        key="page",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")
            
            
    elif page == "Summarize Video":
        if session_state.get("logged_in"):
            st.title("Summarize Video")
            video = st.file_uploader("Upload a video file", type=["mp4"])
            if video:
                st.video(video)
                with open(VIDEO_PATH + "_temp", "wb") as f:
                    f.write(video.read())
                with st.spinner("Processing video..."):
                    subprocess.run(["ffmpeg", "-y", "-i", VIDEO_PATH + "_temp", VIDEO_PATH], check=True)
                summary = generate_summary()
                if os.path.exists(VIDEO_PATH + "_temp"):
                    os.remove(VIDEO_PATH + "_temp")
                if os.path.exists(DIRECTORY):
                    shutil.rmtree(DIRECTORY)
                if os.path.exists("frames"):
                    shutil.rmtree("frames")
                if os.path.exists("VIDEO_PATH"):
                    os.remove("VIDEO_PATH")
                    
                if summary:
                    st.markdown("## Summary")
                    st.markdown(f"<div style='font-size: 16px; margin-top: 10px;'>{summary}</div>", unsafe_allow_html=True)
                else:
                    st.warning("The summary for the selected video could not be generated. Please try a different video.")
        else:
            st.warning("Please login/signup to summarize a video.")
    
                
if __name__ == "__main__":
    initialize_database()
    main()