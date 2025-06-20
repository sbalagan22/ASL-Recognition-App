import streamlit as st
import cv2 as cv
import numpy as np
import mediapipe as mp
import csv
import copy
import itertools
from collections import deque
import os
import time
from PIL import Image

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier

# Utility functions from app.py
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    for landmark in landmark_point:
        cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

def draw_info(image, fps, mode, number):
    # cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    # cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

def load_labels(label_path):
    with open(label_path, encoding='utf-8-sig') as f:
        return [row[0] for row in csv.reader(f)]

def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center;'>ASL Hand Gesture Recognition</h1>", unsafe_allow_html=True)

    # Initialize session state
    if 'sentence' not in st.session_state:
        st.session_state.sentence = []
    if 'last_sign' not in st.session_state:
        st.session_state.last_sign = None
    if 'sign_hold_start' not in st.session_state:
        st.session_state.sign_hold_start = None
    if 'sign_hold_name' not in st.session_state:
        st.session_state.sign_hold_name = None
    if 'valid_sign_time' not in st.session_state:
        st.session_state.valid_sign_time = 0

    # Load models and labels
    use_static_image_mode = False
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    keypoint_labels_path = os.path.join(script_dir, 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv')
    point_history_labels_path = os.path.join(script_dir, 'model', 'point_history_classifier', 'point_history_classifier_label.csv')

    keypoint_classifier_labels = load_labels(keypoint_labels_path)
    point_history_classifier_labels = load_labels(point_history_labels_path)

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    
    # UI Layout
    col1, col2 = st.columns([5, 2])

    with col1:
        st.header("Live Feed")
        run = st.checkbox('Run Webcam', value=True)
        FRAME_WINDOW = st.image([])
        
        # Controls below the video feed
        st.header("Controls")
        control_cols = st.columns(2)
        with control_cols[0]:
            if st.button('Undo Last Word', use_container_width=True):
                if st.session_state.sentence:
                    st.session_state.sentence.pop()
                    st.session_state.last_sign = None # Allow re-adding the previous sign
        with control_cols[1]:
            if st.button('Clear All', use_container_width=True):
                st.session_state.sentence = []
                st.session_state.last_sign = None

    with col2:
        st.header("Recognized Text")
        sentence_box = st.empty()

    while run:
        fps = cvFpsCalc.get()
        ret, image = cap.read()
        if not ret:
            st.warning("Could not read frame from webcam.")
            break
        
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        hand_sign_text = ""
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                hand_sign_text = keypoint_classifier_labels[hand_sign_id]
                
                point_history.append(landmark_list[8])
                
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                
                finger_gesture_history.append(finger_gesture_id)
                
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, hand_sign_text, "")

                # Sign Hold Logic
                sign_hold_duration = 0
                if hand_sign_text == st.session_state.sign_hold_name and hand_sign_text != " ":
                    if st.session_state.sign_hold_start is None:
                        st.session_state.sign_hold_start = time.time()
                    
                    sign_hold_duration = time.time() - st.session_state.sign_hold_start

                    if sign_hold_duration > 1.0: # 1 second hold
                        if st.session_state.last_sign != hand_sign_text:
                            st.session_state.sentence.append(hand_sign_text)
                            st.session_state.last_sign = hand_sign_text
                            st.session_state.valid_sign_time = time.time() # Trigger VALID notification
                        st.session_state.sign_hold_start = None # Reset after adding
                else:
                    st.session_state.sign_hold_name = hand_sign_text
                    st.session_state.sign_hold_start = None
                    st.session_state.last_sign = None

        # Draw Timer on top right of the webcam feed
        if st.session_state.sign_hold_start is not None and st.session_state.last_sign != st.session_state.sign_hold_name:
            progress = min((time.time() - st.session_state.sign_hold_start) / 1.0, 1.0)
            bar_width = int(progress * 100)
            
            # Black background for the bar
            cv.rectangle(debug_image, (debug_image.shape[1] - 130, 20), (debug_image.shape[1] - 30, 50), (0, 0, 0), -1)
            # Green progress indicator
            cv.rectangle(debug_image, (debug_image.shape[1] - 130, 20), (debug_image.shape[1] - 130 + bar_width, 50), (144, 238, 144), -1)
            # White border
            cv.rectangle(debug_image, (debug_image.shape[1] - 130, 20), (debug_image.shape[1] - 30, 50), (255, 255, 255), 2)

        # Draw "VALID" notification
        if time.time() - st.session_state.valid_sign_time < 0.4:
            cv.rectangle(debug_image, (0, debug_image.shape[0] - 50), (180, debug_image.shape[0]), (144, 238, 144), -1)
            cv.putText(debug_image, "VALID", (10, debug_image.shape[0] - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv.LINE_AA)

        debug_image = draw_info(debug_image, fps, " ", " ")
        FRAME_WINDOW.image(debug_image, channels="BGR")
        
        sentence_str = " ".join(st.session_state.sentence)
        sentence_box.markdown(f"### {sentence_str}")

    else:
        st.write('Webcam is stopped.')
        cap.release()

if __name__ == '__main__':
    main() 