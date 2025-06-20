import sys
import cv2 as cv
import numpy as np
import mediapipe as mp
import csv
import copy
import itertools
from collections import Counter, deque
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QPushButton, QScrollArea, QDialog, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier
import time
import os

# Utility functions from app.py (copy as needed)
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
    # Draw finger lines with 30% transparent grey
    overlay = image.copy()
    line_color = (180, 180, 180)  # grey
    alpha_line = 0.3
    alpha_circle = 0.5
    circle_color = (220, 220, 220)  # light grey
    if len(landmark_point) > 0:
        # Thumb
        cv.line(overlay, tuple(landmark_point[2]), tuple(landmark_point[3]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[3]), tuple(landmark_point[4]), line_color, 6)
        # Index finger
        cv.line(overlay, tuple(landmark_point[5]), tuple(landmark_point[6]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[6]), tuple(landmark_point[7]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[7]), tuple(landmark_point[8]), line_color, 6)
        # Middle finger
        cv.line(overlay, tuple(landmark_point[9]), tuple(landmark_point[10]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[10]), tuple(landmark_point[11]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[11]), tuple(landmark_point[12]), line_color, 6)
        # Ring finger
        cv.line(overlay, tuple(landmark_point[13]), tuple(landmark_point[14]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[14]), tuple(landmark_point[15]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[15]), tuple(landmark_point[16]), line_color, 6)
        # Little finger
        cv.line(overlay, tuple(landmark_point[17]), tuple(landmark_point[18]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[18]), tuple(landmark_point[19]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[19]), tuple(landmark_point[20]), line_color, 6)
        # Palm
        cv.line(overlay, tuple(landmark_point[0]), tuple(landmark_point[1]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[1]), tuple(landmark_point[2]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[2]), tuple(landmark_point[5]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[5]), tuple(landmark_point[9]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[9]), tuple(landmark_point[13]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[13]), tuple(landmark_point[17]), line_color, 6)
        cv.line(overlay, tuple(landmark_point[17]), tuple(landmark_point[0]), line_color, 6)
    # Blend overlay with original image for lines
    cv.addWeighted(overlay, alpha_line, image, 1 - alpha_line, 0, image)
    # Draw keypoints as light grey, 50% transparent
    for index, landmark in enumerate(landmark_point):
        if index in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
            # Draw on overlay, then blend
            circ_overlay = image.copy()
            cv.circle(circ_overlay, (landmark[0], landmark[1]), 5, circle_color, -1)
            cv.circle(circ_overlay, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            cv.addWeighted(circ_overlay, alpha_circle, image, 1 - alpha_circle, 0, image)
        if index in [4, 8, 12, 16, 20]:
            circ_overlay = image.copy()
            cv.circle(circ_overlay, (landmark[0], landmark[1]), 8, circle_color, -1)
            cv.circle(circ_overlay, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            cv.addWeighted(circ_overlay, alpha_circle, image, 1 - alpha_circle, 0, image)
    return image

def load_labels(label_path):
    with open(label_path, encoding='utf-8-sig') as f:
        return [row[0] for row in csv.reader(f)]

class SignsDialog(QDialog):
    def __init__(self, signs_folder, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Available Hand Signs')
        self.setStyleSheet('background-color: #181a20; color: #f4f4f4;')
        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setSpacing(20)
        sign_files = [f for f in os.listdir(signs_folder) if f.lower().endswith('.png')]
        sign_files.sort()
        for idx, fname in enumerate(sign_files):
            word = fname.replace('.png','').replace('_',' ')
            # Capitalize I if present
            word = ' '.join(['I' if w=='I' else w.capitalize() for w in word.split()])
            label = QLabel(word)
            label.setStyleSheet('color: #f4f4f4; font-size: 18px;')
            img = QPixmap(os.path.join(signs_folder, fname)).scaled(120, 120, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            img_label = QLabel()
            img_label.setPixmap(img)
            img_label.setStyleSheet('background: #23272f; border-radius: 8px;')
            grid.addWidget(img_label, idx, 0)
            grid.addWidget(label, idx, 1)
        content.setLayout(grid)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.setMinimumSize(400, 600)

class ASLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ASL Recognition Professional')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet('background-color: #181a20;')  # Dark mode background
        self.init_ui()
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()
        self.keypoint_classifier_labels = load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')
        self.point_history_classifier_labels = load_labels('model/point_history_classifier/point_history_classifier_label.csv')
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        self.sentence = []
        self.last_sign = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)
        self.sign_hold_start = None
        self.sign_hold_name = None
        self.sign_hold_time = 0
        self.timer_visible = False
        self.checkmark_time = None
        self.signs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'signs')
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        # Title bar at the top
        title_bar = QLabel('ASL Recognition')
        title_bar.setAlignment(Qt.AlignCenter)
        title_bar.setFont(QFont('Arial', 28, QFont.Bold))
        title_bar.setStyleSheet('background: #23272f; color: #fff; padding: 24px 0 18px 0; letter-spacing: 2px; border-bottom: 2px solid #333;')
        main_layout.addWidget(title_bar, 0)
        # Main content area (horizontal split)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        # Left column: Facecam on top, buttons below (centered in dead space)
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(0)
        self.video_label = QLabel()
        self.video_label.setStyleSheet('background-color: #23272f; border-radius: 18px;')
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setMaximumSize(1920, 1080)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_col.addWidget(self.video_label, 4)
        # Buttons area (centered both vertically and horizontally)
        btns_container = QFrame()
        btns_container.setStyleSheet('background: transparent;')
        btns_outer_layout = QVBoxLayout(btns_container)
        btns_outer_layout.setContentsMargins(0, 0, 0, 0)
        btns_outer_layout.setSpacing(0)
        btns_outer_layout.addStretch(1)
        btns_layout = QHBoxLayout()
        btns_layout.setSpacing(60)
        btns_layout.setAlignment(Qt.AlignCenter)
        # Undo button
        undo_btn = QPushButton('Undo')
        undo_btn.setFont(QFont('Arial', 28, QFont.Bold))
        undo_btn.setMinimumHeight(90)
        undo_btn.setMinimumWidth(180)
        undo_btn.setStyleSheet('''
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2e3138, stop:1 #23272f);
                color: #fff;
                border-radius: 22px;
                padding: 32px 64px;
                border: 2px solid #444;
                font-weight: bold;
                box-shadow: 0px 8px 32px #00000044;
            }
            QPushButton:hover {
                background: #353a40;
                border: 2px solid #5eea7c;
                color: #5eea7c;
            }
        ''')
        undo_btn.clicked.connect(self.undo_last_word)
        btns_layout.addWidget(undo_btn)
        # Clear button
        clear_btn = QPushButton('Clear')
        clear_btn.setFont(QFont('Arial', 28, QFont.Bold))
        clear_btn.setMinimumHeight(90)
        clear_btn.setMinimumWidth(180)
        clear_btn.setStyleSheet('''
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2e3138, stop:1 #23272f);
                color: #fff;
                border-radius: 22px;
                padding: 32px 64px;
                border: 2px solid #444;
                font-weight: bold;
                box-shadow: 0px 8px 32px #00000044;
            }
            QPushButton:hover {
                background: #353a40;
                border: 2px solid #e74c3c;
                color: #e74c3c;
            }
        ''')
        clear_btn.clicked.connect(self.clear_sentence_box)
        btns_layout.addWidget(clear_btn)
        # Signs button
        signs_btn = QPushButton('Signs')
        signs_btn.setFont(QFont('Arial', 28, QFont.Bold))
        signs_btn.setMinimumHeight(90)
        signs_btn.setMinimumWidth(180)
        signs_btn.setStyleSheet('''
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2e3138, stop:1 #23272f);
                color: #fff;
                border-radius: 22px;
                padding: 32px 64px;
                border: 2px solid #444;
                font-weight: bold;
                box-shadow: 0px 8px 32px #00000044;
            }
            QPushButton:hover {
                background: #353a40;
                border: 2px solid #5dade2;
                color: #5dade2;
            }
        ''')
        signs_btn.clicked.connect(self.show_signs_dialog)
        btns_layout.addWidget(signs_btn)
        btns_outer_layout.addLayout(btns_layout)
        btns_outer_layout.addStretch(1)
        left_col.addWidget(btns_container, 1)
        content_layout.addLayout(left_col, 5)  # 5/7 of width
        # Right: Tall sentence box
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(0)
        sentence_box_container = QFrame()
        sentence_box_container.setStyleSheet('background: transparent;')
        sentence_box_layout = QVBoxLayout(sentence_box_container)
        sentence_box_layout.setContentsMargins(0, 0, 0, 0)
        sentence_box_layout.setSpacing(0)
        self.sentence_box = QTextEdit()
        self.sentence_box.setFont(QFont('Arial', 22))
        self.sentence_box.setStyleSheet('background: #23272f; color: #f4f4f4; border: none; border-radius: 4px; padding: 16px;')
        sentence_box_layout.addWidget(self.sentence_box, 10)
        right_col.addWidget(sentence_box_container, 10)
        content_layout.addLayout(right_col, 2)  # 2/7 of width
        main_layout.addLayout(content_layout, 1)
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv.flip(frame, 1)
        debug_image = frame.copy()
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        sign_text = None
        brect = None
        timer_drawn = False
        now = time.time()
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, self.point_history)
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == "Not applicable":
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()
                debug_image = draw_landmarks(debug_image, landmark_list)
                # Overlay hand sign name on video near bounding box
                sign_text = self.keypoint_classifier_labels[hand_sign_id]
                # Timer logic for holding sign
                if self.sign_hold_name == sign_text:
                    self.sign_hold_time = now - self.sign_hold_start
                else:
                    self.sign_hold_name = sign_text
                    self.sign_hold_start = now
                    self.sign_hold_time = 0
                    self.timer_visible = True
                # Only add word if held for 1 second
                if self.sign_hold_time >= 1.0:
                    if len(self.sentence) == 0 or self.sentence[-1] != sign_text:
                        self.sentence.append(sign_text)
                        self.update_sentence_box()
                        self.timer_visible = False
                        self.checkmark_time = time.time()
                # Draw the hand sign name on the video (dark mode style)
                if sign_text:
                    x, y, x2, y2 = brect
                    cv.rectangle(debug_image, (x, y-35), (x2, y), (30, 30, 30), -1)
                    cv.putText(debug_image, sign_text, (x+8, y-10), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
                # Draw timer in top right of facecam only if visible
                if self.timer_visible and self.sign_hold_time < 1.0:
                    timer_text = f"{self.sign_hold_time:.1f}s"
                    cv.rectangle(debug_image, (debug_image.shape[1]-120, 10), (debug_image.shape[1]-10, 60), (30,30,30), -1)
                    cv.putText(debug_image, timer_text, (debug_image.shape[1]-110, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv.LINE_AA)
                    timer_drawn = True
        else:
            self.point_history.append([0, 0])
            self.sign_hold_name = None
            self.sign_hold_start = None
            self.sign_hold_time = 0
            self.timer_visible = False
        # Draw VALID if needed
        if self.checkmark_time is not None and (time.time() - self.checkmark_time) < 0.3:
            h, w = debug_image.shape[:2]
            # Neon green background rectangle
            cv.rectangle(debug_image, (20, h-80), (220, h-20), (57, 255, 20), -1)
            cv.putText(debug_image, "VALID", (40, h-35), cv.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), 5, cv.LINE_AA)
        elif self.checkmark_time is not None and (time.time() - self.checkmark_time) >= 0.3:
            self.checkmark_time = None
        # Convert to QImage and display
        img = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = qt_img.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_img))
    def update_sentence_box(self):
        self.sentence_box.setPlainText(' '.join(self.sentence))
    def undo_last_word(self):
        text = self.sentence_box.toPlainText().strip().split()
        if text:
            text = text[:-1]
            self.sentence = text
            self.update_sentence_box()
    def clear_sentence_box(self):
        self.sentence_box.clear()
        self.sentence = []
    def show_signs_dialog(self):
        dlg = SignsDialog(self.signs_folder, self)
        dlg.exec_()
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ASLApp()
    window.showMaximized()
    sys.exit(app.exec_()) 