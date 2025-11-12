import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import time
import math
from deepface import DeepFace

st.set_page_config(layout="wide", page_title="Gesture & Eye-Controlled Presenter")

# -------------------------
# Custom CSS for Dark Violet UI
# -------------------------
css = """
<style>
    /* --- Main App Background --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #2a0a4f; /* Dark Violet */
    }
    
    /* --- Main Content --- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* --- All Text (FIX: Increased font-weight) --- */
    body, .stMarkdown, .stRadio, .stFileUploader, .stExpander, .stButton {
        color: #262626; /* Dark text (for inside white panel) */
        font-weight: 700; /* Bolder text */
    }
    h3 {
        color: #262626;
        font-weight: 700; /* Bolder title */
    }

    /* --- Left Column (Controls) --- */
    [data-testid="stHorizontalBlock"] > div:nth-child(1) > [data-testid="stVerticalBlock"] {
        background-color: #FFFFFF; /* White panel */
        border: 2px solid #1a0630; /* FIX: Dark and Bold Border */
        padding: 1rem;
        border-radius: 10px;
    }

    /* --- Webcam Video --- */
    video {
        border-radius: 8px;
    }

    /* --- Start/Stop Buttons (Kept Orange as requested) --- */
    .stButton > button {
        background-color: #FFA500; /* Brighter Orange */
        color: #FFFFFF; /* White text */
        border: 1px solid #E69500; /* Brighter Orange Shade */
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #E69500; /* Brighter Orange Shade */
        color: #FFFFFF;
        border-color: #CC8400; /* Brighter Orange Shade */
    }

    /* --- Radio Buttons (Control Mode) --- */
    .stRadio > label {
        font-weight: 700; /* "Select Control Mode:" (EXTRA BOLD) */
        color: #262626;
    }
    
    /* --- NEW: Slide Button for Radio --- */
    .stRadio > div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        width: 100%;
        background-color: #4a1a8f; /* Lighter Violet */
        border: 1px solid #1a0630; /* Dark Violet Border */
        border-radius: 8px;
        padding: 4px;
    }
    .stRadio > div[role="radiogroup"] > label {
        flex: 1;
        text-align: center;
        padding: 10px 16px; /* FIX: Increased padding to be taller */
        margin: 0;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    /* This targets the div *inside* the label that contains the text */
    .stRadio > div[role="radiogroup"] > label > div { 
        color: #FFFFFF; /* FIX: Light text for unselected */
        font-weight: 700; /* Bolder text */
    }
    
    /* Hide the actual radio button circle */
    .stRadio > div[role="radiogroup"] input {
        display: none;
    }
    
    /* Style for the *selected* label */
    .stRadio > div[role="radiogroup"] > label:has(input:checked) {
        background-color: #FFA500; /* Brighter Orange */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Style for the *selected* label's text */
    .stRadio > div[role="radiogroup"] > label:has(input:checked) > div {
        color: #FFFFFF; /* White text for selected */
        font-weight: 700; /* Bolder text */
    }


    /* --- File Uploader --- */
    .stFileUploader > label {
        font-weight: 700; /* Bolder text */
    }
    .stFileUploader button {
        background-color: #FFA500; /* Brighter Orange */
        color: white;
        border: none;
        border-radius: 6px;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border-color: #4a1a8f; /* Lighter Violet Border */
    }

    /* --- Expander (Gesture Guide) --- */
    .stExpander {
        background-color: #4a1a8f; /* Lighter Violet */
        border: 1px solid #1a0630; /* Dark Violet Border */
        border-radius: 8px;
    }
    .stExpander summary {
        color: #FFFFFF; /* FIX: Light Text */
        font-weight: 700; /* Bolder text */
    }
    .stExpander div { /* Text inside expander */
        color: #E0E0E0; /* Lighter text */
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# -------------------------
# Session State Initialization
# -------------------------
if "presenting" not in st.session_state:
    st.session_state.presenting = False
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False
if "last_action_time" not in st.session_state:
    st.session_state.last_action_time = 0
if "movement_history" not in st.session_state:
    st.session_state.movement_history = deque(maxlen=12)
if "left_eye_open" not in st.session_state:
    st.session_state.left_eye_open = True
if "right_eye_open" not in st.session_state:
    st.session_state.right_eye_open = True
if "control_mode" not in st.session_state:
    st.session_state.control_mode = "Hand Gestures"

# --- Drawing State ---
if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "current_path" not in st.session_state:
    st.session_state.current_path = []
if "cursor_pos" not in st.session_state:
    st.session_state.cursor_pos = (0, 0)
if "draw_color" not in st.session_state:
    st.session_state.draw_color = (255, 255, 0) # Yellow highlighter
if "draw_size" not in st.session_state:
    st.session_state.draw_size = 20
if "draw_alpha" not in st.session_state:
    st.session_state.draw_alpha = 128 # Semi-transparent
if "last_tool_click_time" not in st.session_state:
    st.session_state.last_tool_click_time = 0
if "slide_dims" not in st.session_state:
    st.session_state.slide_dims = (800, 600) # (width, height) placeholder

# --- Emotion State ---
if "last_emotion_time" not in st.session_state:
    st.session_state.last_emotion_time = 0
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "N/A"
if "current_emotion_score" not in st.session_state:
    st.session_state.current_emotion_score = 0.0


# -------------------------
# Toolbar Definition
# -------------------------
TOOLBAR_HEIGHT = 50
BUTTON_WIDTH = 70 # Smaller buttons for more tools
toolbar_buttons = {
    # --- Colors ---
    "Highlighter": {
        "rect": [0, 0, BUTTON_WIDTH, TOOLBAR_HEIGHT],
        "tool": "color", "color": (255, 255, 0), "size": 20, "alpha": 128
    },
    "Red": {
        "rect": [BUTTON_WIDTH, 0, BUTTON_WIDTH*2, TOOLBAR_HEIGHT],
        "tool": "color", "color": (255, 0, 0), "size": 5, "alpha": 255
    },
    "Blue": {
        "rect": [BUTTON_WIDTH*2, 0, BUTTON_WIDTH*3, TOOLBAR_HEIGHT],
        "tool": "color", "color": (0, 0, 255), "size": 5, "alpha": 255
    },
    "Green": {
        "rect": [BUTTON_WIDTH*3, 0, BUTTON_WIDTH*4, TOOLBAR_HEIGHT],
        "tool": "color", "color": (0, 255, 0), "size": 5, "alpha": 255
    },
    "Black": {
        "rect": [BUTTON_WIDTH*4, 0, BUTTON_WIDTH*5, TOOLBAR_HEIGHT],
        "tool": "color", "color": (0, 0, 0), "size": 5, "alpha": 255
    },
    # --- Sizes ---
    "Size -": {
        "rect": [BUTTON_WIDTH*5, 0, BUTTON_WIDTH*6, TOOLBAR_HEIGHT],
        "tool": "size", "action": "decrease"
    },
    "Size +": {
        "rect": [BUTTON_WIDTH*6, 0, BUTTON_WIDTH*7, TOOLBAR_HEIGHT],
        "tool": "size", "action": "increase"
    },
    # --- Actions ---
    "Undo": {
        "rect": [BUTTON_WIDTH*7, 0, BUTTON_WIDTH*8, TOOLBAR_HEIGHT],
        "tool": "action", "action": "undo"
    },
    "Clear": {
        "rect": [BUTTON_WIDTH*8, 0, BUTTON_WIDTH*9, TOOLBAR_HEIGHT],
        "tool": "action", "action": "clear"
    }
}


# -------------------------
# Helper: Draw Toolbar (Dark Violet Theme)
# -------------------------
def draw_toolbar_image(slide_width, selected_color, selected_size, selected_alpha, cursor_pos):
    # Create toolbar image (Dark Violet background)
    toolbar_img = Image.new('RGBA', (max(slide_width, BUTTON_WIDTH * 9 + 150), TOOLBAR_HEIGHT), (42, 10, 79, 255))
    draw_context = ImageDraw.Draw(toolbar_img)
    
    # Draw line separator (Darker)
    draw_context.line([0, TOOLBAR_HEIGHT-1, slide_width, TOOLBAR_HEIGHT-1], fill=(26, 6, 48, 255), width=2)
    
    for tool_name, props in toolbar_buttons.items():
        x1, y1, x2, y2 = props["rect"]
        
        # Determine fill color (Lighter violet buttons)
        fill_color = (74, 26, 143) # Default for actions/sizes
        if props.get("tool") == "color":
            fill_color = props["color"]
        
        # --- Highlight selected tool ---
        is_selected = False
        if props.get("tool") == "color":
            if props.get("color") == selected_color and props.get("alpha") == selected_alpha:
                is_selected = True # Exact match
        
        outline_color = (59, 130, 246) if is_selected else (26, 6, 48) # Blue outline, dark border
        draw_context.rectangle([x1+2, y1+2, x2-2, y2-4], fill=fill_color + (200,), outline=outline_color, width=3)
        
        # --- Draw Text (Light text) ---
        text_fill = (230, 230, 230) 

        text_to_draw = tool_name
        
        # Calculate text position to center it
        try:
            # Use textbbox if available (Pillow >= 9.2.0)
            text_bbox = draw_context.textbbox((0,0), text_to_draw)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # Fallback for older Pillow versions
            text_w, text_h = draw_context.textsize(text_to_draw)

        text_x = x1 + (BUTTON_WIDTH - text_w) // 2
        text_y = (TOOLBAR_HEIGHT - text_h) // 2 - 2
        
        # FIX: FAKE BOLD - Draw text twice with 1px offset
        draw_context.text((text_x+1, text_y), text_to_draw, fill=text_fill)
        draw_context.text((text_x, text_y), text_to_draw, fill=text_fill)
    
    # Draw current size
    size_text = f"Size: {selected_size}"
    try:
        size_bbox = draw_context.textbbox((0,0), size_text)
        size_w = size_bbox[2] - size_bbox[0]
        size_h = size_bbox[3] - size_bbox[1]
    except AttributeError:
        size_w, size_h = draw_context.textsize(size_text)

    size_text_x = BUTTON_WIDTH * 9 + 20 # Position after last button
    draw_context.rectangle([size_text_x - 5, 5, size_text_x + size_w + 10, TOOLBAR_HEIGHT - 5], fill=(74, 26, 143)) # Lighter violet bg
    
    # FIX: FAKE BOLD - Draw text twice with 1px offset
    draw_context.text((size_text_x+1, (TOOLBAR_HEIGHT - size_h) // 2 - 2), size_text, fill=(230, 230, 230)) # Light text
    draw_context.text((size_text_x, (TOOLBAR_HEIGHT - size_h) // 2 - 2), size_text, fill=(230, 230, 230)) # Light text


    # NEW: Draw cursor if it's on the toolbar
    if cursor_pos:
        cx, cy = cursor_pos
        if cy <= TOOLBAR_HEIGHT:
            r = 8 # Fixed size for toolbar cursor
            draw_context.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(59, 130, 246, 150), outline=(255,255,255), width=2) # Blue cursor

    return toolbar_img


# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([2, 8])  # 20% for controls, 80% for slide

with col1:
    st.markdown("### 📷 Webcam & Controls")
    webcam_placeholder = st.empty()
    
    st.markdown("---")
    
    # --- Start/Stop Controls (FIX: Removed columns for vertical layout) ---
    start_btn = st.button("▶️ Start", use_container_width=True)
    stop_btn = st.button("⏹ Stop", use_container_width=True)
    
    # --- NEW: Emotion Display ---
    st.markdown("### 😃 Emotion")
    emotion_placeholder = st.empty()
    st.markdown("---")
    # --- End New Section ---

    # --- Mode Selection (FIX: Set to horizontal) ---
    st.session_state.control_mode = st.radio(
        "Select Control Mode:",
        ["Hand Gestures", "Eye Blinks"],
        horizontal=True 
    )
    
    st.markdown("---")

    # --- PDF Uploader ---
    pdf_path = st.file_uploader("Upload slides (PDF)", type=["pdf"])
    
    # --- Gesture Info ---
    with st.expander("Show Gesture Guide"):
        st.markdown("**Hand Gestures:**")
        st.write("• Wave Left/Right (Open Palm) -> Next/Prev Slide")
        st.write("• 2 Fingers -> Zoom In")
        st.write("• 3 Fingers -> Zoom Out")
        st.write("• Little Finger -> Close Presentation")
        st.write("• **Index Finger** -> Move Cursor & Draw")
        st.write("• **Thumb Up Only** -> Select Tool")
        st.markdown("**Eye Gestures:**")
        st.write("• Left Eye Blink -> Previous Slide")
        st.write("• Right Eye Blink -> Next Slide")

with col2:
    # NEW: Separate placeholders for toolbar and slide
    toolbar_placeholder = st.empty()
    slide_placeholder = st.empty()

# -------------------------
# Helper: PDF -> Image
# -------------------------
def render_pdf_page(doc, page_index, zoom, crop_center=None):
    try:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if crop_center and zoom > 1.0:
            cx, cy = crop_center
            w, h = img.size
            base_w, base_h = 800, 600 
            crop_w, crop_h = int(base_w / (zoom * 0.5)), int(base_h / (zoom * 0.5))
            left = max(0, int(cx * zoom - crop_w // 2))
            top = max(0, int(cy * zoom - crop_h // 2))
            right = min(w, left + crop_w)
            bottom = min(h, top + crop_h)
            
            if left < right and top < bottom:
                img = img.crop((left, top, right, bottom))
        
        return img
    except Exception as e:
        # Fallback for any rendering error
        # MODIFIED: Standard Bright Orange
        img = Image.new("RGB", (800, 600), color=(255, 165, 0)) 
        draw_context = ImageDraw.Draw(img)
        draw_context.rectangle([0, 0, 799, 599], outline=(138, 43, 226), width=15) # Violet border
        draw_context.text((50, 100), "Error loading page.", fill=(0,0,0))
        return img


# -------------------------
# MediaPipe & DeepFace Solutions
# -------------------------
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# -------------------------
# Gesture Detection Utils
# -------------------------

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_ear(landmarks, eye_indices, frame_shape):
    try:
        p1 = landmarks[eye_indices[1]] # Top
        p5 = landmarks[eye_indices[5]] # Bottom
        p2 = landmarks[eye_indices[2]] # Top
        p4 = landmarks[eye_indices[4]] # Bottom
        p0 = landmarks[eye_indices[0]] # Corner
        p3 = landmarks[eye_indices[3]] # Corner
        
        v1 = distance(p1, p5)
        v2 = distance(p2, p4)
        h1 = distance(p0, p3)
        
        if h1 == 0: return 0.0
        ear = (v1 + v2) / (2.0 * h1)
        return ear
    except Exception:
        return 0.0

def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    coords = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    fingers = []
    
    try:
        thumb_tip_x = coords[4][0]
        thumb_ip_x = coords[3][0]
        index_x = coords[8][0]
        pinky_x = coords[20][0]
        hand_is_right = index_x < pinky_x
        
        if hand_is_right:
            thumb_extended = thumb_tip_x < thumb_ip_x
        else: # Left hand
            thumb_extended = thumb_tip_x > thumb_ip_x
        fingers.append(thumb_extended)
        
        for t, p in zip(tips_ids[1:], pip_ids[1:]):
            fingers.append(coords[t][1] < coords[p][1])
    except Exception:
        return [False] * 5
    return fingers

# -------------------------
# Load PDF or placeholder
# -------------------------
if pdf_path is not None:
    pdf_bytes = pdf_path.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
else:
    # --- MODIFIED: Placeholder Styling ---
    doc = fitz.open()
    
    page_rect = fitz.Rect(0, 0, 800, 600) # Standard slide size
    border_width = 15
    
    # --- Colors (0-1 scale for fitz) ---
    # MODIFIED: Standard Orange (255, 165, 0)
    bg_color = (1.0, 0.6471, 0.0)
    # Violet: (138, 43, 226)
    border_color = (0.5412, 0.1686, 0.8863)
    # Black
    text_color = (0, 0, 0)
    
    # --- Page 1 (Instructions) ---
    page = doc.new_page(width=page_rect.width, height=page_rect.height)
    # Background (Standard Orange)
    page.draw_rect(page_rect, color=bg_color, fill=bg_color)
    # Border (Violet)
    page.draw_rect(page_rect, color=border_color, width=border_width)
    # Text
    page.insert_text((50, 100), "No PDF uploaded.", fontsize=20, color=text_color)
    page.insert_text((50, 130), "Please upload a PDF to begin.", fontsize=20, color=text_color)

    # --- Page 2 (Sample) ---
    page2 = doc.new_page(width=page_rect.width, height=page_rect.height)
    page2.draw_rect(page_rect, color=bg_color, fill=bg_color)
    page2.draw_rect(page_rect, color=border_color, width=border_width)
    page2.insert_text((50, 100), "Sample Slide 2", fontsize=60, color=text_color)
    
    # --- Page 3 (Sample) ---
    page3 = doc.new_page(width=page_rect.width, height=page_rect.height)
    page3.draw_rect(page_rect, color=bg_color, fill=bg_color)
    page3.draw_rect(page_rect, color=border_color, width=border_width)
    page3.insert_text((50, 100), "Sample Slide 3", fontsize=60, color=text_color)
    # --- End Modified Block ---

num_pages = doc.page_count
current_page = 0
zoom_factor = 1.0
zoom_center = None

# -------------------------
# Start / Stop handlers
# -------------------------
if start_btn:
    st.session_state.presenting = True
    st.session_state.stop_flag = False
    st.session_state.last_action_time = time.time()
if stop_btn:
    st.session_state.presenting = False
    st.session_state.stop_flag = True

# -------------------------
# Main loop
# -------------------------
if st.session_state.presenting:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam.")
    else:
        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands, \
             mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            
            try:
                while st.session_state.presenting and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = frame.shape
                    now = time.time()
                    st.session_state.cursor_pos = None # Reset cursor

                    # --- Process BOTH hands and face every frame ---
                    results_hands = hands.process(frame_rgb)
                    results_face = face_mesh.process(frame_rgb)
                    
                    # --- NEW: Emotion Detection (on a timer) using DeepFace ---
                    emotion_check_interval = 2.0 # Check every 2 seconds
                    if (now - st.session_state.last_emotion_time) > emotion_check_interval:
                        st.session_state.last_emotion_time = now
                        
                        try:
                            # Detect emotions using DeepFace
                            # detector_backend='opencv' is faster
                            emotion_results = DeepFace.analyze(
                                frame_rgb, 
                                actions=['emotion'], 
                                enforce_detection=False, 
                                detector_backend='opencv'
                            )
                            
                            # deepface returns a list, get the first face
                            if emotion_results and isinstance(emotion_results, list):
                                result = emotion_results[0]
                                top_emotion = result['dominant_emotion']
                                # Convert score from 0-100 to 0.0-1.0
                                top_score = result['emotion'][top_emotion] / 100.0 
                                
                                st.session_state.current_emotion = top_emotion
                                st.session_state.current_emotion_score = top_score
                            else:
                                st.session_state.current_emotion = "N/A"
                                st.session_state.current_emotion_score = 0.0
                        except Exception as e:
                            # This can happen if no face is detected
                            st.session_state.current_emotion = "No Face"
                            st.session_state.current_emotion_score = 0.0
                    
                    # --- End New Emotion Logic ---
                    
                    
                    tool_click_cooldown = 0.5 # Cooldown for tool selection
                    ready_for_tool_click = (now - st.session_state.last_tool_click_time) > tool_click_cooldown

                    # --- GESTURE LOGIC ---
                    gesture_drawing = False
                    gesture_selecting = False
                    
                    # --- UNIVERSAL HAND PROCESSING (Drawing & Selection) ---
                    if results_hands.multi_hand_landmarks:
                        hand_landmarks = results_hands.multi_hand_landmarks[0]
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        fingers = fingers_up(hand_landmarks)
                        count_up = sum(fingers)
                        
                        # --- Define Gestures ---
                        one_finger_point = (count_up == 1) and fingers[1] # DRAW
                        thumb_up_only = (count_up == 1) and fingers[0] # SELECT (NEW)

                        # 1. Check for Draw or Select gestures first
                        if one_finger_point:
                            gesture_drawing = True
                            index_tip = hand_landmarks.landmark[8]
                        elif thumb_up_only: # GESTURE CHANGE
                            gesture_selecting = True
                            index_tip = hand_landmarks.landmark[4] # Use thumb tip
                        
                        # --- Cursor Logic ---
                        if gesture_drawing or gesture_selecting:
                            slide_w, slide_h = st.session_state.slide_dims
                            # Map cursor to full display area (toolbar + slide)
                            cursor_x = int(np.interp(index_tip.x, [0.1, 0.9], [0, slide_w]))
                            cursor_y = int(np.interp(index_tip.y, [0.1, 0.9], [0, slide_h + TOOLBAR_HEIGHT]))
                            st.session_state.cursor_pos = (cursor_x, cursor_y)
                        
                        # --- Free-Draw Logic (Index Finger) ---
                        if gesture_drawing and st.session_state.cursor_pos:
                            cx, cy = st.session_state.cursor_pos
                            # Only draw if on the slide
                            if cy > TOOLBAR_HEIGHT:
                                # This is a new line, save current style
                                if not st.session_state.current_path:
                                    path_style = (st.session_state.draw_color, st.session_state.draw_size, st.session_state.draw_alpha)
                                    st.session_state.current_path_style = path_style
                                
                                # Append point
                                st.session_state.current_path.append((cx, cy - TOOLBAR_HEIGHT))
                        
                        # --- Tool Selection Logic (Thumb Up Only) ---
                        if gesture_selecting and ready_for_tool_click and st.session_state.cursor_pos:
                            cx, cy = st.session_state.cursor_pos
                            # Check for Toolbar Click
                            if cy <= TOOLBAR_HEIGHT:
                                st.session_state.last_tool_click_time = now # Add cooldown
                                for tool_name, props in toolbar_buttons.items():
                                    x1, y1, x2, y2 = props["rect"]
                                    if x1 <= cx < x2:
                                        tool_type = props.get("tool")
                                        
                                        if tool_type == "action":
                                            action = props.get("action")
                                            if action == "undo":
                                                if current_page in st.session_state.annotations and st.session_state.annotations[current_page]:
                                                    st.session_state.annotations[current_page].pop()
                                            elif action == "clear":
                                                st.session_state.annotations[current_page] = []
                                        
                                        elif tool_type == "color":
                                            st.session_state.draw_color = props["color"]
                                            st.session_state.draw_size = props["size"]
                                            st.session_state.draw_alpha = props["alpha"]
                                        
                                        elif tool_type == "size":
                                            if st.session_state.draw_alpha == 255: # Only for pens
                                                action = props.get("action")
                                                if action == "increase":
                                                    st.session_state.draw_size = min(50, st.session_state.draw_size + 2)
                                                elif action == "decrease":
                                                    st.session_state.draw_size = max(1, st.session_state.draw_size - 2)
                                            else:
                                                st.session_state.draw_size = 20 # Reset highlighter size
                                        
                                        break # Found clicked button

                        # --- Save Path Logic ---
                        # If we were drawing, but are not anymore, save the path
                        if (not gesture_drawing) and len(st.session_state.current_path) > 1:
                            if current_page not in st.session_state.annotations:
                                st.session_state.annotations[current_page] = []
                            # BUG FIX: Save the path with the style it had when it *started*
                            st.session_state.annotations[current_page].append(
                                (st.session_state.current_path.copy(), st.session_state.current_path_style)
                            )
                            st.session_state.current_path = [] # Clear for next line

                    else: # No hand detected
                        st.session_state.movement_history.clear()
                        # If we were drawing, save the path
                        if len(st.session_state.current_path) > 1:
                            if current_page not in st.session_state.annotations:
                                st.session_state.annotations[current_page] = []
                            st.session_state.annotations[current_page].append(
                                (st.session_state.current_path.copy(), st.session_state.current_path_style)
                            )
                        st.session_state.current_path = []
                    
                    
                    # --- MODE-SPECIFIC NAVIGATION ---
                    hand_action_cooldown = 1.0
                    ready_for_action = (now - st.session_state.last_action_time) > hand_action_cooldown
                    
                    if st.session_state.control_mode == "Hand Gestures":
                        if results_hands.multi_hand_landmarks and ready_for_action:
                            hand_landmarks = results_hands.multi_hand_landmarks[0]
                            fingers = fingers_up(hand_landmarks)
                            count_up = sum(fingers)
                            
                            # Define navigation gestures
                            pinky_up = fingers[4] and not any(fingers[:4])
                            two_fingers = (count_up == 2) and fingers[1] and fingers[2]
                            three_fingers = (count_up == 3) and fingers[1] and fingers[2] and fingers[3]
                            all_fingers_up = (count_up == 5)

                            # Only act if not drawing or selecting
                            if not gesture_drawing and not gesture_selecting:
                                # *** FIX IS HERE ***
                                wrist = hand_landmarks.landmark[0]
                                st.session_state.movement_history.append(wrist.x * w)
                                dx = st.session_state.movement_history[-1] - st.session_state.movement_history[0] if len(st.session_state.movement_history) >= 6 else 0

                                if dx > 80 and all_fingers_up:
                                    current_page = max(0, current_page - 1)
                                    st.session_state.last_action_time = now
                                    st.session_state.movement_history.clear()
                                    zoom_factor = 1.0
                                elif dx < -80 and all_fingers_up:
                                    current_page = min(num_pages - 1, current_page + 1)
                                    st.session_state.last_action_time = now
                                    st.session_state.movement_history.clear()
                                    zoom_factor = 1.0
                                elif two_fingers:
                                    zoom_factor = min(3.0, zoom_factor + 0.2)
                                    zoom_center = (int(wrist.x * w), int(wrist.y * h))
                                    st.session_state.last_action_time = now
                                elif three_fingers:
                                    zoom_factor = max(1.0, zoom_factor - 0.2)
                                    zoom_center = (int(wrist.x * w), int(wrist.y * h))
                                    st.session_state.last_action_time = now
                                elif pinky_up:
                                    st.session_state.presenting = False
                                    st.session_state.stop_flag = True
                                    st.session_state.last_action_time = now
                        elif not results_hands.multi_hand_landmarks:
                             st.session_state.movement_history.clear()


                    elif st.session_state.control_mode == "Eye Blinks":
                        eye_action_cooldown = 0.4
                        ready_for_eye_action = (now - st.session_state.last_action_time) > eye_action_cooldown
                        
                        if results_face.multi_face_landmarks:
                            face_landmarks = results_face.multi_face_landmarks[0].landmark
                            
                            EAR_BLINK_THRESHOLD = 0.18
                            EAR_OPEN_THRESHOLD = 0.22

                            left_ear = calculate_ear(face_landmarks, LEFT_EYE_INDICES, (h, w))
                            right_ear = calculate_ear(face_landmarks, RIGHT_EYE_INDICES, (h, w))

                            if ready_for_eye_action:
                                if right_ear < EAR_BLINK_THRESHOLD and left_ear > EAR_OPEN_THRESHOLD and st.session_state.right_eye_open:
                                    current_page = max(0, current_page - 1)
                                    st.session_state.last_action_time = now
                                    st.session_state.right_eye_open = False
                                    zoom_factor = 1.0
                                
                                elif left_ear < EAR_BLINK_THRESHOLD and right_ear > EAR_OPEN_THRESHOLD and st.session_state.left_eye_open:
                                    current_page = min(num_pages - 1, current_page + 1)
                                    st.session_state.last_action_time = now
                                    st.session_state.left_eye_open = False
                                    zoom_factor = 1.0
                            
                            if right_ear > EAR_OPEN_THRESHOLD:
                                st.session_state.right_eye_open = True
                            if left_ear > EAR_OPEN_THRESHOLD:
                                st.session_state.left_eye_open = True

                    # --- Display Updates ---
                    frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    webcam_placeholder.image(frame_rgb_display, channels="RGB")
                    
                    # --- NEW: Update Emotion UI ---
                    with emotion_placeholder.container():
                        st.write(f"**Emotion:** {st.session_state.current_emotion.capitalize()}")
                        st.progress(float(st.session_state.current_emotion_score))

                    # --- Render Slide + Toolbar ---
                    slide_img_pil = render_pdf_page(doc, current_page, zoom_factor, crop_center=zoom_center)
                    
                    if st.session_state.slide_dims != slide_img_pil.size:
                         st.session_state.slide_dims = slide_img_pil.size
                    slide_w, slide_h = st.session_state.slide_dims

                    # 1. Draw Toolbar and display it
                    toolbar_img = draw_toolbar_image(slide_w, st.session_state.draw_color, st.session_state.draw_size, st.session_state.draw_alpha, st.session_state.cursor_pos)
                    toolbar_placeholder.image(toolbar_img, use_container_width=True)

                    # 2. Draw Annotations on slide
                    final_display = slide_img_pil.convert('RGBA')
                    overlay = Image.new('RGBA', final_display.size, (255, 255, 255, 0))
                    draw_context = ImageDraw.Draw(overlay)

                    # 2a. Draw saved annotations for this page
                    for path_data in st.session_state.annotations.get(current_page, []):
                        path, (color, size, alpha) = path_data # Unpack saved style
                        if len(path) > 1:
                            draw_context.line(path, fill=color + (alpha,), width=size, joint="curve")
                    
                    # 2b. Draw current path (live)
                    if len(st.session_state.current_path) > 1:
                        # Use the style saved when the line *started*
                        color, size, alpha = st.session_state.current_path_style
                        draw_context.line(st.session_state.current_path, fill=color + (alpha,), width=size, joint="curve")
                    
                    # 2c. Draw cursor
                    if st.session_state.cursor_pos:
                        cx, cy = st.session_state.cursor_pos
                        # Do not draw cursor if it's on the toolbar
                        if cy > TOOLBAR_HEIGHT:
                            # Adjust y-coordinate to be relative to slide
                            cy_rel = cy - TOOLBAR_HEIGHT
                            r = st.session_state.draw_size // 2
                            # Draw cursor outline
                            draw_context.ellipse([cx-r-1, cy_rel-r-1, cx+r+1, cy_rel+r+1], outline=(0,0,0), width=2)
                            draw_context.ellipse([cx-r, cy_rel-r, cx+r, cy_rel+r], outline=st.session_state.draw_color, width=2)

                    # Composite overlay onto slide
                    final_display = Image.alpha_composite(final_display, overlay)
                    slide_placeholder.image(final_display, use_container_width=True)
                    
                    time.sleep(0.01)
                    if st.session_state.stop_flag:
                        st.session_state.presenting = False
                        break
            finally:
                cap.release()
                webcam_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8))

else:
    # --- Idle State ---
    
    # NEW: Show idle emotion state
    with emotion_placeholder.container():
        st.write("**Emotion:** N/A")
        st.progress(0.0)

    try:
        # Show preview with annotations
        preview_img = render_pdf_page(doc, current_page, 1.0)
        slide_w, slide_h = preview_img.size
        
        # Draw toolbar
        toolbar_img = draw_toolbar_image(slide_w, st.session_state.draw_color, st.session_state.draw_size, st.session_state.draw_alpha, None) # No cursor in idle state
        toolbar_placeholder.image(toolbar_img, use_container_width=True)

        # Draw annotations
        final_preview = preview_img.convert('RGBA')
        overlay = Image.new('RGBA', final_preview.size, (255, 255, 255, 0))
        draw_context = ImageDraw.Draw(overlay)
        
        for path_data in st.session_state.annotations.get(current_page, []):
            path, (color, size, alpha) = path_data
            if len(path) > 1:
                draw_context.line(path, fill=color + (alpha,), width=size, joint="curve")
        
        final_preview = Image.alpha_composite(final_preview, overlay)
        slide_placeholder.image(final_preview, use_container_width=True)
    except Exception as e:
        toolbar_placeholder.empty() # Clear toolbar if error
        slide_placeholder.info("Upload a PDF and click Start Presentation to begin.")
        print(f"Error in idle state: {e}") # Print error for debugging
    webcam_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8))