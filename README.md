# A-Gesture-Emotion-Controlled-Presentation-App

# 🎯 GestureDeck: Gesture & Emotion Controlled Presentation App

GestureDeck is a **Streamlit-based interactive presentation controller** that lets you navigate, draw, and control your slides using **hand gestures, eye blinks**, and even displays **real-time emotions** detected from your face.

This project integrates **MediaPipe**, **OpenCV**, **DeepFace**, and **PyMuPDF** to create a hands-free presentation experience perfect for virtual lectures, accessibility demos, or smart classrooms.

---

## 🚀 Features

✅ **Hand Gesture Controls**
- ✋ Swipe Right / Left → Next / Previous Slide  
- ✌️ Two Fingers → Zoom In  
- 🤟 Three Fingers → Zoom Out  
- 👆 One Finger (Index) → Draw on Slide  
- 👍 Thumb Up → Select Color or Tool  
- 🤚 All Fingers → Page navigation (wave detection)  
- 🤏 Pinky Up → Exit Presentation  

✅ **Eye Blink Controls**
- 😉 Left Eye Blink → Next Slide  
- 😌 Right Eye Blink → Previous Slide  

✅ **Emotion Detection**
- Detects your dominant facial emotion in real-time using DeepFace.  
- Displays live emotion confidence using Streamlit progress bars.

✅ **Interactive Drawing Toolbar**
- Tools: Highlighter, Colored Pens, Size Control, Undo, and Clear.  
- Dynamic overlay rendering on slides.  
- Semi-transparent drawing with per-stroke memory.

✅ **Smart PDF Integration**
- Upload any PDF as your presentation slides.  
- Zoom & crop features for better visibility.  

✅ **Dark Violet Modern UI**
- Elegant custom CSS theme with orange-highlighted controls.

---

## 🧰 Tech Stack

| Library | Purpose |
|----------|----------|
| **Streamlit** | Web UI Framework |
| **MediaPipe** | Hand & Face Landmark Detection |
| **OpenCV** | Real-Time Camera Feed Processing |
| **DeepFace** | Emotion Recognition |
| **PyMuPDF (fitz)** | PDF Rendering & Slide Display |
| **Pillow (PIL)** | Image Drawing & Toolbar UI |
| **NumPy** | Coordinate and Math Operations |

---
## ⚙️ Installation dependencies
pip install streamlit opencv-python mediapipe deepface PyMuPDF Pillow numpy
## Run it by
streamlit run filename.py
### 1️⃣ Clone this repository
```bash
git clone https://github.com/RathlavathArun/A-Gesture-Emotion-Controlled-Presentation-App.git
cd GestureDeck
