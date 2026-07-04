# Sign Language Detection System

Real-time computer vision application that translates static hand gestures from the **LSM (Lengua de Señas Mexicana)** alphabet into text, using a standard webcam.

**Author:** José Abraham Díaz Uh
**Program:** Data and AI Engineering — Universidad Politécnica de Yucatán (UPY)
**Course:** Programming — 2nd Quarter, May-August 2026

---

## 🎯 Objective

To design a real-time software system that captures live video frames from a webcam, extracts geometric hand landmark coordinates, and translates those spatial patterns into corresponding text characters to assist in sign language translation.

## 🧩 Problem Statement

- **The Communication Gap:** Lack of accessible, real-time translation tools between the Deaf community and non-signers.
- **Hardware Limitations:** Current solutions are often expensive or require specialized hardware, like smart gloves.
- **The Need:** A software solution integrated into standard webcams.

## ⚙️ How It Works (IPO)

| Stage | Description |
|---|---|
| **Input** | `video_frame` (webcam image), `min_detection_confidence` (accuracy threshold), `max_num_hands` (max hands to track) |
| **Process** | Capture frame → detect 21 hand landmarks (MediaPipe) → normalize coordinates relative to the wrist → calculate geometric distances/angles → match pattern against LSM reference dataset |
| **Output** | Real-time text overlay showing the detected letter (`"Detected Sign: A"`) |

## 🛠️ Technical Stack

- **Language:** Python
- **Computer Vision:** OpenCV (frame capture, text overlay)
- **Hand Tracking:** MediaPipe Hands (21 landmark extraction)
- **Logic:** Lightweight geometric classification (angles/distances between joints) — no heavy ML model required, ensuring smooth FPS on standard hardware

## 📂 Repository Structure

```
Sign-Language-Detection-System/
├── README.md
├── PPP.txt                      # Pseudocode (INPUT → PROCESS → OUTPUT)
├── AI_use_declaration.txt       # AI usage transparency
└── sign_language_detector.py    # (coming soon)
```

## 🌱 Engineering Week Alignment

This project connects to the **Agents** theme — it is a perception-and-action system: it perceives hand gestures through a webcam and MediaPipe, then acts by translating and displaying the corresponding letter in real time, following a continuous sense-and-respond loop.

It also connects to **Sustainability** — by working with any standard webcam instead of specialized hardware, it offers a low-cost, accessible communication tool that could support Deaf tour guides and accessibility kiosks in ecotourism and cultural heritage sites in Yucatán.

## 🚧 Project Status

Currently in the design and pseudocode stage. Python implementation in progress.
