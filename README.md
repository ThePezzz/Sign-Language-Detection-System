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
| **Input** | `video_frame` (webcam image), `min_hand_detection_confidence` / `min_tracking_confidence` (accuracy thresholds), `num_hands` (capped at 1) |
| **Process** | Capture frame → detect 21 hand landmarks (MediaPipe `HandLandmarker`) → classify finger states (extended/curled) → apply scale-invariant geometric rules per letter → resolve ambiguity groups (M/N/S/T, H/U/V/R/P, G/L) |
| **Output** | Real-time text overlay showing the detected letter, drawn hand skeleton + bounding box on the video feed |

## 🔤 Alphabet Coverage

Covers **21 of the 27 letters** of the static LSM alphabet:

```
A  B  C  D  E  F  G  H  I  L  M  N  O  P  R  S  T  U  V  W  Y
```

**Not implemented (excluded on purpose):** `J, K, Ñ, Q, X, Z` — these require hand *movement* (dynamic gestures) rather than a static pose, which this version's frame-by-frame classifier can't capture. Adding them would require tracking motion across multiple frames, planned as future work.

### Confidence levels

Not all 21 letters are equally reliable — geometric rules were tuned against a reference prototype for A–F, while G–Y are best-effort estimates pending real-hand testing:

| Confidence | Letters | Why |
|---|---|---|
| **High** | I, W, Y | Distinctive finger-state pattern with no other letter sharing it |
| **Medium** | O, R, L, G, U, V | Ratio-based secondary checks, reasonably distinct but not battle-tested |
| **Low** | H, P, M, N, S, T | Classic ambiguous groups in one-handed manual alphabets (small thumb-tuck or hand-tilt differences) — flagged in code comments for manual tuning after real-hand testing |

## 🛠️ Technical Stack

- **Language:** Python
- **Computer Vision:** OpenCV (frame capture, drawing, text overlay)
- **Hand Tracking:** MediaPipe **Tasks API** — `HandLandmarker` (21 landmark extraction). Uses `mediapipe.tasks.python.vision` exclusively; the legacy `mp.solutions.hands` API was removed in MediaPipe ≥0.10.30 and is not used anywhere in this project.
- **Classification approach:** Rule-based geometric classifier — no external dataset or ML training required. Every threshold is expressed as a **ratio of palm size** (wrist-to-middle-MCP distance) rather than a fixed pixel value, so classification stays consistent regardless of how far the user is from the camera.

## ⚡ Setup

```bash
py -m pip install opencv-python mediapipe
py sign_language_detector.py
```

> ⚠️ Requires **MediaPipe ≥ 0.10.31**. Version 0.10.30 has a known Windows bug (`AttributeError: function 'free' not found`) — see [mediapipe#6187](https://github.com/google-ai-edge/mediapipe/issues/6187). If you hit that error, run `py -m pip install --upgrade mediapipe`.

On first run, the script automatically downloads the `hand_landmarker.task` model (~7.5 MB) into a local `models/` folder — no manual download needed.

Press **Esc** to close the video window and exit.

## 📂 Repository Structure

```
Sign-Language-Detection-System/
├── README.md
├── PPP.txt                      # Pseudocode (INPUT → PROCESS → OUTPUT)
├── AI_use_declaration.txt       # AI usage transparency
├── sign_language_detector.py    # Main application
├── models/                      # Auto-downloaded on first run (not tracked in git)
│   └── hand_landmarker.task
└── logs/
    └── app.log                  # Runtime audit log (auto-generated)
```

## 📝 Logging

The application uses Python's built-in `logging` module to write an audit trail to `logs/app.log` on every run — created automatically via `os.makedirs()` + `FileHandler`, no manual setup needed.

| Level | Example |
|---|---|
| `INFO` | Program started/terminated, model download progress, sign detected |
| `WARNING` | Frame could not be read from the webcam — skipped |
| `ERROR` | Webcam failed to initialize |
| `DEBUG` | Per-frame palm size and key landmark coordinates (verbose, off by default) |

**What it tracks and why:** program lifecycle events, model download progress, webcam initialization failures, dropped frames, and every successfully classified sign — so that if the detector misbehaves or crashes, it's possible to trace exactly which stage (model loading, camera access, or the classification logic itself) caused the problem.

## 🌱 Engineering Week Alignment

**Agents:** This project is a perceive-and-act system — it perceives hand gestures through a webcam and MediaPipe's `HandLandmarker`, then acts by classifying the gesture and displaying the corresponding letter in real time, in a continuous sense-and-respond loop.

**Sustainability:** By working with any standard webcam instead of specialized hardware, it offers a low-cost, accessible communication tool that could support Deaf tour guides and accessibility kiosks in ecotourism and cultural heritage sites in Yucatán.

## 📚 LSM Reference Sources

- CONAPRED & Libre Acceso A.C., *"Manos con voz: Diccionario de Lengua de Señas Mexicana"* (Fleischmann & González Pérez)
- Salgado Martínez et al. (2024), *"Reconocimiento de señas de la Lengua de Señas Mexicana mediante técnicas de Machine Learning"*, XIKUA Boletín Científico de la Escuela Superior de Tlahuelilpan, vol. 12
- *"Detección del abecedario de Lengua de Señas Mexicanas (LSM) usando MediaPipe, SVM y Random Forest"* (ResearchGate)
- ITAIPBC, *"Diccionario de Lengua de Señas Mexicana"* / UTT Tijuana, *"Alfabeto en Lengua de Señas Mexicana"* (handshape reference images)

## 🚧 Project Status & Known Limitations

- Static alphabet (21/27 letters) implemented with geometric rule-based classification
- Dynamic letters (J, K, Ñ, Q, X, Z) **not implemented** — require multi-frame motion tracking, planned as future work
- G–Y letters are **untested against a real hand** — thresholds are best-effort estimates and will likely need manual tuning, especially the Low Confidence group (H, P, M, N, S, T)
- Single-hand detection only (`num_hands=1`)
- No word/phrase-level recognition — letter-by-letter only
