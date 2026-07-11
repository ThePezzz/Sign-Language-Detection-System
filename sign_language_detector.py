import logging
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),  # keep console output too
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model download logic (HandLandmarker Tasks API model bundle)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        logger.debug(f"Model already present at {MODEL_PATH}, skipping download.")
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Downloading HandLandmarker model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("Model download complete.")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d


def normalize_landmarks(hand_landmarks):
    """Return landmarks translated so the wrist (landmark 0) sits at the origin.

    Kept for reuse by future letters / analyze_joint_distances(); the current
    A-F classifier normalizes distances by palm size directly instead (see
    classify_letter_from_rules) but this wrist-relative translation is useful
    for any comparison that cares about direction rather than absolute scale.
    """
    wrist = hand_landmarks[0]
    normalized = []
    for lm in hand_landmarks:
        normalized.append((lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z))
    return normalized


def analyze_joint_distances(hand_landmarks):
    """Compute a dict of pairwise fingertip-to-wrist distances.

    Not used by the current A-F rules, kept available for future letters
    that need distance-based (rather than pure y-comparison) features.
    """
    wrist = hand_landmarks[0]
    tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    distances = {}
    for name, idx in tips.items():
        lm = hand_landmarks[idx]
        distances[name] = distancia_euclidiana(
            (wrist.x, wrist.y), (lm.x, lm.y)
        )
    return distances


def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0

    for landmark in hand_landmarks:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


# Optional connection-line drawing: only use it if the new Tasks API exposes
# a connections table, never fall back to the legacy mp.solutions.drawing_utils.
try:
    from mediapipe.tasks.python.vision import HandLandmarksConnections
    HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
except (ImportError, AttributeError):
    HAND_CONNECTIONS = None
    logger.info(
        "HandLandmarksConnections not available in this MediaPipe version; "
        "skipping connection-line drawing (bounding box + points still drawn)."
    )


def draw_landmarks_and_connections(image, hand_landmarks):
    image_height, image_width, _ = image.shape

    if HAND_CONNECTIONS is not None:
        for connection in HAND_CONNECTIONS:
            start = hand_landmarks[connection.start]
            end = hand_landmarks[connection.end]
            start_px = (int(start.x * image_width), int(start.y * image_height))
            end_px = (int(end.x * image_width), int(end.y * image_height))
            cv2.line(image, start_px, end_px, (255, 255, 255), 2)

    for lm in hand_landmarks:
        cx, cy = int(lm.x * image_width), int(lm.y * image_height)
        cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)


# ---------------------------------------------------------------------------
# LSM static alphabet reference sources (consulted for this classifier)
#
# - CONAPRED & Libre Acceso A.C., "Manos con voz: Diccionario de Lengua de
#   Señas Mexicana" (Fleischmann & González Pérez) —
#   https://www.conapred.org.mx/publicaciones/manos-con-voz-diccionario-de-lengua-de-senas-mexicana/
# - Salgado Martínez et al. (2024), "Reconocimiento de señas de la Lengua de
#   Señas Mexicana mediante técnicas de Machine Learning", XIKUA Boletín
#   Científico de la Escuela Superior de Tlahuelilpan, vol. 12 — confirms the
#   static (non-movement) letter set used in MediaPipe-landmark LSM research
#   is exactly A, B, C, D, E, F, G, H, I, L, M, N, O, P, R, S, T, U, V, W, Y
#   (i.e. the same 21 letters this file targets, excluding the 6 dynamic ones).
# - "Detección del abecedario de Lengua de Señas Mexicanas (LSM) usando
#   MediaPipe, SVM y Random Forest" (ResearchGate) — same static letter set,
#   landmark-based feature approach.
# - ITAIPBC "Diccionario de Lengua de Señas Mexicana" and UTT Tijuana's
#   "Alfabeto en Lengua de Señas Mexicana" — standard manual-alphabet
#   reference images (image-based PDFs; consulted for the conventional
#   handshape each letter uses, not machine-readable text).
#
# IMPORTANT HONESTY NOTE: the sources above confirm *which* letters are static
# and worth building rules for, but none of them were extractable as
# structured per-letter text descriptions (the CONAPRED/ITAIPBC references are
# scanned/image-based PDFs). The handshapes coded below follow the
# conventional one-handed dactylological alphabet that LSM shares with ASL/LSE
# for most letters. Every letter below is commented with a confidence level;
# LOW CONFIDENCE letters are the ones most likely to need hand-tuning or to
# turn out to have an LSM-specific variant not captured here.
# ---------------------------------------------------------------------------


def get_finger_states(hand_landmarks, image_width, image_height):
    """Classify each of the 5 fingers as EXTENDED or CURLED.

    Scale-invariant by construction: every comparison here is a distance
    compared against another distance from the SAME hand in the SAME frame
    (never against a fixed pixel constant), so no palm-size ratio is needed.

    Index/middle/ring/pinky: a finger is EXTENDED if its tip sits farther
    from the wrist than its own PIP joint does, CURLED otherwise.

    Thumb: doesn't fold like the other fingers, and "extended" means swung
    away from the palm (as in G/L/Y) rather than pointing further from the
    wrist. Compare thumb tip vs. thumb MCP distance to the *palm center*
    (middle finger MCP, landmark 9): a thumb swung out to the side moves
    noticeably farther from the palm center than its own MCP is, whereas a
    thumb tucked anywhere against the front of the fist (A/E/S/M/N/T/etc.)
    stays about as close to the palm center as its MCP, regardless of which
    specific spot on the fist front it tucks against. (An earlier version of
    this check compared against the pinky MCP instead, which misread
    front-tucked positions like T's — thumb poking up between index/middle,
    which sits on the index side of the hand — as EXTENDED. Palm-center is
    the more robust reference.)
    """

    def px(lm):
        return (lm.x * image_width, lm.y * image_height)

    wrist = px(hand_landmarks[0])

    def farther_from_wrist(tip_idx, pip_idx):
        tip = px(hand_landmarks[tip_idx])
        pip = px(hand_landmarks[pip_idx])
        return distancia_euclidiana(wrist, tip) > distancia_euclidiana(wrist, pip)

    states = {
        "index": "EXTENDED" if farther_from_wrist(8, 6) else "CURLED",
        "middle": "EXTENDED" if farther_from_wrist(12, 10) else "CURLED",
        "ring": "EXTENDED" if farther_from_wrist(16, 14) else "CURLED",
        "pinky": "EXTENDED" if farther_from_wrist(20, 18) else "CURLED",
    }

    thumb_tip = px(hand_landmarks[4])
    thumb_mcp = px(hand_landmarks[2])
    palm_center = px(hand_landmarks[9])
    states["thumb"] = (
        "EXTENDED"
        if distancia_euclidiana(thumb_tip, palm_center) > distancia_euclidiana(thumb_mcp, palm_center)
        else "CURLED"
    )
    return states


def _angle_between_degrees(v1, v2):
    """Angle in degrees between two 2D vectors. Angle comparisons are
    inherently scale-invariant (they don't depend on hand distance from the
    camera), so unlike distance checks they need no palm-size normalization."""
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


# ---------------------------------------------------------------------------
# Rule-based geometric classifier for the 21 static LSM alphabet letters
# (A-F ported previously; G,H,I,L,M,N,O,P,R,S,T,U,V,W,Y added below).
# Excludes the 6 dynamic/movement letters: J, K, Ñ, Q, X, Z.
# ---------------------------------------------------------------------------
def classify_letter_from_rules(hand_landmarks, image_width, image_height):
    """Classify one detected hand into a static LSM letter using geometric rules.

    `hand_landmarks` is the raw per-hand landmark list from
    `result.hand_landmarks[i]` (Tasks API), indexed directly (no `.landmark`
    attribute as in the legacy API).

    Thresholds are expressed as ratios of "palm size" (wrist-to-middle-MCP
    distance, landmarks 0 and 9) instead of fixed pixel counts, so the rules
    keep working regardless of how far the hand is from the camera. The
    reference prototype's fixed thresholds (45/30/40/360/65/100 px) were tuned
    on footage where palm size was roughly 100px; each ratio below is that
    original constant divided by 100, then reapplied to the live palm size.

    G-Y additions use `get_finger_states()` as a first-pass filter (which
    fingers are up vs. curled), then a secondary geometric check to break
    ties within letters that share the same finger pattern. See the
    per-letter comments below for confidence level and, where relevant, which
    other letters it's commonly confused with.

    Returns the matched letter as a string, or "?" if no rule matches.
    """

    def px(lm):
        return (int(lm.x * image_width), int(lm.y * image_height))

    thumb_tip = px(hand_landmarks[4])
    thumb_pip = px(hand_landmarks[2])
    index_finger_tip = px(hand_landmarks[8])
    index_finger_pip = px(hand_landmarks[6])
    middle_finger_tip = px(hand_landmarks[12])
    middle_finger_pip = px(hand_landmarks[10])
    ring_finger_tip = px(hand_landmarks[16])
    ring_finger_pip = px(hand_landmarks[14])
    pinky_tip = px(hand_landmarks[20])
    pinky_pip = px(hand_landmarks[18])
    wrist = px(hand_landmarks[0])
    # Landmark 5 (index finger MCP) — kept as `ring_finger_pip2` to preserve
    # the variable's role from the reference logic it was ported from.
    ring_finger_pip2 = px(hand_landmarks[5])
    # Clearer alias for the new (G-Y) rules below; same point as ring_finger_pip2.
    index_finger_mcp = ring_finger_pip2
    middle_finger_mcp = px(hand_landmarks[9])

    palm_size = distancia_euclidiana(wrist, middle_finger_mcp)
    if palm_size == 0:
        return "?"

    # Scale-invariant equivalents of the reference logic's fixed-pixel
    # thresholds, derived as (original_px / 100) * live_palm_size.
    BASELINE_PALM_PX = 100.0
    t45 = (45 / BASELINE_PALM_PX) * palm_size
    t30 = (30 / BASELINE_PALM_PX) * palm_size
    t40 = (40 / BASELINE_PALM_PX) * palm_size
    t360 = (360 / BASELINE_PALM_PX) * palm_size
    t65 = (65 / BASELINE_PALM_PX) * palm_size
    t100 = (100 / BASELINE_PALM_PX) * palm_size

    # Additional scale-invariant ratios for the newly added letters below.
    # Unlike t45/t30/etc. above (ported from a tuned fixed-pixel reference),
    # these are fresh estimates based on typical hand proportions rather than
    # a tuned source — treat all as best-effort starting points to retune
    # after real-hand testing (see per-letter confidence notes below).
    t_o_thumb_index = 0.55 * palm_size
    t_o_thumb_middle = 0.65 * palm_size
    t_uv_gap = 0.28 * palm_size

    # Finger extended/curled pattern, used as the first-pass filter for the
    # G-Y letters added below (A-F above don't use this).
    fs = get_finger_states(hand_landmarks, image_width, image_height)

    logger.debug(
        f"palm_size={palm_size:.1f}px thumb_tip_y={thumb_tip[1]} "
        f"index_tip_y={index_finger_tip[1]}"
    )

    # Letter A: closed fist, thumb resting alongside the curled fingers at
    # roughly the same height as their PIP joints.
    if abs(thumb_tip[1] - index_finger_pip[1]) < t45 \
            and abs(thumb_tip[1] - middle_finger_pip[1]) < t30 \
            and abs(thumb_tip[1] - ring_finger_pip[1]) < t30 \
            and abs(thumb_tip[1] - pinky_pip[1]) < t30:
        return "A"

    # Letter B: four fingers extended straight up (tip above pip), thumb
    # folded across the palm near the index MCP.
    elif index_finger_pip[1] - index_finger_tip[1] > 0 \
            and pinky_pip[1] - pinky_tip[1] > 0 \
            and middle_finger_pip[1] - middle_finger_tip[1] > 0 \
            and ring_finger_pip[1] - ring_finger_tip[1] > 0 \
            and middle_finger_tip[1] - ring_finger_tip[1] < 0 \
            and abs(thumb_tip[1] - ring_finger_pip2[1]) < t40:
        return "B"

    # Letter C: fingers curved into a C shape; index tip stays below its own
    # pip (curled) while remaining roughly level with the thumb tip.
    elif abs(index_finger_tip[1] - thumb_tip[1]) < t360 \
            and index_finger_tip[1] - middle_finger_pip[1] < 0 \
            and index_finger_tip[1] - middle_finger_tip[1] < 0 \
            and index_finger_tip[1] - index_finger_pip[1] > 0:
        return "C"

    # Letter D: index finger extended, thumb tip pinched close to the curled
    # middle/ring fingertips, pinky curled.
    elif distancia_euclidiana(thumb_tip, middle_finger_tip) < t65 \
            and distancia_euclidiana(thumb_tip, ring_finger_tip) < t65 \
            and pinky_pip[1] - pinky_tip[1] < 0 \
            and index_finger_pip[1] - index_finger_tip[1] > 0:
        return "D"

    # Letter E: all four fingers curled down (tip above pip), thumb tucked in
    # below every fingertip.
    elif index_finger_pip[1] - index_finger_tip[1] < 0 \
            and pinky_pip[1] - pinky_tip[1] < 0 \
            and middle_finger_pip[1] - middle_finger_tip[1] < 0 \
            and ring_finger_pip[1] - ring_finger_tip[1] < 0 \
            and abs(index_finger_tip[1] - thumb_tip[1]) < t100 \
            and thumb_tip[1] - index_finger_tip[1] > 0 \
            and thumb_tip[1] - middle_finger_tip[1] > 0 \
            and thumb_tip[1] - ring_finger_tip[1] > 0 \
            and thumb_tip[1] - pinky_tip[1] > 0:
        return "E"

    # Letter F: index and thumb tips pinched together, middle/ring/pinky
    # extended.
    elif pinky_pip[1] - pinky_tip[1] > 0 \
            and middle_finger_pip[1] - middle_finger_tip[1] > 0 \
            and ring_finger_pip[1] - ring_finger_tip[1] > 0 \
            and index_finger_pip[1] - index_finger_tip[1] < 0 \
            and abs(thumb_pip[1] - thumb_tip[1]) > 0 \
            and distancia_euclidiana(index_finger_tip, thumb_tip) < t65:
        return "F"

    # Letter O: fingers curved into a circle, all fingertips touching the
    # thumb tip. Same base handshape as ASL/the common convention.
    # MEDIUM CONFIDENCE — ratio thresholds are estimated, not tuned like A-F's.
    # Checked before the M/N/S/T fist group below since O also shows up as
    # "all four fingers curled" and would otherwise be swallowed by it.
    elif fs["index"] == "CURLED" and fs["middle"] == "CURLED" \
            and fs["ring"] == "CURLED" and fs["pinky"] == "CURLED" \
            and distancia_euclidiana(thumb_tip, index_finger_tip) < t_o_thumb_index \
            and distancia_euclidiana(thumb_tip, middle_finger_tip) < t_o_thumb_middle:
        return "O"

    # Letter I: only the pinky extended, everything else (including thumb)
    # curled. Same base handshape as ASL's I. Distinctive pattern — HIGH
    # CONFIDENCE.
    elif fs["thumb"] == "CURLED" and fs["index"] == "CURLED" \
            and fs["middle"] == "CURLED" and fs["ring"] == "CURLED" \
            and fs["pinky"] == "EXTENDED":
        return "I"

    # Letter Y: thumb and pinky extended ("hang loose"), other three curled.
    # Same base handshape as ASL's Y. Distinctive pattern — HIGH CONFIDENCE.
    elif fs["thumb"] == "EXTENDED" and fs["pinky"] == "EXTENDED" \
            and fs["index"] == "CURLED" and fs["middle"] == "CURLED" \
            and fs["ring"] == "CURLED":
        return "Y"

    # Letter W: index, middle and ring extended and spread, pinky curled,
    # thumb curled/resting. Same base handshape as ASL's W. Only letter with
    # exactly these 3 fingers up — HIGH CONFIDENCE.
    elif fs["thumb"] == "CURLED" and fs["index"] == "EXTENDED" \
            and fs["middle"] == "EXTENDED" and fs["ring"] == "EXTENDED" \
            and fs["pinky"] == "CURLED":
        return "W"

    # --- Ambiguity group: closed-fist letters M / N / S / T ---
    # All four share the same coarse finger-state pattern (everything
    # curled) and are distinguished only by where the thumb tip tucks in.
    # LSM vs ASL note: M/N/T/S are the classic hard-to-verify group across
    # most one-handed manual alphabets (small thumb-tuck differences) — could
    # not confirm an LSM-specific deviation from ASL from the sources checked
    # this session (see citations above this function), so all four are
    # FLAGGED LOW CONFIDENCE — check each one carefully by hand.
    elif fs["thumb"] == "CURLED" and fs["index"] == "CURLED" \
            and fs["middle"] == "CURLED" and fs["ring"] == "CURLED" \
            and fs["pinky"] == "CURLED":

        dist_to_index_pip = distancia_euclidiana(thumb_tip, index_finger_pip)
        dist_to_middle_pip = distancia_euclidiana(thumb_tip, middle_finger_pip)
        dist_to_ring_pip = distancia_euclidiana(thumb_tip, ring_finger_pip)
        thumb_above_index_pip = distancia_euclidiana(wrist, thumb_tip) > \
            distancia_euclidiana(wrist, index_finger_pip)
        nearest_pip = min(
            ("index", dist_to_index_pip),
            ("middle", dist_to_middle_pip),
            ("ring", dist_to_ring_pip),
            key=lambda pair: pair[1],
        )[0]

        # T: thumb tip pokes up between the index/middle PIP joints, nearer
        # the fingertips than the PIP row itself.
        if nearest_pip == "index" and thumb_above_index_pip:
            return "T"
        # S: thumb lies flat across the front of the fist, below the PIP row.
        elif nearest_pip == "index" and not thumb_above_index_pip:
            return "S"
        # N: thumb tucked under the first two (index+middle) fingers.
        elif nearest_pip == "middle":
            return "N"
        # M: thumb tucked under the first three (index+middle+ring) fingers.
        else:
            return "M"

    # --- Ambiguity group: H / U / V / R / P (index + middle extended, ring
    # and pinky curled) ---
    elif fs["index"] == "EXTENDED" and fs["middle"] == "EXTENDED" \
            and fs["ring"] == "CURLED" and fs["pinky"] == "CURLED":

        index_dx = index_finger_tip[0] - index_finger_mcp[0]
        index_dy = index_finger_tip[1] - index_finger_mcp[1]
        middle_dx = middle_finger_tip[0] - middle_finger_mcp[0]
        middle_dy = middle_finger_tip[1] - middle_finger_mcp[1]

        # R: same handshape as ASL R (index and middle fingers crossed).
        # Detected as a left/right order swap between the two fingers' MCPs
        # vs. their tips — mirror-safe, no distance threshold needed.
        # MEDIUM CONFIDENCE.
        index_order_at_mcp = index_finger_mcp[0] - middle_finger_mcp[0]
        index_order_at_tip = index_finger_tip[0] - middle_finger_tip[0]
        if index_order_at_mcp * index_order_at_tip < 0:
            return "R"

        # P: assumed same base handshape as ASL P (like K, but the hand tips
        # to point downward/forward, toward/away from the camera). This is a
        # genuinely 3D distinction that x/y-only comparisons can't represent
        # (a finger rotated to point at the camera still measures as
        # "extended" by get_finger_states' wrist-distance test, and pointing
        # it back down toward the wrist would just make it read as CURLED
        # instead — an earlier version of this rule tried the latter and was
        # unreachable, since the outer elif already requires EXTENDED).
        # Landmarks do carry a z (depth) coordinate, so use that instead: if a
        # meaningful share of the finger's length is along z rather than in
        # the xy plane, the hand is tilted toward/away from the camera.
        # Ratio-based (no fixed threshold), but still LOW CONFIDENCE — MediaPipe's
        # z estimate is less reliable than x/y, and this hasn't been tuned
        # against a real hand at all.
        def depth_ratio(tip_idx, mcp_idx):
            tip, mcp = hand_landmarks[tip_idx], hand_landmarks[mcp_idx]
            dx, dy, dz = tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z
            xy_len = math.hypot(dx, dy)
            xyz_len = math.hypot(dx, dy, dz)
            return (xy_len / xyz_len) if xyz_len > 0 else 1.0

        if depth_ratio(8, 5) < 0.85 and depth_ratio(12, 9) < 0.85:
            return "P"

        # H: same handshape as ASL H (index+middle extended together,
        # pointing sideways instead of up). Orientation inferred from whether
        # the fingers extend more horizontally than vertically. LOW
        # CONFIDENCE — sensitive to hand tilt/rotation toward the camera.
        elif abs(index_dx) > abs(index_dy) and abs(middle_dx) > abs(middle_dy):
            return "H"

        # U vs V: both point straight up; the only difference is whether the
        # tips are held together (U) or spread apart (V). Uses a palm-size
        # ratio, consistent with the A-F thresholds. MEDIUM CONFIDENCE.
        elif distancia_euclidiana(index_finger_tip, middle_finger_tip) < t_uv_gap:
            return "U"
        else:
            return "V"

    # --- Ambiguity group: G / L (thumb + index extended, others curled) ---
    elif fs["thumb"] == "EXTENDED" and fs["index"] == "EXTENDED" \
            and fs["middle"] == "CURLED" and fs["ring"] == "CURLED" \
            and fs["pinky"] == "CURLED":

        # thumb_pip here is landmark 2 (actually the thumb MCP — kept from
        # the ported reference logic's naming); it's the thumb's base joint,
        # which is what the angle check needs.
        thumb_vec = (thumb_tip[0] - thumb_pip[0], thumb_tip[1] - thumb_pip[1])
        index_vec = (index_finger_tip[0] - index_finger_mcp[0], index_finger_tip[1] - index_finger_mcp[1])
        angle_deg = _angle_between_degrees(thumb_vec, index_vec)

        # L: same handshape as ASL L (thumb and index roughly perpendicular,
        # forming a right angle). MEDIUM CONFIDENCE.
        if angle_deg > 55:
            return "L"
        # G: same handshape as ASL G (thumb and index roughly parallel, both
        # pointing sideways together). MEDIUM CONFIDENCE.
        else:
            return "G"

    return "?"


def match_pattern_to_letter(hand_landmarks, image_width, image_height):
    return classify_letter_from_rules(hand_landmarks, image_width, image_height)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    logger.info("Program started")
    ensure_model_downloaded()

    # FIX 3: max_num_hands -> num_hands, explicitly capped at 1.
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Could not open webcam (device index 0).")
        cap.set(3, 1920)
        cap.set(4, 1080)
    except Exception as exc:
        logger.error(f"Webcam initialization failed: {exc}")
        return

    with HandLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                logger.warning("Failed to read frame from webcam; skipping.")
                continue

            # FIX 1: refresh the frame each iteration instead of reusing a
            # stale buffer/colorspace conversion from the previous pass.
            image.flags.writeable = False
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            image.flags.writeable = True
            image_height, image_width, _ = image.shape

            # FIX 2: reset the detected letter every frame so a stale value
            # from a previous hand detection never lingers once the hand
            # disappears or changes shape.
            detected_letter = "?"

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks_and_connections(image, hand_landmarks)
                    draw_bounding_box(image, hand_landmarks)

                    detected_letter = match_pattern_to_letter(
                        hand_landmarks, image_width, image_height
                    )

                    if detected_letter != "?":
                        logger.info(f'Detected sign: "{detected_letter}"')

                    cv2.putText(
                        image, detected_letter, (700, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6,
                    )
            else:
                # FIX 5 (ELSE branch): no hand in frame — nothing to draw or
                # classify, detected_letter stays "?" from the reset above.
                pass

            cv2.imshow("Sign Language Detector", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Program terminated")


if __name__ == "__main__":
    main()