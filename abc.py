import cv2
import numpy as np
import os
import time

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SKETCH  = "radha.jpg"        # blue-pen sketch (image 1)
IMAGE_COLOR   = "radha_color.png"  # colored version (image 2)

SCREEN_W = 1920
SCREEN_H = 1080  
FPS = 60

# Timing (seconds)
PARTICLE_TIME  = 5.0   # particles fly into sketch shape  
FORM_TIME      = 2.0   # particles → clean sketch
SKETCH_HOLD    = 1  # hold the clean sketch
COLOR_TIME     = 3.0   # sketch → colored transition  
COLOR_HOLD     = 6.0   # hold final colored image

PARTICLE_COUNT = 5000


# ============================================================
# LOAD IMAGES
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

sketch_path = os.path.join(script_dir, IMAGE_SKETCH)
color_path  = os.path.join(script_dir, IMAGE_COLOR)

sketch_orig = cv2.imread(sketch_path)
color_orig  = cv2.imread(color_path)

if sketch_orig is None:
    print(f"ERROR: '{IMAGE_SKETCH}' nahi mili → {sketch_path}")
    input("Enter dabao...")
    raise SystemExit

if color_orig is None:
    print(f"ERROR: '{IMAGE_COLOR}' nahi mili → {color_path}")
    print("Colored image bhi same folder mein rakhni hai.")
    input("Enter dabao...")
    raise SystemExit


# ============================================================
# FIT BOTH IMAGES TO SCREEN (same canvas size)
# ============================================================

def fit_image(img, W, H):
    h, w = img.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x = (W - nw) // 2
    y = (H - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


sketch_base = fit_image(sketch_orig, SCREEN_W, SCREEN_H)
color_base  = fit_image(color_orig,  SCREEN_W, SCREEN_H)


# ============================================================
# BUILD CLEAN LINE-ART FROM SKETCH
# ============================================================

def build_line_art(img):
    """
    Extract blue-ink strokes from notebook sketch.
    Returns a BGR image: dark lines on black background.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Blue ink range
    lower_blue = np.array([85, 20, 20])
    upper_blue = np.array([145, 255, 240])
    blue_mask  = cv2.inRange(hsv, lower_blue, upper_blue)

    # Dark strokes (very dark pixels = ink over white paper)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Canny edges for thin detail
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, 60, 150)

    # Combine all three
    mask = cv2.bitwise_or(blue_mask, dark_mask)
    mask = cv2.bitwise_or(mask, edges)

    # ---- Remove long horizontal notebook ruling lines ----
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 1))
    h_lines  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
    mask     = cv2.subtract(mask, h_lines)

    # ---- Remove long vertical margin lines ----
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 90))
    v_lines  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)
    mask     = cv2.subtract(mask, v_lines)

    # ---- Slight morphological cleanup ----
    close_k = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    # Build BGR line-art: white/blue lines on black background
    line_art = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # Slightly warm blue-white color for ink lines
    line_art[mask > 0] = (220, 200, 160)   # B G R → warm bluish-white

    return line_art, mask


print("Building line-art from sketch...")
line_art, line_mask = build_line_art(sketch_base)


# ============================================================
# COLLECT PARTICLE TARGET POINTS (from line-art mask)
# ============================================================

ys, xs = np.where(line_mask > 0)
points  = np.column_stack((xs, ys))
print(f"Raw drawing points: {len(points)}")

rng = np.random.default_rng(42)

if len(points) > PARTICLE_COUNT:
    idx    = rng.choice(len(points), PARTICLE_COUNT, replace=False)
    points = points[idx]

targets = points.astype(np.float32)
print(f"Particles used: {len(targets)}")


# ============================================================
# RANDOM START POSITIONS (off-screen edges)
# ============================================================

starts = np.zeros_like(targets)
starts[:, 0] = rng.integers(0, SCREEN_W, len(targets))
starts[:, 1] = rng.integers(0, SCREEN_H, len(targets))

sizes = rng.choice([1, 1, 2, 2, 2, 3], size=len(targets))

# Particle colours: warm gold / amber sparks
base_colors = np.array([
    [255, 200,  80],
    [255, 220, 120],
    [255, 180,  50],
], dtype=np.uint8)
particle_colors = base_colors[rng.integers(0, 3, len(targets))]


# ============================================================
# EASING FUNCTIONS
# ============================================================

def ease_out(t):
    return 1 - (1 - t) ** 3

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def smoothstep(t):
    return t * t * (3 - 2 * t)


# ============================================================
# DRAW PARTICLES ONTO A FRAME
# ============================================================

def render_particles(frame, positions):
    for i, (x, y) in enumerate(positions):
        xi, yi = int(x), int(y)
        if 0 <= xi < SCREEN_W and 0 <= yi < SCREEN_H:
            col = (
                int(particle_colors[i, 0]),
                int(particle_colors[i, 1]),
                int(particle_colors[i, 2]),
            )
            cv2.circle(frame, (xi, yi), int(sizes[i]), col, -1, cv2.LINE_AA)


# ============================================================
# WINDOW SETUP
# ============================================================

WINDOW = "Vasudeva - Divine Particle Reveal"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ============================================================
# PHASE 1 — BLACK SCREEN (brief intro)
# ============================================================

print("Phase 1: Black screen...")
intro_frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
cv2.imshow(WINDOW, intro_frame)
cv2.waitKey(800)


# ============================================================
# PHASE 2 — PARTICLES FLY INTO SKETCH SHAPE
# ============================================================

print("Phase 2: Particles forming sketch...")
start = time.time()

while True:
    elapsed = time.time() - start
    t       = min(elapsed / PARTICLE_TIME, 1.0)
    eased   = ease_out(t)

    positions = starts + (targets - starts) * eased

    frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    render_particles(frame, positions)

    # Subtle particle glow (very light)
    glow  = cv2.GaussianBlur(frame, (0, 0), 2)
    frame = cv2.addWeighted(frame, 1.1, glow, 0.4, 0)

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(int(1000 / FPS)) & 0xFF
    if key in (27, ord('q')):
        cv2.destroyAllWindows(); raise SystemExit
    if t >= 1.0:
        break


# ============================================================
# PHASE 3 — PARTICLES → CLEAN LINE-ART
# ============================================================

print("Phase 3: Resolving to clean line-art...")
start = time.time()

# Snapshot of fully-formed particle frame
particle_final = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
render_particles(particle_final, targets)

while True:
    elapsed = time.time() - start
    t       = min(elapsed / FORM_TIME, 1.0)
    t_ease  = ease_in_out(t)

    frame = cv2.addWeighted(particle_final, 1.0 - t_ease, line_art, t_ease, 0)

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(int(1000 / FPS)) & 0xFF
    if key in (27, ord('q')):
        cv2.destroyAllWindows(); raise SystemExit
    if t >= 1.0:
        break


# ============================================================
# PHASE 4 — HOLD CLEAN SKETCH
# ============================================================

print("Phase 4: Holding clean sketch...")
start = time.time()

while True:
    cv2.imshow(WINDOW, line_art)
    key = cv2.waitKey(30) & 0xFF
    if key in (27, ord('q')):
        cv2.destroyAllWindows(); raise SystemExit
    if time.time() - start >= SKETCH_HOLD:
        break


# ============================================================
# PHASE 5 — LINE-ART → COLORED IMAGE TRANSITION
# ============================================================

print("Phase 5: Colorizing...")

# The colored image is already loaded as color_base.
# We blend: line_art (0..1) → color_base (1..0) over COLOR_TIME seconds.

start = time.time()

while True:
    elapsed = time.time() - start
    t       = min(elapsed / COLOR_TIME, 1.0)
    t_ease  = smoothstep(t)

    # Blend sketch → color
    frame = cv2.addWeighted(
        line_art,   1.0 - t_ease,
        color_base, t_ease,
        0
    )

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(int(1000 / FPS)) & 0xFF
    if key in (27, ord('q')):
        cv2.destroyAllWindows(); raise SystemExit
    if t >= 1.0:
        break


# ============================================================
# PHASE 6 — HOLD FINAL COLORED IMAGE
# ============================================================

print("Phase 6: Final colored image — hold.")
start = time.time()

while True:
    cv2.imshow(WINDOW, color_base)
    key = cv2.waitKey(30) & 0xFF
    if key in (27, ord('q')):
        break
    if time.time() - start >= COLOR_HOLD:
        break


# ============================================================
# END
# ============================================================

cv2.destroyAllWindows()
print("\nAnimation complete! ✓")
