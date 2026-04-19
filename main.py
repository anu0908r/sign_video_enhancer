import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
FACE_MESH = None

if mp is not None:
    try:
        FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )
    except AttributeError:
        FACE_MESH = None
        mp = None


def _clamp_box(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _expand_box(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
    pad_x: float,
    pad_y: float,
) -> tuple[int, int, int, int] | None:
    box_w = x1 - x0
    box_h = y1 - y0
    pad_w = int(box_w * pad_x)
    pad_h = int(box_h * pad_y)
    return _clamp_box(x0 - pad_w, y0 - pad_h, x1 + pad_w, y1 + pad_h, width, height)


def _mediapipe_face_boxes(frame: np.ndarray) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:
    if FACE_MESH is None:
        return None, []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = FACE_MESH.process(rgb)
    if not result.multi_face_landmarks:
        return None, []

    h, w = frame.shape[:2]
    landmarks = np.asarray(
        [[landmark.x * w, landmark.y * h] for landmark in result.multi_face_landmarks[0].landmark],
        dtype=np.float32,
    )

    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    face_box = _expand_box(
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
        w,
        h,
        0.08,
        0.12,
    )

    eye_landmark_groups = [
        (33, 133, 159, 145),
        (362, 263, 386, 374),
    ]
    eye_boxes: list[tuple[int, int, int, int]] = []
    for group in eye_landmark_groups:
        points = landmarks[list(group)]
        eye_box = _expand_box(
            int(points[:, 0].min()),
            int(points[:, 1].min()),
            int(points[:, 0].max()),
            int(points[:, 1].max()),
            w,
            h,
            0.65,
            0.85,
        )
        if eye_box is not None:
            eye_boxes.append(eye_box)

    return face_box, eye_boxes


def find_default_video(input_dir: Path) -> Path:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    videos = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )

    if not videos:
        raise FileNotFoundError(
            f"No video file found in {input_dir}. Supported: {sorted(VIDEO_EXTENSIONS)}"
        )
    return videos[0]


def read_frames_and_fps(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames: list[np.ndarray] = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def read_frames_from_dir(frames_dir: Path) -> list[np.ndarray]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames folder not found: {frames_dir}")

    frame_files = sorted(
        [
            p
            for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    if not frame_files:
        raise FileNotFoundError(
            f"No image frames found in {frames_dir}. Supported: {sorted(IMAGE_EXTENSIONS)}"
        )

    frames: list[np.ndarray] = []
    for path in frame_files:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frames.append(frame)

    return frames


def smooth_trajectory(data: np.ndarray, radius: int) -> np.ndarray:
    if len(data) == 0:
        return data

    radius = max(1, radius)
    kernel_size = 2 * radius + 1
    x = np.arange(kernel_size, dtype=np.float32) - radius
    sigma = max(radius / 2.0, 1.0)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= np.sum(kernel)

    smoothed = np.copy(data).astype(np.float32)
    for col in range(data.shape[1]):
        padded = np.pad(data[:, col], (radius, radius), mode="edge").astype(np.float32)
        smoothed[:, col] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def rolling_median_1d(series: np.ndarray, radius: int) -> np.ndarray:
    if len(series) == 0:
        return series

    radius = max(1, int(radius))
    out = np.zeros_like(series, dtype=np.float32)
    for i in range(len(series)):
        a = max(0, i - radius)
        b = min(len(series), i + radius + 1)
        out[i] = np.median(series[a:b]).astype(np.float32)
    return out


def suppress_glitch_transforms(
    transforms_np: np.ndarray,
    frame_w: int,
    frame_h: int,
    max_shift_pct: float,
    max_rotate_deg: float,
) -> np.ndarray:
    if len(transforms_np) == 0:
        return transforms_np

    max_dx = max(2.0, frame_w * max(0.001, max_shift_pct))
    max_dy = max(2.0, frame_h * max(0.001, max_shift_pct))
    max_da = np.deg2rad(max(0.1, max_rotate_deg))

    filtered = np.copy(transforms_np)
    for i in range(len(filtered)):
        dx, dy, da = filtered[i]
        if abs(dx) > max_dx or abs(dy) > max_dy or abs(da) > max_da:
            filtered[i] = filtered[i - 1] if i > 0 else np.array([0.0, 0.0, 0.0], dtype=np.float32)

    med_radius = 2
    filtered[:, 0] = rolling_median_1d(filtered[:, 0], med_radius)
    filtered[:, 1] = rolling_median_1d(filtered[:, 1], med_radius)
    filtered[:, 2] = rolling_median_1d(filtered[:, 2], med_radius)
    return filtered


def stabilize_frames_once(
    frames: list[np.ndarray],
    radius: int,
    strength: float,
    anti_glitch: bool,
    max_shift_pct: float,
    max_rotate_deg: float,
) -> list[np.ndarray]:
    if len(frames) <= 1:
        return frames

    transforms = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=1200,
            qualityLevel=0.004,
            minDistance=12,
            blockSize=3,
        )

        if prev_pts is None:
            transforms.append([0.0, 0.0, 0.0])
            prev_gray = curr_gray
            continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if curr_pts is None or status is None:
            transforms.append([0.0, 0.0, 0.0])
            prev_gray = curr_gray
            continue

        status = status.reshape(-1)
        good_prev = prev_pts.reshape(-1, 2)[status == 1]
        good_curr = curr_pts.reshape(-1, 2)[status == 1]

        if len(good_prev) < 20:
            transforms.append([0.0, 0.0, 0.0])
            prev_gray = curr_gray
            continue

        m, _ = cv2.estimateAffinePartial2D(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=5000,
            confidence=0.995,
            refineIters=20,
        )
        if m is None:
            dx, dy, da = 0.0, 0.0, 0.0
        else:
            dx = float(m[0, 2])
            dy = float(m[1, 2])
            da = float(np.arctan2(m[1, 0], m[0, 0]))

        transforms.append([dx, dy, da])
        prev_gray = curr_gray

    transforms_np = np.asarray(transforms, dtype=np.float32)
    if anti_glitch:
        frame_h, frame_w = frames[0].shape[:2]
        transforms_np = suppress_glitch_transforms(
            transforms_np,
            frame_w,
            frame_h,
            max_shift_pct,
            max_rotate_deg,
        )

    trajectory = np.cumsum(transforms_np, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory, radius)
    correction = (smoothed_trajectory - trajectory) * np.float32(max(0.1, strength))
    transforms_smooth = transforms_np + correction

    stabilized = [frames[0]]
    for i in range(len(transforms_smooth)):
        dx, dy, da = transforms_smooth[i]

        m = np.array(
            [[np.cos(da), -np.sin(da), dx], [np.sin(da), np.cos(da), dy]],
            dtype=np.float32,
        )

        fixed = cv2.warpAffine(
            frames[i + 1],
            m,
            (frames[i + 1].shape[1], frames[i + 1].shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        stabilized.append(fixed)

    return stabilized


def stabilize_frames(
    frames: list[np.ndarray],
    radius: int,
    strength: float,
    passes: int,
    anti_glitch: bool,
    max_shift_pct: float,
    max_rotate_deg: float,
) -> list[np.ndarray]:
    output = frames
    passes = max(1, passes)
    for _ in range(passes):
        output = stabilize_frames_once(
            output,
            radius,
            strength,
            anti_glitch,
            max_shift_pct,
            max_rotate_deg,
        )
    return output


def detect_primary_face_center(frame: np.ndarray) -> tuple[float, float] | None:
    if FACE_CASCADE.empty():
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(48, 48),
    )
    if len(faces) == 0:
        return None

    areas = [w * h for (_, _, w, h) in faces]
    x, y, w, h = faces[int(np.argmax(areas))]
    return float(x + 0.5 * w), float(y + 0.45 * h)


def build_face_center_track(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) == 0:
        return np.empty((0, 2), dtype=np.float32)

    h, w = frames[0].shape[:2]
    fallback = np.array([w * 0.5, h * 0.5], dtype=np.float32)

    centers = np.full((len(frames), 2), np.nan, dtype=np.float32)
    for i, frame in enumerate(frames):
        c = detect_primary_face_center(frame)
        if c is not None:
            centers[i, 0] = c[0]
            centers[i, 1] = c[1]

    valid = np.where(~np.isnan(centers[:, 0]))[0]
    if len(valid) == 0:
        centers[:, 0] = fallback[0]
        centers[:, 1] = fallback[1]
        return centers

    first = int(valid[0])
    centers[:first] = centers[first]

    last_valid = centers[first].copy()
    for i in range(first, len(centers)):
        if np.isnan(centers[i, 0]) or np.isnan(centers[i, 1]):
            centers[i] = last_valid
        else:
            last_valid = centers[i].copy()

    return centers


def face_lock_stabilize(
    frames: list[np.ndarray],
    radius: int,
    strength: float,
) -> list[np.ndarray]:
    if len(frames) == 0:
        return frames

    track = build_face_center_track(frames)
    smooth_track = smooth_trajectory(track, max(2, radius))
    shifts = (smooth_track - track) * np.float32(max(0.1, strength))

    output: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        tx, ty = float(shifts[i, 0]), float(shifts[i, 1])
        m = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
        fixed = cv2.warpAffine(
            frame,
            m,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        output.append(fixed)

    return output


def save_frames(frames: list[np.ndarray], output_dir: Path, image_ext: str = "jpg") -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob(f"frame_*.{image_ext}"):
        old.unlink()

    frame_index = 0
    for frame in frames:

        frame_name = f"frame_{frame_index:06d}.{image_ext}"
        frame_path = output_dir / frame_name
        ok = cv2.imwrite(str(frame_path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write frame: {frame_path}")

        frame_index += 1

    return frame_index


def save_video(frames: list[np.ndarray], output_path: Path, fps: float) -> None:
    if not frames:
        raise RuntimeError("Cannot save empty video.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    h, w = frames[0].shape[:2]
    writer = None
    for codec in ["mp4v", "avc1", "H264"]:
        candidate = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (w, h),
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        raise RuntimeError(f"Could not open video writer: {output_path}")

    for frame in frames:
        writer.write(frame)

    writer.release()


def _unsharp(frame: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blur = cv2.GaussianBlur(frame, (0, 0), sigma)
    return cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)


def enhance_face_eyes(frame: np.ndarray, backend: str = "auto") -> np.ndarray:
    # Keep global sharpening mild; apply stronger sharpening only on eye regions.
    enhanced = _unsharp(frame, sigma=1.0, amount=0.28)

    use_mediapipe = backend == "mediapipe" or (backend == "auto" and FACE_MESH is not None)
    if backend == "mediapipe" and FACE_MESH is None:
        use_mediapipe = False

    if use_mediapipe:
        face_box, eye_boxes = _mediapipe_face_boxes(enhanced)
        if face_box is not None:
            x0, y0, x1, y1 = face_box
            face_roi = enhanced[y0:y1, x0:x1]
            face_roi = _unsharp(face_roi, sigma=0.9, amount=0.22)

            for ex0, ey0, ex1, ey1 in eye_boxes:
                eye_roi = face_roi[ey0:ey1, ex0:ex1]
                eye_roi = _unsharp(eye_roi, sigma=0.75, amount=0.9)
                face_roi[ey0:ey1, ex0:ex1] = eye_roi

            enhanced[y0:y1, x0:x1] = face_roi
            return enhanced

    if FACE_CASCADE.empty() or EYE_CASCADE.empty():
        return enhanced

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(60, 60),
    )

    for (x, y, w, h) in faces:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(enhanced.shape[1], x + w), min(enhanced.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            continue

        face_gray = gray[y0:y1, x0:x1]
        face_bgr = enhanced[y0:y1, x0:x1]

        eyes = EYE_CASCADE.detectMultiScale(
            face_gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(14, 14),
        )

        for (ex, ey, ew, eh) in eyes:
            pad_w = int(0.35 * ew)
            pad_h = int(0.45 * eh)
            rx0 = max(0, ex - pad_w)
            ry0 = max(0, ey - pad_h)
            rx1 = min(face_bgr.shape[1], ex + ew + pad_w)
            ry1 = min(face_bgr.shape[0], ey + eh + pad_h)
            if rx1 <= rx0 or ry1 <= ry0:
                continue

            eye_roi = face_bgr[ry0:ry1, rx0:rx1]
            eye_roi = _unsharp(eye_roi, sigma=0.9, amount=0.85)
            face_bgr[ry0:ry1, rx0:rx1] = eye_roi

        enhanced[y0:y1, x0:x1] = face_bgr

    return enhanced


def enhance_frames_for_clarity(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [enhance_face_eyes(frame) for frame in frames]


def slow_down_frames(frames: list[np.ndarray], slow_factor: int) -> list[np.ndarray]:
    slow_factor = max(1, int(slow_factor))
    if slow_factor == 1:
        return frames

    slowed: list[np.ndarray] = []
    for frame in frames:
        for _ in range(slow_factor):
            slowed.append(frame)
    return slowed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract every frame from a video with optional jitter stabilization."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Video file path. If omitted, first video from input1/ is used.",
    )
    parser.add_argument(
        "--frames-input",
        type=str,
        default="",
        help="Folder containing input frame images. Overrides --input when provided.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output FPS used when --frames-input is provided.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output1/frames",
        help="Folder where extracted frames will be saved.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Image format for extracted frames.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual processed frames to the output folder.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=55,
        help="Stabilization smoothing radius (higher = steadier, lower = more natural).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.05,
        help="How strongly correction is applied. Typical range: 0.8 to 2.0.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="How many stabilization passes to run.",
    )
    parser.add_argument(
        "--disable-anti-glitch",
        action="store_true",
        help="Disable anti-glitch filtering of motion spikes.",
    )
    parser.add_argument(
        "--max-shift-pct",
        type=float,
        default=0.05,
        help="Max per-frame shift as percentage of frame size before spike rejection.",
    )
    parser.add_argument(
        "--max-rotate-deg",
        type=float,
        default=3.5,
        help="Max per-frame rotation in degrees before spike rejection.",
    )
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Disable jitter removal and save raw frames.",
    )
    parser.add_argument(
        "--video-output",
        type=str,
        default="output1/output_stabilized.mp4",
        help="Output video file path for stabilized result.",
    )
    parser.add_argument(
        "--slow-factor",
        type=int,
        default=1,
        help="Repeat each frame this many times for slower frame-by-frame playback.",
    )
    parser.add_argument(
        "--no-eye-clarity",
        action="store_true",
        help="Disable face/eye clarity enhancement.",
    )
    parser.add_argument(
        "--clarity-backend",
        type=str,
        default="auto",
        choices=["auto", "opencv", "mediapipe"],
        help="Choose the face/eye detector used for clarity enhancement.",
    )
    parser.add_argument(
        "--disable-face-lock",
        action="store_true",
        help="Disable face-centered jitter correction.",
    )
    parser.add_argument(
        "--face-lock-radius",
        type=int,
        default=30,
        help="Smoothing radius for face-lock jitter correction.",
    )
    parser.add_argument(
        "--face-lock-strength",
        type=float,
        default=1.0,
        help="Correction strength for face-lock jitter correction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    video_output_path = Path(args.video_output)

    if args.frames_input:
        frames_dir = Path(args.frames_input)
        print(f"Input frames: {frames_dir}")
        frames = read_frames_from_dir(frames_dir)
        fps = max(1.0, float(args.fps))
    else:
        if args.input:
            video_path = Path(args.input)
        else:
            video_path = find_default_video(Path("input1"))
        print(f"Input video: {video_path}")
        frames, fps = read_frames_and_fps(video_path)

    print(f"Output dir : {output_dir}")
    print(f"Output video: {video_output_path}")

    if not frames:
        print("No frames extracted. The video may be empty or unreadable.")
        return

    if args.no_stabilize:
        print("Stabilization: disabled")
        final_frames = frames
    else:
        print(
            "Stabilization: enabled "
            f"(radius={max(1, args.radius)}, strength={max(0.1, args.strength):.2f}, passes={max(1, args.passes)})"
        )
        final_frames = stabilize_frames(
            frames,
            max(1, args.radius),
            max(0.1, args.strength),
            max(1, args.passes),
            not args.disable_anti_glitch,
            max(0.001, args.max_shift_pct),
            max(0.1, args.max_rotate_deg),
        )

    if args.no_eye_clarity:
        print("Eye clarity: disabled")
        enhanced_frames = final_frames
    else:
        print(f"Eye clarity: enabled (backend={args.clarity_backend})")
        enhanced_frames = [enhance_face_eyes(frame, args.clarity_backend) for frame in final_frames]

    if args.disable_face_lock:
        print("Face lock: disabled")
        face_stable_frames = enhanced_frames
    else:
        print(
            "Face lock: enabled "
            f"(radius={max(2, args.face_lock_radius)}, strength={max(0.1, args.face_lock_strength):.2f})"
        )
        face_stable_frames = face_lock_stabilize(
            enhanced_frames,
            max(2, args.face_lock_radius),
            max(0.1, args.face_lock_strength),
        )

    slow_factor = max(1, args.slow_factor)
    if slow_factor > 1:
        print(f"Slow mode: enabled (slow_factor={slow_factor})")
    slowed_frames = slow_down_frames(face_stable_frames, slow_factor)

    if args.save_frames:
        count = save_frames(slowed_frames, output_dir, args.ext)
    else:
        count = len(slowed_frames)

    save_video(slowed_frames, video_output_path, fps)

    if count == 0:
        print("No frames processed. The video may be empty or unreadable.")
        return

    print(f"Extracted {count} processed frames successfully.")
    print(f"Saved stabilized video: {video_output_path}")


if __name__ == "__main__":
    main()