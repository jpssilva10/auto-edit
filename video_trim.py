import cv2
import numpy as np
import os

# === Config ===
input_path = os.path.join(os.getcwd(), "videos\\test.mp4")  # Input video path
output_path = os.path.join(os.getcwd(), "videos\\test_trimmed.mp4")
motion_threshold = 15000          # Adjust: number of changed pixels to trigger motion
min_motion_frames = 10            # Only keep segments with at least this many frames of motion
padding_frames = 3                # Add frames before/after motion for smoother cuts

# === Init ===
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

frame_buffer = []
segment_buffer = []
motion_detected = False
motion_counter = 0
no_motion_counter = 0

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = fgbg.apply(frame)
    motion_score = np.sum(mask == 255)

    # === Show preview ===
    display = frame.copy()
    cv2.putText(display, f"Motion Score: {motion_score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if motion_score > motion_threshold else (0, 0, 255), 2)

    stacked = np.hstack([
        display,
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    ])
    cv2.imshow("Frame | Mask", cv2.resize(stacked, (stacked.shape[1] // 2, stacked.shape[0] // 2)))

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit early
        break

    # === Rolling buffer for pre-padding ===
    frame_buffer.append(frame)
    if len(frame_buffer) > padding_frames:
        frame_buffer.pop(0)

    if motion_score > motion_threshold:
        if not motion_detected:
            segment_buffer.extend(frame_buffer)
            motion_detected = True
            motion_counter = 0
        segment_buffer.append(frame)
        motion_counter += 1
        no_motion_counter = 0
    else:
        if motion_detected:
            no_motion_counter += 1
            segment_buffer.append(frame)

            if no_motion_counter > padding_frames:
                if motion_counter >= min_motion_frames:
                    print(f"[WRITE] Segment ending at frame {frame_idx}")
                    for f in segment_buffer:
                        out.write(f)
                segment_buffer = []
                motion_detected = False

    frame_idx += 1

# Write last segment
if motion_detected and motion_counter >= min_motion_frames:
    print(f"[WRITE] Final segment at frame {frame_idx}")
    for f in segment_buffer:
        out.write(f)

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done. Output saved to:", output_path)
