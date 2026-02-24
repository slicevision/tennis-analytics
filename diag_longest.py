"""Find the longest non-horizontal lines (sidelines) and add padded parallel lines."""
import cv2
import numpy as np

PADDING_PX = 350  # pixels outward from sideline — adjustable

cap = cv2.VideoCapture("data/videos/test_clip.mp4")
_, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 100)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=150, maxLineGap=15)
lines = lines.reshape(-1, 4).astype(np.float64)
lengths = np.sqrt((lines[:,2]-lines[:,0])**2 + (lines[:,3]-lines[:,1])**2)
angles = np.degrees(np.arctan2(np.abs(lines[:,3]-lines[:,1]), np.abs(lines[:,2]-lines[:,0])))

# Filter out horizontal lines
steep_mask = angles >= 20
lines = lines[steep_mask]
lengths = lengths[steep_mask]
angles = angles[steep_mask]

# Split into left and right sideline groups by centroid x
mid_x = (lines[:, 0] + lines[:, 2]) / 2.0
left_mask = mid_x < w / 2
right_mask = ~left_mask

# Collect all points from each group and fit a line
def fit_and_extend(group_lines):
    pts = np.vstack([group_lines[:, :2], group_lines[:, 2:4]]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    # Ensure vy > 0 (direction goes top to bottom)
    if vy < 0:
        vx, vy = -vx, -vy
    t_top = (0 - y0) / vy
    t_bot = (h - y0) / vy
    p_top = (int(x0 + vx * t_top), 0)
    p_bot = (int(x0 + vx * t_bot), h)
    return p_top, p_bot, vx, vy

def shift_line(p_top, p_bot, vx, vy, px_outward):
    # Normal perpendicular to line direction (pointing outward)
    nx, ny = -vy, vx  # perpendicular
    norm = np.sqrt(nx**2 + ny**2)
    nx, ny = nx / norm, ny / norm
    # Shift
    return ((int(p_top[0] + nx * px_outward), p_top[1]),
            (int(p_bot[0] + nx * px_outward), p_bot[1]))

left_top, left_bot, lvx, lvy = fit_and_extend(lines[left_mask])
right_top, right_bot, rvx, rvy = fit_and_extend(lines[right_mask])

print(f"DEBUG left group: {left_mask.sum()} lines, top={left_top}, bot={left_bot}")
for l in lines[left_mask]:
    print(f"  [{l[0]:.0f},{l[1]:.0f}]->[{l[2]:.0f},{l[3]:.0f}]")
print(f"DEBUG right group: {right_mask.sum()} lines, top={right_top}, bot={right_bot}")
for l in lines[right_mask]:
    print(f"  [{l[0]:.0f},{l[1]:.0f}]->[{l[2]:.0f},{l[3]:.0f}]")

# Ensure left is actually left (smaller x at mid-frame)
left_mid_x = (left_top[0] + left_bot[0]) / 2
right_mid_x = (right_top[0] + right_bot[0]) / 2
if left_mid_x > right_mid_x:
    left_top, right_top = right_top, left_top
    left_bot, right_bot = right_bot, left_bot
    lvx, rvx = rvx, lvx
    lvy, rvy = rvy, lvy

# Shift outward: left sideline shifts LEFT (negative x), right shifts RIGHT
# The normal (-vy, vx) may point left or right depending on direction — pick the exterior side
# For left line: exterior = smaller x → check which direction the normal points
left_pad_top, left_pad_bot = shift_line(left_top, left_bot, lvx, lvy, -PADDING_PX)
right_pad_top, right_pad_bot = shift_line(right_top, right_bot, rvx, rvy, PADDING_PX)

# If the padded line moved inward instead of outward, flip sign
if left_pad_top[0] > left_top[0]:  # went right instead of left
    left_pad_top, left_pad_bot = shift_line(left_top, left_bot, lvx, lvy, PADDING_PX)
if right_pad_top[0] < right_top[0]:  # went left instead of right
    right_pad_top, right_pad_bot = shift_line(right_top, right_bot, rvx, rvy, -PADDING_PX)

print(f"Left sideline:  {left_top} -> {left_bot}")
print(f"Left padded:    {left_pad_top} -> {left_pad_bot}")
print(f"Right sideline: {right_top} -> {right_bot}")
print(f"Right padded:   {right_pad_top} -> {right_pad_bot}")
print(f"Padding: {PADDING_PX}px")

vis = frame.copy()
# Sidelines (green)
cv2.line(vis, left_top, left_bot, (0, 255, 0), 2)
cv2.line(vis, right_top, right_bot, (0, 255, 0), 2)
# Padded lines (cyan)
cv2.line(vis, left_pad_top, left_pad_bot, (255, 255, 0), 2)
cv2.line(vis, right_pad_top, right_pad_bot, (255, 255, 0), 2)
# Labels
cv2.putText(vis, "L", (left_top[0]-20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
cv2.putText(vis, "R", (right_top[0]+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
cv2.imwrite("data/output/sidelines_padded.jpg", vis)
print("Saved: data/output/sidelines_padded.jpg")
