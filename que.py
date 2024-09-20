from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

def two_line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    if x1 == x2 and x3 == x4:
        return False
    elif x1 == x2:
        x_intersect = x1
        m2 = (y4 - y3) / (x4 - x3)
        c2 = y3 - m2 * x3
        y_intersect = m2 * x_intersect + c2
    elif x3 == x4:
        x_intersect = x3
        m1 = (y2 - y1) / (x2 - x1)
        c1 = y1 - m1 * x1
        y_intersect = m1 * x_intersect + c1
    else:
        m1 = (y2 - y1) / (x2 - x1)
        m2 = (y4 - y3) / (x4 - x3)
        c1 = y1 - m1 * x1
        c2 = y3 - m2 * x3
        if m1 == m2:
            return False
        x_intersect = (c2 - c1) / (m1 - m2)
        y_intersect = m1 * x_intersect + c1

    if (min(x1, x2) <= x_intersect <= max(x1, x2)) and (min(x3, x4) <= x_intersect <= max(x3, x4)) and \
       (min(y1, y2) <= y_intersect <= max(y1, y2)) and (min(y3, y4) <= y_intersect <= max(y3, y4)):
        return True
    return False

def click_event(event, x, y, flags, params):
    global LINE_START, LINE_END
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if LINE_START is None:
            LINE_START = (x, y)
            print(f"Line Start set to: {LINE_START}")
        elif LINE_END is None:
            LINE_END = (x, y)
            print(f"Line End set to: {LINE_END}")

def is_in_queue_area(position):
    # Check if the last position of the vehicle is within the queue area
    return QUEUE_START[0] <= position[0] <= QUEUE_END[0] and QUEUE_START[1] <= position[1] <= QUEUE_END[1]

LINE_START = None 
LINE_END = None
model = YOLO("yolov8s.pt")
SOURCE_VIDEO_PATH = 'I:/Git/YOLO_Transportation_Engineering/obdetcfromyoutbue.mp4'

cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
   
success, frame = cap.read()
if not success:
    print("Error: Could not read the frame.")
    cap.release()
    exit()

cv2.imshow("Frame", frame)
print("Click on the video frame to select the line coordinates.")
print("Left-click to select the start and end points of the line.")

cv2.setMouseCallback("Frame", click_event)
    
while LINE_END is None:
    cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()
    
if LINE_START and LINE_END:
    print(f"LINE_START: {LINE_START}")
    print(f"LINE_END: {LINE_END}")
else:
    print("Error: Could not determine line coordinates.")
    exit()

cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

track_history = defaultdict(lambda: [])
speeds = {}  # To store speed of each object
SPEED_THRESHOLD = 10  # Speed threshold in pixels per second

# Define the queue length area
QUEUE_START = (LINE_START[0], LINE_START[1] - 50)  # Adjust height as needed
QUEUE_END = (LINE_START[0], LINE_END[1] + 50)      # Adjust height as needed

def calculate_speed(track, fps):
    if len(track) < 2:
        return 0
    x1, y1 = track[-2]
    x2, y2 = track[-1]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    speed = distance * fps  # Pixels per second
    return speed

frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print("Starting processing...")
line_crossing_counts = defaultdict(lambda: defaultdict(int))  # To track line crossing counts by class and track_id

with sv.VideoSink("output_with_queue_length.mp4", video_info) as sink:
    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1
        
        if not success:
            print("Error: Could not read frame.")
            break

        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        annotated_frame = results[0].plot()

        if track_ids is not None:
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))

                if len(track) > 100:
                    track.pop(0)

                speed = calculate_speed(track, fps)
                speeds[track_id] = speed

                # Count vehicles in the queue area if speed is below the threshold
                if speed < SPEED_THRESHOLD:
                    queue_count = sum(1 for pos in track_history.values() if is_in_queue_area(pos[-1]))
                    cv2.putText(annotated_frame, f"Queue Length: {queue_count} vehicles", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw the movement track
                points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                # Display speed and track ID on top of the vehicle
                cv2.putText(annotated_frame, f"ID: {track_id} Speed: {speed:.2f} px/s", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Check if the object has crossed the line
                if len(track) > 1:
                    prev_x = track[-2][0]
                    prev_y = track[-2][1]

                    if two_line_intersect(x, y, prev_x, prev_y, LINE_START[0], LINE_START[1], LINE_END[0], LINE_END[1]):
                        line_crossing_counts[class_id][track_id] = 1

            cv2.line(annotated_frame, LINE_START, LINE_END, (0, 0, 255), 2)

        # Draw the queue length area for visualization
        cv2.rectangle(annotated_frame, QUEUE_START, QUEUE_END, (255, 0, 0), 2)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        sink.write_frame(annotated_frame)

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")

total_count = sum([sum(counts.values()) for counts in line_crossing_counts.values()])
print(f"Total objects crossed the line: {total_count}")
for class_id, counts in line_crossing_counts.items():
    class_name = model.names[class_id]
    print(f"Total objects of class {class_id} {class_name} crossed the line: {sum(counts.values())}")
