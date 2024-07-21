from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
from shapely import Point, Polygon
import cvzone


# Initializing the YOLO model
model = YOLO('yolov8l.pt')

classnames = []
with open('classnames.txt', 'r') as f:
    classnames = f.read().splitlines()

# cap = cv2.VideoCapture(0)

# Initializing a variable for collecting the points of each ID
track_history = defaultdict(lambda: [])

points = []
collecting_flag = None
cap = cv2.VideoCapture(2)

first_frame = cap.retrieve(cap.grab())[1]
first_frame_poly = first_frame.copy()



def collect_point(event, x, y, flags, params):
    global polygon_s, polygon_ocv

    if collecting_flag == True:
        if event == cv2.EVENT_LBUTTONDOWN:

            cv2.circle(first_frame_poly, (x,y), 4, (255, 0, 0), -1)
            cv2.imshow("Window", first_frame_poly)
            points.append((x,y))
    
    elif collecting_flag == False:
        if len(points) > 2:
            polygon_s = Polygon(points)
            polygon_ocv = np.array(points, np.int32)

            cv2.polylines(first_frame_poly, [polygon_ocv], isClosed=True, color=(0,255,0), thickness=2)
            cv2.fillPoly(first_frame_poly, [polygon_ocv], (0,255,0))
            img_final = cv2.addWeighted(first_frame_poly, 0.4, first_frame, 0.6, 0)
            cv2.imshow("Window", img_final)

def checking_poly():
    global collecting_flag

    cv2.namedWindow('Window')
    cv2.setMouseCallback("Window", collect_point)
    cv2.imshow('Window', first_frame_poly)

    while True:
        key = cv2.waitKey() & 0xFF
        
        if key == ord('s'):
            collecting_flag = True
            break
            

    while collecting_flag:
        key = cv2.waitKey() & 0xFF
        if key == ord('f'):
            collecting_flag = False
            break

    # cv2.waitKey()
    # cv2.destroyAllWindows()

def main():

    checking_poly()

    np_zonePoints = np.array([points], np.int32)
    cars_in_zone = 0
    counted_car_ids = set()
    while cap.isOpened():

        ret, frame = cap.read()
        
        if ret:
            
            poly_frame = frame.copy()
            results = model.track(frame, conf=0.50, persist=True, tracker="bytetrack.yaml")
            if results[0].boxes.id == None: # This check  is for the frames in which nothing is detected. We ignore that frame and continue to the next frame  # noqa: E711
                continue
            
            boxes = results[0].boxes
            track_ids = results[0].boxes.id.int().tolist()

            MAIN_boxes = []
            box_id_flag = False
            for box, track_id in zip(boxes, track_ids):

                if classnames[int(box.cls)] == 'person':
                    MAIN_boxes.append(box)
            
            # for main_box, track_id in zip(MAIN_boxes, track_ids):
                    box_id = int(box.id)
                    x,y,w,h = box.xywh[0]
                    top_left = int(x - (w/2)), int(y - (h/2))
                    bottom_right = int(x + (w/2)), int(y + (h/2))

                    
                    
                    rect_img = cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)
                    track = track_history[track_id] #Here we're collecting all the center points of each ID
                    track.append((float(x), float(y)))
                    if len(track) > 30: # removing the point at index 0 once we reach 30 points
                        track.pop(0)
                    track_points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    ann_frame = cv2.polylines(rect_img, [track_points], isClosed=False, color=(255, 0, 255), thickness=2)
                    
                    polygon_ocv = np.array(points, np.int32)
                    
                    
                    new_img = cv2.polylines(poly_frame, [polygon_ocv], isClosed=True, color=(0,255,0), thickness=2)
                    reshaped_points = polygon_ocv.reshape(-1,1,2)
                    counts = cv2.pointPolygonTest(reshaped_points, pt=(int(x),int(y)), measureDist=False)
                    
                    new_img = cv2.fillPoly(new_img, [polygon_ocv], (0,255,0))
                    ann_frame_plus_new_img = cv2.addWeighted(new_img, 0.4, ann_frame, 0.6, 0)
                    final_image = cv2.addWeighted(ann_frame_plus_new_img, 0.4, frame, 0.6, 0)
                    text_img = cv2.putText(final_image, f'Number of cars in the zone: {cars_in_zone}', (50, 100), 4, 1.0, (255,255,0), 2)
                    if counts == 1 and box_id not in counted_car_ids:
                        cars_in_zone += 1
                        text_img = cv2.putText(final_image, f'Number of cars in the zone: {cars_in_zone}', (50, 100), 4, 1.0, (255,255,0), 2)
                        counted_car_ids.add(box_id)

                    text_img = cv2.putText(final_image, f'Number of cars in the zone: {cars_in_zone}', (50, 100), 4, 1.0, (255,255,0), 2)

                    cv2.imshow("Window", text_img)

                    if cv2.waitKey(30) & 0xFF == ord('q'):
                    
                        cap.release()

                        cv2.destroyAllWindows()
                        break
        else:
            break

    

if __name__ == '__main__':
    main()