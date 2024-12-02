import cv2
import dlib
import time
import math
WIDTH, HEIGHT = 1280, 720
VIDEO_PATH = 'carsVid.mp4'
CASCADE_PATH = 'vech.xml'
PPM = 8.8
FPS = 18
DETECTION_INTERVAL = 10
Y_THRESHOLD = 300

car_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def estimate_speed(loc1, loc2):
    d_pixels = math.sqrt((loc2[0] - loc1[0]) ** 2 + (loc2[1] - loc1[1]) ** 2)
    d_meters = d_pixels / PPM
    return d_meters * FPS * 3.6  # convert m/s to km/h


def detect_cars(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=13, minSize=(24, 24))


def create_tracker(image, x, y, w, h):
    tracker = dlib.correlation_tracker()
    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
    return tracker


def track_objects(video_path):
    video = cv2.VideoCapture(video_path)

    frame_counter, car_id_counter = 0, 0
    car_trackers = {}
    car_locations = {}
    speeds = {}
    average_speeds = {}

    while True:
        start_time = time.time()
        ret, frame = video.read()
        if frame is None:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame_counter += 1
        for car_id, tracker in list(car_trackers.items()):
            tracking_quality = tracker.update(frame)
            if tracking_quality < 7:
                car_trackers.pop(car_id, None)
                car_locations.pop(car_id, None)
                speeds.pop(car_id, None)
                average_speeds.pop(car_id, None)
                continue
            tracked_pos = tracker.get_position()
            x, y, w, h = (int(tracked_pos.left()), int(tracked_pos.top()),
                          int(tracked_pos.width()), int(tracked_pos.height()))

            if car_id in car_locations:
                prev_location = car_locations[car_id]
                speed = estimate_speed(prev_location, [x, y, w, h])

                if car_id not in average_speeds:
                    average_speeds[car_id] = []
                average_speeds[car_id].append(speed)

                speeds[car_id] = speed

            car_locations[car_id] = [x, y, w, h]


        if frame_counter % DETECTION_INTERVAL == 0:
            for (x, y, w, h) in detect_cars(frame):

                matched = False
                x_bar, y_bar = x + w // 2, y + h // 2
                for car_id, tracker in car_trackers.items():
                    tracked_pos = tracker.get_position()
                    t_x, t_y = int(tracked_pos.left()), int(tracked_pos.top())
                    t_w, t_h = int(tracked_pos.width()), int(tracked_pos.height())
                    if (t_x <= x_bar <= t_x + t_w) and (t_y <= y_bar <= t_y + t_h):
                        matched = True
                        break

                if not matched:
                    car_trackers[car_id_counter] = create_tracker(frame, x, y, w, h)
                    car_locations[car_id_counter] = [x, y, w, h]
                    speeds[car_id_counter] = None
                    average_speeds[car_id_counter] = []
                    car_id_counter += 1

        for car_id, (x, y, w, h) in car_locations.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)


            if y > Y_THRESHOLD and average_speeds[car_id]:
                avg_speed = sum(average_speeds[car_id]) / len(average_speeds[car_id])
                cv2.putText(frame, f"{int(avg_speed)} km/h", (x + w // 2, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow('Vehicle Tracking', frame)

        if cv2.waitKey(1) == 27:  # Esc to quit
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    track_objects(VIDEO_PATH)

