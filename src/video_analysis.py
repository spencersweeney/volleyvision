import os
from roboflow import Roboflow
import json
import cv2
import numpy as np
from collections import deque
from path_manager import PathManager, PathStatus

API_KEY = "" # INSERT YOUR API KEY HERE
PROJECT = "volleyball_v2"

# how many seconds before the start of tracking a ball in motion the video should start
START_VIDEO_ON_PATH_BUFFER = 1
# how many seconds after the end of tracking a ball in motion the video should cut
STOP_VIDEO_ON_PATH_BUFFER = 3

VIEWING_AREA = {
    'top_cut_percentage': 0.1,
    'bottom_cut_percentage': 0.5,
    'left_cut_percentage': 0.05,
    'right_cut_percentage': 0.05
}
CONFIDENCE_VALUE = 0.65
SQUARENESS_VALUE = 0.9

TRAIL_LENGTH = 6 # for visualizing ball tracking


class VideoAnalysis():
    """
    Class to handle analysis operations and store information for a single video
    """

    def __init__(self, video_path):
        self.video_path = video_path

        cap = cv2.VideoCapture(video_path)

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))

    def init_detections(self, output_path="src/assets/inference_outputs/", api_key=API_KEY, project=PROJECT):
        self.output_path = output_path
        self.api_key = api_key
        self.project = project

        self.inference_results = self.get_detections_json()
        
        self.predictions = self._init_detections_map()
        
    def _init_detections_map(self):
        frames_with_predictions = self.inference_results['frame_offset']
        
        predictions_map = {}
        for i in range(len(frames_with_predictions)):
            predictions_map[f'{frames_with_predictions[i]}'] = self.inference_results[self.project][i]['predictions']
            
        return predictions_map

    def init_paths(self):
        self.path_manager = PathManager()
        for frame_count, predictions in self.predictions.items():
            for prediction in predictions:
                x, y, width, height = int(prediction['x']), int(
                    prediction['y']), int(prediction['width']), int(prediction['height'])
                confidence = prediction['confidence']

                # if self._validate_detection(x, y, width, height, confidence):
                self.path_manager.add_detection(x, y, int(frame_count))

        self.paths = self.path_manager.get_paths()
        
    def split_paths(self):
        self.path_manager.split_paths()
        self.paths = self.path_manager.get_paths()

    def get_detections_json(self):
        input_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        results_file = os.path.join(
            self.output_path, f"{input_filename}_results.txt")

        if not os.path.exists(results_file):
            results_file = self._generate_detections()

        with open(results_file, 'r') as f:
            content = f.read()

        formated_results_file = content.replace("'", '"')

        return json.loads(formated_results_file)

    def _generate_detections(self):
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace().project(self.project)
        model = project.version(2).model

        job_id, signed_url, expire_time = model.predict_video(
            self.video_path,
            fps=self.fps,
            prediction_type="batch-video"
        )
        
        print(f'job_id: {job_id}') # to have job_id incase something crashes

        results = model.poll_until_video_results(job_id)

        input_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        output_filename = os.path.join(
            self.output_path, f"{input_filename}_results.txt")

        with open(output_filename, "w") as f:
            f.write(str(results))

        return output_filename

    def plot_paths(self, output_file):
        cap = cv2.VideoCapture(self.video_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        for path in self.paths:
            status = path.get_status()

            if status == PathStatus.BALL_IN_MOTION:
                color = (0, 255, 0)
            elif status == PathStatus.BALL_STATIC:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            for detection in path.get_detections():
                cv2.circle(
                    black_frame, (detection['x'], detection['y']), 5, color, -1)

        cv2.imwrite(output_file, black_frame)
        
    def plot_paths_one_by_one(self):
        cap = cv2.VideoCapture(self.video_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for path in self.paths:
            black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            status = path.get_status()

            if status == PathStatus.BALL_IN_MOTION:
                color = (0, 255, 0)
            elif status == PathStatus.BALL_STATIC:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            for detection in path.get_detections():
                cv2.circle(
                    black_frame, (detection['x'], detection['y']), 5, color, -1)
                
            cv2.imshow('path', black_frame)
            cv2.waitKey(0)
            
    def plot_paths_video(self, output_path="src/outputs/"):
        input_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        results_filename = os.path.join(
            output_path, f"{input_filename}_plotting_paths.mp4")
        
        cap = cv2.VideoCapture(self.video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(results_filename, fourcc,
                              self.fps, (self.width, self.height))
        
        black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if f'{frame_count}' in self.predictions.keys():
                predictions = self.predictions[f'{frame_count}']
                
                for prediction in predictions:
                    x, y = int(prediction['x']), int(prediction['y'])
                    
                    cv2.circle(
                    black_frame, (x, y), 5, (255, 255, 255), -1)              

            out.write(black_frame)

            frame_count += 1
            
        for path in self.paths:
            status = path.get_status()

            if status == PathStatus.BALL_IN_MOTION:
                color = (0, 255, 0)
            elif status == PathStatus.BALL_STATIC:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            for detection in path.get_detections():
                cv2.circle(
                    black_frame, (detection['x'], detection['y']), 5, color, -1)
                
            out.write(black_frame)
            
        

        cap.release()
        out.release()
        

    def video_with_detections(self, output_path="src/outputs/", with_trail=False):
        input_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        results_filename = os.path.join(
            output_path, f"{input_filename}_with_detections.mp4")

        cap = cv2.VideoCapture(self.video_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(results_filename, fourcc,
                              self.fps, (self.width, self.height))

        frame_count = 0
        ball_trail = deque(maxlen=TRAIL_LENGTH)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if f'{frame_count}' in self.predictions.keys():
                predictions = self.predictions[f'{frame_count}']
                
                for prediction in predictions:
                    x, y, width, height = int(prediction['x']), int(
                        prediction['y']), int(prediction['width']), int(prediction['height'])
                    confidence = prediction['confidence']
                    class_name = prediction['class']
                    
                    # if self._validate_detection(x, y, width, height, confidence):
                    cv2.rectangle(frame, (x - width // 2, y - height // 2),
                                (x + width // 2, y + height // 2), (0, 255, 0), 2)

                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame, label, (x - width // 2, y - height //
                                2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if with_trail:
                        ball_trail.append((x, y))
                        
                        for x, y in ball_trail:
                            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)                    

            out.write(frame)

            frame_count += 1

        cap.release()
        out.release()

        return results_filename

    def chop_up_video(self, output_path="src/outputs/", with_detections=False, with_trail=False):
        input_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        results_filename = os.path.join(
            output_path, f"{input_filename}_cut_up.mp4")

        cap = cv2.VideoCapture(self.video_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(results_filename, fourcc,
                              self.fps, (self.width, self.height))

        necessary_frames = self._generate_neccessary_frames()

        frame_count = 0
        ball_trail = deque(maxlen=TRAIL_LENGTH)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count in necessary_frames:
                if with_detections:
                    if f'{frame_count}' in self.predictions.keys():
                        predictions = self.predictions[f'{frame_count}']
                        
                        for prediction in predictions:
                            x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
                            confidence = prediction['confidence']
                            class_name = prediction['class']
                            
                            if self._validate_detection(x, y, width, height, confidence): 
                                cv2.rectangle(frame, (x - width // 2, y - height // 2),
                                            (x + width // 2, y + height // 2), (0, 255, 0), 2)

                                label = f"{class_name} {confidence:.2f}"
                                cv2.putText(frame, label, (x - width // 2, y - height //
                                            2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                if with_trail:
                                    ball_trail.append((x, y))
                                    
                                    for x, y in ball_trail:
                                        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        return results_filename
    
    def watch_paths(self, output_path="src/outputs/", with_detections=False, with_trail=False):
        input_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        results_filename = os.path.join(
            output_path, f"{input_filename}_paths.mp4")

        cap = cv2.VideoCapture(self.video_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(results_filename, fourcc,
                              self.fps, (self.width, self.height))

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            
        for path in self.paths:    
            ball_trail = deque(maxlen=TRAIL_LENGTH)        
            for frame_count in range(path.get_start_frame(), path.get_stop_frame() + 1):
                output_frame = frames[frame_count]
                
                if with_detections:
                    if f'{frame_count}' in self.predictions.keys():
                        predictions = self.predictions[f'{frame_count}']
                        
                        for prediction in predictions:
                            x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
                            confidence = prediction['confidence']
                            class_name = prediction['class']
                        
                            if self._validate_detection(x, y, width, height, confidence): 
                                status = path.get_status()
                                if status == PathStatus.BALL_IN_MOTION:
                                    color = (0, 255, 0)
                                elif status == PathStatus.BALL_STATIC:
                                    color = (255, 0, 0)
                                else:
                                    color = (0, 0, 255)
                                
                                cv2.rectangle(output_frame, (x - width // 2, y - height // 2),
                                            (x + width // 2, y + height // 2), color, 2)

                                label = f"{class_name} {confidence:.2f}"
                                cv2.putText(output_frame, label, (x - width // 2, y - height //
                                            2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                if with_trail:
                                    ball_trail.append((x, y))
                                    
                if with_trail:
                    for x, y in ball_trail:
                                        cv2.circle(output_frame, (x, y), radius=5, color=color, thickness=-1)
                                        
                out.write(output_frame)
                
        cap.release()
        out.release()

        return results_filename
    
    def draw_set_trajectories(self, output_path="src/outputs/"):
        """
        WIP semi-hardcoded right now
        """
        input_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        results_filename = os.path.join(
            output_path, f"{input_filename}_set_trajectories.mp4")

        cap = cv2.VideoCapture(self.video_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(results_filename, fourcc,
                              self.fps, (self.width, self.height))

        frame_count = 0
        trajectory_points = {}
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            for path in self.paths:
                detections = path.get_detections()
                if (path.get_status() == PathStatus.BALL_IN_MOTION
                    and len(detections) >= 8
                    and detections[0]['x'] > self.width * 0.5
                    and detections[0]['x'] < self.width - self.width * 0.3
                    and detections[0]['y'] > self.height * 0.2
                    and detections[0]['y'] < self.height - self.height * 0.5):
                    
                    current_detections = [detection for detection in detections if detection['frame_count'] <= frame_count]
                    
                    if len(current_detections) >= 5:
                        x_vals = np.array([detection['x'] for detection in current_detections])
                        y_vals = np.array([detection['y'] for detection in current_detections])

                        coefficientss = np.polyfit(x_vals, y_vals, 2)
                        poly_func = np.poly1d(coefficientss)

                        # Generate smooth trajectory points
                        x_fit = np.linspace(min(x_vals), max(x_vals), num=100)
                        y_fit = poly_func(x_fit)

                        # Convert to integer coordinates
                        trajectory_points = np.array(list(zip(x_fit.astype(int), y_fit.astype(int))), np.int32)

                        # Draw the parabolic trajectory
                        for i in range(len(trajectory_points) - 1):
                            cv2.line(frame, tuple(trajectory_points[i]), tuple(trajectory_points[i + 1]), (0, 255, 0), 2)                

            out.write(frame)

            frame_count += 1

        cap.release()
        out.release()

        return results_filename

    def _generate_neccessary_frames(self):
        """
        based on all the paths determine which frames have ball in motion then add padding on either side
        then create a list that includes every frame number that should be included in the output video
        """
        if not len(self.paths) > 0:
            raise ValueError('there must be paths to chop up video')

        path_copy = list(filter(lambda path: path.get_status() == PathStatus.BALL_IN_MOTION, self.paths.copy()))
        
        if not len(path_copy):
            return []

        path_copy.sort(key=lambda path: path.get_start_frame())

        start_frame = path_copy[0].get_start_frame()
        if start_frame - self._seconds_to_frames(START_VIDEO_ON_PATH_BUFFER) < 0:
            start_frame = 0
        else:
            start_frame = start_frame - \
                self._seconds_to_frames(START_VIDEO_ON_PATH_BUFFER)
        stop_frame = path_copy[0].get_stop_frame(
        ) + self._seconds_to_frames(STOP_VIDEO_ON_PATH_BUFFER)

        frame_intervals = [[start_frame, stop_frame]]
        for path in path_copy:
            previous_start_frame, previous_stop_frame = frame_intervals[-1]
            start_frame = path.get_start_frame(
            ) - self._seconds_to_frames(START_VIDEO_ON_PATH_BUFFER)
            stop_frame = path.get_stop_frame() + self._seconds_to_frames(STOP_VIDEO_ON_PATH_BUFFER)

            if start_frame <= previous_stop_frame:
                frame_intervals[-1] = [previous_start_frame,
                                       max(previous_stop_frame, stop_frame)]
            else:
                frame_intervals.append([start_frame, stop_frame])

        saved_frames = []
        for interval in frame_intervals:
            frame_nums = range(interval[0], interval[1] + 1)

            saved_frames += frame_nums

        return saved_frames

    def _seconds_to_frames(self, seconds):
        return int(seconds * self.fps)
    
    def _validate_detection(self, x, y, width, height, confidence):
        return (
            x > self.width * VIEWING_AREA['left_cut_percentage'] 
            and x < self.width - self.width * VIEWING_AREA['right_cut_percentage'] 
            and y > self.height * VIEWING_AREA['top_cut_percentage'] 
            and y < self.height - self.height * VIEWING_AREA['bottom_cut_percentage']
            and 1 - abs((width / height) - 1) > SQUARENESS_VALUE
            and confidence > 0.5
        )
