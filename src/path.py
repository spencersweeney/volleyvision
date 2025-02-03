from enum import Enum
import math

class PathStatus(Enum):
    RANDOM = 1
    BALL_IN_MOTION = 2
    BALL_STATIC = 3

class Path:
    """
    Class for a specific path of a volleyball
    """

    def __init__(self, x, y, frame_count, max_distance=100, recency_value=7):
        self.detections = [{'x': x, 'y': y, 'frame_count': frame_count}]
        self.max_distance = max_distance
        self.recency_value = recency_value # value in frames of classifying recency
        self.status = PathStatus.RANDOM
    
    def add_detection(self, x, y, frame_count):
        """
        adds new detection to the list and updates the state depending on how the 
        """
        self.detections.append({'x': x, 'y': y, 'frame_count': frame_count})

        if len(self.detections) > 2:
            if self.status == PathStatus.RANDOM or self.status == PathStatus.BALL_STATIC:
                delta_x_1 = self.detections[-2]['x'] - self.detections[-3]['x']
                delta_y_1 = self.detections[-2]['y'] - self.detections[-3]['y']

                delta_x_2 = self.detections[-1]['x'] - self.detections[-2]['x']
                delta_y_2 = self.detections[-1]['y'] - self.detections[-2]['y']
                
                detections_copy = self.detections.copy()
                detections_copy.sort(key=lambda detection: detection['x'])
                
                furthest_left = detections_copy[0]['x']
                furthest_right = detections_copy[-1]['x']
                
                detections_copy.sort(key=lambda detection: detection['y'])
                
                furthest_up = detections_copy[0]['y']
                furthest_down = detections_copy[-1]['y']
                
                detections_copy.sort(key=lambda detection: detection['frame_count'])
                
                age = detections_copy[-1]['frame_count'] - detections_copy[0]['frame_count']

                if (delta_x_1 * delta_x_2 >= 0 
                    and delta_y_1 * delta_y_2 >= 0 
                    and (furthest_right - furthest_left > self.max_distance / 2
                    or furthest_down - furthest_up > self.max_distance / 2
                    or age > self.recency_value * 2)
                ):
                    self.status = PathStatus.BALL_IN_MOTION
                else:
                    self.status = PathStatus.BALL_STATIC

    def compute_distance(self, x, y):
        most_recent_x = self.detections[-1]['x']
        most_recent_y = self.detections[-1]['y']

        return math.sqrt((most_recent_x - x) ** 2 + (most_recent_y - y) ** 2)

    def check_detection_path_compatability(self, x, y):
        distance = self.compute_distance(x, y)

        return distance < self.max_distance
    
    def check_detection_recency_compatability(self, frame_count):
        return abs(frame_count - self.detections[-1]['frame_count']) <= self.recency_value
    
    def split(self):
        """
        splits this path based on amount of individual parabolic motions
        
        WIP only used for drawing set trajectories
        """
        if self.status != PathStatus.BALL_IN_MOTION:
            return [self]
        
        if len(self.detections) < 2:
            return [self]
        
        return_paths = []
        
        current_path = Path(self.detections[0]['x'], self.detections[0]['y'], self.detections[0]['frame_count'])
        for i in range(1, len(self.detections)):
            if len(current_path.detections) < 2:
                current_path.add_detection(self.detections[i]['x'], self.detections[i]['y'], self.detections[i]['frame_count'])
                continue
            
            delta_x_1 = current_path.detections[-1]['x'] - current_path.detections[-2]['x']
            delta_y_1 = current_path.detections[-1]['y'] - current_path.detections[-2]['y']
            
            delta_x_2 = self.detections[i]['x'] - current_path.detections[-1]['x']
            delta_y_2 = self.detections[i]['y'] - current_path.detections[-1]['y']
            
            if (delta_x_1 * delta_x_2 >= 0 
                and delta_y_1 * delta_y_2 >= 0):
                current_path.add_detection(self.detections[i]['x'], self.detections[i]['y'], self.detections[i]['frame_count'])
            else:
                current_path.set_status(PathStatus.BALL_IN_MOTION)
                return_paths.append(current_path)
                current_path = Path(self.detections[i]['x'], self.detections[i]['y'], self.detections[i]['frame_count'])

        current_path.set_status(PathStatus.BALL_IN_MOTION)
        return_paths.append(current_path)   
        
        return return_paths         
    
    def get_status(self):
        return self.status
    
    def set_status(self, status):
        self.status = status

    def get_detections(self):
        return self.detections
    
    def get_start_frame(self):
        return self.detections[0]['frame_count']
    
    def get_stop_frame(self):
        return self.detections[-1]['frame_count']
    