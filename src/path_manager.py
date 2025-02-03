from path import Path, PathStatus

class PathManager():
    """
    Class to handle all paths that exist within the video maintains the additions of locations to path and classification of paths
    """

    def __init__(self):
        self.paths = []

    def add_detection(self, x, y, frame_count):
        """
        Add detection to the correct path or if no path matches it create a new path
        """
        path = self._match_path(x, y, frame_count)

        if path:
            path.add_detection(x, y, frame_count)
        else:
            self.paths.append(Path(x, y, frame_count))

    def get_paths(self):
        return self.paths

    def split_paths(self):
        """
        This is used for taking one large path that includes multiple parabolic travels
        and splits it into its individual parabolic travel
        
        Does this for each path
        
        WIP only used for set trajectories
        """
        new_paths = []
        for path in self.paths:
            paths_to_add = path.split()
            
            for path_to_add in paths_to_add:
                new_paths.append(path_to_add)
        self.paths = new_paths

    def _match_path(self, x, y, frame_count):
        """
        Find the path that the detection belongs to, if no path return None

        Possible paths are paths that are paths that have been updated recently
        Possible static paths are for static balls or consistent false positives
        """
        possible_paths = []
        possible_static_paths = []
        for path in self.paths:
            if path.check_detection_path_compatability(x, y):
                if path.check_detection_recency_compatability(frame_count):
                    possible_paths.append(path)
                elif path.get_status == PathStatus.BALL_STATIC:
                    possible_static_paths.append(path)
            
        if len(possible_paths) > 0:
            possible_paths.sort(key=lambda path: path.compute_distance(x, y))
            return possible_paths[0]
        
        if len(possible_static_paths) > 0:
            possible_static_paths.sort(key=lambda path: path.compute_distance(x, y))
            return possible_static_paths[0]
        
        return None

