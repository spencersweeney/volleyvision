import timeit

from video_analysis import VideoAnalysis

INPUT_VIDEO = "" # PUT YOUR INPUT VIDEO HERE

t_0 = timeit.default_timer()

video_analyzer = VideoAnalysis(INPUT_VIDEO)

video_analyzer.init_detections()
video_analyzer.init_paths()

video_analyzer.chop_up_video()

t_1 = timeit.default_timer()

elapsed_time = round((t_1 - t_0))
print(f"Elapsed time: {elapsed_time} s")
