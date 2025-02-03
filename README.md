# VolleyVision

**Breakdown volleyball film to cut out down time in between plays.**

## 🎥 Demo Video

### Original Uncut Clip
[![Watch the Video](https://img.youtube.com/vi/Q-eyubyxllo/0.jpg)](https://www.youtube.com/watch?v=Q-eyubyxllo)

### Output Cut Up Clip
[![Watch the Video](https://img.youtube.com/vi/yUwK1aLkqzc/0.jpg)](https://www.youtube.com/watch?v=yUwK1aLkqzc)

## Further Development

### Set Trajectory Tracking (WIP)
[![Watch the Video](https://img.youtube.com/vi/_tEciWYCacY/0.jpg)](https://www.youtube.com/watch?v=_tEciWYCacY)

## Installation

```bash
git clone https://github.com/spencersweeney/volleyvision.git
cd volleyvision
pip install -r requirements.txt
```

## Usage

CLI is currently in the works.

To use the system currently.
- Put your API Key for Roboflow in the top of ```video_analysis.py```
- Put your input video in the top of ```main.py```.
- Call whatever methods you want done on the video_analyzer object in ```main.py```

## Citation/Acknowledgements

- [Roboflow Dataset/Model](https://universe.roboflow.com/shukur-sabzaliev1/volleyball_v2)
- [Idea for ball tracking technique](https://github.com/tprlab/vball)
