# Tennis Play/Dead Time Detection

Automatically classifies tennis video frames as **play** or **dead time** using TrackNet ball detection, YOLO player detection, and temporal state classification.

## Setup

```bash
git clone --recurse-submodules https://github.com/rlxai/tennis-analytics-phase-1.git
cd tennis-analytics-phase-1
python3 -m venv venv
source venv/bin/activate
pip install -r tracknet/requirements.txt
pip install ultralytics

sudo apt install ffmpeg   # required for H.264 video encoding
```

## Usage

Place input videos in `data/videos/`, then run:

```bash
python src/pipeline.py --video data/videos/clip.mp4
python src/pipeline.py --video data/videos/clip.mp4 --output data/output
```

## Output

- Annotated video with play/dead overlay