# Tennis Play/Dead Time Detection

Automatically classifies tennis video frames as **play** or **dead time** using TrackNet ball detection, YOLO player detection, and temporal state classification.

## Setup

```bash
pip install -r tracknet/requirements.txt
```

## Usage

```bash
python src/pipeline.py --video data/videos/clip.mp4
python src/pipeline.py --video data/videos/clip.mp4 --output data/output
```

## Output

- Annotated video with play/dead overlay