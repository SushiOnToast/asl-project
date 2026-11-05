import urllib.request
from asl import paths

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/latest/hand_landmarker.task")

paths.MODELS.mkdir(exist_ok=True)
if paths.LANDMARKER_TASK.exists():
    print("Hand model already present, skipping.")
else:
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(URL, paths.LANDMARKER_TASK)
    print("Done:", paths.LANDMARKER_TASK)