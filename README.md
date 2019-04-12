# single-face-tracking-demo
A small demo of how to track a face in a video with python and opencv.
To add some excitement, I compare the speed of detecting the face in
every frame with an off-the-shelf cnn from dlib running in my CPU vs
detecting once and then traking the detected region with a CSRT tracker.

# Requirements
Code is written in python 3.6. Everything else can be installed with:
```
$ pip install -r requirements.txt
```
# Results
Processing is printed in the notebook. An example of the ouput for both
methonds is given under ``data/output``, do check it out.
The tracker is of course faster and the fallback to face detection when
the tracked region is lost allows for extra robustness. Look like this:

![Output sample](https://github.com/DunaiFuentes/single-face-tracking-demo/blob/master/data/output/track_my_video.gif)

# How to use
To try it out on your own videos, you may replace the input video path
in the jupyter notebook and run again, or you may execute:
```
$ python find_and_track_principal_face.py -i <path/to/video>
```
which will create an output of the same name under /data/output.
You may optionally indicate the mode with -m, it will defaut to "track"