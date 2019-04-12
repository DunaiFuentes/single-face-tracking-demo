import os
import cv2
import dlib
import skvideo.io
import time
import operator
import argparse
from src import det_modes as dm

# Settings
## paths
face_det_model_path = 'res/mmod_human_face_detector.dat'
output_folder = 'data/output'
## ffmpeg encoding options
outputdict={'-c:v': 'libx264', '-crf': 0}


def main(args):

    # Load face detector
    cnn_face_detector = dlib.cnn_face_detection_model_v1(face_det_model_path)

    # Load video
    video_reader = skvideo.io.vreader(args.input_path)

    # Set destination
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = args.mode + '_' + os.path.basename(args.input_path)
    output_path = os.path.join(output_folder, filename)

    # Open writer
    video_writer = skvideo.io.FFmpegWriter(output_path)
    
    # Start processing
    video_start = time.time()
    try:
        if args.mode == 'full_recog':
            dm.full_recog(video_reader, video_writer, cnn_face_detector)
        if args.mode == 'track':
            dm.track(video_reader, video_writer, cnn_face_detector)
    except Exception as e:
        print(e)
    finally:
        video_reader.close()
        video_writer.close()
        video_end = time.time()
        print('Full Video: eplapsed {}s'.format(video_end - video_start))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, type=str,
                        help="Path to input video, data/input/my_video.mp4")
    parser.add_argument('-m', '--mode', default='track', type=str,
                        choices=['track','full_recog'], help='How to do it')
    args = parser.parse_args()

    main(args)
