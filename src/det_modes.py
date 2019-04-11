import time
import cv2
import operator


def full_recog(video_reader, video_writer, cnn_face_detector):
    '''
    Looks for a face in input video, draws a rectangle around it,
    and writes to output video. Looks for face in every frame
    '''

    # Loop over frames in the video
    for i, frame in enumerate(video_reader):
        frame_start = time.time()
        
        # Forced resized during processing for speed.
        scale_factor = 360 / frame.shape[1]
        cv2_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        # Dlib detect
        dets = cnn_face_detector(cv2_frame, 0)
        if dets:
            dets = [det for det in dets]
            #sorts by confidence, pics the most confident prediction, goes with it.
            dets.sort(key=operator.attrgetter('confidence'))
            det = dets[0]
            
            # Adjust coords for drawing bbox on original frame
            x_min = int(det.rect.left() / scale_factor)
            y_min = int(det.rect.top() / scale_factor)
            x_max = int(det.rect.right() / scale_factor)
            y_max = int(det.rect.bottom() / scale_factor)
            
            # Plot expanded bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
            
        frame_end = time.time()
        print('Frame: {:0>4} eplapsed {}s'.format(i,frame_end-frame_start))
        video_writer.writeFrame(frame)

    return


def track(video_reader, video_writer, cnn_face_detector):
    '''
    Looks for a face in input video, draws a rectangle around it,
    and writes to output video. Finds face once and tracks in following
    frames.
    '''
    # Tracker
    tracker = cv2.TrackerCSRT_create()
    bbox = None

    # Loop over frames in the video
    for i, frame in enumerate(video_reader):
        frame_start = time.time()

        # Forced resized for speed
        scale_factor = 360 / frame.shape[1]
        cv2_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        
        if bbox is not None:
            # Track from previous frame
            success, bbox = tracker.update(cv2_frame)
            if success:  # Skip frame if no match found
                x_min, y_min, w, h = [int(z) for z in bbox]
                x_max, y_max = x_min + w, y_min + h

                # Adjust coords for drawing bbox on original frame
                x_min = int(x_min / scale_factor)
                y_min = int(y_min / scale_factor)
                x_max = int(x_max / scale_factor)
                y_max = int(y_max / scale_factor)
                
                # Plot expanded bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            else:
                bbox = None
                
        if bbox is None:
            # Dlib detect
            dets = cnn_face_detector(cv2_frame, 0)
            if dets:
                dets = [det for det in dets]
                #sorts by confidence, pics the most confident prediction, goes with it.
                dets.sort(key=operator.attrgetter('confidence'))
                det = dets[0]

                # Adjust coords for bbox
                x_min = int(det.rect.left())
                y_min = int(det.rect.top())
                x_max = int(det.rect.right())
                y_max = int(det.rect.bottom())

                bbox = (x_min, y_min, x_max-x_min, y_max-y_min)
                tracker.init(cv2_frame, bbox)

                # Adjust coords for drawing bbox on original frame
                x_min = int(det.rect.left() / scale_factor)
                y_min = int(det.rect.top() / scale_factor)
                x_max = int(det.rect.right() / scale_factor)
                y_max = int(det.rect.bottom() / scale_factor)

                # Plot expanded bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
            
        frame_end = time.time()
        print('Frame: {:0>4} eplapsed {}s'.format(i,frame_end-frame_start))
        video_writer.writeFrame(frame)

    return