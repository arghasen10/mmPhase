import cv2
import time

def capture_video(duration, video_name):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"./scene_annotation/{video_name}", fourcc, 20.0, (640, 480))

    # Capture video from the webcam
    cap = cv2.VideoCapture(0)
    end_time = time.time() + duration

    while(cap.isOpened() and time.time()<end_time):
        ret, frame = cap.read()
        if ret:
            # Write the frame into the file
            out.write(frame)

            # # Display the resulting frame
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
 

    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()