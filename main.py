import time
from utils import *

feature_params = dict(maxCorners=100,
                      qualityLevel=0.5,
                      minDistance=7,
                      blockSize=7)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
start_flag = True
cap = cv2.VideoCapture('test_video.mp4')  # reading the video
while cap.isOpened():  # run on the video
    ret, frame = cap.read()
    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray scale
    if start_flag:  # skipping on the first frame
        start_flag = False
        old_gray = gray.copy()
        continue
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # find good features
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None,
                                           **lk_params)  # calculate the optical flow new location
    changing_inside_frames = mean_frame_changing(p0, p1, st)  # find the mean change between frames
    old_gray = gray.copy()  # save old frame to comparison
    if cv2.waitKey(1) & 0xFF == ord('w'):  # stop for deeper look fo frame difference
        show_frame_diff(old_gray, gray, p0, p1, st)
    text = "the frame diff is: " + str(changing_inside_frames)
    cv2.putText(gray, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 255, 255))  # print the average distance between etch frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # emergency stop of playing
        break

cap.release()
cv2.destroyAllWindows()
