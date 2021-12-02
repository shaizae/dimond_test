import cv2

from utils import *

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
start_flag = True
cap = cv2.VideoCapture('test_video.mp4')  # reading the video
changing_between_frames = []
while cap.isOpened():  # run on the video
    ret, frame = cap.read()
    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray scale
    if start_flag:  # skipping on the first frame
        start_flag = False
        old_gray = gray.copy()
        continue
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None,
                                           **lk_params)  # calculate the optical flow new location
    changing_inside_frames = mean_frame_changing(p0, p1, st)  # find the mean change between frames

    changing_between_frames.append(changing_inside_frames)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    old_gray = gray.copy()
mean_chang_between_frames=np.array(changing_between_frames).mean()
print(mean_chang_between_frames)
cap.release()
cv2.destroyAllWindows()
