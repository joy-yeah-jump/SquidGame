# code from
# https://076923.github.io/posts/Python-opencv-4/
# https://blog.naver.com/msnayana/220865581946
import cv2

capture = cv2.VideoCapture("./data/video/cars.mp4")

# while cv2.waitKey(33) < 0:  # 33 means 33ms
#     if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
#         capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # repeat play
#         # break
#
#     ret, frame = capture.read()
#     cv2.imshow("VideoFrame", frame)

while capture.isOpened():
    ret, frame = capture.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(33) == 27:
        break  # ESC
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        break

capture.release()
cv2.destroyAllWindows()
