import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    maska = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(gray, maska, iterations=1)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 110;
    params.maxThreshold = 200;

    params.filterByArea = True
    params.minArea = 35
    params.maxArea = 150

    params.filterByCircularity = True
    params.minCircularity = 0.8

    params.filterByConvexity = True
    params.minConvexity = 0.8

    params.filterByInertia = True
    params.minInertiaRatio = 0.6


    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(dilation)


    select = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(select, "Liczba: " + str(len(keypoints)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Keypoints", select)


    #cv2.imshow('frame', detector)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()