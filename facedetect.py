#! /usr/bin/python2.7

#       Author:         Jayant Kashyap
#       Date:           08 January, 2016
#       Description:    Face detection

#importing necessary packages
import cv2
import sys
import argparse
import datetime
import imutils
import time

#Face cascade classifier
face_cascade = 'c:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(face_cascade)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    video_capture = cv2.VideoCapture(0)
    time.sleep(0.25)

# otherwise, we are reading from a video file
else:
    video_capture = cv2.VideoCapture(args["video"])
	
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Converting to the grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        flags = 0
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

