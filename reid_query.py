import argparse
import imutils
import time
import cv2
import os
import pandas as pd
from torchreid.utils import FeatureExtractor

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
args = ap.parse_args()

# Using CSRT Tracker : Discriminative Correlation Filter (with Channel and Spatial)

tracker = cv2.TrackerCSRT_create()

#OSNet Feature Extractor
extractor = FeatureExtractor(
	model_name='osnet_x1_0',
	 model_path='model_weights/model.pth.tar-250',
	 device = 'cuda'
)


# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

query_features = []

if __name__ == "__main__":

    vs = cv2.VideoCapture(args.video)

    #Make directory for storing Query Pedestrain
    path = "Query Person"
    os.makedirs(path)
    os.chdir(path)

    # loop over frames from the video stream
    while True:
        ret, frame = vs.read()

        # check to see if we are currently tracking an object
        if initBB is not None:
            i = 0
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
                croppedImg = frame[y:(y+h+1), x:(x+w+1)]
                filename = "Query_Person-1" +str(i)+ ".jpg"
                cv2.imwrite(filename , croppedImg )
                i += 1

				# getting features from OSnet
                feature = extractor(croppedImg)

                # Storing features in array
                f = features.cpu().numpy()
                query_features.append(numpy.append(f))

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    	# if the 's' key is selected, we are going to "select" a bounding
    	# box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)

        # if the `q` key is pressed,
        # break from the loop
        elif key == ord("q"):
			df = pd.DataFrame({'Query Features' : query_features})
			df.to_CSV('query_features.csv', index = False)
            break

    # go to parent directory
    os.chdir(os.path.dirname(os.getcwd()))

    vs.release()
    cv2.destroyAllWindows()
