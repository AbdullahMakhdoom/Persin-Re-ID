import numpy
import tensorflow as tf
import cv2
import time
import os
import pandas as pd
import argparse
import numpy as np
from torchreid.utils import FeatureExtractor

class PersonDetector:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":


    # Arguments Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type = str,
            help = "Surviellance video file path")
    parser.add_argument("-s", "--save", type = bool, default = True,
            help = "Save cropped Images of Pedestrain")
    args = parser.parse_args()

    # OSNet Feature Extractor
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
         model_path='model_weights/osnet_weights.tar-250',
         device = 'cuda'
    )

    model_path = 'model_weights/ssd_weights.pb'
    odapi = PersonDetector(path_to_ckpt=model_path)
    threshold = 0.7

    cap = cv2.VideoCapture(args.video)
    frame_counter = 0

    # Make a directory for saving cropped images of pedestrian
    gallery_path = "Gallery"
    os.makedirs(gallery_path, exist_ok=True)
    os.chdir(gallery_path)

    Features = []
    UTCtime = []

    while True:

        #make directory for each frame
        frame_counter += 1
        frame_path = "Frame" + str(frame_counter)
        os.makedirs(frame_path, exist_ok=True)
        os.chdir(frame_path)

        #reading and resizing frame
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.
        k = 0
        for i in range(len(boxes)):
            # Class 1 represents person
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                # save the cropped image of person
                filename = "Pedestrian-" + str(k) + ".jpg"
                croppedImg = img[box[0]:box[2]+1, box[1]:box[3]+1]
                cv2.imwrite(filename, croppedImg)

                # # getting features from OSnet
                feature = extractor(croppedImg)

                # Storing features in array
                f = features.cpu().numpy()
                Features.append(numpy.append(f))

                # Storing time in array
                UTCtime.append(time.time())

                k += 1
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)

        # go to parent directory
        os.chdir(os.path.dirname(os.getcwd()))

        if key & 0xFF == ord('q'):

            #saving pandas dataFrame of feature maps
            df = pd.DataFrame({'Features' : Features,
                               'UTCtime' : UTCtime})
            df.to_CSV('gallery.csv', index = False)
            break
