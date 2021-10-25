import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0: tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils

# from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg

# from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_string('framework', 'tf', 'tf, tflite, trt')
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
# flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416', 'path to weights file')



def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    vid = cv2.VideoCapture(int(video_path))

    out = None

    is_1st = False
    coord_1st = None
    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_num += 1
        # print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        import person_track_id as pti
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*8, int(bbox[1])), color, -1)
            cv2.putText(frame, "id-" + str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255),2)

            #red target circle
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            red = (255, 0, 0)
            cv2.circle(frame, center, 20, red, 2)
            cv2.circle(frame, center, 30, red, 2)
            cv2.line(frame, (center[0] - 15, center[1]), (center[0] - 40, center[1]), red, 2)
            cv2.line(frame, (center[0] + 15, center[1]), (center[0] + 40, center[1]), red, 2)
            cv2.line(frame, (center[0], center[1] - 15), (center[0], center[1] - 40), red, 2)
            cv2.line(frame, (center[0], center[1] + 15), (center[0], center[1] + 40), red, 2)

            if not is_1st : # id별로 잡을것
                coord_1st = bbox[:]
                is_1st = True
                print(coord_1st)

            # print(track.track_id) # id
            # print(track.to_tlbr()) # coords
            cv2.rectangle(frame, (int(coord_1st[0]), int(coord_1st[1])), (int(coord_1st[2]), int(coord_1st[3])), color, 2)
            cv2.rectangle(frame, (int(coord_1st[0]), int(coord_1st[1] - 30)), (int(coord_1st[0]) + (len(class_name) + len(str(track.track_id)))*12, int(coord_1st[1])), color, -1)
            cv2.putText(frame, str(track.track_id) + '-first', (int(coord_1st[0]), int(coord_1st[1]-10)), 0, 0.75, (255, 255, 255), 2)
            coord_now = bbox[:]
            now_area = (coord_now[2] - coord_now[0]) * (coord_now[3] - coord_now[1])
            inter_area = 0
            inter = [max(coord_1st[0], coord_now[0]), max(coord_1st[1], coord_now[1]), min(coord_1st[2], coord_now[2]), min(coord_1st[3], coord_now[3])]
            if inter[0] < inter[2] and inter[1] < inter[3] :
                inter_area = (inter[2] - inter[0]) * (inter[3] - inter[1])

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        if frame_num % 10 == 0 :
            try:
                print('Frame # : {:>4} | FPS : {:>5.2f} | {:>6.2f}%'.format(frame_num, fps, inter_area / now_area * 100))
            except :
                print("nothing detected or error occured")
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Output Video", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
