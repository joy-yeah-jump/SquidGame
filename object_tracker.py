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

from moviepy.editor import *
import pygame
pygame.mixer.pre_init(44100, 16, 2, 4096)
pygame.init()
import threading

flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_string('framework', 'tf', 'tf, tflite, trt')
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('score', 0.50, 'score threshold')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
# flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
# flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416', 'path to weights file')

def play_op_audio(audio_ready, video_ready) :
    audio = AudioFileClip('OpeningMovie.mp4')
    audio_ready.set()
    video_ready.wait()
    audio.preview(fps=44100)

def play_op_video(audio_ready, video_ready) :
    video = VideoFileClip('OpeningMovie.mp4', audio=False).resize(width=1920)
    video_ready.set()
    audio_ready.wait()
    video.preview(fps=24)

def opening_video() :
    # http://zulko.github.io/blog/2013/09/19/a-basic-example-of-threads-synchronization-in-python/
    audio_ready = threading.Event()
    video_ready = threading.Event()

    pygame.display.set_caption('opening movie')
    threading.Thread(target=play_op_audio, args=(audio_ready, video_ready)).start()
    play_op_video(audio_ready, video_ready)

def intro_screen() :
    pygame.display.set_caption('init screen')
    screen = pygame.display.set_mode((1920, 1080))
    screen.fill([0, 0, 0])  # [0, 0, 0] = black
    bg = pygame.transform.scale(pygame.image.load("pic3.png"), (1920, 1080))
    screen.blit(bg, [0, 0])
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        TextSurf = pygame.font.Font('freesansbold.ttf', 180).render("SQUID GAME", True, [255, 0, 0])
        TextRect = TextSurf.get_rect()

        TextRect.center = (1920 / 2, 180 / 2)  # 1920 = width, 180 = font size
        screen.blit(TextSurf, TextRect)

        # left button
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        # w = width, h = height, c = coordinate, l = length
        b1_wc, b1_hc, b1_wl, b1_hl = 600, 450, 100, 50
        if b1_wc + b1_wl > mouse[0] > b1_wc and b1_hc + b1_hl > mouse[1] > b1_hc:
            pygame.draw.rect(screen, [127, 255, 127], (b1_wc, b1_hc, b1_wl, b1_hl))
            if click[0]:
                screen.fill([0, 0, 0])
                screen.blit(pygame.transform.scale(pygame.image.load('pic1.png'), (1920, 1080)), [0, 0])

                b3Surf = pygame.font.Font('freesansbold.ttf', 40).render("NOW LOADING...", True, [0, 255, 255])
                b3Rect = b3Surf.get_rect()

                b3Rect.center = (1920 - 180, 1080 - 20)
                screen.blit(b3Surf, b3Rect)

                pygame.display.update()
                break
        else:
            pygame.draw.rect(screen, [0, 255, 0], (b1_wc, b1_hc, b1_wl, b1_hl))

        b1Surf = pygame.font.Font('freesansbold.ttf', 20).render("GO!", True, [0, 0, 0])
        b1Rect = b1Surf.get_rect()

        b1Rect.center = ((b1_wc + (b1_wl / 2)), (b1_hc + (b1_hl / 2)))
        screen.blit(b1Surf, b1Rect)

        # right button
        b2_wc, b2_hc, b2_wl, b2_hl = 1200, 450, 100, 50
        pygame.draw.rect(screen, [255, 0, 0], (b2_wc, b2_hc, b2_wl, b2_hl))

        b2Surf = pygame.font.Font('freesansbold.ttf', 20).render("QUIT", True, [0, 0, 0])
        b2Rect = b2Surf.get_rect()

        b2Rect.center = ((b2_wc + (b2_wl / 2)), (b2_hc + (b2_hl / 2)))
        screen.blit(b2Surf, b2Rect)

        if b2_wc + b2_wl > mouse[0] > b2_wc and b2_hc + b2_hl > mouse[1] > b2_hc:
            if click[0]:
                pygame.quit()
                sys.exit()

        pygame.display.update()

def play_tagger_voice() :
    pygame.mixer.music.load('ko.mp3')
    pygame.mixer.music.play()

def get_go_forward_time() :
    threading.Thread(target=play_tagger_voice()).start()
    return 3

def get_tagger_time() :
    return 5

def person_track(vid, tracker, infer, encoder, surface) :
    frame_num = 0
    standard_time = time.time()
    coord_1st = list()
    area_percent_list = list()
    turn = False # True : tagger see you
    go_forward_time = get_go_forward_time()
    tagger_time = get_tagger_time()
    sec_color = [0, 255, 0] # green
    while True:
        return_value, frame = vid.read()
        try :
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except :
            print('cv2 cannot get frame. fuck')
            continue

        frame_num += 1
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (416, 416)) # 416 = input_size
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
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, 1.0, scores) # 1.0 = nms_max_overlap
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                pygame.quit()
                sys.exit()
                break

        import person_track_id as pti
        # update tracks
        for track in tracker.tracks:
            if not turn : continue
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            if len(coord_1st) < 1:
                coord_1st.append([int(track.track_id), int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            else :
                is_exist = False
                for item in coord_1st :
                    if item[0] == int(track.track_id) : is_exist = True
                if not is_exist : coord_1st.append([int(track.track_id), int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

            now = list()
            for item in coord_1st :
                if item[0] == int(track.track_id) : now = item

            # draw bbox(now coordinate) on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[0]) + 64, int(bbox[1])), color, -1)
            cv2.putText(frame, "id-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)

            # cv2.rectangle(frame, (now[1], now[2]), (now[3], now[4]), color, 2)
            # cv2.rectangle(frame, (now[1], now[2] - 30), (now[1] + 64, now[2]), color, -1)
            # cv2.putText(frame, str(now[0]) + "-id", (now[1], now[2] - 10), 0, 0.75, (255, 255, 255), 2)

            # red target circle
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            red = (255, 0, 0)
            cv2.circle(frame, center, 20, red, 2)
            cv2.circle(frame, center, 30, red, 2)
            cv2.line(frame, (center[0] - 15, center[1]), (center[0] - 40, center[1]), red, 2)
            cv2.line(frame, (center[0] + 15, center[1]), (center[0] + 40, center[1]), red, 2)
            cv2.line(frame, (center[0], center[1] - 15), (center[0], center[1] - 40), red, 2)
            cv2.line(frame, (center[0], center[1] + 15), (center[0], center[1] + 40), red, 2)

            # draw rectangle by first coordinate, calculate intersection area percentage
            if now :
                cv2.rectangle(frame, (int(now[1]), int(now[2])), (int(now[3]), int(now[4])), color, 2)
                cv2.rectangle(frame, (int(now[1]), int(now[2] - 30)), (int(now[1]) + 120, int(now[2])), color, -1)
                cv2.putText(frame, str(track.track_id) + '-first', (int(now[1]), int(now[2] - 10)), 0, 0.75, (255, 255, 255), 2)
                coord_now = bbox[:]
                inter_area = 0
                now_area = (coord_now[2] - coord_now[0]) * (coord_now[3] - coord_now[1])
                inter = [max(now[1], coord_now[0]), max(now[2], coord_now[1]), min(now[3], coord_now[2]), min(now[4], coord_now[3])]
                if inter[0] < inter[2] and inter[1] < inter[3] : inter_area = (inter[2] - inter[0]) * (inter[3] - inter[1])
                area_percent_list.append((track.track_id, inter_area / now_area * 100))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        if frame_num % 10 == 0:
            try:
                # print('Frame # : {:>4} | FPS : {:>5.2f} | {:>6.2f}% | {:>6.2f}sec'
                # .format(frame_num, fps, inter_area / now_area * 100, time.time() - standard_time))
                print('Frame # : {:>4} | FPS : {:>5.2f}'.format(frame_num, fps))
                for key, value in area_percent_list :
                    print('id : {:>3} | area : {:>6.2f}%'.format(key, value))
            except:
                print("nothing detected or error occured")

        area_percent_list = list()

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        surf = pygame.surfarray.make_surface(cv2.cvtColor((np.rot90(np.fliplr(result))), cv2.COLOR_BGR2RGB))
        if turn :
            surface.blit(pygame.transform.scale(surf, (1920, 1080)), (0, 0))
            if time.time() - standard_time > tagger_time : # tagger -> runner phase
                go_forward_time = get_go_forward_time()
                standard_time = time.time()
                turn = False
                sec_color = [0, 255, 0] # red -> green
        else :
            surface.blit(pygame.transform.scale(pygame.image.load("pic4.png"), (1920, 1080)), [0, 0])
            if time.time() - standard_time > go_forward_time : # runner -> tagger phase
                tagger_time = get_tagger_time()
                standard_time = time.time()
                turn = True
                sec_color = [255, 0, 0] # green -> red
                coord_1st = list()

        # for debug, timer show
        secSurf = pygame.font.Font('freesansbold.ttf', 180).render('{:>5.2f}'.format(time.time() - standard_time), True, sec_color)
        secRect = secSurf.get_rect()
        secRect.center = (180, 90)  # 1920 = width, 180 = font size
        surface.blit(secSurf, secRect)

        pygame.display.flip()

        # cv2.imshow("detecting window", result) # no webcam window
        # if cv2.waitKey(1) & 0xFF == ord('q'): break

def main(_argv):
    # opening phase
    opening_video()

    # intro phase
    intro_screen()

    # loading phase
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None # nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
    tracker = Tracker(metric) # initialize tracker

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    # load model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default'] # out = None

    # main game phase
    # begin video capture
    vid = cv2.VideoCapture(0) # 0 = int(video_path) = webcam
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # will get from webcam max constant?
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    pygame.display.set_caption("main game playing")
    surface = pygame.display.set_mode((1920, 1080)) # screen succeed?
    surface.blit(pygame.transform.scale(pygame.image.load("pic4.png"), (1920, 1080)), [0, 0])
    pygame.display.update()

    person_track(vid, tracker, infer, encoder, surface) # while video is running
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
