from asyncore import file_dispatcher
import io
import os
from pyexpat import XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE
import torch
import json
import numpy as np
import six
import sys
import time
import timeit
import glob
import math
from IPython.display import display
from six import BytesIO
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import cv2
import tensorflow as tf
from logging import getLogger

print(tf.__version__)  # for Python 3
from google.protobuf import text_format
from string_int_label_map_pb2 import StringIntLabelMap

from Inference.images import legacy_fetch_images, gather_api_image
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum
from typing import List, Optional, Dict
from uuid import uuid4

from Inference import serialization
from Inference.envvar import EnvConfig
from Inference.images import find_image, find_single_image_regular_model
from Inference.queueing import ApiVersionOne, ApiVersionTwo

print("inference imports done")
RESOLUTION = 1024
CAM_TO_SUFFIX_MAP = {"": ""}
logger = getLogger(__name__)


class InferenceResultType(str, Enum):
    SUCCESSFUL_DETECTION = "SUCCESSFUL_DETECTION"
    NO_DETECTION = "NO_DETECTION"
    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"
    IMAGE_INVALID_SIZE = "IMAGE_INVALID_SIZE"


class DetectionType(str, Enum):
    SUCCESSFUL_DETECTION = "SUCCESSFUL_DETECTION"
    ROI_OUTSIDE_REGION = "ROI_OUTSIDE_REGION"
    SUCCESSFUL_LEGACY_DETECTION = "SUCCESSFUL_LEGACY_DETECTION"


@dataclass_json
@dataclass
class Detection:
    detection_type: DetectionType
    box: List
    roi: List
    probability: float
    label: str
    image_path: str
    tile_num: int = -1
    next_image_path: str = ""
    side: str = ""
    image: Image = None
    sub_detections: List["Detection"] = field(default_factory=list)
    datapoints: Dict = field(default_factory=dict)
    comment : str = ""

@dataclass_json
@dataclass
class InferenceResult:
    result_type: InferenceResultType
    detections: List[Detection]
    elapsed: float
    positive_detections: Optional[List[Detection]]
    id: str = field(default_factory=lambda: str(uuid4()))


class TFModel:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        """Load the PyTorch model into memory"""
        model = torch.hub.load('./yolov5', 'custom', './yolov5/runs/train/missing_plug_door_roller/weights/best.pt', source='local')
        return model

    def detect_image(
        self,
        model,
        tile_num: int,
        image: np.ndarray,
        image_path: str = "",
        min_confidence: float = 0.0,
        check_image_ratio: bool = True,
    ) -> List[Detection]:
        image = Image.fromarray(image)
        image_width, image_height = image.size

        # Detections
        result = model.model(image)      

        # Confidence
        confs_roi = result.pandas().xyxy[0].confidence

        # ROI
        x_mins_roi = result.pandas().xyxy[0].xmin
        y_mins_roi = result.pandas().xyxy[0].ymin
        x_maxs_roi = result.pandas().xyxy[0].xmax
        y_maxs_roi = result.pandas().xyxy[0].ymax

        # Build detection obj
        detections: List[Detection] = []
        draw = ImageDraw.Draw(image)
        logger.info('completely raw detections:')
        for conf, x_min, y_min, x_max, y_max in zip(confs_roi, x_mins_roi, y_mins_roi, x_maxs_roi, y_maxs_roi):
            logger.info(f'ROI: {x_min} {y_min} {x_max} {y_max}')
            # Exclude detections
            if conf < 0.15:
                continue

            # Box
            x_min_box = x_min / image_width
            y_min_box = y_min / image_height
            x_max_box = x_max / image_width
            y_max_box = y_max / image_height
            a_box = [y_min_box, x_min_box, y_max_box, x_max_box]
            logger.info(f'BOX: {a_box}')
            # ROI
            x_min_box = int(x_min)
            y_min_box = int(y_min)
            x_max_box = int(x_max)
            y_max_box = int(y_max)
            a_roi = [x_min_box, y_min_box, x_max_box, y_max_box]

            # Is roller too long?
            if a_box[3] - a_box[1] > .3:
                continue
            # Is roller too tall?
            if a_box[2] - a_box[0] > .2:
                continue
            # Does roller appear too low?
            if a_box[2] > .8:
                continue
            # Does roller appear too high?
            if a_box[0] < .08:
                continue

            # Draw detections on image
            y_min_roller = y_min
            x_min_roller = x_min
            y_max_roller = y_max
            x_max_roller = x_max
            draw.rectangle(((x_min_roller, y_min_roller), (x_max_roller, y_max_roller)), outline="red", width=6)

            if a_roi[0] < image_width / 2:
                side = "left"
            else:
                side = "right"

            detections.append(
                Detection(
                    detection_type=DetectionType.SUCCESSFUL_DETECTION,
                    box=a_box,
                    roi=a_roi,
                    side=side,
                    probability=conf,
                    label='Camel Roller',
                    image_path=image_path,
                    image=None,
                    tile_num=tile_num,
                )
            )

        sorted_detections = []
        sorted_detections = sorted(detections, key=lambda x: x.box[1])
        return sorted_detections

class InferenceScript:
    def gather_detections(self, timestamp: float, car_id: str) -> InferenceResult:
        pass


class MissingPlugDoorRollerInference(InferenceScript):
    def __init__(self, env: EnvConfig):
        self.env = env
        logger.info("Loading the missing plug door roller detector...")
        self.missing_plug_door_roller_model = TFModel()

    def roi_to_pixels(box: List[float], height: int, width: int):
        x_1 = np.float64(box[1] * width)
        y_1 = np.float64(box[0] * height)
        x_2 = np.float64(box[3] * width)
        y_2 = np.float64(box[2] * height)
        return [x_1, y_1, x_2, y_2]
    
    def tile_image(img):
        # convert PIL img to cv2 img
        img = np.array(img)
        img = img[:, :].copy()

        tiled_imgs = []
        height = img.shape[0]
        width = img.shape[1]
        tile_size = (width / 7, height)

        for i in range(int(math.ceil(height / (tile_size[1] * 1.0)))):
            for j in range(int(math.ceil(width / (tile_size[0] * 1.0)))):
                tiled_img = img[
                    int(tile_size[1] * i) : int(
                        min(tile_size[1] * i + tile_size[1], height)
                    ),
                    int(tile_size[0] * j) : int(
                        min(tile_size[0] * j + tile_size[0], width)
                    ),
                ]
                tiled_imgs.append(tiled_img)
        return width / 7, height, tiled_imgs    

    def gather_detections( self, timestamp: float, car_id: str, request=None, image_path=None):
        tf_run_start_at = timeit.default_timer()
        images = []
        image_paths = []

        """
            Test only images
        """
        # path_to_test_images = 'images/bent_brake_beams/test_cases/'
        # # new_width = 15000
        # # new_height = 2048
        # for filename in os.listdir(path_to_test_images):
        #     if filename.endswith(".jpg"):
        #         path_to_test_image = path_to_test_images + filename
        #         print('opening image at: ' + str(path_to_test_image))
        #         image = Image.open(path_to_test_image)
        #         # image = image.resize((new_width, new_height), Image.ANTIALIAS)
        #         images.append(image)
        #         path_to_output_image = 'saved_images/' + filename
        #         image_paths.append(path_to_output_image)
        """
            Prod only images
        """
        if type(request) == ApiVersionOne:
            for camera, suffix in CAM_TO_SUFFIX_MAP.items():
                image_path, image = legacy_fetch_images(
                    self.env,
                    timestamp=request.timestamp,
                    camera=camera,
                    car_id=request.car_id,
                    resolution=RESOLUTION,
                    suffix=suffix
                )
                image_paths.append(image_path)
                images.append(image)
        elif type(request) == ApiVersionTwo:
            for entry in request.algorithms[0].images:
                camera = "SPA1" if "SPA1" in entry.image_url else "SPB1"
                image_path, image = gather_api_image(
                    entry=entry,
                    resolution=RESOLUTION,
                    env=self.env,
                    timestamp=request.timestamp,
                    car_id=request.car_id,
                    camera=camera,
                    cam_to_suffix_map=CAM_TO_SUFFIX_MAP
                )
                image_paths.append(image_path)
                images.append(image)

        if len(images) == 0:
            raise Exception(
                f"Could not find any images for: ts - {timestamp}, carid - {car_id}"
            )

        """
            Save transformed images to dir
        """
        missing_plug_door_roller_detections = []
        for (image, image_path) in zip(images, image_paths):
            # Convert from np array to PIL image - prod only
            image = Image.fromarray(image)
            new_width = 15000
            new_height = 2048
            image = image.resize((new_width, new_height), Image.ANTIALIAS)

            # Image dims
            img_width, img_height = image.size

            # Crop image
            if img_width > 10000:
                area = (3100, 900, 11800, 1700)
                image = image.crop(area)

            # Tile images
            (tile_width, tile_height, tiled_images_np) = MissingPlugDoorRollerInference.tile_image(image)

            # Get detections
            raw_roller_detections = []
            tile_num = 1
            for idx, tiled_img_np in enumerate(tiled_images_np):
                raw_roller_detections = raw_roller_detections + self.missing_plug_door_roller_model.detect_image(image=tiled_img_np, tile_num=tile_num, model=self.missing_plug_door_roller_model)
                tile_num = tile_num + 1

            """
                Identify positive detections
            """
            # All boxes for true rollers in current image
            true_rollers = []
            # iterator for raw_roller_detections
            raw_roller_cnt = 1
            # Is current roller left/right part of clip?
            clp_1_2, clp_2_2 = False, False
            # Is there partial evidence of current/previous roller being clipped?
            ptntl_clip_crnt, ptntl_clip_prev = False, False

            # ID the true rollers
            raw_boxes = []
            for raw_roller_detection in raw_roller_detections:
                raw_boxes.append(raw_roller_detection.box)
                if raw_roller_detection.tile_num == 1:
                    ptntl_clip_prev = False

                # Instantiate current, previous and next boxes
                (y_min_crnt, x_min_crnt, y_max_crnt, x_max_crnt,) = raw_roller_detection.box
                if raw_roller_cnt < len(raw_roller_detections):
                    (y_min_next, x_min_next, y_max_next, x_max_next,) = raw_roller_detections[raw_roller_cnt].box
                if raw_roller_cnt > 0:
                    (y_min_prev, x_min_prev, y_max_prev, x_max_prev,) = raw_roller_detections[raw_roller_cnt - 2].box

                # Instantiate + print roller info
                probability = raw_roller_detection.probability
                tile_num = raw_roller_detection.tile_num

                # Instantiate + print roller spatial info
                rlr_len = x_max_crnt - x_min_crnt
                min_rlr_len = 0.75 * 0.165
                dtcn_side = "middle"
                if x_min_crnt > 0.8:
                    dtcn_side = "right"
                elif x_min_crnt < 0.2:
                    dtcn_side = "left"

                print('image path: ' + image_path)
                print("Tile #" + str(tile_num))
                # print("Roller detection #" + str(raw_roller_cnt))
                print("Probability #" + str(probability))
                print("Box: " + str(raw_roller_detection.box))
                print('roller length: ' + str(rlr_len))
                """
                    Very long set of conditionals which loads a 2-d list
                    with only "true" roller boxes in image
                """
                # prev box not clipped
                if ptntl_clip_prev == True and rlr_len < min_rlr_len and dtcn_side != 'left' :
                    print("Previous object wasn't clipped.")
                    print(f"Saving true coords for roller at: {y_min_prev, x_min_prev, y_max_prev, x_max_prev}")
                    true_rollers.append([y_min_prev,[x_min_prev, ptntl_clip_tile_iter],y_max_prev,[x_max_prev, ptntl_clip_tile_iter], 1]) 

                # prev + crnt box not clipped 
                if ptntl_clip_prev == True and rlr_len > min_rlr_len:
                    print("Previous object wasn't clipped and neither is current one.")
                    print(f"Saving true coords for roller at: {y_min_prev, x_min_prev, y_max_prev, x_max_prev}")
                    true_rollers.append([y_min_prev,[x_min_prev, ptntl_clip_tile_iter],y_max_prev,[x_max_prev, ptntl_clip_tile_iter], 2])

                # prev box - 1/2 clip
                if (ptntl_clip_prev == True and dtcn_side == "left"and rlr_len < min_rlr_len):
                    print("Previous object was clipped.")
                    print(f"Saving true coords for roller at: {y_min_crnt, x_min_prev, y_max_crnt, x_max_crnt}")
                    true_rollers.append([y_min_crnt,[x_min_prev, ptntl_clip_tile_iter],y_max_crnt,[x_max_crnt, tile_num], 3])

                else:
                    # 1/2 clip - right
                    if dtcn_side == "right" and rlr_len < min_rlr_len:
                        clp_1_2 = True
                        print("1/2 clip detected.")
                        print(f"saving true label coords for label at: {y_min_crnt, x_min_crnt, y_max_crnt, x_max_next}")
                        true_rollers.append([y_min_crnt,[x_min_crnt, tile_num],y_max_crnt,[x_max_next, tile_num + 1], 4])

                    # 2/2 clip - do nothing
                    elif clp_2_2 == True:
                        print("2/2 clip detected.")
                        print("already appended object, standing by.")
                        pass

                    # potential 1/2 clip - right 
                    elif dtcn_side == "right" and rlr_len > min_rlr_len:
                        ptntl_clip_crnt = True
                        ptntl_clip_tile_iter = tile_num
                        print("Potential clip detected...")
                        print("More info needed...")

                    # non-clipped roller - middle
                    elif dtcn_side == "middle":
                        print("Object detected.")
                        print(f"Saving true label coords for label at: {y_min_crnt, x_min_crnt, y_max_crnt, x_max_crnt}")
                        true_rollers.append([y_min_crnt,[x_min_crnt, tile_num],y_max_crnt,[x_max_crnt, tile_num], 5])

                    # non-clipped roller - left 
                    elif dtcn_side == "left" and rlr_len > min_rlr_len:
                        print("left side non-clip detected.")
                        print(f"Saving true label coords for label at: {y_min_crnt, x_min_crnt, y_max_crnt, x_max_crnt}")
                        true_rollers.append([y_min_crnt, [x_min_crnt, tile_num], y_max_crnt, [x_max_crnt, tile_num], 6])
                    # roller was partially detected and appears smaller than is
                    elif rlr_len > 0.07:
                        print("Roller appears smaller than expected")
                        print(f"Saving true label coords for label at: {y_min_crnt, x_min_crnt, y_max_crnt, x_max_crnt}")
                        true_rollers.append([y_min_crnt, [x_min_crnt, tile_num], y_max_crnt, [x_max_crnt, tile_num], 7])
                    else:
                        print("uh-oh!: unidentified use case reached.")
                        pass

                clp_2_2 = False
                clp_2_2 = clp_1_2
                clp_1_2 = False
                ptntl_clip_prev = False
                ptntl_clip_prev = ptntl_clip_crnt
                ptntl_clip_crnt = False
                raw_roller_cnt = raw_roller_cnt + 1
                print('true roller current count: ' + str(len(true_rollers)))

            # non-clipped roller b/c no rollers left
            if ptntl_clip_prev:
                print(f"Reached end of detections, roller not clipped: {y_min_crnt, x_min_crnt, y_max_crnt, x_max_crnt}")
                true_rollers.append([y_min_crnt,[x_min_crnt, tile_num],y_max_crnt,[x_max_crnt, tile_num], 8])
                ptntl_clip_prev = False

            # Re-assign the CR coordinates from tiled-image box to full-sized-image roi
            for box in true_rollers: 
                box[0] = int(box[0] * tile_height)
                box[2] = int(box[2] * tile_height)
                box[1] = int((box[1][1] - 1) * tile_width + (box[1][0] * tile_width))
                box[3] = int((box[3][1] - 1) * tile_width + (box[3][0] * tile_width))

            """
                Post label-stitching Business Rules
            """

            if true_rollers:
                ## Remove detections with abnormal heights
                # Extract y_min vals 
                y_mins = []
                for roller in true_rollers:
                    y_min = roller[0]
                    y_mins.append(y_min)

                # Vote which label is near most amt of other labels
                y_min_votes = []
                for y1 in y_mins:
                    # Take vote off for self
                    y1_votes = -1
                    y1_range = range(y1 - 70, y1 + 70)
                    # Add vote to y1 for every y2 that is within range
                    for y2 in y_mins:
                        if y2 in y1_range:
                            y1_votes = y1_votes + 1
                    y_min_votes.append(y1_votes)
                    
                # Get height of 'winner' label
                true_roller_height = y_mins[y_min_votes.index(max(y_min_votes))]

                # Exclude labels that are not close enough to winners roller height
                temp_rollers = []
                inclusion_range = range(true_roller_height - 70, true_roller_height + 70)
                for roller in true_rollers:
                    if roller[0] not in inclusion_range:
                        continue
                    temp_rollers.append(roller)
                true_rollers = temp_rollers

                ## Determine indexes of overlapping detections
                indexes_to_delete = []
                for id_a, roller_a in enumerate(true_rollers):
                    x_min_a, x_max_a = roller_a[1], roller_a[3]
                    width_a = x_max_a - x_min_a
                    for id_b, roller_b in enumerate(true_rollers):
                        # dont compare same detections
                        if id_a == id_b:
                            continue
                        x_min_b, x_max_b = roller_b[1], roller_b[3]
                        width_b = x_max_b - x_min_b

                        # if roller_a starts within boundaries of roller_b
                        if x_min_a >= x_min_b and x_min_a <= x_max_b:
                            # record detection index with the shorter width
                            if width_a < width_b:
                                indexes_to_delete.append(id_a)
                            else:
                                indexes_to_delete.append(id_b)

                # Remove overlapping detections                
                for index_to_delete in indexes_to_delete:                 
                    true_rollers.pop(index_to_delete)

            true_roller_cnt = len(true_rollers)
            # Detection = True
            if true_roller_cnt == 1 or true_roller_cnt == 3:
                logger.info("DETECTION ALERT")
                logger.info('CR count: ' + str(true_roller_cnt))
                logger.info('raw boxes:')
                for b in raw_boxes:
                    logger.info(b)

                # CR area box vals
                width, height = image.size
                x_min_ra = width * .15
                y_min_ra = height * .20
                x_max_ra = width * .85
                y_max_ra = height * .80
                box_ra = [0.20, 0.15, 0.80, 0.85]
                roi_ra = [x_min_ra, y_min_ra, x_max_ra, y_max_ra]
                
                # Append CR area
                missing_plug_door_roller_detections.append(
                    Detection(
                        detection_type=DetectionType.SUCCESSFUL_DETECTION,
                        label="Missing Plug Door Roller",
                        image_path=image_path,
                        probability=1,
                        image=image,
                        box=box_ra,  
                        roi=roi_ra,     
                    )
                )

                # Create camel roller detections
                for roller in true_rollers:
                    logger.info('formatted boxes:')
                    logger.info(roller)
                    x_min_cr = roller[1]
                    y_min_cr = roller[0]
                    x_max_cr = roller[3]
                    y_max_cr = roller[2]
                    box_cr = [x_min_cr / width, y_min_cr / height, x_max_cr / width, y_max_cr / height]
                    roi_cr = [x_min_cr, y_min_cr, x_max_cr, y_max_cr]

                    # Append CRs
                    missing_plug_door_roller_detections.append(
                        Detection(
                            detection_type=DetectionType.SUCCESSFUL_DETECTION,
                            label="Camel Roller",
                            image_path=image_path,
                            probability=1,
                            image=image,
                            box=box_cr,  
                            roi=roi_cr,     
                        )
                    )
                """
                    -create detection result images
                    -test only
                """ 
                # # Image vals
                # font_fname = 'FreeSansBold.ttf'
                # font_size = 140
                # font = ImageFont.truetype(font_fname, font_size)
                # draw = ImageDraw.Draw(image)

                # # Roller area box vals
                # width, height = image.size
                # x_min = width * .15
                # y_min = height * .20
                # x_max = width * .85
                # y_max = height * .80

                # # Draw text
                # draw.text(((width / 20) * 7, height / 30), "Missing Plug Door Roller", font=font, fill='rgb(255, 0, 0)')

                # # Draw bounding box (roller area)
                # draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=13)

                # # Draw bounding box (camel rollers)
                # for roller in true_rollers:
                #     y_min_roller = roller[0]
                #     x_min_roller = roller[1]
                #     y_max_roller = roller[2]
                #     x_max_roller = roller[3]
                #     draw.rectangle(((x_min_roller, y_min_roller), (x_max_roller, y_max_roller)), outline="red", width=13)
                # # Save image
                # image.save(image_path)
            # Detection = False
            else:
                logger.info("NO DETECTION")
                logger.info('CR count: ' + str(true_roller_cnt))

                """
                    Test only
                """
                # # Image val
                # draw = ImageDraw.Draw(image)

                # # Draw bounding box (camel rollers)
                # for roller in true_rollers:
                #     y_min_roller = roller[0]
                #     x_min_roller = roller[1]
                #     y_max_roller = roller[2]
                #     x_max_roller = roller[3]
                #     draw.rectangle(((x_min_roller, y_min_roller), (x_max_roller, y_max_roller)), outline="red", width=13)
                # # Save image
                # image.save(image_path)

        if missing_plug_door_roller_detections:
            return InferenceResult(
                result_type=InferenceResultType.SUCCESSFUL_DETECTION,
                detections=missing_plug_door_roller_detections,
                elapsed=timeit.default_timer() - tf_run_start_at,
                positive_detections=missing_plug_door_roller_detections,
            )

        return InferenceResult(
            result_type=InferenceResultType.NO_DETECTION,
            detections=[],
            elapsed=timeit.default_timer() - tf_run_start_at,
            positive_detections=[],
    )


if __name__ == "__main__":
    pass


