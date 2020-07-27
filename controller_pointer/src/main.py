"""Gaze pointer."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import cv2
import logging

from argparse import ArgumentParser
from base_pointer import BasePointer
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController
from input_feeder import InputFeeder
from sys import platform

CODEC = cv2.VideoWriter_fourcc('M','J','P','G')

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m1", "--model1", required=True, type=str,
                        help="Path to an xml file with a trained model1.")
    parser.add_argument("-m2", "--model2", required=True, type=str,
                        help="Path to an xml file with a trained model2.")
    parser.add_argument("-m3", "--model3", required=True, type=str,
                        help="Path to an xml file with a trained model3.")
    parser.add_argument("-m4", "--model4", required=True, type=str,
                        help="Path to an xml file with a trained model4.")
    parser.add_argument("-i", "--input_file", required=True, type=str,
                        help="Path to video file")
    parser.add_argument("-t", "--input_type", required=True, type=str,
                        help="video, cam")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU")
                             #output of intermediate models
    parser.add_argument("-o", "--output_intermediate_model", type=str, default="true",
                        help="Outputs of intermediate models")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")

    return parser

def crop_face(coords, frame, output_intermediate_model):
    delta_y = coords[0][3] - coords[0][1]
    delta_x = coords[0][2] - coords[0][0]
    frame = frame[coords[0][1]:coords[0][1]+delta_y, coords[0][0]:coords[0][0]+delta_x]
    if output_intermediate_model == 'true':
        cv2.imwrite('output_image_crop_face.jpg', frame)
    return frame

def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """

    #if args.input == 'cam':
    #    args.input = 0
    output_intermediate_model = args.output_intermediate_model

    ### TODO: Handle the input stream ###
    feed=InputFeeder(input_type=args.input_type, input_file=args.input_file)
    cap = feed.load_data()
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))

    # Initialise the class
    try:
        infer_network_face_detection = BasePointer()
        infer_network_head_pose_estimation = BasePointer()
        infer_network_landmarks_regression_retail = BasePointer()
        infer_network_gaze_estimation = GazeEstimation()
    except:
        logging.error("Error in initializing models")
        exit(1)
    ### TODO: Load the model through `infer_network_face_detection` ###
    try:
        start_loading_time_face_detection = time.time()
        infer_network_face_detection.load_model(args.model1, args.device)
        load_model_face_detection_time_taken = time.time() - start_loading_time_face_detection

        start_loading_time_head_pose_estimation = time.time()
        infer_network_head_pose_estimation.load_model(args.model2, args.device)
        load_model_head_pose_estimation_time_taken = time.time() - start_loading_time_head_pose_estimation

        start_loading_time_landmarks_regression_retail = time.time()
        infer_network_landmarks_regression_retail.load_model(args.model3, args.device)
        load_model_landmarks_regression_retail_time_taken = time.time() - start_loading_time_landmarks_regression_retail

        start_loading_time_gaze_estimation = time.time()
        infer_network_gaze_estimation.load_model(args.model4, args.device)
        load_model_gaze_estimation_time_taken = time.time() - start_loading_time_gaze_estimation
    except:
        logging.error("Error in loading the models")
        exit(1)

    logging.debug("Loading times for facial detection : {} , landmark detection : {} , head pose detection : {} , gaze estimation : {} ".format(load_model_face_detection_time_taken, load_model_landmarks_regression_retail_time_taken, load_model_head_pose_estimation_time_taken, load_model_gaze_estimation_time_taken))

    if output_intermediate_model == 'true':
        out = cv2.VideoWriter('out.mp4', CODEC, fps, (width,height))

    total_time_taken_to_infer_inf_face_detection = 0
    total_time_taken_to_infer_landmarks_regression_retail = 0
    total_time_taken_to_infer_inf_head_pose_estimation = 0
    total_time_taken_to_infer_gaze_estimation = 0

    ### TODO: Loop until stream is over ###
    for batch in feed.next_batch():
        ### TODO: Read from the video capture ###

        flag, frame  = batch
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Start inference for face detection ###
        start_inf_face_detection = time.time()
        outputs_face_detection = infer_network_face_detection.predict(frame)
        time_taken_to_infer_inf_face_detection = time.time() - start_inf_face_detection
        coords, frame = infer_network_face_detection.preprocess_output_face_detection(outputs_face_detection, width, height, args.prob_threshold, frame)
        if output_intermediate_model == 'true':
            out.write(frame)

        frame_crop_face = crop_face(coords, frame, output_intermediate_model)

        start_inf_head_pose_estimation = time.time()
        outputs_head_pose_estimation = infer_network_head_pose_estimation.predict(frame_crop_face)
        time_taken_to_infer_inf_head_pose_estimation = time.time() - start_inf_head_pose_estimation

        yaw, pitсh, roll = infer_network_head_pose_estimation.preprocess_output_head_pose_estimation(outputs_head_pose_estimation, frame_crop_face)
        head_pose_angles = [yaw, pitсh, roll]

        if output_intermediate_model == 'true':
            cv2.putText(frame,("Yaw: " + str(int(yaw))), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.putText(frame,("Pitch: " + str(int(pitсh))), (100,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.putText(frame,("Roll: " + str(int(roll))), (100,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        height_crop_face = coords[0][3] - coords[0][1]
        width_crop_face = coords[0][2] - coords[0][0]

        start_inf_landmarks_regression_retail = time.time()
        outputs_landmarks_regression_retail = infer_network_landmarks_regression_retail.predict(frame_crop_face)
        time_taken_to_infer_landmarks_regression_retail = time.time() - start_inf_landmarks_regression_retail

        coord_landmarks_regression_retail = infer_network_landmarks_regression_retail.preprocess_output_landmarks_regression_retail(outputs_landmarks_regression_retail, width_crop_face, height_crop_face, args.prob_threshold, frame)
        center_left_eye = ((coords[0][0]+coord_landmarks_regression_retail[0]),coords[0][1]+coord_landmarks_regression_retail[1])
        center_right_eye = ((coords[0][0]+coord_landmarks_regression_retail[2]),coords[0][1]+coord_landmarks_regression_retail[3])


        xmin_left_eye = center_left_eye[0] - 30
        ymin_left_eye = center_left_eye[1] - 30
        xmax_left_eye = center_left_eye[0] + 30
        ymax_left_eye = center_left_eye[1] + 30
        xmin_right_eye = center_right_eye[0] - 30
        ymin_right_eye = center_right_eye[1] - 30
        xmax_right_eye = center_right_eye[0] + 30
        ymax_right_eye = center_right_eye[1] + 30

        frame_landmarks_regression_retail  = cv2.circle(frame,center_left_eye, 2, (0, 255, 0), thickness=3)
        frame_landmarks_regression_retail  = cv2.circle(frame,center_right_eye , 2, (0, 255, 0), thickness=3)
        box_left_eye = cv2.rectangle(frame, (xmin_left_eye, ymin_left_eye), (xmax_left_eye, ymax_left_eye), (0,255,0), 3)
        box_right_eye = cv2.rectangle(frame, (xmin_right_eye, ymin_right_eye), (xmax_right_eye, ymax_right_eye), (0,255,0), 3)
        if output_intermediate_model == 'true':
            out.write(frame_landmarks_regression_retail)

        ### TODO: Start inference for gaze estimation ###
        start_inf_gaze_estimation = time.time()
        outputs_gaze_estimation = infer_network_gaze_estimation.predict(box_left_eye, box_right_eye, head_pose_angles)
        time_taken_to_infer_gaze_estimation = time.time() - start_inf_gaze_estimation

        total_time_taken_to_infer_inf_face_detection = time_taken_to_infer_inf_face_detection + total_time_taken_to_infer_inf_face_detection
        total_time_taken_to_infer_landmarks_regression_retail = time_taken_to_infer_landmarks_regression_retail + total_time_taken_to_infer_landmarks_regression_retail
        total_time_taken_to_infer_inf_head_pose_estimation = time_taken_to_infer_inf_head_pose_estimation + total_time_taken_to_infer_inf_head_pose_estimation
        total_time_taken_to_infer_gaze_estimation = time_taken_to_infer_gaze_estimation + total_time_taken_to_infer_gaze_estimation

        arrow = 100
        g_x = int(outputs_gaze_estimation[0]*arrow)
        g_y = int(-(outputs_gaze_estimation[1])*arrow)

        frame = cv2.arrowedLine(frame, (center_left_eye), ((center_left_eye[0]+g_x), (center_left_eye[1]+g_y)), (0, 0, 255), 3)
        frame = cv2.arrowedLine(frame, (center_right_eye), ((center_right_eye[0]+g_x), (center_right_eye[1]+g_y)), (0, 0, 255), 3)

        if output_intermediate_model == 'true':
                out.write(frame)

        mouse_controler_pc = MouseController("high", "fast")
        mouse_controler_pc.move(outputs_gaze_estimation[0], outputs_gaze_estimation[1])

        if key_pressed == 27:
            break
    feed.close()

    logging.debug("total inference times for facial detection : {} , landmark detection : {} , head pose detection : {} , gaze estimation : {} ".format(total_time_taken_to_infer_inf_face_detection, total_time_taken_to_infer_landmarks_regression_retail, total_time_taken_to_infer_inf_head_pose_estimation, total_time_taken_to_infer_gaze_estimation))
    if output_intermediate_model == 'true':
        out.release()
    #cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    logging.basicConfig(filename="app.log", level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

    # Grab command line args
    args = build_argparser().parse_args()

    # Perform inference on the input stream
    infer_on_stream(args)

if __name__ == '__main__':
    main()
