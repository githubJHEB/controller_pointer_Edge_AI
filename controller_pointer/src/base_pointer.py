"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import os
import cv2
from openvino.inference_engine import IENetwork, IECore
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarkDetection
class BasePointer:
    """
    Class for the Face Detection Model.
    """
    def __init__(self):
        """
        TODO: Use this to set your instance variables.
        """
        self.plugin = None
        self.network = None
        self.input_name = None
        self.output_name = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU"):
        """
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        """
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.plugin = IECore()
        self.network  = self.plugin.read_network(model=model_xml, weights=model_bin)

        self.exec_network = self.plugin.load_network(self.network, device)

        self.input_name=next(iter(self.network.inputs))
        self.input_shape=self.network.inputs[self.input_name].shape
        self.output_name=next(iter(self.network.outputs))
        self.output_shape=self.network.outputs[self.output_name].shape
        return

    def predict(self, image):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        p_frame = self.preprocess_input(image)
        input_dict={self.input_name:[p_frame]}
        #return self.exec_network.requests[0].outputs[self.output_name]
        result = self.exec_network.infer(input_dict)
        ##return self.exec_network.requests[0].outputs[self.output_name]
        return result

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output_face_detection(self, outputs, width, height, threshold, frame):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        face_detection = FaceDetection()

        coords = []
        coords, frame = face_detection.preprocess_output(outputs, width, height, threshold, frame, self.output_name)
        return coords, frame

    def preprocess_output_head_pose_estimation(self, outputs, frame):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        head_pose_estimation = HeadPoseEstimation()

        yaw, pitсh, roll = head_pose_estimation.preprocess_output(outputs, frame)
        return (yaw, pitсh, roll)

    def preprocess_output_landmarks_regression_retail(self, outputs, width, height, threshold, frame):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        landmarks_regression_retail = FacialLandmarkDetection()

        coords = []
        coords = landmarks_regression_retail.preprocess_output(outputs, width, height, threshold, frame, self.output_name)
        return coords
