"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import cv2

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
         
    def preprocess_output(self, outputs, width, height, threshold, frame, output_name):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = []
        for box in outputs[output_name][0][0]: # Output shape is 1x1x100x7
            conf = box[2]
            if conf >= threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                coord = [xmin, ymin, xmax, ymax]
                coords.append(coord)
                for coord in coords:
                    (xmin, ymin, xmax, ymax) = coord
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 3)
        return coords, frame
