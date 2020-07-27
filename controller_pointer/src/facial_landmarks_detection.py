"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import cv2

class FacialLandmarkDetection:
    '''
    Class for the Face Detection Model.
    '''

    def preprocess_output(self, outputs, width, height, threshold, frame, output_name):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        x0 = outputs[output_name][0][0]*width
        y0 = outputs[output_name][0][1]*height
        x1 = outputs[output_name][0][2]*width
        y1 = outputs[output_name][0][3]*height
        x2 = outputs[output_name][0][4]*width
        y2 = outputs[output_name][0][5]*height
        x3 = outputs[output_name][0][6]*width
        y3 = outputs[output_name][0][7]*height
        x4 = outputs[output_name][0][8]*width
        y4 = outputs[output_name][0][9]*height
        coords = []
        coords = [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4]
        #cv2.imwrite('output_image.jpg', frame)


        #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 3)
        return coords
