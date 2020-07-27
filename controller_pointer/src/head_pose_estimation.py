"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import cv2

class HeadPoseEstimation:
        '''
        Class for the Face Detection Model.
        '''
        def preprocess_output(self, outputs, frame):
            '''
            Before feeding the output of this model to the next model,
            you might have to preprocess the output. This function is where you can do that.
            '''
            #logging.warning('%s before you %s', p)
            font = cv2.FONT_HERSHEY_SIMPLEX
            p = outputs["angle_p_fc"][0][0]
            r = outputs["angle_r_fc"][0][0]
            y = outputs["angle_y_fc"][0][0]
            #text1 = "Head pose estimation " + "p: " + str(int(p)) + " r: " + str(int(r)) + " y: " + str(int(y))
            #cv2.putText(frame, text1, (5, 5 ), font, 1, (0, 255, 0), 3)
            #print("Result for P {} R {} Y {}".format(int(p), int(r), int(y)))
            return (y, p, r)
