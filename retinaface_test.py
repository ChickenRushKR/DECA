from retinaface import RetinaFace
import numpy as np
from skimage.io import imread, imsave
img_path = "TestSamples/examples/LeonardoDiCaprio.jpg"
resp = RetinaFace.detect_faces(img_path=img_path)
print(resp)

class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(img_path)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'


image = np.array(imread(img_path))
fan = FAN()
res = fan.run(image)
print(res)
