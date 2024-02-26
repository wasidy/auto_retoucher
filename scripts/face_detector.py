from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import numpy as np
from scripts.imageutils import get_custom_tile, smooth_mask


class Face():
    def __init__(self, image, mask, coords):
        self.image = image
        self.mask = mask
        self.coords = coords


class FaceDetector():
    ''' Face detector. Arg for init - clip segmentation pipeline
        detect_faces returns combined mask and list of Face class (image, mask, coords) '''

    def __init__(self, clip_seg):
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        self.model = YOLO(model_path)
        self.clip_seg = clip_seg

    def detect_faces(self,
                     image,
                     expand_value=1.4,
                     threshold=127,
                     mask_smooth=10,
                     mask_expand=10):

        output = self.model(image, verbose=False)
        results = Detections.from_ultralytics(output[0])
        combined_mask = np.zeros(image.shape[0:2], dtype=np.uint8)

        faces = []
        for i in results:
            faces.append(i[0][0:4])

        detected_faces = []        
        for face in faces:
            x1, y1, x2, y2 = (round(x) for x in face[0:4])
            width = x2-x1
            height = y2-y1
            tile_size = int(max(width, height)*expand_value)
            center_x = int(x1+width/2)
            center_y = int(y1+height/2)
            face, fx1, fy1, fx2, fy2 = get_custom_tile(image, tile_size, (center_x, center_y))
            face_mask = self.clip_seg.predict(image=face, prompt='face', threshold=threshold)
            if face_mask.max()>0:
                face_mask = smooth_mask(face_mask, mask_smooth, mask_expand)
                detected_faces.append(Face(image[fy1:fy2, fx1:fx2, :], face_mask, (fx1,fy1,fx2,fy2)))
                combined_mask[fy1:fy2, fx1:fx2] = ((combined_mask[fy1:fy2, fx1:fx2] == 255) | (face_mask==255))*255
                #combined_mask[fy1:fy2, fx1:fx2] += face_mask  # Wrong! Overlapping masks
        return combined_mask, detected_faces
