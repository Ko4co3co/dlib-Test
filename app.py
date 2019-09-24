import sys 
import os 
import cv2 
import dlib 
import glob 
import numpy as np
from imutils.video import FPS
import datetime
#https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

def reshape_for_polyline(array):
        return np.array(array, np.int32).reshape((-1, 1, 2))

def VideoFrame(videos,detector,predictor):
    print(videos)
    past = datetime.datetime.now()
    cap = cv2.VideoCapture(videos)
    fps = FPS().start()
    idx = 0
        
   # fourcc = cv2.VideoWriter_fourcc(*'H264')
   # out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

    while cap.isOpened():
            ret, frame = cap.read() 
           # frame = cv2.resize(frame,None,fx =1 /4, fy =1 /4 )
            vid = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            vid = cv2.transpose(vid)
            vid = cv2.flip(vid,1)
            faces = detector(vid,1)
            if len(faces) == 1:
                for face in faces:
                    landmark = predictor(vid,face).parts()
                    landmarks = [[p.x, p.y] for p in landmark]
                    jaw = reshape_for_polyline(landmarks[0:17])
                    left_eyebrow = reshape_for_polyline(landmarks[22:27])
                    right_eyebrow = reshape_for_polyline(landmarks[17:22])
                    nose_bridge = reshape_for_polyline(landmarks[27:31])
                    lower_nose = reshape_for_polyline(landmarks[30:35])
                    left_eye = reshape_for_polyline(landmarks[42:48])
                    right_eye = reshape_for_polyline(landmarks[36:42])
                    outer_lip = reshape_for_polyline(landmarks[48:60])
                    inner_lip = reshape_for_polyline(landmarks[60:68])

                    color = (255,255,255)
                    thickness = 2
                    cv2.polylines(vid, [jaw], False, color, thickness)
                    cv2.polylines(vid, [left_eyebrow], False, color, thickness)
                    cv2.polylines(vid, [right_eyebrow], False, color, thickness)
                    cv2.polylines(vid, [nose_bridge], False, color, thickness)
                    cv2.polylines(vid, [lower_nose], True, color, thickness)
                    cv2.polylines(vid, [left_eye], True, color, thickness)
                    cv2.polylines(vid, [right_eye], True, color, thickness)
                    cv2.polylines(vid, [outer_lip], True, color, thickness)
                    cv2.polylines(vid, [inner_lip], True, color, thickness)
                    
            fps.update()
        
            cv2.imshow('file',vid)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cv2.destroyAllWindows()
    out.release()
    fps.close()
    cap.release()
def ImageFrame(image,detector,predictor):
    
    img = cv2.imread(image,cv2.IMREAD_COLOR)
    faces = detector(img, 1)
#    shape = predictor(img, d)
    black = np.zeros(img.shape)
    output = [img,black]
    print(len(faces))
    if len(faces) >= 1:
        for face in faces:
            landmark = predictor(img,face).parts()
            landmarks = [[p.x, p.y] for p in landmark]
            jaw = reshape_for_polyline(landmarks[0:17])
            left_eyebrow = reshape_for_polyline(landmarks[22:27])
            right_eyebrow = reshape_for_polyline(landmarks[17:22])
            nose_bridge = reshape_for_polyline(landmarks[27:31])
            lower_nose = reshape_for_polyline(landmarks[30:35])
            left_eye = reshape_for_polyline(landmarks[42:48])
            right_eye = reshape_for_polyline(landmarks[36:42])
            outer_lip = reshape_for_polyline(landmarks[48:60])
            inner_lip = reshape_for_polyline(landmarks[60:68])
            color = (255,255,255)
            thickness = 2
            filename_idx = 1
            for i in output:
                cv2.polylines(i, [jaw], False, color, thickness)
                cv2.polylines(i, [left_eyebrow], False, color, thickness)
                cv2.polylines(i, [right_eyebrow], False, color, thickness)
                cv2.polylines(i, [nose_bridge], False, color, thickness)
                cv2.polylines(i, [lower_nose], True, color, thickness)
                cv2.polylines(i, [left_eye], True, color, thickness)
                cv2.polylines(i, [right_eye], True, color, thickness)
                cv2.polylines(i, [outer_lip], True, color, thickness)
                cv2.polylines(i, [inner_lip], True, color, thickness) 
                filepath = './Output/'+'test' + str(filename_idx) + '.jpg'
               
                cv2.imwrite(filepath,i)
                filename_idx +=1 

if __name__ == '__main__':
    VIDEO_PATH = './Video'
    IMG_PATH = './Image'
    video = glob.glob(VIDEO_PATH + '/*')
    images = glob.glob(IMG_PATH + './*')
    face_landmark_shape_file = glob.glob('./model/*.dat')
    if face_landmark_shape_file != None:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(face_landmark_shape_file[0])     
        #ImageFrame(images[1],detector,predictor) 
        VideoFrame(video[0],detector,predictor)
    else:
        raise print('Model is Not Exist')
  