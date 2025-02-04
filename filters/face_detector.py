import cv2
import dlib

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_face_landmarks(image):
    """
    Detect faces and return facial landmarks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    landmarks_list = []
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_list.append([(p.x, p.y) for p in landmarks.parts()])
    
    return landmarks_list, faces
