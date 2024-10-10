import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
from collections import OrderedDict


class Face_utilities():
    def __init__(self, face_width = 200):
        # Bu metot, sınıfın başlangıç durumunu yapılandırır. 
        # Yüz algılama, yüz şekli tahmini, yaş ve cinsiyet tespiti için kullanılan modelleri yükler. 
        # Ayrıca yüz hizalama işlemi için gerekli olan bazı parametreleri ayarlar.
        self.detector = None
        self.predictor = None 
        self.age_net = None
        self.gender_net = None
        
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
        self.gender_list = ['Male', 'Female']
        
        self.desiredLeftEye=(0.35, 0.35)
        self.desiredFaceWidth = face_width
        self.desiredFaceHeight = None
        
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
        #For dlib’s 81-point facial landmark detector:
        self.FACIAL_LANDMARKS_81_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17)),
            ("forehead", (17, 27))
        ])
        
        self.FACIAL_LANDMARKS_5_IDXS = OrderedDict([
            ("right_eye", (2, 3)),
            ("left_eye", (0, 1)),
            ("nose", (4))   
        ])
        
        #last params
        self.last_age = None
        self.last_gender = None
        self.last_rects = None
        self.last_shape = None
        self.last_aligned_shape = None

    
    def face_alignment(self, frame, shape):
        # Verilen bir yüz ve yüz şekli (81 veya 5 nokta) üzerinde dikey eksene göre hizalama yapar. 
        # Bu işlem, yüzü dikey eksende hizalayarak daha tutarlı sonuçlar elde etmeyi amaçlar.
        if (len(shape)==81):
			# extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = self.FACIAL_LANDMARKS_81_IDXS["left_eye"]
            (rStart, rEnd) = self.FACIAL_LANDMARKS_81_IDXS["right_eye"]
        else:
            (lStart, lEnd) = self.FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = self.FACIAL_LANDMARKS_5_IDXS["right_eye"]
        
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))
        
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned_face = cv2.warpAffine(frame, M, (w, h),
            flags=cv2.INTER_CUBIC)
        
        if(len(shape)==81):
            shape = np.reshape(shape,(81,1,2))
        else:
            shape = np.reshape(shape,(5,1,2))
        aligned_shape = cv2.transform(shape, M)
        aligned_shape = np.squeeze(aligned_shape)        
        return aligned_face,aligned_shape
    
    def face_detection(self, frame):
        # Verilen bir karedeki yüzleri algılar ve dikdörtgen olarak sınırlayıcı kutular oluşturur.
        if self.detector is None:
            self.detector = dlib.get_frontal_face_detector()
        
        if frame is None:
            return
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)        
        return rects
    
    def age_gender_detection(self, face):
        # Verilen bir yüzden yaş ve cinsiyet tahmini yapar. 
        # Bu işlem, önceden eğitilmiş derin öğrenme modelleri kullanarak gerçekleştirilir.
        if self.age_net is None:
            print("[INFO] load age and gender models ...")
            self.age_net = cv2.dnn.readNetFromCaffe("age_gender_models/deploy_age.prototxt", 
                                                    "age_gender_models/age_net.caffemodel")
            self.gender_net = cv2.dnn.readNetFromCaffe("age_gender_models/deploy_gender.prototxt", 
                                                        "age_gender_models/gender_net.caffemodel")
            print("[INFO] Load models - DONE!")
        
        if face is None:
            return
        
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
        # Predict gender
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]
        # Predict age
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]
        
        return age, gender
        
    def get_landmarks(self, frame, type):
    #    Verilen bir karedeki yüz için 81 veya 5 noktalı yüz hatlarını alır. 
    #    Bu noktalar, yüz hatlarını tanımlayan koordinatları içerir.
        if self.predictor is None:
            print("[INFO] load " + type + " facial landmarks model ...")
            self.predictor = dlib.shape_predictor("shape_predictor_" + type + "_face_landmarks.dat")
            print("[INFO] Load model - DONE!")
        
        if frame is None:
            return None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.face_detection(frame)
        
        if len(rects)<0 or len(rects)==0:
            return None, None
            
        shape = self.predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)        
        return shape, rects
    
    def ROI_extraction(self, face, shape):
        #  ROI çıkarma
        if (len(shape)==81):
            ROI1 = face[shape[29][1]:shape[33][1], #right cheek
                    shape[54][0]:shape[12][0]]
                    
            ROI2 =  face[shape[29][1]:shape[33][1], #left cheek
                    shape[4][0]:shape[48][0]]
                    
        else:
            ROI1 = face[int((shape[4][1] + shape[2][1])/2):shape[4][1], #right cheek
                    shape[2][0]:shape[3][0]]
                    
            ROI2 =  face[int((shape[4][1] + shape[2][1])/2):shape[4][1], #left cheek
                    shape[1][0]:shape[0][0]]
                        
        return ROI1, ROI2 

   
    def facial_landmarks_remap(self,shape):
        # Bazı yüz hatlarını yeniden sıralamak için kullanılır. 
        # Bu, daha sonra cv2.fillConvexPoly fonksiyonuyla kullanılmak üzere hazırlanmıştır.
        remapped_shape = shape.copy()
        # left eye brow
        remapped_shape[17] = shape[26]
        remapped_shape[18] = shape[25]
        remapped_shape[19] = shape[24]
        remapped_shape[20] = shape[23]
        remapped_shape[21] = shape[22]
        # right eye brow
        remapped_shape[22] = shape[21]
        remapped_shape[23] = shape[20]
        remapped_shape[24] = shape[19]
        remapped_shape[25] = shape[18]
        remapped_shape[26] = shape[17]
        # neatening 
        remapped_shape[27] = shape[0]
        remapped_shape = cv2.convexHull(shape)
        return remapped_shape       
    
    def no_age_gender_face_process(self, frame, type):
        #  Verilen bir karedeki yüzü işlemek için kullanılır. 
        # Bu işlem, yüz algılama, yüz hizalama, yaş ve cinsiyet tespiti gibi işlemleri içerir. 
        # no_age_gender_face_process metodu yaş ve cinsiyet tespiti yapmazken, face_full_process metodu bu özellikleri de içerir.
        if(type=="5"):
            shape, rects = self.get_landmarks(frame, "5")
            
            if shape is None:
                return None
        else:    
            shape, rects = self.get_landmarks(frame, "81")
            if shape is None:
                return None
        
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        
        face = frame[y:y+h,x:x+w]
        aligned_face,aligned_shape = self.face_alignment(frame, shape)
                
        return rects, face, shape, aligned_face, aligned_shape
        
    def face_full_process(self, frame, type, face_detect_on, age_gender_on):

        #assign from last params
        age = self.last_age
        gender = self.last_gender
        rects = self.last_rects
        shape = self.last_shape
        aligned_shape = self.last_aligned_shape
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if face_detect_on:
            if(type=="5"):
                shape, rects = self.get_landmarks(frame, "5")
                if shape is None:
                    return None
            else:    
                shape, rects = self.get_landmarks(frame, "81")
                if shape is None:
                    return None
        
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        
        face = frame[y:y+h,x:x+w]        
        
        if age_gender_on:
            age, gender = self.age_gender_detection(face)
        
        aligned_face, aligned_face = self.face_alignment(frame, shape)
        self.last_age = age
        self.last_gender = gender
        self.last_rects = rects
        self.last_shape = shape
        self.last_aligned_shape = aligned_shape

        return rects, face, (age, gender), shape, aligned_face, aligned_shape
        


