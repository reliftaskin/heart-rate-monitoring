import cv2
import numpy as np
import time

class Webcam(object):
    def __init__(self):
        self.dirname = ""
        self.cap = None
    
    def start(self):
        # Webcam'in başlatılması için gereken işlemleri gerçekleştirir.  
        print("[INFO] Start webcam")
        time.sleep(1) #Önce bir saniye bekler (time.sleep(1)) ve sonra cv2.VideoCapture kullanarak kameraya erişmeye çalışır
        self.cap = cv2.VideoCapture(0)
        self.valid = False
        try: # Eğer kamera erişilebilirse (cap.read() çağrısında hata alınmazsa), valid bayrağını True yapar ve kameranın şeklini (shape) kaydeder.
            resp = self.cap.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None
    
    def get_frame(self):
    #  Kameradan bir kare alır ve bu kareyi döndürür. 
    #  Eğer kamera erişilemezse (valid bayrağı False ise), bir hata mesajı içeren bir kare döndürür.
        if self.valid:
            _,frame = self.cap.read()
            frame = cv2.flip(frame,1)
        else:
            frame = np.ones((480,640,3), dtype=np.uint8)
            col = (0,256,256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                       (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame

    def stop(self):
        # Kamerayı serbest bırakır (release metoduyla). 
        # Bu, kameranın kullanımını bırakır ve başka bir uygulama tarafından kullanılabilir hale getirir.
        if self.cap is not None:
            self.cap.release()
            print("[INFO] Stop webcam")
        
