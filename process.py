import cv2
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal
from face_utilities import Face_utilities
from signal_processing import Signal_processing
from imutils import face_utils

class Process(object):
    def __init__(self):
        #  __init__ metodu sınıfın başlangıç durumunu ayarlar. Gerekli değişkenler ve nesneler 
        #  (örneğin, görüntü işleme için FaceDetection, Face_utilities, Signal_processing gibi yardımcı sınıflar) başlatılır.
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        self.fu = Face_utilities()
        self.sp = Signal_processing()

    def extractColor(self, frame):
        # Görüntüden bir renk özelliği (green_val) çıkarır. Bu özellik, görüntünün yeşil tonlarının ortalamasını hesaplar.
        g = np.mean(frame[:,:,1])
        return g
        
    def run(self):
        # Ana işleme adımı burada gerçekleşir. Öncelikle, yüz bölgesini algılar ve gerekli işlemleri yapar. 
        # Ardından, alınan verileri işler, renk içeriğini çıkarır ve sinyal işleme teknikleri uygular. 
        # Son olarak, hesaplanan verileri kullanarak kalp atış hızını tahmin eder.
        frame = self.frame_in #self.frame_in içindeki görüntüden yüz bölgesi algılanır ve gerekli işlemler yapılır.
        ret_process = self.fu.no_age_gender_face_process(frame, "81")
        if ret_process is None:
            return False
        rects, face, shape, aligned_face, aligned_shape = ret_process

        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        
        if(len(aligned_shape)==81):
            #draw rectangle on right and left cheeks
            cv2.rectangle(aligned_face,(aligned_shape[54][0], aligned_shape[29][1]), 
                    (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
            
            cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]), 
                    (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)   
            
            #draw forehead
            x_center = (aligned_shape[18][0] + aligned_shape[25][0]) // 2  # Alın bölgesinin merkez noktasının x koordinatı
            y_center = (aligned_shape[70][1] + aligned_shape[24][1]) // 2  # Alın bölgesinin merkez noktasının y koordinatı
            width = aligned_shape[25][0] - aligned_shape[18][0]  # Alın bölgesinin genişliği
            height = aligned_shape[24][1] - aligned_shape[70][1]  # Alın bölgesinin yüksekliği
            for _ in range(2):
                width = int(width * 0.85)
                height = int(height * 0.85)
                x1 = x_center - width // 2
                y1 = y_center - height // 2
                x2 = x_center + width // 2
                y2 = y_center + height // 2
            cv2.rectangle(aligned_face, (x1, y1), (x2, y2), (0, 255, 0), 0)

        else:
            cv2.rectangle(aligned_face, (aligned_shape[0][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[1][0],aligned_shape[4][1]), (0,255,0), 0)
            
            cv2.rectangle(aligned_face, (aligned_shape[2][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[3][0],aligned_shape[4][1]), (0,255,0), 0)
        
        for (x, y) in aligned_shape: 
            cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)

        #ROI_extraction ve extract_color fonksiyonları ile yüz bölgesinden belirli bölgeler (ROI) ve bu bölgelerin renk içerikleri elde edilir.
        ROIs = self.fu.ROI_extraction(aligned_face, aligned_shape)
        green_val = self.sp.extract_color(ROIs)

        self.frame_out = frame
        self.frame_ROI = aligned_face
        
        L = len(self.data_buffer)

        g = green_val
        
        if(abs(g-np.mean(self.data_buffer))>5 and L>99): #remove sudden change, if the avg value change is over 5, use the mean of the data_buffer
            g = self.data_buffer[-1]
        
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)

        #only process in a fixed-size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        
        # ilk 10 kareden sonra hesaplamaya başlar
        if L == self.buffer_size:
            #HR'yi kameranın sağladığı fps'yi değil, bilgisayarın işlemcisinin gerçek fps'sini kullanarak hesaplar
            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            
            processed = signal.detrend(processed)#ışık değişiminin girişimini önlemek için sinyal azaltılır
            interpolated = np.interp(even_times, self.times, processed) #interpolation by 1
            interpolated = np.hamming(L) * interpolated #Sinyalin daha periyodik olması sağlanır (spektral sızıntıyı önler)
            norm = interpolated/np.linalg.norm(interpolated)

            # Aşağıdaki satır, FFT (Hızlı Fourier Dönüşümü) işlemi için gerekli olan verileri hazırlar. 
            # norm adlı dizinin her elemanı 30 ile çarpılarak genişletilir ve ardından bu genişletilmiş veri üzerinde FFT işlemi uygulanır.
            # Genişletme işlemi, sinyalin frekans bileşenlerini daha belirgin hale getirmeyi amaçlar. 
            # FFT, bir sinyalin frekans bileşenlerini analiz ederek, sinyalin hangi frekanslarda hangi şiddette olduğunu gösteren bir dizi (spektrum) 
            # elde etmeye yarar.Bu spektrum, sinyalin frekans bileşenlerini görselleştirmeye ve analiz etmeye yardımcı olur.
            raw = np.fft.rfft(norm*30) 

            #FFT işleminden elde edilen veri üzerinden frekansları hesaplar. 
            #Bu adım, FFT sonucundaki frekans bileşenlerine karşılık gelen frekans değerlerini hesaplamak için kullanılır.
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs

            # FFT sonucundan elde edilen veri, amplitüd spektrumu olarak adlandırılan ve 
            # frekans bileşenlerinin şiddetini gösteren bir dizi elde etmek için kullanılır.
            self.fft = np.abs(raw)**2 #get amplitude spectrum
        
            #HR'nin içinde olması gereken frekans aralığı
            idx = np.where((freqs > 50) & (freqs < 180))

            #Belirlenen frekans aralığına göre amplitüd spektrumunu filtreler.
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            self.freqs = pfreq 
            self.fft = pruned
            
            #Filtrelenmiş amplitüd spektrumunda en yüksek değere sahip olan frekansı bulur. Bu genellikle kalp atış hızını temsil eder.
            idx2 = np.argmax(pruned) #aralıktaki en yüksek HR
            
            #Bulunan en yüksek amplitüd değerine karşılık gelen frekans, tahmini kalp atış hızı olarak kaydedilir.
            self.bpm = self.freqs[idx2]

            #Tahmini kalp atış hızını bir listeye ekler. Bu liste istatistiksel analiz, görselleştirme için kullanılır.
            self.bpms.append(self.bpm)
            
            # Son olarak, işlenmiş sinyal, kalp atış hızının daha doğru bir şekilde tahmin edilmesine yardımcı olmak için 
            # bir Butterworth band geçiren filtreden geçirilir. 
            # Bu filtre, istenmeyen frekans bileşenlerini ortadan kaldırmaya yardımcı olur ve sinyali daha düzgün hale getirir.
            processed = self.butter_bandpass_filter(processed,0.8,3,self.fps,order = 3)
            # 0.8: Düşük kesim frekansı. Bu değer, filtrelemenin başlayacağı düşük frekans sınırını belirler. 
            #   3: Yüksek kesim frekansı. Bu değer, filtrelemenin sona ereceği yüksek frekans sınırını belirler.)

        self.samples = processed
        return True
    
    def reset(self):
        #  reset metodu, sınıfın durumunu sıfırlar. Bu, yeni bir işlem için sınıfın hazır hale getirilmesini sağlar.
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
    
    # Bu iki metot, sinyaller üzerinde bant geçiren filtreleme işlemlerini gerçekleştirir. 
    # Bu, sinyallerdeki istenmeyen frekansları filtrelemek için kullanılır.
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y 
    
    