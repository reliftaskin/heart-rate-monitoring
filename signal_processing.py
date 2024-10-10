import cv2
import numpy as np
import time
from scipy import signal


class Signal_processing():
    def __init__(self):
        self.a = 1
        
    def extract_color(self, ROIs):
        # Verilen ROI'ler (Region of Interest) için yeşil rengin ortalamasını hesaplar. 
        # Her bir ROI için yeşil renk kanalındaki piksel değerlerinin ortalamasını alır ve bu değerlerin ortalamasını döndürür.
        g = []
        for ROI in ROIs:
            g.append(np.mean(ROI[:,:,1]))
        output_val = np.mean(g)
        return output_val
    
    def normalization(self, data_buffer):
        # Verilen veri tamponunu normalize eder. Verilen veri tamponunun normu kullanılarak veriler normalize edilir.
        normalized_data = data_buffer/np.linalg.norm(data_buffer)
        return normalized_data
    
    def signal_detrending(self, data_buffer):
        # Verilen veri tamponundan genel trendi kaldırır. signal.detrend fonksiyonu kullanılarak veri tamponundan genel trend çıkarılır.
        detrended_data = signal.detrend(data_buffer)
        return detrended_data
        
    def interpolation(self, data_buffer, times):
        #  Verilen veri tamponunu interpolasyon yaparak daha periyodik hale getirir. 
        #  Veri tamponunun zamanlarını kullanarak, verinin periyodik olmasını sağlamak için np.interp ve np.hamming fonksiyonları kullanılır.
        L = len(data_buffer)
        even_times = np.linspace(times[0], times[-1], L)
        
        interp = np.interp(even_times, times, data_buffer)
        interpolated_data = np.hamming(L) * interp
        return interpolated_data
        
    def fft(self, data_buffer, fps):
        # Verilen veri tamponu için FFT (Hızlı Fourier Dönüşümü) işlemi uygular. 
        # FFT sonucunda elde edilen frekanslar arasından belirli bir aralıktaki frekans bileşenlerini seçer.
        L = len(data_buffer)
        freqs = float(fps) / L * np.arange(L / 2 + 1) 
        freqs_in_minute = 60. * freqs
        raw_fft = np.fft.rfft(data_buffer*30)
        fft = np.abs(raw_fft)**2
        
        interest_idx = np.where((freqs_in_minute > 50) & (freqs_in_minute < 180))[0]
        print(freqs_in_minute)
        interest_idx_sub = interest_idx[:-1].copy()
        freqs_of_interest = freqs_in_minute[interest_idx_sub]
        
        fft_of_interest = fft[interest_idx_sub]
        return fft_of_interest, freqs_of_interest


    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=5):
        # Verilen veri tamponuna bir Butterworth band geçiren filtre uygular. 
        # Bu, belirli bir frekans bandındaki sinyalleri geçirirken diğer frekansları baskılar.
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.lfilter(b, a, data_buffer)
        return filtered_data
