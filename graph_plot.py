import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import scipy.fftpack
from scipy.signal import butter, lfilter


arr_red = []
arr_green = []
arr_blue = []

def butter_bandpass(lowcut, highcut, fs, order=5):
    # Bu fonksiyon, verilen bir geçiş bant genişliği içindeki frekansları geçiren bir Butterworth band geçiren filtresinin katsayılarını hesaplar. 
    # lowcut ve highcut frekansları filtreleme aralığını belirlerken, fs örnekleme frekansı ve order ise filtre sırasını belirler.
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # Bu fonksiyon, bir veri dizisine Butterworth band geçiren filtresini uygular. 
    # Önceki fonksiyonu kullanarak filtre katsayılarını alır ve lfilter ile veriye uygular.
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y  

#read file signal.dat
with open("signal.dat") as f:
    lines = f.readlines()
    for i in range(lines.__len__()):
        r,g,b = lines[i].split("%")
        arr_red.append(float(r))
        arr_green.append(float(g))
        arr_blue.append(float(b))
      
# Bu fonksiyon, bir veri dizisinden lineer bir trendi çıkarır. Veri dizisini alır ve trendi çıkarılmış veriyi döndürür.
green_detrended = signal.detrend(arr_blue)
L = len(arr_red)


bpf = butter_bandpass_filter(green_detrended,0.8,3,fs=30,order = 3)

even_times = np.linspace(0, L, L)
# np.interp : Bu fonksiyon, bir veri dizisini belirli bir sayıda noktaya doğrusal olarak interpolasyon yaparak yeniden örnekler. 
# Bu durumda, her örnek arasındaki süreyi eşit olacak şekilde örneklenen bir zaman dizisi elde edilir.
interpolated = np.interp(even_times, even_times, bpf) 
# np.hamming : Bu fonksiyon, Hamming penceresini oluşturur. Bu durumda, Hamming penceresi ile örneklenmiş veriye uygulanır.
interpolated = np.hamming(L)*interpolated
norm = interpolated/np.linalg.norm(interpolated)
# np.fft.rfft : Bu fonksiyon, FFT'yi uygulayarak verinin frekans alanında gösterimini hesaplar. Burada, önceki işlemlerle hazırlanan veri FFT'ye verilir.
raw = np.fft.rfft(norm*30)
# np.fft.rfftfreq : Bu fonksiyon, FFT sonucunda elde edilen frekans alanında gösterimi belirleyen frekans noktalarını hesaplar.
freq = np.fft.rfftfreq(L, 1/30)*60
fft = np.abs(raw)**2


g = plt.figure("green")
ax2 = g.add_subplot(111)    
ax2.set_title("band pass filter")
ax2.set_xlabel("time")
ax2.set_ylabel("magnitude")
# plt.plot : Bu fonksiyon, belirtilen x ve y değerlerini içeren bir grafik çizer. 
# Burada, FFT sonucunun frekans alanında gösterimini çizmek için kullanılır.
plt.plot(freq,fft, color = "blue")
g.show()

input("Press Enter to exit...")    
    
