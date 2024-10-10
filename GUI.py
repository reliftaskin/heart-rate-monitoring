import cv2
import numpy as np
from PyQt5 import QtCore

from PyQt5.QtCore import QThread
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QPushButton, QApplication, QComboBox, QLabel, QFileDialog, QStatusBar, QDesktopWidget, QMessageBox, QMainWindow, QLineEdit

import pyqtgraph as pg
import sys
import time
from process import Process
from webcam import Webcam
from interface import waitKey, plotXY

class GUI(QMainWindow, QThread):
    def __init__(self):
        super(GUI,self).__init__()
        self.initUI()
        self.webcam = Webcam()
        self.input = self.webcam
        self.dirname = ""
        print("Input: webcam")
        self.statusBar.showMessage("Input: webcam",5000)
        self.btnOpen.setEnabled(False)
        self.process = Process()
        self.status = False
        self.frame = np.zeros((10,10,3),np.uint8)
        self.bpm = 0
        self.terminate = False

    def set_background_image(self, image_path):
        background_label = QLabel(self)
        pixmap = QPixmap(image_path)

        if not pixmap.isNull():
            background_label.setPixmap(pixmap)
            background_label.setGeometry(0, 0, self.width(), self.height())
            background_label.lower()

    def set_background_image(self, image_path):
        background_label = QLabel(self)
        pixmap = QPixmap(image_path)

        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.size(), aspectRatioMode=QtCore.Qt.KeepAspectRatio)
            background_label.setPixmap(scaled_pixmap)
            background_label.setGeometry(0, 0, self.width(), self.height())
            background_label.lower()
        
    def initUI(self):
        font = QFont()
        font.setPointSize(16)
        self.lineEdit = QLineEdit("Click the start button to start the camera.", self)
        self.lineEdit.setGeometry(180, 10, 400, 50) 
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setStyleSheet("background-color: rgba(0, 0, 0, 0);border: none; font-size: 18px; color:white")

        self.btnStart = QPushButton("Start", self)
        self.btnStart.move(190, 60) 
        self.btnStart.setFixedWidth(300)
        self.btnStart.setFixedHeight(50)
        self.btnStart.setFont(font)
        self.btnStart.clicked.connect(self.run)

        self.btnOpen = QPushButton("Open", self)
        self.btnOpen.move(230,520)
        self.btnOpen.setFixedWidth(200)
        self.btnOpen.setFixedHeight(50)
        self.btnOpen.setFont(font)
        self.btnOpen.clicked.connect(self.openFileDialog)
        
        self.cbbInput = QComboBox(self)
        self.cbbInput.addItem("Webcam")
        self.cbbInput.setCurrentIndex(0)
        self.cbbInput.setFixedWidth(200)
        self.cbbInput.setFixedHeight(50)
        self.cbbInput.move(20,520)
        self.cbbInput.setFont(font)
        self.cbbInput.activated.connect(self.selectInput)
        self.cbbInput.hide()
        
        self.lblDisplay = QLabel(self) #label to show frame from camera
        self.lblDisplay.setGeometry(60, 180, 550, 400) 
        self.lblDisplay.setStyleSheet("background-color: #cdc5bf")
        
        self.lblROI = QLabel(self) #label to show face with ROIs
        self.lblROI.setGeometry(660,415,200,200)
        self.lblROI.setStyleSheet("background-color: #cdc5bf")
        
        self.lblHR = QLabel(self)  # label to show HR change over time
        self.lblHR.setGeometry(900, 450, 300, 40)  
        self.lblHR.setFont(font)
        self.lblHR.setText("Frequency: ")
        self.lblHR.setStyleSheet("color:white;")

        self.lblHR2 = QLabel(self)  # label to show stable HR
        self.lblHR2.setGeometry(900, 500, 300, 40)  
        self.lblHR2.setFont(font)
        self.lblHR2.setText("Heart rate: ")
        self.lblHR2.setStyleSheet("color:white;")

       # dynamic plot
        self.signal_Plt = pg.PlotWidget(self)
        self.signal_Plt.move(660, 50)
        self.signal_Plt.resize(450, 160)
        self.signal_Plt.setLabel('bottom', "Signal")

        self.fft_Plt = pg.PlotWidget(self)
        self.fft_Plt.move(660, 220) 
        self.fft_Plt.resize(450, 160) 
        self.fft_Plt.setLabel('bottom', "FFT")

        
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)
        
        
        self.statusBar = QStatusBar()
        self.statusBar.setFont(font)
        self.setStatusBar(self.statusBar)

        self.setGeometry(100,100,1160,640)
        self.setWindowTitle("Pulse Meter")
        self.set_background_image("arkaplan5.jpg")
        self.show()
        
    def update(self):
        self.signal_Plt.clear()
        self.signal_Plt.plot(self.process.samples[20:],pen='g')

        self.fft_Plt.clear()
        self.fft_Plt.plot(np.column_stack((self.process.freqs, self.process.fft)), pen = 'g')
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self,"Message", "Are you sure want to quit",
            QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
            self.input.stop()
            self.terminate = True
            self.timer.stop() 
            self.close()  # PyQt uygulamasını kapat

        else: 
            event.ignore()
    
    def selectInput(self):
        self.reset()
        if self.cbbInput.currentIndex() == 0:
            self.input = self.webcam
            print("Input: webcam")
            self.btnOpen.setEnabled(False)
    
    def key_handler(self):
        """
        cv2 window must be focused for keypresses to be detected.
        """
        self.pressed = waitKey(1) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("[INFO] Exiting")
            self.webcam.stop()
            sys.exit()
    
    def openFileDialog(self):
        self.dirname = QFileDialog.getOpenFileName(self, 'OpenFile')
    
    def reset(self):
        self.process.reset()
        self.lblDisplay.clear()
        self.lblDisplay.setStyleSheet("background-color: #cdc5bf")

    def main_loop(self):
        frame = self.input.get_frame()

        self.process.frame_in = frame
        if self.terminate == False:
            ret = self.process.run()
        
        # cv2.imshow("Processed", frame)
        if ret == True:
            self.frame = self.process.frame_out #get the frame to show in GUI
            self.f_fr = self.process.frame_ROI #get the face to show in GUI
            #print(self.f_fr.shape)
            self.bpm = self.process.bpm #get the bpm change over the time
        else:
            self.frame = frame
            self.f_fr = np.zeros((10, 10, 3), np.uint8)
            self.bpm = 0
        
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        img = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], 
                        self.frame.strides[0], QImage.Format_RGB888)
        self.lblDisplay.setPixmap(QPixmap.fromImage(img))
        
        self.f_fr = cv2.cvtColor(self.f_fr, cv2.COLOR_RGB2BGR)
        self.f_fr = np.transpose(self.f_fr,(0,1,2)).copy()
        f_img = QImage(self.f_fr, self.f_fr.shape[1], self.f_fr.shape[0], 
                       self.f_fr.strides[0], QImage.Format_RGB888)
        self.lblROI.setPixmap(QPixmap.fromImage(f_img))
        
        self.lblHR.setText("Frequency: " + str(float("{:.2f}".format(self.bpm))))
        
        if self.process.bpms.__len__() >50:
            if(max(self.process.bpms-np.mean(self.process.bpms))<5): #show HR if it is stable -the change is not over 5 bpm- for 3s
                self.lblHR2.setText("Heart rate: " + str(float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm")
    
        self.key_handler()  #if not the GUI cant show anything

    def run(self, input):
        print("run")
        self.reset()
        input = self.input
        self.input.dirname = self.dirname
        if self.status == False:
            self.status = True
            input.start()
            self.btnStart.setText("Stop")
            self.cbbInput.setEnabled(False)
            self.btnOpen.setEnabled(False)
            self.lblHR2.clear()
            while self.status == True:
                self.main_loop()

        elif self.status == True:
            self.status = False
            input.stop()
            self.btnStart.setText("Start")
            self.cbbInput.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())
