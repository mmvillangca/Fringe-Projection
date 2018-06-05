#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:14:42 2018

@author: mmvillangca

"""

__author__      = "Mark Jayson Villangca"
__copyright__   = "Copyright 2018"
__version__ = "1.0"
__license__ = "MIT"


from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QDesktopWidget, QMainWindow, QLabel, QSlider, QInputDialog, QMessageBox
import numpy as np
import cv2
import time
from threading import Thread
import plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

form_class = uic.loadUiType("FringGUI_RasPi.ui")[0]


class webcamVideoStream():
    
    def __init__(self,src=0,res=(480,360)):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,res[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,res[1])
        (self.grabbed,self.frame) = self.stream.read()
        
        self.stopped = False
    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            else:
                (self.grabbed,self.frame) = self.stream.read()
    
    def readRecent(self):
        return self.frame
        
    def width(self):
        return self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    def height(self):
        return self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def stop(self):
        self.stopped = True
        self.stream.release()

    
class MainWindow(QMainWindow, form_class):
    
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        
        #fringe dimension along the X
        self.dimX = int(self.lineEditDimXpx.text())
        self.dimY = int(self.lineEditDimYpx.text())
        self.lineEditDimXpx.returnPressed.connect(self.updateSliderCurrentValue)
        
        #slider properties
        self.sliderFringePeriod.setTickInterval(1)
        self.sliderFringePeriod.setMaximum(100)
        self.sliderFringePeriod.setValue(40)
        self.labelPeriodMax.setText('100')
        self.sliderFringePeriod.sliderReleased.connect(self.updateSliderCurrentValue)
        
        #projection window position
        self.projWinPosX.setText('0')
        self.projWinPosY.setText('0')
        
        #radio button label
        self.radiopi4.setText(U'\U000003c0/2') #not yet implemented
        self.radio2pi3.setText(U'2\U000003c0/3')
        
        #slider current value indicator
        self.labelSliderValue.setText(str(self.sliderFringePeriod.value()))
        
        #create statusbar
        self.statusBar()        
        
        # create OpenCV camera object using webcamVideoStream class
        self.cap = webcamVideoStream(0,(1280,960)).start()
        
        # if alignment tool is selected
        self.alignTool.stateChanged.connect(self.showAlignTool)
        
        #if calibrate button is clicked
        self.pushCalibrate.clicked.connect(self.calibratePressed)
        
        #if measure button is clicked
        self.pushMeasure.clicked.connect(self.measurePressed)
        
        #id save button is clicked
        self.pushSave.clicked.connect(self.savePressed)
        
        #initialize fringe image
        self.I = 0
        
        #initialize mouse click
        self.clkX = []
        self.clkY = []
        
        #initialize Ref, Obj and Res
        self.RefImag = 0
        self.ObjImag = 0
        self.ResImag = 0
        self.objPhi = 0
        self.basePhi = 0
        self.realX = 0
        self.realY = 0
        
        #initialize fringe pitch
        self.pitchFringe = 0
        self.pitchFringePX = 1
        
        #load alignment target
        self.alignImg = cv2.imread("alignmentTarget.png")

        self.img = 0
        
        self.show()
        
    def generateFringe(self):
        #Generate Fringes
        self.dimX = int(self.lineEditDimXpx.text())
        self.dimY = int(self.lineEditDimYpx.text())
        x = np.linspace(0, self.dimX-1, self.dimX, dtype=float)
        y = np.linspace(0, self.dimY-1, self.dimY, dtype=float)
        
        X,Y = np.meshgrid(x,y)
        
        L = float(self.sliderFringePeriod.value())
        self.I = np.zeros([self.dimY, self.dimX,3])
        
        dom = [0,1]
        newRange = [float(self.lineEditMinGray.text()),float(self.lineEditMaxGray.text())]
        m = (newRange[1]-newRange[0])/(dom[1]-dom[0])
        b = newRange[0]-m*dom[0]
        f = lambda I: m*I+b
        
        if self.comboFringe.currentIndex() == 0:
            #sinusoid
            if self.radiopi4.isChecked() == True:
                #pi/2 phase shift
#                self.I0 = (np.cos(2*np.pi*X/L-0)+1)/2
#                self.I1 = (np.cos(2*np.pi*X/L-np.pi/2)+1)/2
#                self.I2 = (np.cos(2*np.pi*X/L-np.pi)+1)/2
#                self.I3 = (np.cos(2*np.pi*X/L-3*np.pi/2)+1)/2
                pass
            if self.radio2pi3.isChecked() == True:
                #2pi/3 phase shift
                self.I[:,:,0] = f((np.cos(2*np.pi*X/L-2*np.pi/3)+1)/2)
                self.I[:,:,1] = f((np.cos(2*np.pi*X/L-0)+1)/2)
                self.I[:,:,2] = f((np.cos(2*np.pi*X/L+2*np.pi/3)+1)/2)

        elif self.comboFringe.currentIndex() == 1:
            #blaze
            pass
        elif self.comboFringe.currentIndex() == 2:
            #binary
            pass
        
        return self.I
        
            
    def updateSliderCurrentValue(self):
        #update slider value display based on current position
        if len(self.lineEditDimXpx.text()) == 0:
            self.sliderFringePeriod.setMaximum(100)
            self.labelPeriodMax.setText('100')
            self.lineEditDimXpx.setText('800')
        if int(self.lineEditDimXpx.text()) < self.sliderFringePeriod.minimum():
            self.sliderFringePeriod.setMaximum(100)
            self.labelPeriodMax.setText('100')
        
        self.labelSliderValue.setText(str(self.sliderFringePeriod.sliderPosition()))
#        self.generateFringe()
    
    
    def calibratePressed(self):
        self.pushCalibrate.setText('Calibrating')  
        imagHeight = self.cap.height()
        imagWidth = self.cap.width()
        
        self.I = self.generateFringe()
        fringeCount = np.size(self.I, 2)
        self.RefImag = np.zeros([int(imagHeight), int(imagWidth), 3])
        movX = int(self.projWinPosX.text()); movY = int(self.projWinPosY.text())
        
        cv2.namedWindow('fringe')
        cv2.moveWindow('fringe',movX,movY)
        for i in range(fringeCount):

            cv2.imshow('fringe',self.I[:,:,i]/255)
            #cv2.moveWindow('fringe',movX,movY)
            cv2.waitKey(500)
            frame = self.cap.readRecent()
            self.RefImag[:,:,i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.waitKey(500)
        
        cv2.imshow('fringe',self.I[:,:,i]/255)
        cv2.moveWindow('fringe',movX,movY)
        
        self.pitchDialogBox()
        
        cv2.destroyAllWindows()
        
        #add window to get pixel distance between fringes. this will be used 
        #in px-to-real world conversion. in particular to the x and y coordinates
        QMessageBox.information(self, "Fringe pitch in px","In the next image, select consecutive fringes. Press 'q' when done",QMessageBox.Ok)
        
        
        #show image here
        cv2.namedWindow('Captured Fringe')
        cv2.setMouseCallback('Captured Fringe', self.getMousePoints)
        
        while True:
            cv2.imshow('Captured Fringe', self.RefImag[:,:,0]/255)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print(self.clkX)
                self.pitchFringePX = np.floor(np.mean(np.diff(self.clkX)))
                print(self.pitchFringePX)
                self.clkX = []
                self.clkY = []
                break
        
        cv2.destroyAllWindows()
        
#       
       
        self.pushCalibrate.setText('Calibrate')
        
        self.statusBar().showMessage('Calibration done.')
    
    def getMousePoints(self,event, x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x)
            self.clkX.append(x)
            self.clkY.append(y)
            
            
    
    def calcPhasemap(self, I0, I1, I2, unwrap=False):
        #this function will calculate the phase map based on 3 fringe images that has 2pi/3 phase shifts
        #where I1, I2, and I3 are intensity images
        
        phi = np.arctan2(np.sqrt(3)*(I0-I2),2*I1-I0-I2)
        
        if unwrap == True:
            phi = np.unwrap(phi,discont=np.pi,axis=1)
        
        return phi
        
    
    def measurePressed(self):
        self.pushMeasure.setText('in progress')
        imagWidth = self.cap.width()
        imagHeight = self.cap.height()
        
        self.I = self.generateFringe()
        fringeCount = np.size(self.I, 2)
        self.ObjImag = np.zeros([int(imagHeight), int(imagWidth), 3])
        movX = int(self.projWinPosX.text()); movY = int(self.projWinPosY.text())
        
        cv2.namedWindow('fringe')
        cv2.moveWindow('fringe',movX,movY)
        for i in range(fringeCount):

            cv2.imshow('fringe',self.I[:,:,i]/255)
            #cv2.moveWindow('fringe',movX,movY)
            cv2.waitKey(500)
            frame = self.cap.readRecent()
            self.ObjImag[:,:,i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.waitKey(500)
        
        cv2.destroyAllWindows()
        
        #add window to get pixel distance between fringes. this will be used 
        #in px-to-real world conversion. in particular to the x and y coordinates
        QMessageBox.information(self, "Select ROI","In the next image, click two points to define a rectangle. Press 'q' when done",QMessageBox.Ok)
        
        
        #show image here
        cv2.namedWindow('ROI')
        cv2.setMouseCallback('ROI', self.getMousePointsRect)
        
        self.img = self.ObjImag[:,:,0]/255
        while True:
            cv2.imshow('ROI', self.img)
            #draw rect
            if cv2.waitKey(10) & 0xFF == ord('q'):
                x0 = min(self.clkX[-2],self.clkX[-1])
                y0 = min(self.clkY[-2],self.clkY[-1])
                x1 = max(self.clkX[-2],self.clkX[-1])
                y1 = max(self.clkY[-2],self.clkY[-1])
                self.clkX = []
                self.clkY = []
                break
        
        cv2.destroyAllWindows()
    
        
        #Ref phase
        B0 = self.RefImag[y0:y1,x0:x1,0]
        B1 = self.RefImag[y0:y1,x0:x1,1]
        B2 = self.RefImag[y0:y1,x0:x1,2]
        
         #Gaussian smoothing
        B0 = cv2.GaussianBlur(B0,(5,5),10)
        B1 = cv2.GaussianBlur(B1,(5,5),10)
        B2 = cv2.GaussianBlur(B2,(5,5),10)
        
        #calculate phase
        self.basePhi = self.calcPhasemap(B0,B1,B2, unwrap=True)
        
        #Obj phase
        I0 = self.ObjImag[y0:y1,x0:x1,0]
        I1 = self.ObjImag[y0:y1,x0:x1,1]
        I2 = self.ObjImag[y0:y1,x0:x1,2]       
              
        I0 = cv2.GaussianBlur(I0,(5,5),10)
        I1 = cv2.GaussianBlur(I1,(5,5),10)
        I2 = cv2.GaussianBlur(I2,(5,5),10)
        
        #calculate phase
        self.objPhi = self.calcPhasemap(I0,I1,I2, unwrap=True)
        phasemap = self.objPhi-self.basePhi
        
        
        print(self.pitchFringePX)
        print(self.pitchFringe)
        
        #calculate calibration constant for z-coordinate
        p = self.pitchFringe
        l = float(self.lineEditProjObjDist.text())
        d = float(self.lineEditProjCamDist.text())
        K = p*l/(2*np.pi*d)
        #print(K)
        
        #calculate calib constant for lateral dim
        K2 = float(self.pitchFringe)/float(self.pitchFringePX)
        
        
        #do some phase to real world mapping here. Unit is in cm
        self.ResImag = K*phasemap
        self.realX = K2*np.arange(0,np.shape(self.ResImag)[1])
        self.realY = K2*np.arange(0,np.shape(self.ResImag)[0])
        
        self.pushMeasure.setText('Measure')
        
        
        if self.plotRes.isChecked()==True:
            self.plotResult(self.realX,self.realY,self.ResImag)
        
        if self.matplotlibPlot.isChecked()==True:
            self.plotInMatplotlib(self.realX,self.realY,self.ResImag)
        
        self.statusBar().showMessage('Phase measurement done.')
        
    def getMousePointsRect(self,event, x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            
            self.clkX.append(x)
            self.clkY.append(y)
            print(len(self.clkX))
            
            if len(self.clkX)>=2:
                x0 = min(self.clkX[-2],self.clkX[-1])
                y0 = min(self.clkY[-2],self.clkY[-1])
                x1 = max(self.clkX[-2],self.clkX[-1])
                y1 = max(self.clkY[-2],self.clkY[-1])
                self.img = self.ObjImag[:,:,0]/255
                cv2.rectangle(self.img,(x0,y0),(x1,y1),255,3)
        
    def msgBoxFringe(self):
         msgBox = QMessageBox(self)
         msgBox.setIcon(QMessageBox.Information)
         msgBox.setText('Fringe pitch in px')
         msgBox.setInformativeText('In the next image, select four consecutive fringes.')
         msgBox.setWindowTitle('Fringe period in px')
         msgBox.setStandardButtons(QMessageBox.Ok)
         msgBox.show()
        
    def pitchDialogBox(self):
        text, ok = QInputDialog.getText(self,'Fringe Pitch',
                                        'Enter the pitch of the fringe (cm):')
        
        if ok:
            self.pitchFringe = float(text)
        
        
    def plotResult(self, x, y, Result):
        # test some plotting
        data = [
        go.Surface(
            z=Result
            )
            ]
        layout = go.Layout(
            #title='Phase Map',
            autosize=False,
            scene = dict(
                    xaxis = dict(nticks = 4, range=[0,max(x)],),
                    yaxis = dict(nticks = 4, range=[0,max(y)],),
                    zaxis = dict(nticks = 4, range=[-1,3*np.max(Result)],),
                    ),
            width=800,
            height=800,
            margin=dict(
                l=65,
                r=50,
                b=65,
                t=90
            )
        )
        fig = go.Figure(data=data, layout=layout)
        now = QDateTime.currentDateTime()
        if len(self.filename.text()) == 0:
            py.offline.plot(fig, filename='Result - %s.html' %now.toString('yyyy-mm-dd hhmmss'),auto_open=False)
        else:
            py.offline.plot(fig, filename='Result - %s.html' %self.filename.text(),auto_open=False)
    
    def plotInMatplotlib(self, x, y, Result):
#        fig = Figure()
#        ax =fig.add_subplot(111,projection='3d')
#        ax = plt.axes(projection = '3d')
#        X,Y = np.meshgrid(x,y)
        
#        ax.plot_surface(X,Y,Result,rstride=20,cstride=20,cmap='viridis',edgecolor='none')
        plt.imshow(Result)
        plt.show()
        
    
    def savePressed(self):
        now = QDateTime.currentDateTime()
        if len(self.filename.text()) == 0:
            if self.saveImages.isChecked():
                for i in range(3): #update range depending on number of fringes
                    cv2.imwrite('Ref %s - %s.png' % (now.toString(Qt.ISODate),i), self.RefImag[:,:,i])
                    cv2.imwrite('Obj %s - %s.png' % (now.toString(Qt.ISODate),i), self.ObjImag[:,:,i])
            
            if self.savePointCloud.isChecked():
                #save x,y,z coordinates of reconstructed object
    #            pass
                X,Y = np.meshgrid(self.realX,self.realY)
                tempX = np.reshape(X,[np.prod(np.shape(X)),1])
                tempY = np.reshape(Y,[np.prod(np.shape(Y)),1])
                tempRes = np.reshape(self.ResImag,[np.prod(np.shape(self.ResImag)),1])
                tempArray = np.concatenate([tempX,tempY,tempRes],axis=1)
                
                np.savetxt('PointCloud - %s.txt' % now.toString(Qt.ISODate),
                           tempArray, delimiter='\t')
        else:
            if self.saveImages.isChecked():
                for i in range(3): #update range depending on number of fringes
                    cv2.imwrite('Ref %s - %s.png' % (self.filename.text(),i), self.RefImag[:,:,i])
                    cv2.imwrite('Obj %s - %s.png' % (self.filename.text(),i), self.ObjImag[:,:,i])
            
            if self.savePointCloud.isChecked():
                #save x,y,z coordinates of reconstructed object
    #            pass
                X,Y = np.meshgrid(self.realX,self.realY)
                tempX = np.reshape(X,[np.prod(np.shape(X)),1])
                tempY = np.reshape(Y,[np.prod(np.shape(Y)),1])
                tempRes = np.reshape(self.ResImag,[np.prod(np.shape(self.ResImag)),1])
                tempArray = np.concatenate([tempX,tempY,tempRes],axis=1)
                
                np.savetxt('PointCloud - %s.txt' % self.filename.text(),
                           tempArray, delimiter='\t')
            
        self.statusBar().showMessage('Saved.')
        
    def showAlignTool(self):
        height, width, channels = self.alignImg.shape
        self.dimX = float(self.lineEditDimXpx.text())
        self.dimY = float(self.lineEditDimYpx.text())
        hScale = (self.dimY/height)
        wScale = (self.dimX/width)
        scale = min([hScale, wScale])
        img = cv2.resize(self.alignImg, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        movX = int(self.projWinPosX.text()); movY = int(self.projWinPosY.text())
        
        
        if self.alignTool.isChecked() == True:
            cv2.imshow('align',img)
            cv2.moveWindow('align',movX,movY)
            cv2.waitKey(10)
        else:
            cv2.destroyWindow('align')
    
    def closeEvent(self,event):
        self.cap.stop()
        cv2.destroyAllWindows()
    
    
    
        
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
