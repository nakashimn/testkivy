#testkivy.py
import os, sys, glob
import numpy as np
import cv2
from functools import partial
from kivy.app import App
from kivy.base import EventLoop
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, StringProperty, ObjectProperty
from kivy.graphics.texture import Texture
from kivy.clock import Clock

# define Window profile
Window.fullscreen = 0
Window.size = (375,667)

class CameraData():
    config = {"CAM_ID"  : 0,
              "FPS"     : 29.97,
              "Size"    : (640, 480)}
    cap = cv2.VideoCapture()
    frame = np.zeros(config["Size"],dtype = "uint8")
    frame_gray = np.zeros(config["Size"],dtype = "uint8")
    extructedface = np.zeros((256,256), dtype = "uint8")
    facerects = []

class ClassifierData():
    config = {"ScaleFactor" : 1.1,
              "minNeighbors": 1,
              "minSize"     : (1, 1)}
    dirname = os.path.dirname(os.path.abspath(__file__))
    classifier_name = "../haarcascades/haarcascade_frontalface_alt.xml"
    classifier_path = os.path.normpath(os.path.join(dirname,classifier_name))
    cascade = cv2.CascadeClassifier(classifier_path)

class DrawingData():
    xStart = 0
    yStart = 0
    xEnd = 0
    yEnd = 0
    xUpperLeft = 0
    yUpperLeft = 0
    xLowerRight = 0
    yLowerRight = 0

    drawLayer = np.zeros((480, 640, 3), dtype = "uint8")
    posTouchDown = (0, 0)
    posTouchUp = (0, 0)

    def UpperLeft(self):
#        upperLeft = (min(self.xStart, self.xEnd), min(self.yStart, self.yEnd))
        upperLeft = (min(self.posTouchDown[0], self.posTouchUp[0]), min(self.posTouchDown[1], self.posTouchUp[1]))
        self.xUpperLeft = upperLeft[0]
        self.yUpperLeft = upperLeft[1]
        return upperLeft

    def LowerRight(self):
#        lowerRight = (max(self.xStart, self.xEnd), max(self.yStart, self.yEnd))
        lowerRight = (max(self.posTouchDown[0], self.posTouchUp[0]), max(self.posTouchDown[1], self.posTouchUp[1]))
        self.xLowerRight = lowerRight[0]
        self.yLowerRight = lowerRight[1]
        return lowerRight

class FlagData():
    camera = False
    detectface = False
    drawrect = False

class Test(Widget):
    text_input = StringProperty("test")
    value_input = NumericProperty(1)
    image_input = StringProperty("../Image/lena.jpg")
    texture_input = ObjectProperty()
    texture_input_small = ObjectProperty()
    file_input = StringProperty("NULL")
    # ------------------------------------------------
    color = (255,255,255)
    # ------------------------------------------------

    def __init__(self):
        # enable "Drag and Drop"
        super(Test, self).__init__()
        self._file = Window.bind(on_dropfile=self.Drop)
        # make Instance
        self.Camera = CameraData()
        self.Classifier = ClassifierData()
        self.Draw = DrawingData()
        self.Flag = FlagData()
        # initialize Value
        self.texture_input = Texture.create(self.Camera.config["Size"], colorfmt='bgr')
        self.texture_input_small = Texture.create((256, 256), colorfmt='bgr')
        self.event = Clock

    # TestCode ---------------------------------------
    # ------------------------------------------------
    def PressA(self):
        if self.Flag.camera == False:
            self.activateCamera(self.Camera, self.Classifier, self.Draw, self.Flag)
            self.Flag.camera = True
        elif self.Flag.camera == True:
            self.inactivateCamera(self.Camera)
            self.Flag.camera = False
    def PressB(self):
        self.Flag.detectface = not(self.Flag.detectface)
    def PressC(self):
        self.drawContour(self.Draw)
    # ------------------------------------------------
    # ------------------------------------------------

    # ------------------------------------------------
    # ------------------------------------------------
    def showImage(self, Camera):
        Camera.cap = cv2.VideoCapture(Camera.config["CAM_ID"])
        while True:
            ref, Camera.frame = Camera.cap.read()
            if ref == True:
                cv2.imshow("test",Camera.frame)
                key = cv2.waitKey(50)
                if key == 27:
                    cv2.destroyAllWindows()

    # show Image -------------------------------------
    # ------------------------------------------------
    def activateCamera(self, Camera, Classifier, Draw, Flag):
        Camera.cap = cv2.VideoCapture(Camera.config["CAM_ID"])
        self.event = Clock.schedule_interval(partial(self.updateImage, Camera, Classifier, Draw, Flag), 1/Camera.config["FPS"])
    # ------------------------------------------------
    # ------------------------------------------------

    # show Image -------------------------------------
    # ------------------------------------------------
    def inactivateCamera(self, Camera):
        Camera.cap.release()
        self.event.cancel()
        Camera.frame = np.array(Camera.frame/2, dtype = "uint8")
        texture = Texture.create(Camera.config["Size"], colorfmt = "bgr")
        texture.blit_buffer(Camera.frame.tostring(), colorfmt = "bgr", bufferfmt="ubyte")
        self.texture_input = texture
    # ------------------------------------------------
    # ------------------------------------------------

    # classify Frame ---------------------------------
    # ------------------------------------------------
    def classifyFace(self, Camera, Classifier):
        Camera.frame_gray = cv2.cvtColor(Camera.frame, cv2.COLOR_BGR2GRAY)
        Camera.facerects = Classifier.cascade.detectMultiScale(Camera.frame_gray, scaleFactor = Classifier.config["ScaleFactor"],
                                                               minNeighbors = Classifier.config["minNeighbors"], minSize = Classifier.config["minSize"])
        if len(Camera.facerects) > 0:
            for facerect in Camera.facerects:
                cv2.rectangle(Camera.frame,tuple(facerect[0:2]),tuple(facerect[0:2]+facerect[2:4]),self.color, thickness = 2)
    # ------------------------------------------------
    # ------------------------------------------------

    # extruct Rect -----------------------------------
    # ------------------------------------------------
    def extructRect(self, Camera, Draw, Flag):
        if Flag.drawrect == True:    # 中身があればTrue
            UpperLeft = (int(Draw.xUpperLeft/0.997*640), int((1-(Draw.yUpperLeft-0.542)/0.42)*480))
            LowerRight = (int(Draw.xLowerRight/0.997*640), int((1-(Draw.yLowerRight-0.542)/0.42)*480))
            print(UpperLeft)
            print(LowerRight)
            Camera.extructedface = cv2.resize(Camera.frame[LowerRight[1]:UpperLeft[1],
                                                           UpperLeft[0]:LowerRight[0]],
                                                           (256,256))
            Camera.extructedface = cv2.flip(Camera.extructedface, 0)
            texture = Texture.create((256,256), colorfmt = "bgr")
            texture.blit_buffer(Camera.extructedface.tostring(), colorfmt = "bgr", bufferfmt = "ubyte")
            self.texture_input_small = texture
    # ------------------------------------------------
    # ------------------------------------------------

    # extruct Face -----------------------------------
    # ------------------------------------------------
    def extructFace(self, Camera):
        if len(Camera.facerects) != 0:    # 中身があればTrue
            facerect = Camera.facerects[0]
            Camera.extructedface = cv2.resize(Camera.frame[facerect[1]:facerect[1]+facerect[3],
                                                           facerect[0]:facerect[0]+facerect[2]],
                                                           (256,256))
            Camera.extructedface = cv2.flip(Camera.extructedface, 0)
            texture = Texture.create((256,256), colorfmt = "bgr")
            texture.blit_buffer(Camera.extructedface.tostring(), colorfmt = "bgr", bufferfmt = "ubyte")
            self.texture_input_small = texture
    # ------------------------------------------------
    # ------------------------------------------------

    # update Image -----------------------------------
    # ------------------------------------------------
    def updateImage(self, Camera, Classifier, Draw, Flag, dt):
        ref, Camera.frame = Camera.cap.read()
        if ref == True:
            # Processing
            if Flag.detectface == True:
                self.classifyFace(Camera, Classifier)
                self.extructFace(Camera)
            if Flag.drawrect == True:
                Camera.frame = cv2.addWeighted(Camera.frame, 1, Draw.drawLayer, 1, 0)
                self.extructRect(Camera, Draw, Flag)
            # Show Image
            Camera.frame = cv2.flip(Camera.frame, 0)
            texture = Texture.create(Camera.config["Size"], colorfmt = "bgr")
            texture.blit_buffer(Camera.frame.tostring(), colorfmt = "bgr", bufferfmt="ubyte")
            self.texture_input = texture
    # ------------------------------------------------
    # ------------------------------------------------

    # update Image -----------------------------------
    # ------------------------------------------------
    def drawRect(self, touch, act):
        Camera = self.Camera
        Draw = self.Draw
        if act == "down":
            self.Flag.drawrect = False
            Draw.posTouchDown = touch.spos
            Draw.drawLayer = np.zeros((480, 640, 3), dtype = "uint8")
        elif act == "up":
            self.Flag.drawrect = True
            Draw.posTouchUp = touch.spos
            Draw.UpperLeft()
            Draw.LowerRight()
            UpperLeft = (int(Draw.xUpperLeft/0.997*640), int((1-(Draw.yUpperLeft-0.542)/0.42)*480))
            LowerRight = (int(Draw.xLowerRight/0.997*640), int((1-(Draw.yLowerRight-0.542)/0.42)*480))
            print(UpperLeft)
            cv2.rectangle(Draw.drawLayer, UpperLeft, LowerRight, self.color, thickness = 2)
    # ------------------------------------------------
    # ------------------------------------------------

    # Draw Contour -----------------------------------
    # ------------------------------------------------
    # Safety
    def drawContour(self, Draw):
        if self.file_input == "NULL":
            return 0
        file_name,ext = os.path.splitext(self.file_input)
        window_name = "test"
        def MouseEvent(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.Draw.xStart = x
                self.Draw.yStart = y
                self.image = cv2.imread(self.file_input,cv2.IMREAD_GRAYSCALE)
                print("LeftButtonDown")
            elif event == cv2.EVENT_LBUTTONUP:
                self.Draw.xEnd = x
                self.Draw.yEnd = y
                cv2.rectangle(self.image, self.Draw.UpperLeft(), self.Draw.LowerRight(), self.color, thickness = 2)
        # Show Image
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, MouseEvent)
        self.image = cv2.imread(self.file_input,cv2.IMREAD_GRAYSCALE)
        while True:
            cv2.imshow(window_name,self.image)
            key = cv2.waitKey(10)
            if key == 27:
                cv2.destroyWindow(window_name)
                break
            elif key == 13:
                self.image = cv2.imread(self.file_input,cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(file_name+"_gray.jpg",self.image)
                cv2.destroyWindow(window_name)
                with open("D:/naoto/Python/Image/positive.txt","a") as file:
                    file.write(file_name+"_gray.jpg,"+str(Draw.xUpperLeft)+","+str(Draw.yUpperLeft)+","+str(Draw.xLowerRight)+","+str(Draw.yLowerRight)+"\n")
                break
    # ------------------------------------------------
    # ------------------------------------------------

    # Gray Image Read --------------------------------
    # ------------------------------------------------
    def GrayImageRead(self):
        file_name,ext = os.path.splitext(self.file_input)
        image = cv2.imread(self.file_input,cv2.IMREAD_GRAYSCALE)
        cv2.imshow("window",image)
        cv2.imwrite(file_name+"_gray.jpg",image)
        pass
    # ------------------------------------------------
    # ------------------------------------------------


    # Write File Name --------------------------------
    # ------------------------------------------------
    def WriteFileName(self):
        file_path = self.dir_input + "/*.jpg"
        self.file_names = glob.glob(file_path)
        with open("D:/naoto/Python/Image/gray/positive.txt","w") as file:
            for file_name in self.file_names:
                file.write(file_name+"\n")
        pass
    # ------------------------------------------------
    # ------------------------------------------------



    # TestCode FileDrop ------------------------------
    # ------------------------------------------------
    def Drop(self,window,drop_path):
        if os.path.isdir(drop_path.decode("utf-8")):
            self.dir_input = drop_path.decode("utf-8")
        else:
            self.file_input = drop_path.decode("utf-8")
        self.text_input = drop_path.decode("utf-8")
    # ------------------------------------------------
    # ------------------------------------------------

class TestKivyApp(App):
    def build(self):
        return Test()

# Run
if __name__ == "__main__":
    TestKivyApp().run()
