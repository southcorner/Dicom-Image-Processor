import sys
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QWidget, QMenu, QAction, QFileDialog, QApplication

class mywin(QMainWindow):
    def __init__(self):
        super(mywin, self).__init__()

        self.setWindowTitle("Dicom")
        self.setGeometry(QRect(600,300,400,100) )

        self.setCentralWidget(QWidget(self))
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint);
        self.addMenus()
        self.w = self.h = 512


        self.winCount = 0
        app.aboutToQuit.connect(self.closeApp)

    def addMenus(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        openMenu = QMenu('Open', self)
        impAct = QAction('Open image', self)
        impAct.triggered.connect(self.fileopen)
        openMenu.addAction(impAct)

        exmenu = QMenu('Exit', self)
        exact = QAction('Close File', self)
        exact.triggered.connect(self.closeApp)
        exmenu.addAction(exact)

        fileMenu.addMenu(openMenu)
        fileMenu.addMenu(exmenu)

        editMenu = menubar.addMenu('&Edit')
        impMenu2 = QMenu('Edit', self)
        editMenu.addMenu(impMenu2)


        edit1Menu = QMenu('Lookup Tables', self)
        editMenu.addMenu(edit1Menu)
        impact1 = QAction('Red', self)
        impact1.triggered.connect(self.showRed)
        edit1Menu.addAction(impact1)
        impact2 = QAction('Green', self)
        impact2.triggered.connect(self.showGreen)
        edit1Menu.addAction(impact2)
        impact3 = QAction('Light', self)
        impact3.triggered.connect(self.showLighter)
        edit1Menu.addAction(impact3)
        impact4 = QAction('XRay', self)
        impact4.triggered.connect(self.showXray)
        edit1Menu.addAction(impact4)



        impacta = QAction('Flip Vertical', self)
        impacta.triggered.connect(self.flipvertical)
        impMenu2.addAction(impacta)
        impactc = QAction('Equalised Image', self)
        impactc.triggered.connect(self.histeq)
        impMenu2.addAction(impactc)
        impactd = QAction('Brightness Adjustment', self)
        impactd.triggered.connect(self.nothing)
        impMenu2.addAction(impactd)
        impactk = QAction('Gamma Correction', self)
        impactk.triggered.connect(self.nothing)
        impMenu2.addAction(impactk)
        impacte = QAction('Laplacian Image', self)
        impacte.triggered.connect(self.laplacian)
        impMenu2.addAction(impacte)
        impactf = QAction('Sharpen Image', self)
        impactf.triggered.connect(self.nothing)
        impMenu2.addAction(impactf)
        impactg = QAction('Magnitude Spectrum', self)
        impactg.triggered.connect(self.magn)
        editMenu.addAction(impactg)
        impacth = QAction('Segmented Image', self)
        impacth.triggered.connect(self.segment)
        impMenu2.addAction(impacth)
        impactz = QAction('Morphological Processing', self)
        impactz.triggered.connect(self.nothing)
        impMenu2.addAction(impactz)
        impacty = QAction('Show Gradient', self)
        impacty.triggered.connect(self.gradient)
        impMenu2.addAction(impacty)
        impactx = QAction('Show Dilation', self)
        impactx.triggered.connect(self.dilate)
        impMenu2.addAction(impactx)
        impactm = QAction('Erosion', self)
        impactm.triggered.connect(self.erosion)
        impMenu2.addAction(impactm)
        # impactn = QAction('Low Pass Filter', self)
        # impactn.triggered.connect(self.nothing)
        # impMenu2.addAction(timpactn)id P
        ## impacto = QAction('Mass Filer', self)
        # impacto.triggered.connect(self.nothing)
        # impMenu2.addAction(impacto)


        histmenu = QMenu("Histogram", self)
        editMenu.addMenu(histmenu)
        impactb = QAction('Histogram', self)
        impactb.triggered.connect(self.histogram)
        histmenu.addAction(impactb)
        impacteq = QAction("Equalised Histogram", self)
        impacteq.triggered.connect(self.eqhist)
        histmenu.addAction(impacteq)

        binmenu = QMenu('Binary', self)
        editMenu.addMenu(binmenu)
        impactj = QAction('Select Image', self)
        impactj.triggered.connect(self.binary)
        binmenu.addAction(impactj)
        impacti = QAction('Add Images', self)
        impacti.triggered.connect(self.addImage)
        binmenu.addAction(impacti)
        impactj = QAction("Bitwise Operation", self)
        impactj.triggered.connect(self.bitwise)
        binmenu.addAction(impactj)


        self.show()

    def fileopen(self):
        dlg = QFileDialog()
        if(dlg.exec()):
            openfilepath = dlg.selectedFiles()
            print(len(openfilepath))
            if(len(openfilepath) == 1):
                dicomDataset = pydicom.read_file(openfilepath[0])
                self.Image1 = self.showImage(openfilepath[0], dicomDataset)

        else:
            print("No file name chosen!!")
            return

    def DicomToRGB(self,img, bt, wt):
        # enforce boundary conditions
        img = np.clip(img, bt, wt)
        img = np.multiply(img, 255 / (wt - bt)).astype(np.int16)
        rgb_img = np.stack([img] * 3, axis=-1)
        return rgb_img

    def showImage(self,path,dicomf):
        self.filename = path.split("/")[-1]
        print(self.filename)
        self.npimage = self.DicomToRGB(dicomf.pixel_array, 0, 1400)
        self.slice1Copy = np.uint8(self.npimage)
        self.openImage = cv2.cvtColor(self.slice1Copy, cv2.COLOR_RGB2GRAY)
        self.openImage = cv2.resize(self.openImage,(self.w,self.h))
        self.winCount = self.winCount+1
        imgTitle = self.filename+" "+str(self.winCount)
        cv2.imshow(imgTitle, self.openImage)
        return self.openImage

    def gamma(self, x):
          # location of the image
        original = self.slice1Copy.copy()
        self.gamma = x  # change the value here to g
        adjusted = self.adjust_gamma(original)
        cv2.imshow("Gamma Image", adjusted)
        cv2.waitKey(0)

    def adjust_gamma(self, image):
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def histogram(self):
        histogram = plt.hist(self.openImage.ravel(), 256, [0, 256])
        show = plt.show(histogram)
        return show

    def eqhist(self):
        histogram = plt.hist(self.eq.ravel(), 256, [0, 256])
        plt.show(histogram)

    def histeq(self):
        img_output = self.openImage.copy()
        self.eq = cv2.equalizeHist(img_output)
        cv2.imshow("Equalized Histogram", self.eq)
        cv2.waitKey(0)

    def showRed(self):
        redCopy = self.slice1Copy
        self.red = cv2.cvtColor(redCopy, cv2.COLOR_RGB2HSV_FULL)
        self.red = cv2.resize(self.red, (self.w, self.h))
        cv2.imshow("Red "+self.filename, self.red)


    def showGreen(self):
        greenCopy = self.slice1Copy.copy()
        self.green = cv2.cvtColor(greenCopy, cv2.COLOR_RGB2HLS)
        self.green = cv2.resize(self.green, (self.w, self.h))
        cv2.imshow("Green "+self.filename, self.green)


    def showLighter(self):
        lighterCopy = self.slice1Copy
        self.light = cv2.cvtColor(lighterCopy, cv2.COLOR_RGB2YUV)
        self.light = cv2.resize(self.light, (self.w, self.h))
        cv2.imshow("Light "+self.filename, self.light)


    def showXray(self):
        xrayCopy = self.slice1Copy
        self.xray = cv2.cvtColor(xrayCopy, cv2.COLOR_RGB2XYZ)
        self.xray = cv2.resize(self.xray, (self.w, self.h))
        cv2.imshow("Xray "+self.filename, self.xray)

    #
    # def duplicate(self):
    #     orig = cv2.cvtColor(self.slice1Copy, cv2.COLOR_RGB2GRAY)
    #     orig = cv2.resize(orig, (self.w, self.h))
    #     cv2.imshow("Duplicate "+self.filename, orig)
    #     cv2.setMouseCallback("Duplicate Of  " + self.filename, self.duplicateImageClickTransform)
    #
    # def redImageClickTransform(self, event, x, y, flags, param):
    #     if (event == cv2.EVENT_LBUTTONDOWN):
    #         print("Red Image Clicked at ", x, " ", y)
    #         self.redActive = True
    #
    # def greenImageClickTransform(self, event, x, y, flags, param):
    #     if (event == cv2.EVENT_LBUTTONDOWN):
    #         print("Green Image Clicked at ", x, " ", y)
    #         self.greenActive = True
    #
    # def lightImageClickTransform(self, event, x, y, flags, param):
    #     if (event == cv2.EVENT_LBUTTONDOWN):
    #         print("Light Image Clicked at ", x, " ", y)
    #         self.lightActive = True
    #
    # def xrayImageClickTransform(self, event, x, y, flags, param):
    #     if (event == cv2.EVENT_LBUTTONDOWN):
    #         print("Xray Image Clicked at ", x, " ", y)
    #         self.xrayActive = True

    def flipvertical(self):


        self.imgTitle = "flip Vert" + self.filename
        self.flippedCopy = self.slice1Copy
        self.flippedCopy = self.flippedCopy[::-1]
        self.flippedCopy = cv2.resize(self.flippedCopy, (self.w, self.h))
        cv2.imshow(self.imgTitle, self.flippedCopy)

    def lowFilter(self,x):
        img = self.slice1Copy.copy()
        blur = cv2.blur(img, (2,x))
        cv2.imshow("Blur", blur)

    def sharpen(self,x):
        img = self.openImage.copy()
        median =  cv2.GaussianBlur(img, (3, 3), 0)
        sharp = cv2.addWeighted(img, x+1, median, -x, 0)
        cv2.imshow("sharp", sharp)

    # def MedianFilter(self, x):
    #     img = self.openImage.copy()
    #     median = cv2.GaussianBlur(img, (x, x), 0)
    #     cv2.imshow("median", median)

    def laplacian(self):
        img = self.openImage.copy()
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
        plt.imshow(laplacian, cmap='gray')
        plt.show()

    def morphology(self,x):
        origImg = self.openImage.copy()
        ret, thresh = cv2.threshold(origImg, 0, x, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.kernel = np.ones((3, 3), np.uint8)
        self.opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel, iterations=2)
        cv2.imshow("open", self.opening)

    def dilate(self):
        img = self.openImage.copy()
        kernel = np.ones((5,5), np.uint8)
        self.dilate = cv2.dilate(img, kernel, iterations=1)
        cv2.imshow("dilate",self.dilate)

    def erosion(self):
        img = self.openImage.copy()
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        cv2.imshow("Erosion", erosion)

    def gradient(self):
        img = self.openImage.copy()
        kernel = np.ones((3,3), np.uint8)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow("Gradient", gradient)

    def magn(self):
        img = self.openImage.copy()
        img_float32 = np.float32(img)
        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum')
        plt.show()

    def segment(self):
        origImg = self.openImage.copy()
        ret, thresh = cv2.threshold(origImg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
                                   iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, cv2.THRESH_BINARY)
        print(sure_fg)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        print(markers)
        markers = markers + 1
        markers[unknown == 255] = 0
        markedImage = self.slice1Copy.copy()
        markers = cv2.watershed(self.slice1Copy, markers)
        markedImage[markers == -1] = [255, 0, 0]
        cv2.imshow("Segmented Image", markedImage)

    def binary(self):
        dlg1 = QFileDialog()
        if (dlg1.exec()):
            openfilepath1 = dlg1.selectedFiles()
            print(len(openfilepath1))
            if (len(openfilepath1) == 1):
                dicomDataset1 = pydicom.read_file(openfilepath1[0])
                print("okay")
                self.Image = self.showImage(openfilepath1[0], dicomDataset1)
                print("okay")
        else:
            print("Choose a file for the Binary Operation!!")
            return

    def addImage(self):
        dst = cv2.addWeighted(self.Image1, 0.5, self.Image, 0.5, 0)
        cv2.imshow('Addition', dst)

    def bitwise(self):
        rows, cols = self.Image.shape
        roi = self.Image1[0:rows, 0:cols]
        # Now create a mask of logo and create its inverse mask also
        img2gray = self.Image
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(self.Image, self.Image1, mask=mask)
        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        cv2.imshow("Bitwise", dst)
        cv2.waitKey(0)

    def brightness(self, x):
        img = self.slice1Copy.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)


        lim = 255 - x
        v[v > lim] = 255
        v[v <= lim] += x

        self.final_hsv = cv2.merge((h, s, v))
        self.img = cv2.cvtColor(self.final_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("Bright Image", self.img)

    def nothing(self):
        cv2.namedWindow('image')

        cv2.createTrackbar('Brightness', 'image', 0, 300, self.brightness)
        cv2.createTrackbar("Gamma", "image", 1, 10, self.gamma)
        cv2.createTrackbar("Filters", "image", 1, 60, self.lowFilter)
        cv2.createTrackbar("Morph Open", "image", 1, 255, self.morphology)
        cv2.createTrackbar("Sharpen", "image", 0, 50, self.sharpen)
        # cv2.createTrackbar("Median", "image", 0, 5, self.MedianFilter)



    def closeApp(self):
        quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()
    # print("The screen resolution is ", width, "x ", height)
    ex = mywin()
    ex.show()
    cv2.waitKey(0)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # ESC0
            print("Esc")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            ex.closeApp()
            break
    sys.exit(app.exec_())

