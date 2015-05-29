import numpy as np


from matplotlib import pyplot as plt
from matplotlib import colors as cl
import matplotlib.cm as cm


from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt
from skimage import exposure

import cv2
import time
import math

import glob

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    
    def __init__(self, center, radius,color):
        self.center = center
        self.radius = radius
        self.color = color
        
    def area(self):
        return math.pi * self.radius**2
    
    def contains(self, point):
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        #print ("x courant =",math.sqrt(dx*dx + dy*dy),"Rayon=",self.radius)
        return (math.sqrt(dx*dx + dy*dy) <= self.radius) 
    
    def getradius(self):
        return self.radius
    def getcenter(self):
        return self.center
    def getcolor(self):
        return self.color
    def setradius(self, r):
        self.radius = r
    def setcenter(self, c):
        self.center = c


def imageCaracterisation(img):
    imgMIN = img.min()
    imgMAX = img.max()

    print str(img.dtype) + str(type(img))+ str(img.shape) 
    print ("MIN = %d   MAX = %d " % (imgMIN,imgMAX))
    return imgMIN,imgMAX

def grayScale(img):
    imgHSV = convertHSV(img)
    #print "Image HSV"
    #imageCaracterisation(imgHSV)
    imgBW  = imgHSV[:,:,2]

    return imgBW.astype(np.uint8)

def loadImages(path1='./Holes/*.jpeg',path2= './Background/*.jpeg' ):
    imagesTrous = glob.glob(path1)
    imagesBack = glob.glob(path2)

    print imagesTrous[0]
    print type(imagesTrous[0])
    print len(imagesTrous)
    imgTrousList = []
    imgBackList = []


    print "Processing img Trous"

    for fname in imagesTrous:
        print "fname = " + fname
        img = cv2.imread(fname)
        #img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

        #gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_image = grayScale(img)
        #print "Min / max " , min(gray_image),max(gray_image)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_image)
        print " min_val, max_val, min_loc, max_loc ", min_val, max_val, min_loc, max_loc 
        gray_image_low = lowPass(gray_image)
        #gray_image_low_morph = morphoNoiseRemoval(gray_image_low)
        imgTrousList.append(gray_image_low)
    
    print "Processing img Background"

    print imagesBack[0]
    print type(imagesBack[0])

    for fname2 in imagesBack:
        print "fname2 = " + str(fname2)
        img2 = cv2.imread(fname2)
        #img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5) 
        gray_image = grayScale(img2)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_image)
        print " min_val, max_val, min_loc, max_loc ", min_val, max_val, min_loc, max_loc 
        gray_image_low = lowPass(gray_image)
        #gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        #gray_image_low_morph = morphoNoiseRemoval(gray_image_low)
        imgBackList.append(gray_image_low)

    print "imgBackList = "  
    print type(imgBackList) 
    print len(imgBackList) 

    return imgTrousList,imgBackList


def loadImagesNOList():

    img = cv2.imread('Backgrounf.jpeg')
    gray_image_back = grayScale(img)

    img = cv2.imread('Holes.jpeg')
    gray_image_holes = grayScale(img)

    return gray_image_back,gray_image_holes


def convertHSV(img):
    "convert RGBA into HSV color Space"
    if img.shape[2]==4:
        return cl.rgb_to_hsv(img[:,:,0:3])
    else:
        if img.shape[2]==3:
            return cl.rgb_to_hsv(img)
        else:
            print ("Image format not supported")


def averageFrames(imgList):
    averageFrame = np.zeros_like(imgList[0],dtype=np.int16)#,dtype=np.uint16)
    """
    width = imgList[0].shape[0]
    height = imgList[1].shape[1]


    for i in range(width):
        for j in range(height):
            sumPixels = 0 
            for img in imgList:
                sumPixels += img[i,j]

            averageFrame[i,j] =  sumPixels *1.0 / len(imgList)

    print "End average Frame"

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(averageFrame)
    print " min_val, max_val, min_loc, max_loc ", min_val, max_val, min_loc, max_loc 
    """
    print "Hello"
    print type(averageFrame)
    print averageFrame.dtype
    for image in imgList:
        averageFrame += image

    averageFrame = averageFrame/len(imgList)
    
    return averageFrame
 

def show(img1,img2,img3=None):

    plt.figure(1)
    plt.subplot(131)
    plt.imshow(img1, cmap = cm.gray)
    plt.title('Avec Trous')
    plt.subplot(132)
    plt.imshow(img2, cmap = cm.gray)
    plt.title('Background')
    plt.subplot(133)
    plt.imshow(img3, cmap = cm.gray)
    plt.title('Scolytes Holes')
    plt.show()
    return

def lowPass(img):
    #kernel = np.ones((5,5),np.float32)/25
    #imgLow = cv2.filter2D(img,-1,kernel)
    imgLow = cv2.blur(img,(10,10))
    #imgLow = cv2.GaussianBlur(img,(5,5),0)
    #imgLow = cv2.bilateralFilter(img,9,75,75)
    #imgLow = cv2.medianBlur(img,5)

    return imgLow

def morphoNoiseRemoval(img):
    "Removes noise by succession of 5 opening/closing morphological operators"
    kernel = np.ones((5,5),np.uint8)

    #img = cv2.filter2D(img,-1,kernel)
    #img = cv2.dilate(img,kernel,iterations = 1) #dilatation

    for i in range(0,5):
        #img = opening2(img, square(3))
        #img = closing2(img, square(3))
        img = cv2.erode(img,kernel,iterations = 1)  #erosion
        img = cv2.dilate(img,kernel,iterations = 1) #dilatation

        img = cv2.dilate(img,kernel,iterations = 1) #dilatation
        img = cv2.erode(img,kernel,iterations = 1)  #erosion 

    return img

def objectLabeling(img):
    # find connected components
    labeled, nr_objects = ndimage.label(img > 0) 
    print "Number of objects is %d " % nr_objects

    sizes = np.bincount(labeled.ravel())

    #print sizes

    #plotHistogram(sizes[4:])
    return labeled, nr_objects, sizes

def objectRemovalByLabel(imgLabeled, label,circle,img,objectSize = 150):

    if label == 0:
        return imgLabeled,img

    borneInfX= (circle.getcenter()[0]-objectSize).astype(np.int16)
    borneSupX= (circle.getcenter()[0]+objectSize).astype(np.int16)
    borneInfY= (circle.getcenter()[1]-objectSize).astype(np.int16)
    borneSupY= (circle.getcenter()[1]+objectSize).astype(np.int16)
 
    if borneInfX < 0:
        borneInfX =0
    if borneSupX > imgLabeled.shape[0]:
        borneSupX =imgLabeled.shape[0]
    if borneInfY < 0:
        borneInfY =0
    if borneSupY > imgLabeled.shape[1]:
        borneSupY =imgLabeled.shape[1]

    for i in range(borneInfX,borneSupX):
        for j in range(borneInfY,borneSupY):
            if imgLabeled[i,j] ==  label:
                imgLabeled[i,j] = 0
                img[i,j]=0

    return imgLabeled,img


def objectRemoval(sizes,imgLabeled,threshold1 = 500,threshold2 = 0):
    mask_sizes = (sizes < threshold1) & (sizes > threshold2)
    mask_sizes[0] = 0

    img_cleaned = mask_sizes[imgLabeled]

    return img_cleaned



def drawCircle(circle,img,color):
    "Draw a circle of a given color on a map"

    borneInfX= (circle.getcenter()[0]-circle.getradius()).astype(np.int16)
    borneSupX= (circle.getcenter()[0]+circle.getradius()).astype(np.int16)
    borneInfY= (circle.getcenter()[1]-circle.getradius()).astype(np.int16)
    borneSupY=(circle.getcenter()[1]+circle.getradius()).astype(np.int16)
    #print ("Dans drawCircle", circle.getcenter()[0],circle.getcenter()[1],circle.getradius())
    #print (borneInfX,borneSupX,borneInfY,borneSupY)
    if borneInfX < 0:
        borneInfX =0
    if borneSupX > img.shape[0]:
        borneSupX =img.shape[0]
    if borneInfY < 0:
        borneInfY =0
    if borneSupY > img.shape[1]:
        borneSupY =img.shape[1]

    for i in range (borneInfX,borneSupX):
        for j in range (borneInfY,borneSupY):
            #print ("point dans cercle? ",circle.contains ([i,j]))
            if circle.contains ([i,j]):
                img[i,j] = color
                #print("coloration")"""
    return img

def selectRoundShapes(imgToProcess,threshold = 10 ):
    "Select round shapes of a minimum threshold on a binary image"
    
    img = np.copy(imgToProcess)
    labeled, nr_objects,sizes = objectLabeling(img)


    distMap= distance_transform_edt(img)#calculate the first distMap
    go = True

    listCircle=[]
    cmpt = 0 
    while (go):
            print "In the while " + str(cmpt)
            cmpt += 1
            maxIndex = np.unravel_index(distMap.argmax(), distMap.shape) #recherche de l'indice du maximum de distMap
            circle = Circle ([maxIndex[0],maxIndex[1]],distMap[maxIndex[0],maxIndex[1]],2)
            listCircle.append(circle)#memorise les centres et rayon des cercles
            
            label = labeled [circle.getcenter()[0],circle.getcenter()[1]]
            print "size of object" + str(sizes[label]) + "  Label = " + str (label)
            labeled,img = objectRemovalByLabel(labeled, label,circle,img,sizes[label])
            

            if distMap[maxIndex[0],maxIndex[1]] < threshold:
                #If circle size smaller threshold
                go = False

            distMap = distMapUpdate(distMap,img, circle)

    return listCircle

def ConsiderLimitsFrame(lowerLeftCoordX,lowerLeftCoordY,radius,binaryImg):
    "Returns the limits of the dist map to update considering the limits of the image"
    shapeX = binaryImg.shape[0]
    shapeY = binaryImg.shape[1]
    lowerX = lowerLeftCoordX
    lowerY = lowerLeftCoordY
    limitX = 4*radius
    limitY = 4*radius
    
    if (lowerLeftCoordX < 0):
        limitX = 4*radius +  lowerLeftCoordX
        lowerX = 0

    if (lowerLeftCoordY < 0):
        limitY = 4*radius +  lowerLeftCoordY
        lowerY = 0
        
    if ((lowerLeftCoordX + 4*radius) > shapeX):
        limitX =  shapeX - lowerLeftCoordX 
        
    if ((lowerLeftCoordY + 4*radius) > shapeY):
        limitY =  shapeY - lowerLeftCoordY
        
    return lowerX,lowerY,limitX,limitY

def distMapUpdate(distMap, binaryImg, placedCircle):
    "Update a part of the distance map"

    radiusCirc = (placedCircle.radius).astype(np.int16)
    circCenter = placedCircle.center
    lowerLeftCoordX = circCenter[0] - 2* radiusCirc
    lowerLeftCoordY = circCenter[1] - 2* radiusCirc
    mapToUpdate = np.ones((radiusCirc*4,radiusCirc*4))
    #print lowerLeftCoordX, lowerLeftCoordY
    #print radiusCirc
    #print mapToUpdate.shape
    
    lowerLeftCoordX,lowerLeftCoordY,limitX,limitY = ConsiderLimitsFrame(lowerLeftCoordX,lowerLeftCoordY,radiusCirc,binaryImg)
    
    mapToUpdate = np.ones((limitX,limitY))

    for i in range(0,limitX):
        for j in range(0,limitY):
            mapToUpdate[i,j] = binaryImg [lowerLeftCoordX+i,lowerLeftCoordY+j]
    
    distMapUpdate = distance_transform_edt(mapToUpdate)
    
    for i in range(limitX):
        for j in range(limitY):
            distMap[lowerLeftCoordX+i,lowerLeftCoordY+j] = distMapUpdate [i,j] 
    
    return distMap

def holesOnOriginal(listOfHoles,imgOrigin):
    img = np.copy(imgOrigin)
    #imgTest = np.zeros_like(imgOrigin, dtype=np.uint8)

    for i in range (0,len(listOfHoles)):
        print str(listOfHoles[i].getcenter())+"    " + str(listOfHoles[i].getradius())
        cv2.circle(img, (listOfHoles[i].getcenter()[1],listOfHoles[i].getcenter()[0]), 10,(255, 0, 0), thickness=2)

    return img




def main ():
    list_holes_Ordered = []
    trousList,backgroundList = loadImages('./Trou1/*.jpeg','./Back1/*.jpeg')
    #trousList,backgroundList = loadImagesNOList()

    #trousList_low = lowPass(trousList)
    #backgroundList_low = lowPass(backgroundList)
    
    trouAVG = averageFrames(trousList)
    trousOrigin = trouAVG
    backAVG = averageFrames(backgroundList)
    trouAVG = exposure.equalize_hist(trouAVG)
    backAVG = exposure.equalize_hist(backAVG)

    """
    img = (255*img).astype(np.uint8)
    background =  (255*background).astype(np.uint8)
    """

    imageCaracterisation(trouAVG)
    imageCaracterisation(backAVG)

    imgDiff = trouAVG - backAVG
    #imgDiff = trousList_low - backgroundList_low
    #cv2.imwrite('imgDiff2.png',imgDiff)
    #show(trouAVG,backAVG,imgDiff)

    #On remarque un leger point gris. Ptet essayer de trouver un interval qui permet de recup que ce gris final. Sinon, il y a essentiellement du Blanc tres blanc et du noir tres noir.

    imgDiffCleaned =  np.zeros_like(imgDiff)
    #imgDiffCleaned = imgDiff[imgDiff > 0]
    print "imgDiffCleaned",imgDiffCleaned.shape
    #imgDiffTest3 = np.zeros_like(imgDiff)


    for i in range(imgDiff.shape[0]):
        for j in range(imgDiff.shape[1]):
            if imgDiff[i,j] < -0.1 :
                imgDiffCleaned[i,j] = 1
            else:
                imgDiffCleaned[i,j] = 0

    #imgDiffCleaned = [(imgDiff < 230) & (imgDiff > 180)]
    
    """
    plt.figure(2)
    plt.subplot(121)
    plt.imshow(imgDiffTest3, cmap = cm.gray)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(imgDiffCleaned, cmap = cm.gray)
    plt.colorbar()
    plt.show()


    print "Before plotting figures"
    plt.figure(1)
    plt.imshow(imgDiff, cmap = cm.gray)
    plt.colorbar()


    plt.figure(2)
    plt.imshow(imgDiffCleaned, cmap = cm.gray)
    plt.colorbar()
    """

    #show(trouAVG,backAVG,imgDiff)
    
    #plt.show()
    imgDiffCleaned = morphoNoiseRemoval(imgDiffCleaned)

    labeled,nr_Objects,sizes = objectLabeling(imgDiffCleaned)

    img_cleaned = objectRemoval(sizes,labeled)
    #img_cleaned = objectRemovalMinSize(sizes,labeled)
    """
    plt.figure(5)
    plt.imshow(img_cleaned, cmap = cm.gray)
    plt.colorbar()
    """

    list_Holes = selectRoundShapes(img_cleaned)
    
    list_holes_Ordered.append(list_Holes)

    imgWithHoles = holesOnOriginal(list_Holes,trousOrigin)
    
    plt.figure(1)
    plt.imshow(imgWithHoles, cmap = cm.gray)
    plt.title('Image Trou 1')
    plt.show()

    """
    plt.figure(4)
    plt.imshow(trousOrigin, cmap = cm.gray)
    plt.show()
    """

    trousList,backgroundList = loadImages('./Trou2/*.jpeg','./Trou1/*.jpeg')
    #trousList,backgroundList = loadImagesNOList()

    #trousList_low = lowPass(trousList)
    #backgroundList_low = lowPass(backgroundList)
    
    trouAVG = averageFrames(trousList)
    trousOrigin = trouAVG
    backAVG = averageFrames(backgroundList)
    trouAVG = exposure.equalize_hist(trouAVG)
    backAVG = exposure.equalize_hist(backAVG)

    """
    img = (255*img).astype(np.uint8)
    background =  (255*background).astype(np.uint8)
    """

    imageCaracterisation(trouAVG)
    imageCaracterisation(backAVG)

    imgDiff = trouAVG - backAVG
    #imgDiff = trousList_low - backgroundList_low
    #cv2.imwrite('imgDiff2.png',imgDiff)
    #show(trouAVG,backAVG,imgDiff)

    #On remarque un leger point gris. Ptet essayer de trouver un interval qui permet de recup que ce gris final. Sinon, il y a essentiellement du Blanc tres blanc et du noir tres noir.

    imgDiffCleaned =  np.zeros_like(imgDiff)
    #imgDiffCleaned = imgDiff[imgDiff > 0]
    print "imgDiffCleaned",imgDiffCleaned.shape
    #imgDiffTest3 = np.zeros_like(imgDiff)


    for i in range(imgDiff.shape[0]):
        for j in range(imgDiff.shape[1]):
            if imgDiff[i,j] < -0.1 :
                imgDiffCleaned[i,j] = 1
            else:
                imgDiffCleaned[i,j] = 0

    #imgDiffCleaned = [(imgDiff < 230) & (imgDiff > 180)]
    
    """
    plt.figure(2)
    plt.subplot(121)
    plt.imshow(imgDiffTest3, cmap = cm.gray)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(imgDiffCleaned, cmap = cm.gray)
    plt.colorbar()
    

    print "Before plotting figures"
    plt.figure(1)
    plt.imshow(imgDiff, cmap = cm.gray)
    plt.colorbar()


    plt.figure(2)
    plt.imshow(imgDiffCleaned, cmap = cm.gray)
    plt.colorbar()
    """
    #show(trouAVG,backAVG,imgDiff)
    
    #plt.show()
    imgDiffCleaned = morphoNoiseRemoval(imgDiffCleaned)

    labeled,nr_Objects,sizes = objectLabeling(imgDiffCleaned)

    img_cleaned = objectRemoval(sizes,labeled)
    #img_cleaned = objectRemovalMinSize(sizes,labeled)

    plt.figure(5)
    plt.imshow(img_cleaned, cmap = cm.gray)
    plt.colorbar()
    
    list_Holes = selectRoundShapes(img_cleaned)

    list_holes_Ordered.append(list_Holes)

    imgWithHoles = holesOnOriginal(list_Holes,trousOrigin)

    
    plt.figure(2)
    plt.imshow(imgWithHoles, cmap = cm.gray)
    plt.title('Image Trous 2')
    plt.show()
    
    trousList,backgroundList = loadImages('./Trou3/*.jpeg','./Trou2/*.jpeg')
    #trousList,backgroundList = loadImagesNOList()

    #trousList_low = lowPass(trousList)
    #backgroundList_low = lowPass(backgroundList)
    
    trouAVG = averageFrames(trousList)
    trousOrigin = trouAVG
    backAVG = averageFrames(backgroundList)
    trouAVG = exposure.equalize_hist(trouAVG)
    backAVG = exposure.equalize_hist(backAVG)

    """
    img = (255*img).astype(np.uint8)
    background =  (255*background).astype(np.uint8)
    """

    imageCaracterisation(trouAVG)
    imageCaracterisation(backAVG)

    imgDiff = trouAVG - backAVG
    #imgDiff = trousList_low - backgroundList_low
    #cv2.imwrite('imgDiff2.png',imgDiff)
    #show(trouAVG,backAVG,imgDiff)

    #On remarque un leger point gris. Ptet essayer de trouver un interval qui permet de recup que ce gris final. Sinon, il y a essentiellement du Blanc tres blanc et du noir tres noir.

    imgDiffCleaned =  np.zeros_like(imgDiff)
    #imgDiffCleaned = imgDiff[imgDiff > 0]
    print "imgDiffCleaned",imgDiffCleaned.shape
    #imgDiffTest3 = np.zeros_like(imgDiff)


    for i in range(imgDiff.shape[0]):
        for j in range(imgDiff.shape[1]):
            if imgDiff[i,j] < -0.1 :
                imgDiffCleaned[i,j] = 1
            else:
                imgDiffCleaned[i,j] = 0

    #imgDiffCleaned = [(imgDiff < 230) & (imgDiff > 180)]
    
    """
    plt.figure(2)
    plt.subplot(121)
    plt.imshow(imgDiffTest3, cmap = cm.gray)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(imgDiffCleaned, cmap = cm.gray)
    plt.colorbar()
    

    print "Before plotting figures"
    plt.figure(1)
    plt.imshow(imgDiff, cmap = cm.gray)
    plt.colorbar()


    plt.figure(2)
    plt.imshow(imgDiffCleaned, cmap = cm.gray)
    plt.colorbar()

    #show(trouAVG,backAVG,imgDiff)
    """

    #plt.show()
    imgDiffCleaned = morphoNoiseRemoval(imgDiffCleaned)

    labeled,nr_Objects,sizes = objectLabeling(imgDiffCleaned)

    img_cleaned = objectRemoval(sizes,labeled)
    #img_cleaned = objectRemovalMinSize(sizes,labeled)
    """
    plt.figure(5)
    plt.imshow(img_cleaned, cmap = cm.gray)
    plt.colorbar()
    """
    list_Holes = selectRoundShapes(img_cleaned)
    
    list_holes_Ordered.append(list_Holes)

    imgWithHoles = holesOnOriginal(list_Holes,trousOrigin)

    plt.figure(3)
    plt.imshow(imgWithHoles, cmap = cm.gray)
    plt.title('Image trous 3')
    plt.show()

    """
    plt.figure(3)
    plt.subplot(121)
    plt.imshow(labeled)
    plt.title('labeled')
    plt.subplot(122)
    plt.imshow(imgWithHoles, cmap = cm.gray)
    plt.title('imgTh')
    plt.figure(4)
    plt.imshow(trousOrigin, cmap = cm.gray)
    plt.show()
    """
    print "list_holes_Ordered = ",list_holes_Ordered

    imgWithHoles = holesOnOriginal(list_holes_Ordered[0],trousOrigin)
    imgWithHoles = holesOnOriginal(list_holes_Ordered[1],imgWithHoles)
    imgWithHoles = holesOnOriginal(list_holes_Ordered[2],imgWithHoles)

    plt.figure(4)
    plt.imshow(imgWithHoles, cmap = cm.gray)
    plt.title('Image trous Total')
    plt.show()

    #imgWithHoles = holesOnOriginal(list_holes_Ordered[0],imgWithHoles)


    return 0

if __name__ == "__main__":
    main()