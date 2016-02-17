import cv2
import numpy, networktables
from networktables import NetworkTable
import logging, random
from heapq import nlargest

# Configure logging
logging.basicConfig(level=logging.DEBUG)
area,height,width,perimeter,solidity,centroidX,centroidY = [0,0,0,0,0,0,0]
ip = "roborio-4561-frc.local"

# Configure NetworkTables
NetworkTable.setIPAddress(ip)
NetworkTable.setClientMode()
NetworkTable.initialize()

# Create "Vision" Subtable
visionTable = NetworkTable.getTable("Vision")

cam = cv2.VideoCapture(0)

# Define thresholding limits
lower_blue = numpy.array([40, 241, 0])
upper_blue = numpy.array([80, 255,255])

# Create windows to display stream on
#cv2.namedWindow("cameraraw", cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow("threshold", cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow("contours", cv2.WINDOW_AUTOSIZE)

while True:

    # Get frame
    _, img = cam.read()

    # Stream raw image
    print(img.shape[:2])
#    cv2.imshow("cameraraw",img)
#    cv2.waitKey(1)

    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Threshold image
    imgthresh = cv2.inRange(hsv, lower_blue, upper_blue)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
##    imgthresh = cv2.erode(imgthresh, element)
##    imgthresh = cv2.dilate(imgthresh, element)
    imgthresh = cv2.morphologyEx(imgthresh, cv2.MORPH_OPEN, element)
    imgthresh = cv2.dilate(imgthresh, element)
    imgthresh = cv2.dilate(imgthresh, element)
    # Stream thresholded image
#    cv2.imshow("threshold",imgthresh)
#    cv2.waitKey(1)

    # Find contours
    _, contours, hierarchy = cv2.findContours(imgthresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    areaList = []
    heightList = []
    widthList = []
    perimeterList = []
    solidityList = []
    centroidXList = []
    centroidYList = []
    # Get important values about each contour
    for contour in contours:
        areaList.append(cv2.contourArea(contour))
        perimeterList.append(cv2.arcLength(contour, True))
        M = cv2.moments(contour)
        centroidXList.append(int(M['m10']/M['m00']))
        centroidYList.append(int(M['m01']/M['m00']))
        x, y, width, height = cv2.boundingRect(contour)
        widthList.append(width)
        heightList.append(height)
        solidityList.append(float(areaList[len(areaList)-1]/cv2.contourArea(cv2.convexHull(contour))))

    areaList = sorted(areaList)
    perimeterList = sorted(perimeterList)
    centroidXList = sorted(centroidXList)
    centroidYList = sorted(centroidYList)
    widthList = sorted(widthList)
    heightList = sorted(heightList)
    solidityList = sorted(solidityList)

    indexList = list(range(1, len(areaList)))

    if len(areaList) > 0:

        contourNum = 1
        passing = False
        while not passing:
            indexes = []
            indexes = nlargest(contourNum, indexList, key = lambda i: areaList[i])
            if len(areaList) > 1:
                index = indexes[len(indexes)-1]
            else:
                index = 0
            if solidityList[index] > 0.25 and solidityList[index] < 0.5:
                HWRatio = heightList[index]/widthList[index]
                if HWRatio > 0.4 and HWRatio < 0.9:
                    passing = True
                else:
                    passing = False
            else:
                passing = False
            if contourNum == len(areaList):
                passing = True
            contourNum += 1

  #      cv2.drawContours(img, [contours[index]], -1, (255, 0, 0), 2)
#        cv2.imshow("contours",img)
 #       cv2.waitKey(1)

        # Get the properties of the contour with the largest area
        area = areaList[index]
        height = heightList[index]
        width = widthList[index]
        perimeter = perimeterList[index]
        solidity = solidityList[index]
        centroidX = centroidXList[index]
        centroidY = centroidYList[index]

    # Publish NumberArrays to NetworkTables
    visionTable.putNumber("area", area)
    visionTable.putNumber("height", height)
    visionTable.putNumber("width", width)
    visionTable.putNumber("perimeter", perimeter)
    visionTable.putNumber("solidity", solidity)
    visionTable.putNumber("centroidX", centroidX)
    visionTable.putNumber("centroidY", centroidY)
    visionTable.putNumber("coolLookingNumber", random.randint(0,99999))
