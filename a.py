#!/usr/bin/env python

# Code on Github repository: https://github.com/jmssalas/Artificial-Vision-Exercises-2017

# Press 'Esc' key for exit
# Press 'Enter' key for process ROI
# Use mouse for select ROI

##################################
## PROBLEM STATEMENT (IN SPANISH)
## -------------------------------
##
## Programa de reconocimiento de matrículas con la webcam.
##################################

import numpy             as np
import numpy.fft         as fft
import cv2               as cv


programName = 'licence-plate-recognition'   # Program name

# Key's code
escKey          = 27    # Escape key code
enterKey        = 10    # Enter key code


drawing, lButtonUp = False, False   # - 'drawing' indicates that ROI rectangle is drawing
                                    # - 'lButtonUp' indicates that when left button up event happens,
                                    #    if mouse move event happens, then ROI rectangle doesn't change.

initialPosition = -1                        # Initial position

x0, y0 = initialPosition, initialPosition   # ROI's Initial position
xf, yf = initialPosition, initialPosition   # ROI's Final position

# Mouse event handler
def mouseEventHandler(event, x, y, flags, param):
    global x0, y0, xf, yf, drawing, lButtonUp

    if event == cv.EVENT_LBUTTONDOWN:  # Indicate init ROI rectangle
        drawing = True
        lButtonUp = False
        x0, y0 = x, y
        xf, yf = x0, y0

    elif event == cv.EVENT_MOUSEMOVE:  # Indicate changes of ROI rectangles
        if drawing and not lButtonUp:
            xf, yf = x, y

    elif event == cv.EVENT_LBUTTONUP:  # Indicate final ROI rectangle
        if drawing:
            xf, yf = x, y
            # Check ROI's points sign
            if yf < y0:
                aux = y0; y0 = yf; yf = aux

            if xf < x0:
                aux = x0; x0 = xf; xf = aux

            lButtonUp = True

# Set window's name
cv.namedWindow(programName)
# Set mouseCallback to mouseEventHandler
cv.setMouseCallback(programName, mouseEventHandler)

# Function which draw ROI selected above 'frame' param with
# first vertex 'p0' and second vertex 'p1' and color 'color' param
def drawROI(frame, p0, pf, color):
    cv.rectangle(frame, p0, pf, color)



# Utils functions
def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True) >= 0

def fixOrientation(x):
    if orientation(x):
        return x
    else:
        return np.flipud(x)

# Extract contours of 'g' image
def extractContours(g, minlen=50, holes=False):
    if holes:
        mode = cv.RETR_CCOMP
    else:
        mode = cv.RETR_EXTERNAL

    gt = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, -10)
    a, contours, b = cv.findContours(gt.copy(), mode, cv.CHAIN_APPROX_NONE)
    ok = [fixOrientation(c.reshape(len(c), 2)) for c in contours if cv.arcLength(c, closed=True) >= minlen]
    return ok


def readbgr(file):
    return cv.imread('car.jpg')

def bgr2gray(x):
    return cv.cvtColor(x,cv.COLOR_BGR2GRAY)

# Frequency invariant
def invar(c, wmax=10):
    z = c[:, 0] + c[:, 1] * 1j  # Convert 'c' to complex
    f = fft.fft(z)              # Calculate FFT

    fa = abs(f)                 # Get absolute value for get rotation and initial point invariant

    s = fa[1] + fa[-1]
    fp = fa[2:wmax + 2]
    fn = np.flipud(fa)[1:wmax + 1]
    return np.hstack([fp, fn]) / s

# Function which compares a vector 'c' with models 'mods' and returns sorted distances
# and corresponded label
def mindist(c,mods,labs):
    import numpy.linalg as la
    ds = [(la.norm(c-mods[m]),labs[m]) for m in range(len(mods)) ]
    return sorted(ds, key=lambda x: x[0])

# Get input key
def getInputKey():
    return cv.waitKey(1) & 0xFF

# Get current frame
def getFrame(cap):
    ret, frame = cap.read()
    return frame #cv.cvtColor(frame, cv.COLOR_BGR2RGB)

# Load models and calculate invariant features of models
# Return (models, feats)
def loadModels(image):
    # Convert to gray and invert image
    g = 255-bgr2gray(readbgr(image))

    # Models -> Sort the letters for X coordinate
    models = sorted(extractContours(g), key=lambda x: x[0,0])

    # Invariant features of all models
    feats = [invar(m) for m in models]

    return models, feats


# Function which processes current 'frame' searching matches with 'models' and show the licence plat number with 'labels'
def processFrame(frame, models, labels, feats):
    # Convert to gray and invert image
    g = 255 - bgr2gray(x=frame)

    cv.imshow('roi', g)

    # Extract contours
    things = sorted(extractContours(g, holes=True), key=lambda x: x[0,0])

    print('The licence plate is: ')
    for x in things:
        d, l = mindist(invar(x), feats, labels)[0]
        if d < 0.10:
            print(l, end='')

    print('\n')




def play(dev=0):
    global lButtonUp, drawing, x0, xf, y0, yf

    cap = cv.VideoCapture(dev)

    # Labels and templates
    labels = "0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ"
    templates = 'car.jpg'

    # Load models and its invariant features
    models, feats = loadModels(image=templates)

    while True:
        # Get input key
        key = getInputKey()

        # Process input key
        if key == escKey:
            break

        frame = getFrame(cap=cap)

        # Press 'Enter' key for process current frame
        if key == enterKey:
            if x0 != initialPosition and xf != initialPosition and y0 != initialPosition and yf != initialPosition:
                # Restart values of mouse's events
                lButtonUp = False
                drawing = False

                # Select ROI
                roi = np.copy(frame[y0:yf + 1, x0:xf + 1])
                # Process frame
                processFrame(frame=roi, models=models, labels=labels, feats=feats)

                # Delete ROI selection
                x0, y0, xf, yf = initialPosition, initialPosition, initialPosition, initialPosition

        # Draw ROI
        drawROI(frame=frame, p0=(x0, y0), pf=(xf, yf), color=(0, 255, 0))

        # Show frame
        cv.imshow(programName, frame)


if __name__ == "__main__":
    play()
