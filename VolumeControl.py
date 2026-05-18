import cv2 as cv
import mediapipe as mp
import time
import HandDetectorModule as hdm
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wcam,hcam = 640,480

cap=cv.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)

ptime=0
vol=0

detector = hdm.handDetector()

devices = AudioUtilities.GetSpeakers()

interface = devices._dev.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]


print(minVol, maxVol)

while True:
    bool,img = cap.read()
    if not bool :
        break
    img = cv.flip(img,1)
    img = detector.findHands(img)
    lmlist = detector.findposition(img,draw=False)
    if len(lmlist) != 0:
        x1,y1 = lmlist[4][1],lmlist[4][2]
        x2,y2 = lmlist[8][1],lmlist[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        cv.circle(img,(x1,y1),10,(0,255,0),cv.FILLED)
        cv.circle(img,(x2,y2),10,(0,255,0),cv.FILLED)
        cv.circle(img,(cx,cy),10,(0,255,0),cv.FILLED)

        cv.line(img,(x1,y1),(x2,y2),(0,255,0),5)

        length = math.hypot(x2-x1,y2-y1)
        if length<50:
            cv.circle(img,(cx,cy),10,(255,255,0),cv.FILLED)

        print(length)
        # vol = np.interp(length,[20,160],[0.25,1.0])
        vol = np.interp(length,[20,160],[minVol,maxVol])


        volume.SetMasterVolumeLevelScalar(vol, None)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv.putText(img,str(int(fps)),(20,70),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv.imshow("Image",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()