import cv2 as cv
import mediapipe as mp
import time 


mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils


class handDetector():
    def __init__(self,mode=False,maxHands=2,detectioncon=0.5,trackcon=0.5):
        self.mode=False
        self.maxHands=maxHands
        self.detectioncon=detectioncon
        self.trackcon=trackcon

        self.mphands=mp.solutions.hands
        self.hands=mphands.Hands()

        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img,draw=True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlmk in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handlmk,self.mphands.HAND_CONNECTIONS)
        
        return img
    
    def findposition(self,img,handno=0,draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id,lm in enumerate(myhand.landmark):
                h,w,c = img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                # print(id,lm)
                lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),5,(255,0,255),cv.FILLED)
                
        return lmlist


    

def main():

    ctime = 0
    ptime=0
    cap= cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success,img = cap.read()
        if not success:
            break
        img = detector.findHands(img)
        lmlist = detector.findposition(img)
        if len(lmlist) != 0:
            print(lmlist[4])
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime

        cv.imshow("image",img)
        if cv.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()