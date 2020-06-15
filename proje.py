import cv2
import numpy as np
from collections import deque
#vid=cv2.VideoCapture("C:\\Users\\buket\\Desktop\\proje.mp4")

cap = cv2.VideoCapture(0)
mavi_nokta = [deque(maxlen=512)]
yesil_nokta = [deque(maxlen=512)]
kırmızı_nokta = [deque(maxlen=512)]
sarı_nokta = [deque(maxlen=512)]
bindex=0
yindex=0
kindex=0
sindex=0;

color_index = 0

color = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
enaz_mavi=np.array([100,60,60])#en düşük mavi değeri
enyuksek_mavi=np.array([140,255,255])#en yüksek mavideğeri


paintWindow = np.zeros((471,636,3))+255

paintWindow = cv2.rectangle(paintWindow,(1,5),(100,70),(0,0,0),3)
paintWindow = cv2.rectangle(paintWindow,(1,100),(100,160),color[0],3)
paintWindow = cv2.rectangle(paintWindow,(1,200),(100,260),color[1],3)
paintWindow = cv2.rectangle(paintWindow,(1,300),(100,360),color[2],3)
paintWindow = cv2.rectangle(paintWindow,(1,400),(100,460),color[3],3)

#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(paintWindow,"MAVI",(30,130),font,0.5,(255,0,0),2,cv2.LINE_AA)
#cv2.putText(paintWindow,"YESIL",(30,230),font,0.5,(0,255,0),2,cv2.LINE_AA)
#cv2.putText(paintWindow,"KIRMIZI",(30,330),font,0.5,(0,0,255),2,cv2.LINE_AA)
#cv2.putText(paintWindow,"SARI",(30,430),font,0.5,(0,255,255),2,cv2.LINE_AA)
#cv2.putText(paintWindow,"TEMIZLE",(30,30),font,0.5,(255,255,255),2,cv2.LINE_AA)

cv2.namedWindow("Paint")

    
while 1:
    #_,frame = vid.read()
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#her frame i hsv formatına çeviriyorum binary gri format

    

    frame = cv2.rectangle(frame,(1,100),(100,160),color[0],3)
    frame = cv2.rectangle(frame,(1,200),(100,260),color[1],3)
    frame = cv2.rectangle(frame,(1,300),(100,360),color[2],3)
    frame = cv2.rectangle(frame,(1,400),(100,460),color[3],3)
    frame = cv2.rectangle(frame,(1,5),(100,70),(255,255,255),3)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame,"MAVI",(30,130),font,0.5,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"YESIL",(30,230),font,0.5,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,"KIRMIZI",(30,330),font,0.5,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,"SARI",(30,430),font,0.5,(0,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,"TEMIZLE",(30,30),font,0.5,(255,255,255),2,cv2.LINE_AA)

    if ret is False:
        break
 
    mask=cv2.inRange(hsv,enaz_mavi,enyuksek_mavi)#o rengi ayırt ediyor saiyah beyaz şekilde

    
    mask = cv2.erode(mask,(5,5),iterations =2)#inceltiliyor 
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,(5,5))#gürültüyü kaldırıyor
    mask = cv2.dilate(mask,(5,5),iterations = 1)#kalınlaştırma iteration =tekrar sayısı

    contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #kontur yaklaşım metodu
    center=None #daha sonra içini doldurucaz

    if len(contours)>0: #kontur varsa
        #sorted sıralama yapıyor 
        max_contours=sorted(contours,key=cv2.contourArea,reverse=True)[0]#konturları alanlarna göre sıralıcak büyükten küçüğe reversele nu belirledik büyükten küçüğe olduğu içinde . indisi aldık
        ((x,y),radius)=cv2.minEnclosingCircle(max_contours)
        cv2.circle(frame,(int(x),int(y)),int(radius),(255,0,255),3)#çember çiz(üzerine çizdiğimiz yer,kordinatları,yarıçapı,çember rengi,kalınlık)

        center=(int(x),int(y))
        
       
        if center[0] <= 100:
            if 5<=center[1]<=70:
                 mavi_nokta = [deque(maxlen=512)]
                 bindex=0
                 yesil_nokta = [deque(maxlen=512)]
                 yindex=0
                 kırmızı_nokta = [deque(maxlen=512)]
                 kindex=0
                 sarı_nokta = [deque(maxlen=512)]
                 sindex=0
                 paintWindow[:,:,:]=255

                 paintWindow = np.zeros((471,636,3))+255

                 paintWindow = cv2.rectangle(paintWindow,(1,5),(100,70),(0,0,0),3)
                 paintWindow = cv2.rectangle(paintWindow,(1,100),(100,160),color[0],3)
                 paintWindow = cv2.rectangle(paintWindow,(1,200),(100,260),color[1],3)
                 paintWindow = cv2.rectangle(paintWindow,(1,300),(100,360),color[2],3)
                 paintWindow = cv2.rectangle(paintWindow,(1,400),(100,460),color[3],3)

                 #font = cv2.FONT_HERSHEY_SIMPLEX
                 #cv2.putText(paintWindow,"MAVI",(30,130),font,0.5,(255,0,0),2,cv2.LINE_AA)
                 #cv2.putText(paintWindow,"YESIL",(30,230),font,0.5,(0,255,0),2,cv2.LINE_AA)
                 #cv2.putText(paintWindow,"KIRMIZI",(30,330),font,0.5,(0,0,255),2,cv2.LINE_AA)
                 #cv2.putText(paintWindow,"SARI",(30,430),font,0.5,(0,255,255),2,cv2.LINE_AA)
                 #cv2.putText(paintWindow,"TEMIZLE",(30,30),font,0.5,(0,0,0),2,cv2.LINE_AA)

                 cv2.namedWindow("Paint")
            elif 100<=center[1]<=160:
                color_index = 0

            elif 200<=center[1]<=260:
                color_index = 1

            elif 300<=center[1]<=360:
                color_index = 2

            elif 400<=center[1]<=460:
                color_index = 3

        else:
            if color_index == 0:
                mavi_nokta[0].appendleft(center)
                
            elif color_index == 1:    
                yesil_nokta[0].appendleft(center)
                
            elif color_index == 2:
                kırmızı_nokta[0].appendleft(center)
                
            elif color_index == 3:
                sarı_nokta[0].appendleft(center)
    else:
        mavi_nokta.append(deque(maxlen=512))
        bindex+=1
        
        yesil_nokta.append(deque(maxlen=512))
        yindex+=1
        
        kırmızı_nokta.append(deque(maxlen=512))
        kindex+=1
        
        sarı_nokta.append(deque(maxlen=512))
        sindex+=1

       
    
        #cv2.circle(frame,center,5,(0,0,255),5);#başlangıç konumu - bitiş konumu)

    points=[mavi_nokta,yesil_nokta,kırmızı_nokta,sarı_nokta]

    for i in range(len(points)): 
        for j in range(len(points[i])):
            for k in range(len(points[i][j])):
               if points[i][j][k-1] is None or points[i][j][k] is None: #başlangıç ve bitiş noktası boş olmadığına göre 
                    continue
               cv2.line(frame,points[i][j][k-1],points[i][j][k],color[i],2)
               cv2.line(paintWindow,points[i][j][k-1], points[i][j][k], color[i], 2)
                
    cv2.imshow("Frame",frame)
    cv2.imshow("Paint",paintWindow)
    

    if cv2.waitKey(20)&0xFF == ord('q'):
       break
#vid.release()
cap.release()
cv2.destroyAllWindows()
