# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 18:44:41 2023

@author: erdem
"""

import cv2
import numpy as np
import math

# en büyük contours alanı yüzümüz olacağı için en büyüğü arıyoruz
def findMaxContour(contours):
    max_i = 0
    max_area = 0

    for i in range(len(contours)):
        # contours değerleri arasında i gezdiridk bunlarıon alanlarını alta hesaplatıp face area ya atadık
        area_face = cv2.contourArea(contours[i])
        if max_area < area_face:  # burdada en büyük olanı bulmak için if kullandık
            max_area = area_face
            max_i = i  # indexi kaydettik

        try:  # bir şey olmaz ise max indexi atamasını c ye yaptık
            c = contours[max_i]
        except:  # eğer bir hata yada boş dönerse gelen değeri 0 olarak dönüştürdük
            contours = [0]
            c = contours[0]
        return c


cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #           y1  y2  x1  x2
    roi = frame[120:280, 220:380]
    #                     x1 y1        x2   y2     renk
    # rectangelın kalınlığını 0 yaptık çünki çizilen çzigide maskeleme işlemlerine dahil olur sonucu bozar
    cv2.rectangle(frame, (220, 120), (380, 280), (0, 0, 255), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 45, 79], dtype=np.uint8)
    upper_color = np.array([17, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    kernel = np.ones((3, 3), np.uint8)  # kernel ın tam ortasındaki pikselin çevresinde bulunan piksellerin kaça kaç olucağını belirliyoruz 3,3 9,9 gibi ve bu alanda merkez piksel harici bütün pikselleri toplıyarak ortalamasını alıyoruz ve bu değere göre işlem yapıyoruz bir nevi nekadar yoğunluk olucağını hesaplıyoruz
    # işlemin uygulanacapı yerini, uygulama yoğunluğunu ve iterasyon denilen pixellerin genişleme değerini girdik
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 41)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Konturlar, aynı renk veya yoğunluğa sahip tüm sürekli noktaları (sınır boyunca) birleştir
        # en bir eğri olarak basitçe açıklanabilir. Konturlar, şekil analizi ve nesne algılama ve tanıma için yararlı bir araçtır.
        c = findMaxContour(contours)
        # [c[:,:,0].argmin()[]] bu yapı bütün contours ları dolaşıp bana en küçük x değerini vericek
        # extream left en uç kısım
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        # [:,:,0] x yönü için [:,:,1] y yönü için
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])  # aşşağı gittikçe y artar
        
        # arg min ve argmax arasındaki fark ekranın en solu 0 en sağı ise +sonsuza gittiği için arg in ile lefti argmax iile rght alıyoruz
        # argmin aynı zamanda 1 düzeyi için yani y ekseni için de en üstü ıfırdan başlatacağı için kullanırız
        # roi penceresinde merkex noktası extleft değeri radyusu 5 renk kodu yeşil kalınlığı 2
        cv2.circle(roi, extLeft, 5, (0, 255, 0), 2)
        cv2.circle(roi, extRight, 5, (0, 255, 0), 2)
        cv2.circle(roi, extTop, 5, (0, 255, 0), 2)
        cv2.circle(roi, extBot, 5, (0, 255, 0), 2)
       

        cv2.line(roi, extLeft, extTop, (255, 0, 0), 2)
        cv2.line(roi, extRight, extTop, (255, 0, 0), 2)
        cv2.line(roi, extRight, extBot, (255, 0, 0), 2)
        cv2.line(roi, extLeft, extBot, (255, 0, 0), 2)
        
        #extRight[0]-ExtLeft[0] karesi bize bu iki noktanın oluşturduğu üçgenin x tabanının karesini verir
        #extRight[1]-ExtLeft[1] karesi ise bize bu iki noktanın oluşturduğu üçgenin y tabanının karesini verir
        #biz bu iki değeri toplarsak bize hipotenüsü verir
        a = math.sqrt((extRight[0]-extRight[0])**2+(extRight[1]-extTop[1])**2)
        b = math.sqrt((extRight[0]-extBot[0])**2+(extBot[1]-extRight[1])**2)
        c = math.sqrt((extBot[0]-extTop[0])**2+(extBot[1]-extTop[1])**2)
        
        try:
            angle_ab = int(math.acos((a**2+b**2-c**2)/(2*b*c))*57)
            cv2.putText(roi,str(angle_ab),(extRight[0]-50,extRight[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
        except:
            cv2.putText(roi,"?",(extRight[0]-50,extRight[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
        
    cv2.imshow("mask", mask)
    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)
    if cv2.waitKey(5) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
