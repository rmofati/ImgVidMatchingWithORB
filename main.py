# Resolução do Exercício da Aula do dia 25/11 - ORB e Matching com imagem/video
# Aluno: Rafael Mofati Campos

import numpy as np
import cv2

MIN_NUM_MATCHES = 10

img_ref = cv2.imread("img/refQuadro1.jpg", 0)
vid = cv2.VideoCapture("img/vidQuadro.mp4")

orb = cv2.ORB_create(patchSize = 21, fastThreshold = 70, nlevels = 5, scaleFactor = 1.2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print()
print("=> Escolha a região de interesse na imagem utilizando o mouse e depois pressione 'Enter' ou 'Espaço'")

cv2.imshow("Imagem", img_ref)

x,y,w,h = cv2.selectROI("Imagem", img_ref, False)
if w and h:
    img_ref_roi = img_ref[y:y+h, x:x+w]
    cv2.destroyAllWindows()

kp_ref_img, des_ref_img = orb.detectAndCompute(img_ref_roi, None)

if (vid.isOpened() == False):
    print("Erro ao abrir o vídeo")
else:
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret == True:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_vid, des_vid = orb.detectAndCompute(gray_frame, None)
            matches = bf.match(des_ref_img, des_vid)
            matches = sorted(matches, key=lambda x: x.distance)
            
            good_matches = matches[:10]

            src_pts = np.float32([kp_ref_img[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_vid[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img_ref_roi.shape[:2]
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            dst += (w, 0)

            img_matches = cv2.drawMatches(img_ref_roi, kp_ref_img, gray_frame, kp_vid, good_matches, None, flags=2)
            if len(good_matches) >= MIN_NUM_MATCHES:
                img_matches = cv2.polylines(img_matches, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
            cv2.imshow("Matching", img_matches)

            #Apertar Q no teclado para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

vid.release()
cv2.destroyAllWindows()