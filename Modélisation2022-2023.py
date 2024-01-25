# -*- coding: utf-8 -*-
"""
Created on 2 Dec 2020
Updated 9 Jan 2023
@author: chatoux
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob


def CameraCalibration():
    # Préparation des critères d'arrêt de la fonction cornerSubPix
    # Itérations = 30 et Précision = 0.001
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prépare les points 3D de l'échiquier
    objp = np.zeros((5*8, 3), np.float32) #Crée une matrice 3D de zéros avec 5*8 lignes et 3 colonnes
    objp[:, :2] = np.mgrid[0:5, 0:8].T.reshape(-1, 2) #Crée une grille 2D avec les coordonnées x et y des coins de l'échiquier et ensuite les transposent.
    objp[:, :2] *= 35  # taille de la case de l'échiquier en mm
    
    
    objpoints = []  #Points 3D réels
    imgpoints = []  #Points 2D de l'image
    
    #Liste des chemin des imades de l'échiquier par la camera gauche 
    images = glob.glob('Images/cameraBleu/*.jpg')
    for fname in images:
        # img = cv.imread(fname)
        
        img = cv.pyrDown(cv.imread(fname)) #Lit l'image puis réduit de la résolution 
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Definit la couleur en niveau de gris 
        ret, corners = cv.findChessboardCorners(gray, (5, 8), None) #Recherche les coins de l'échiquier
        
        print(ret)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) #Raffine les coins
            imgpoints.append(corners)
            
            cv.drawChessboardCorners(img, (5, 8), corners2, ret) #Dessine les coins sur l'image de l'échiquier
            
            ############################################ 
            cv.namedWindow('img', 0)
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()

    #Calibration de la caméra
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('camraMatrix\n', mtx)
    print('dist\n', dist)

    img = cv.pyrDown(cv.imread('Images/chess2/test2.jpg'))
    
    #La nouvelle matrice de la caméra optimale 
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print('newcameramtx\n', newcameramtx)

    #Utilisation de la fonction de mapping pour enlever la distortion
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    
    #Découper la région d'intérêt
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    ############################################
    cv.namedWindow('img', 0)
    cv.imshow('img', dst)
    cv.waitKey(1)
    
    ############ Enregistrement ##########################
    cv.imwrite('calibresultM.png', dst)
    
    #Calculer l'erreur de la calibration 
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    return newcameramtx


def DepthMapfromStereoImages():
    ############ to adapt ##########################
    imgL = cv.pyrDown(cv.imread('Images/cameraBleu/bleu.jpg'))
    imgR = cv.pyrDown(cv.imread('Images/cameraRouge/rouge.jpg'))
    #################################################
    
    #s'assurer que les images sont de la même taille
    imgL = cv.resize(imgL, (640, 480))
    imgR = cv.resize(imgR, (640, 480))

    #Convertir en niveaux de gris
    imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    #Paramètres de StereoSGBM
    window_size = 15
    min_disp = 0
    num_disp = 160 - min_disp  # Doit être un multiple de 16.
    
    #Création d'un objet StereoSGBM 
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=16,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=16,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=32)
    
    #Calcul la carte de disparité
    
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    #Affichage de la carte de disparité normalisée
    plt.figure('3D')
    plt.imshow((disparity-min_disp)/num_disp, 'gray')
    plt.colorbar()
    plt.show()


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    
    #Dessiner les lignes épipolaires sur les images en définissant les points correspondantes
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def StereoCalibrate(Cameramtx):
    ############ to adapt ##########################
    img1 = cv.pyrDown(cv.imread('Images/cameraBleu/bleu.jpg', 0))
    img2 = cv.pyrDown(cv.imread('Images/cameraRouge/rouge.jpg', 0))
    #################################################
    
    
    # opencv 4.5
    sift = cv.SIFT_create() #Initialise le détecteur de points SIFT
    # opencv 3.4: sift = cv.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    #Les paramétres pour la correspondance des descripteurs
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    #Utiliser le rfatio test pour filtrer les bonnes correspondances 
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    #########################Dessiner#################################
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img3)
    plt.show()

    #Calculer la matrice essentielle 
    E, maskE = cv.findEssentialMat(pts1, pts2, Cameramtx, method=cv.FM_LMEDS)
    print('E\n', E)
    
    #Lapose entre la caméra droite et la caméra gauche 
    retval, R, t, maskP = cv.recoverPose(E, pts1, pts2, Cameramtx, maskE)
    print('R\n', R)
    print('t\n', t)

    #Calculer la matrice fondamentale
    F, maskF = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    print('F\n', F)

    return pts1, pts2, F, maskF, maskE


def EpipolarGeometry(pts1, pts2, F, maskF):
    ############ to adapt ##########################
    img1 = cv.pyrDown(cv.imread('Images/im1.jpeg', 0))
    img2 = cv.pyrDown(cv.imread('Images/im2.jpeg', 0))
    #################################################
    r, c = img1.shape

    #Filter les points correspondants en utilisant le masque de la matrice fondamentale
    pts1F = pts1[maskF.ravel() == 1]
    pts2F = pts2[maskF.ravel() == 1]

    #Calculer les lignes épipolaires pour l'image droite et l'image gauche 
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1F, pts2F)
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    
    #################################################
    plt.figure('Fright')
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img6)
    plt.figure('Fleft')
    plt.subplot(121), plt.imshow(img4)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

    #Estimer les matrices de rectification dans un stereo non calibré  
    retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (c, r))
    print(H1)
    print(H2)
    
    #################################################
    im_dst1 = cv.warpPerspective(img1, H1, (c, r))
    im_dst2 = cv.warpPerspective(img2, H2, (c, r))
    cv.namedWindow('left', 0)
    cv.imshow('left', im_dst1)
    cv.namedWindow('right', 0)
    cv.imshow('right', im_dst2)
    cv.waitKey(0)


if __name__ == "__main__":
    cameraMatrix = CameraCalibration()
    cameraMatrix = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
    dist = [[0, 0, 0, 0, 0]]

    pts1, pts2, F, maskF, FT, maskE = StereoCalibrate(cameraMatrix)

    # EpipolarGeometry(pts1, pts2, F, maskF, FT, maskE)

    DepthMapfromStereoImages()
