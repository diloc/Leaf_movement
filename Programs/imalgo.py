# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:41:45 2018

@author: DLozano
"""

import numpy as np
import cv2
import pandas as pd
import os
from scipy.optimize import curve_fit

eps = np.finfo(float).eps


def grosSegm(src, krn):
    green = 2 * src[:, :, 1] - src[:, :, 0] - src[:, :, 2]
    blur = cv2.GaussianBlur(np.copy(green), (krn, krn), 0)
    blur = np.clip(blur, 0, 255)

    # plotting.im(blur, 'blur grosSegm')

    ret, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    otsu = np.copy(blur)
    otsu[otsu > ret] = 255
    otsu[otsu <= ret] = 0
    otsu = cv2.bitwise_not(otsu)
    potgen = cv2.bitwise_and(src, src, mask=otsu)
    H1, _, V1 = cv2.split(cv2.cvtColor(potgen, cv2.COLOR_BGR2HSV))
    BW1 = np.copy(V1)
    BW1[BW1 <= 40] = 0
    BW1[BW1 >= 180] = 0
    BW1[BW1 > 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(np.copy(BW1), cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing


def otsu(gray, src):
    ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otMask = np.copy(gray)
    otMask[otMask <= ret] = 0
    otMask[otMask > ret] = 255
    otsu = cv2.bitwise_and(src, src, mask=otMask)  # Masked Image
    return otsu, otMask


def gamma(src, gamma):
    power_aux = cv2.pow((src / 255.0), 1.0 / gamma)
    power = np.uint8(power_aux * 255)
    return power


def thrHSV(src, lowerlevel, upperlevel):
    lowerb = np.array(lowerlevel)
    upperb = np.array(upperlevel)
    HSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    maskHSV = cv2.inRange(HSV, lowerb, upperb)
    resHSV = cv2.bitwise_and(src, src, mask=maskHSV)
    return resHSV, maskHSV


def centroids(cx, cy, x_aux, col, y_aux, x, y, offset, pblm):
    if pblm == "yes":
        cxp, cyp = 0, 0

    if pblm == "no":
        cxp = np.int16(x_aux - col + cx + x)
        cyp = np.int16(y_aux + cy + y)
        cxp = cxp + offset
        cyp = cyp + offset
    return cxp, cyp


def spcFilter(BWsrc, bgr, pptn):
    perimeters = []
    xyMeanAll = [0, 0]
    area_all = pd.DataFrame(columns=["area", "cx", "cy"])
    BWdest = BWsrc
    if len(bgr) > 1:
        H1, S1, V1 = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))

    contours, hierarchy = cv2.findContours(
        BWsrc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cntArea = 0

    if len(contours) > 0:
        contMax = max(contours, key=cv2.contourArea)
        areaMax = cv2.contourArea(contMax)
        kernel_size = np.int16(pptn * areaMax)

        for cntl in range(len(contours)):
            BWhue, Hm = [], []
            cntr = contours[cntl]
            area = cv2.contourArea(cntr)

            if len(bgr) > 1:
                BWhue = np.zeros_like(H1)
                cv2.drawContours(BWhue, [cntr], -1, (255, 255, 255), -1)
                Hm = cv2.bitwise_and(H1, H1, mask=BWhue)
                _, Hm_hist, _ = histogram(Hm, 0, 179)
                HmX = np.arange(len(Hm_hist))
                mean = sum(HmX * Hm_hist) / sum(Hm_hist)

                if (mean < 35) or (mean > 80):
                    cv2.drawContours(BWdest, [cntr], -1, (0, 0, 0), -1)

            if area < kernel_size:
                cv2.drawContours(BWdest, [cntr], -1, (0, 0, 0), -1)
                continue

            M = cv2.moments(cntr)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area_all.loc[cntArea, "area"] = area
            area_all.loc[cntArea, "cx"] = cx
            area_all.loc[cntArea, "cy"] = cy
            perimeters.append(cntr)
            cntArea += 1

        xyMeanAll = np.mean(np.where(BWdest > 0))

    area_all["portion"] = area_all["area"].apply(
        lambda x: x / area_all.loc[:, "area"].sum()
    )
    area_all["cxp"] = area_all["cx"] * area_all["portion"]
    area_all["cyp"] = area_all["cy"] * area_all["portion"]

    return BWdest, perimeters, xyMeanAll, area_all


def roiFilter(bwSRC, hole, roi, flag):
    bw = bwSRC.copy()
    row, col = bwSRC.shape

    top = np.int((row / 2) - row * (roi / 2))
    bott = np.int((row / 2) + row * (roi / 2))
    left = np.int((col / 2) - col * (roi / 2))
    rigt = np.int((col / 2) + col * (roi / 2))

    contours, _ = cv2.findContours(bwSRC, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cntl in range(len(contours)):
        arebw3 = 0
        cntr = contours[-cntl]
        area = cv2.contourArea(cntr)
        cv2.drawContours(bw, [cntr], -1, (0, 0, 0), 1)  # Remove a pixel along contour

        if flag and (area < hole):
            cv2.drawContours(bw, [cntr], -1, (0, 0, 0), -1)
            continue

        bw2 = np.uint8(np.zeros((np.shape(bwSRC)[0], np.shape(bwSRC)[1])))
        cv2.drawContours(bw2, [cntr], -1, (255, 255, 255), -1)
        rowbw2, colbw2 = np.where(bw2 > 0)
        bw3 = bw2[top:bott, left:rigt]
        arebw3 = len(np.where(bw3 > 0)[0])
        arebw3T = bw3.shape[0] * bw3.shape[1]

        if arebw3 < arebw3T * 0.002:
            cv2.drawContours(bw, [cntr], -1, (0, 0, 0), -1)
            continue

    return bw


# %%
def joinAreas(BW, area_all):
    BWsrc = np.copy(BW)

    if len(area_all) > 1:
        area_all["cxp"] = area_all["cx"] * area_all["portion"]
        area_all["cyp"] = area_all["cy"] * area_all["portion"]
        xMeanM = int(area_all["cxp"].sum())
        yMeanM = int(area_all["cyp"].sum())
        contours, hierarchy = cv2.findContours(
            BWsrc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if len(contours) > 0:
            contMax = max(contours, key=cv2.contourArea)
            xyM = contMax[:, 0]
            xCompM = xyM[:, 0] - xMeanM  # Translation to the origin
            yCompM = xyM[:, 1] - yMeanM
            distM = np.sqrt(np.power(xCompM, 2) + np.power(yCompM, 2))
            distMax = np.max(distM)

            for cntl in range(len(contours)):
                cntr = contours[cntl]
                xy = cntr[:, 0]  # Contour
                xComp = xy[:, 0] - xMeanM  # Translation to the origin
                yComp = xy[:, 1] - yMeanM
                dist = np.sqrt(np.power(xComp, 2) + np.power(yComp, 2))
                distChk = dist[np.where(dist <= distMax * 2.50)]

                if len(distChk) != 0:
                    minDist = (np.abs(dist - 0)).argmin()
                    xClose = np.int16(xy[minDist, 0])
                    yClose = np.int16(xy[minDist, 1])
                    cv2.line(
                        BWsrc, (xMeanM, yMeanM), (xClose, yClose), (255, 255, 255), 1
                    )

                if len(distChk) == 0:
                    cv2.fillConvexPoly(BWsrc, cntr, 0)

    return BWsrc


# %%
def pheno(binSrc):
    (
        length,
        hull_area,
        _,
        _,
        roundness,
        roundness2,
        compact,
        ellipse,
        eccentricity,
        radius,
    ) = (1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15)
    area = 1e-15
    cntr = []

    cntrs, hier = cv2.findContours(binSrc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cntrs) == 1:
        cntr = cntrs[0]
        area = cv2.contourArea(cntr)
        length = cv2.arcLength(cntr, True)
        hull = cv2.convexHull(cntr, returnPoints=True)
        (x, y), radius = cv2.minEnclosingCircle(cntr)

        # center = (int(x),int(y))
        radius = int(radius)

        if radius == 0:
            radius = 1e-15
        if area == 0:
            area = 1e-15
        if length == 0:
            length = 1e-15

        roundness = (np.pi * 4 * area) / np.power(length, 2)
        hull_area = cv2.contourArea(hull)
        hull_perim = cv2.arcLength(cntr, True)

        if hull_area == 0:
            hull_area = 1e-15
        if hull_perim == 0:
            hull_perim = 1e-15

        roundness2 = (np.pi * 4 * hull_area) / np.power(hull_perim, 2)

        compact = area / hull_area
        if len(hull) < 5:
            a = 1e-15
            b = 1e-15
        else:
            ellipse = cv2.fitEllipse(hull)
            axis1 = np.sqrt(
                np.power((ellipse[1][0] - ellipse[0][0]), 2)
                + np.power((ellipse[1][1] - ellipse[0][1]), 2)
            )
            axis2 = ellipse[2] / 2
            axis = [axis1, axis2]
            a = np.max(axis) / 2
            b = np.min(axis) / 2
        if a == 0:
            a = 1e-15
        if b == 0:
            b = 1e-15

        eccentricity = np.sqrt(1 - (np.power(b, 2) / np.power(a, 2)))
    rosPheno = np.array(
        [area, length, hull_area, roundness, roundness2, compact, eccentricity, radius]
    )
    return cntr, rosPheno


def padding(src, borSiz):
    src[np.where((src < [1, 1, 1]).all(axis=2))] = [255, 255, 255]
    borT = np.int16([borSiz - (src.shape[0] / 2)])
    if borT <= 0:
        borT = 0
    borB = np.int16([borSiz - (src.shape[0] / 2)])
    if borB <= 0:
        borB = 0
    borL = np.int16([borSiz - (src.shape[1] / 2)])
    if borL <= 0:
        borL = 0
    borR = np.int16([borSiz - (src.shape[1] / 2)])
    if borR <= 0:
        borR = 0
    # print(borT,borB,borL,borR)
    border = cv2.copyMakeBorder(
        src,
        top=borT,
        bottom=borB,
        left=borL,
        right=borR,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    return border


def centre_mask(img, gray, winSiz):
    src_gray = np.uint8(np.zeros((2 * winSiz, 2 * winSiz)))
    src_img = np.zeros((2 * winSiz,2 * winSiz,3), np.uint8)

    src_gray[
        int(winSiz - gray.shape[0] / 2) : int(winSiz + gray.shape[0] / 2),
        int(winSiz - gray.shape[1] / 2) : int(winSiz + gray.shape[1] / 2),
    ] = gray
    
    src_img[
        int(winSiz - gray.shape[0] / 2) : int(winSiz + gray.shape[0] / 2),
        int(winSiz - gray.shape[1] / 2) : int(winSiz + gray.shape[1] / 2), :] = img

    # plotting.imT(gray, src_gray, "gray", "src_gray")

    cor_cent = np.where(src_gray > 0)
    row_cent = int(np.round(np.mean(cor_cent[0])))
    col_cent = int(np.round(np.mean(cor_cent[1])))

    gray_out = src_gray[
        int(row_cent - winSiz / 2) : int(row_cent + winSiz / 2),
        int(col_cent - winSiz / 2) : int(col_cent + winSiz / 2),
    ]
    
    img_out = src_img[
        int(row_cent - winSiz / 2) : int(row_cent + winSiz / 2),
        int(col_cent - winSiz / 2) : int(col_cent + winSiz / 2), :]

    # plotting.imT(src_gray, dest,  "src_gray", "dest")

    return img_out, gray_out


def fourier(scr):
    gray = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift))
    return mag


def kmean(img, K, itera):
    dim = len(img.shape)

    centerInfo, center_sort = [], []

    if dim == 3:
        Z = img.reshape((-1, 3))

    if dim == 2:
        Z = img.reshape((-1, 1))

    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, itera, 1.0)
    # K = 8
    # _,labels,centers  = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # Now convert back into uint8, and make original image
    # centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape(img.shape)
    total = 0
    centerInfo = pd.DataFrame(columns=["center", "qty", "percen"])

    for cnt in range(len(np.sort(centers, axis=0))):
        center = centers[cnt]

        qty = len(res2[res2 == center])
        centerInfo.loc[cnt, "center"] = center[0]
        centerInfo.loc[cnt, "qty"] = qty
        total = total + qty

    centerInfo.loc[:, "percen"] = centerInfo.loc[:, "qty"] / total

    centerInfo.sort_values(by="center", inplace=True)
    centerInfo = centerInfo.reset_index(drop=True)
    center_sort = centerInfo.loc[:, "center"].values
    return res2, centerInfo, center_sort


def centroid(root, cam, trayName, posn, mins, ctRow, ctCol):
    pathCVS = root + "centroids" + "/" + cam + "/"
    if not os.path.exists(pathCVS):
        os.makedirs(pathCVS)

    filetext = ".csv"
    filepath = pathCVS + trayName + "_" + posn + filetext

    if not os.path.isfile(filepath):
        df = pd.DataFrame({"mins": [mins], "ctRow": [ctRow], "ctCol": [ctCol]})
        df.to_csv(filepath, sep=",", index=False)

    else:
        df = pd.read_csv(filepath)
        df1 = pd.DataFrame({"mins": [mins], "ctRow": [ctRow], "ctCol": [ctCol]})
        df.append(df1)
        df.to_csv(filepath, sep=",", index=False)
        print("cvs exists", df)

    return df


def histogram(gray, minV, maxV):
    histR = cv2.calcHist([gray], [0], None, [maxV - minV], [minV, maxV])[1:]
    histYmax = np.max(histR)

    # if histYmax == 0: histYmax = eps
    # histR = histR / histYmax

    histY = list()
    histX = list()
    for cntV5 in range(len(histR)):
        histY.append(histR[cntV5][0])
        histX.append(cntV5)

    return np.array(histX), np.array(histY), histYmax


# %%
def drawgrid(src, winSiz):
    img = src.copy()
    h, w, _ = img.shape
    rows = int(h / winSiz)
    cols = int(w / winSiz)
    sRow = 0

    for cnt1 in range(rows + 1):
        eRow = sRow + winSiz

        sCol = 0
        for cn1 in range(cols + 1):
            eCol = sCol + winSiz

            cv2.line(img, (sCol, eRow), (eCol, eRow), (100, 100, 100), 1, 1)
            cv2.line(img, (eCol, sRow), (eCol, eRow), (100, 100, 100), 1, 1)
            sCol = eCol

        sRow = eRow
    return img


def stoll(t,a,b):
    return (a* np.exp(t*b))

def fitting(time,area):
    intCond = [1e-3, 1e-3]
    win = int(np.floor(0.20 * len(time)))
    
    area_out = np.zeros_like(area)
    
    for cnt in range(int(len(time)/win)):
        _start = int(cnt*win)
        _end = int((cnt + 1)*win)
        
        if cnt == int((len(time)/win) - 1):
            _end = int(len(time))
    
        [a,b], pcov = curve_fit(stoll, time[_start:_end], area[_start:_end],intCond)
        areaF  = (a* np.exp(time[_start:_end]*b))
        area_out[_start:_end] = areaF
    
    return time, area_out


def gray_img(winSiz, path):
    src = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    src[np.where((src > [200, 200, 200]).all(axis=2))] = [0, 0, 0]
    aux = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    area = len(aux[aux > 0])
       
    if area > 50:
        img, gray,  = centre_mask(src, aux, winSiz)
    else:
        img, gray, area = [], [], []

    return img, gray, area

