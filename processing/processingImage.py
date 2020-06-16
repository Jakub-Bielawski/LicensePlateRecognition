import numpy as np
import cv2
import math
from skimage.measure import LineModelND, ransac
import csv
import ast
from processing import platesHists
from matplotlib import pyplot as plt


def createHist():
    values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 6.0, 28.0, 88.0, 96.0, 135.0, 194.0, 230.0, 312.0,
              389.0, 453.0, 500.0, 555.0, 679.0, 716.0, 668.0, 436.0, 302.0, 164.0, 120.0, 89.0, 66.0, 56.0, 43.0, 40.0,
              43.0, 49.0, 48.0, 54.0, 53.0, 76.0, 55.0, 58.0, 48.0, 52.0, 38.0, 46.0, 42.0, 42.0, 42.0, 43.0, 54.0,
              75.0, 61.0, 27.0, 34.0, 27.0, 26.0, 22.0, 21.0, 27.0, 34.0, 29.0, 37.0, 24.0, 36.0, 35.0, 35.0, 42.0,
              51.0, 46.0, 35.0, 46.0, 59.0, 45.0, 52.0, 61.0, 70.0, 84.0, 83.0, 64.0, 64.0, 58.0, 63.0, 53.0, 54.0,
              55.0, 54.0, 69.0, 54.0, 54.0, 49.0, 27.0, 37.0, 33.0, 45.0, 32.0, 30.0, 38.0, 27.0, 33.0, 29.0, 34.0,
              18.0, 35.0, 26.0, 27.0, 16.0, 19.0, 24.0, 21.0, 23.0, 19.0, 24.0, 25.0, 25.0, 25.0, 29.0, 32.0, 26.0,
              33.0, 36.0, 32.0, 49.0, 47.0, 44.0, 60.0, 47.0, 60.0, 61.0, 67.0, 74.0, 71.0, 66.0, 82.0, 83.0, 91.0,
              91.0, 87.0, 84.0, 110.0, 101.0, 96.0, 104.0, 110.0, 87.0, 119.0, 128.0, 120.0, 105.0, 108.0, 116.0, 115.0,
              113.0, 119.0, 113.0, 110.0, 125.0, 115.0, 108.0, 132.0, 126.0, 181.0, 208.0, 241.0, 303.0, 297.0, 354.0,
              339.0, 356.0, 351.0, 415.0, 389.0, 390.0, 357.0, 380.0, 364.0, 383.0, 433.0, 426.0, 402.0, 384.0, 346.0,
              400.0, 462.0, 554.0, 555.0, 681.0, 752.0, 766.0, 781.0, 900.0, 1177.0, 1455.0, 1709.0, 1585.0, 995.0,
              439.0, 140.0, 57.0, 48.0, 35.0, 32.0, 23.0, 16.0, 8.0, 6.0, 10.0, 10.0, 7.0, 6.0, 7.0, 6.0, 3.0, 2.0, 3.0,
              0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    hist = [[value] for value in values]
    print(hist)


def show(image):
    while 1:
        if cv2.waitKey(20) & 0xFF == 27:
            break
        # image_resized = cv2.resize(image, (500, 375))
        cv2.imshow("", image)


def empty_callback(nothing):
    pass


def readDataSet():
    f = open('/home/jakub/PycharmProjects/LicensePlateRecognition/signDescriptors/HU_corrected_signs_lenght.csv', 'r')
    signs = []
    huMoments = []
    with f:
        reader = csv.reader(f, delimiter="#")
        index = 0
        for row in reader:
            for sign in row:
                if len(sign) == 1:
                    signs.append(sign)
                else:
                    res = ast.literal_eval(sign)
                    huMoments.append(res)

            index += 1

    data = []
    for sign, hu in zip(signs, huMoments):
        data.append([sign, hu])
    return data


def return_intersection(hist_1, hist_2):
    # minima = np.minimum(hist_1, hist_2)
    # intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    # return intersection
    # hist_2.astype
    hist_2 = np.array(hist_2).astype(np.float32)

    return cv2.compareHist(hist_1,hist_2,method=cv2.HISTCMP_BHATTACHARYYA)


def findPlate(image):
    cv2.namedWindow('image')
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_filtered = cv2.GaussianBlur(image_gray, (9, 9), 0.0)
    image_edge = cv2.Canny(img_filtered, 30, 120, apertureSize=3, L2gradient=True)
    dilation = cv2.dilate(image_edge, (3, 3), iterations=2)

    image_contours, image_he = cv2.findContours(dilation, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    histOfImagePieces = []
    plates = []
    for index, countour in enumerate(image_contours):
        x, y, w, h = cv2.boundingRect(countour)
        aspect_ratio = float(w) / h
        arearec = w * h
        if 50_000 < arearec < 200_000:
            if 1.5 < aspect_ratio < 5.8:
                perimeter = 2 * w + 2 * h
                # print(perimeter)
                if perimeter < 2_000:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)
                    piece = image_gray[y:y + h, x:x + w][:, :, np.newaxis]
                    plates.append(piece)
                    hist = cv2.calcHist([piece], [0], None, [256], [0, 256])
                    histOfImagePieces.append(hist)

    matches = []
    for histogram in histOfImagePieces:
        matches_ = []
        for histogram_ in platesHists.plateHist:
            matches_.append(return_intersection(histogram, histogram_))
        matches.append(np.max(matches_))

    matches = np.array(matches)
    plate = image_edge
    try:
        indexOfPlate = np.argmax(matches)
        print("Posible plate", len(matches))
        print(indexOfPlate)
        plate = np.copy(plates[indexOfPlate])
    except ValueError:
        pass # TODO jeśli nie znalazł tablicy przetwarzaj całe zdjęcie



    while (1):
        if cv2.waitKey(20) & 0xFF == 27:
            break

        img_edge = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow("PLATE", plate)
        cv2.imshow('image', img_edge)


def findBoxes(image):
    cv2.namedWindow('image')
    cv2.createTrackbar('area_min', 'image', 500, 10000, empty_callback)
    cv2.createTrackbar('len_min', 'image', 0, 20000, empty_callback)
    cv2.createTrackbar('len_max', 'image', 1500, 20000, empty_callback)

    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_filtered = cv2.GaussianBlur(image_gray, (9, 9), 0.0)
    image_edge = cv2.Canny(img_filtered, 40, 110, apertureSize=3, L2gradient=True)
    image_contours, image_he = cv2.findContours(image_edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, image_contours, -1, (0, 255, 0), 3)
    print(len(image_contours))
    ilosc = 0

    while (1):
        if cv2.waitKey(20) & 0xFF == 27:
            break
        cv2.drawContours(image, image_contours, -1, (0, 255, 0), 3)

        for index, countour in enumerate(image_contours):
            area = cv2.contourArea(countour)
            perimeter = cv2.arcLength(countour, False)
            area_min = cv2.getTrackbarPos('area_min', 'image')
            len_min = cv2.getTrackbarPos('len_min', 'image')
            len_max = cv2.getTrackbarPos('len_max', 'image')

            rectangle = cv2.minAreaRect(countour)
            box = cv2.boxPoints(rectangle)
            box = np.int0(box)

            if area_min < area and len_min < perimeter < len_max:
                cv2.drawContours(image, image_contours, index, (0, 0, 255), 3)

                cv2.drawContours(image, [box], -1, (155, 0, 0), 3)
        img_edge = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow('image', img_edge)


def findSigns_test(image):
    cv2.namedWindow('image')
    cv2.createTrackbar('area_min', 'image', 3000, 10000, empty_callback)
    cv2.createTrackbar('len_min', 'image', 1000, 20000, empty_callback)
    cv2.createTrackbar('len_max', 'image', 8000, 20000, empty_callback)
    cv2.createTrackbar('t_min', 'image', 30, 255, empty_callback)
    cv2.createTrackbar('t_max', 'image', 120, 255, empty_callback)
    cv2.createTrackbar('G_size', 'image', 9, 15, empty_callback)
    cv2.createTrackbar('sigma', 'image', 0, 30, empty_callback)
    cv2.createTrackbar('ratio', 'image', 57, 100, empty_callback)

    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    image_copy = np.copy(image)
    print("shape after resize: ", image.shape)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    while (1):
        if cv2.waitKey(20) & 0xFF == 27:
            break

        T_min = cv2.getTrackbarPos('t_min', 'image')
        T_max = cv2.getTrackbarPos('t_max', 'image')
        G_size = cv2.getTrackbarPos('G_size', 'image')
        simga = cv2.getTrackbarPos('sigma', 'image') / 10

        if not (G_size % 2):
            G_size = 7
        img_filtered = cv2.GaussianBlur(image_gray, (G_size, G_size), simga)
        image_edge = cv2.Canny(img_filtered, T_min, T_max, apertureSize=3, L2gradient=True)
        dilation = cv2.dilate(image_edge, (3, 3), iterations=2)
        # opening = cv2.morphologyEx(image_edge, cv2.MORPH_CLOSE, (13, 13))
        # show(opening)
        image_contours, image_he = cv2.findContours(dilation, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image, image_contours, -1, (0, 255, 0), 3)
        maybeSigns = []
        for index, countour in enumerate(image_contours):
            area = cv2.contourArea(countour)
            # perimeter = cv2.arcLength(countour, True)
            area_min = cv2.getTrackbarPos('area_min', 'image')
            len_min = cv2.getTrackbarPos('len_min', 'image')
            len_max = cv2.getTrackbarPos('len_max', 'image')

            x, y, w, h = cv2.boundingRect(countour)
            aspect_ratio = float(w) / h
            arearec = w * h
            histOfImagePieces = []
            plates = []
            if 50_000 < arearec < 200_000:
                ratio = cv2.getTrackbarPos('ratio', 'image') / 10
                if 1.5 < aspect_ratio < ratio:
                    perimeter = 2*w+2*h
                    # print(perimeter)
                    if perimeter < 2_000:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)
                        piece = image_gray[y:y + h, x:x + w][:, :, np.newaxis]
                        plates.append(piece)
                        hist = cv2.calcHist([piece], [0], None, [256], [0, 256])
                        histOfImagePieces.append(hist)

            matches = []
            for histogram in histOfImagePieces:
                matches_ = []
                for histogram_ in platesHists.plateHist:
                    matches_.append(return_intersection(histogram, histogram_))
                matches.append(np.max(matches_))

            matches = np.array(matches)
            plate = image_edge
            try:
                indexOfPlate = np.argmax(matches)
                print("Posible plate", len(matches))
                print(indexOfPlate)
                plate = np.copy(plates[indexOfPlate])
            except ValueError:
                pass  # TODO jeśli nie znalazł tablicy przetwarzaj całe zdjęcie

            # if area_min < area and len_min < perimeter < len_max:
            #
            #     area = cv2.contourArea(countour)
            #     hull = cv2.convexHull(countour)
            #     hull_area = cv2.contourArea(hull)
            #     solidity = float(area) / hull_area
            #     if 0.5 < solidity:
            #         # if 2<aspect_ratio<5:
            #         cv2.drawContours(image, [countour], 0, (0, 255, 255), 3)
            #         maybeSigns.append(countour)
            #         ilosc += 1

        # cv2.drawContours(image, maybeSigns, 0, (0, 255, 255), 3)

        img_edge = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow('image', img_edge)
        cv2.imshow("", plate)
        # cv2.waitKey()
        image = np.copy(image_copy)
        image_contours = 0


def findSigns(image):
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_filtered = cv2.GaussianBlur(image_gray, (9, 9), 0.0)
    image_edge = cv2.Canny(img_filtered, 40, 110, apertureSize=3, L2gradient=True)
    # TODO ustawić progi tak aby znajdowało wszystkie znaki, wtedy boundingboxy beda mialy sens
    opening = cv2.morphologyEx(image_edge, cv2.MORPH_CLOSE, (9, 9))
    image_contours, image_he = cv2.findContours(opening, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    best_fitted_contours = []
    new_hierarchy = []
    for index, countour in enumerate(image_contours):
        area = cv2.contourArea(countour)
        perimeter = cv2.arcLength(countour, False)
        if 380 < area and 80 < perimeter < 800:
            best_fitted_contours.append(countour)
            new_hierarchy.append(image_he[0][index])
    cv2.drawContours(image, best_fitted_contours, -1, (0, 0, 255), 3)
    none_duplicated_conturs = []
    for index, contour_1 in enumerate(best_fitted_contours):
        for index_2, contour_2 in enumerate(reversed(best_fitted_contours)):
            ret = cv2.matchShapes(contour_1, contour_2, 1, 0.0)
            if ret < 0.0001:
                try:
                    best_fitted_contours.pop(len(best_fitted_contours) - index_2 - 1)
                    none_duplicated_conturs.append(contour_1)
                    new_hierarchy.pop(len(best_fitted_contours) - index_2 - 1)

                except IndexError:
                    pass

    print("Found ", len(none_duplicated_conturs), " countours that probabli are signs")

    countoursBoxes = []
    for index, countour in enumerate(best_fitted_contours):
        rectangle = cv2.minAreaRect(countour)
        box = cv2.boxPoints(rectangle)
        box = np.int0(box)
        countoursBoxes.append([countour, box])
        ################### TO DRAW BOUNDING BOXES UNCOMMENT THIS #############################
        cv2.drawContours(image, [box], -1, (155, 0, 0), 3)

    cv2.drawContours(image, none_duplicated_conturs, -1, (0, 255, 255), 2)
    cv2.imshow("", image)
    cv2.waitKey()

    return countoursBoxes


def extraxtSigns(conturs_boxes, image, answers, data):
    def key_to_sort(signwithcenter):
        return signwithcenter[1][0]  # x value of contour center

    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    contours = [contour[0] for contour in conturs_boxes]
    boxes = [contour[1] for contour in conturs_boxes]

    contour_centers = []
    box_centers = []

    for contour, box in zip(contours, boxes):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contour_centers.append((cX, cY))
        M = cv2.moments(box)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        box_centers.append((cX, cY))

    box_centers = np.array(box_centers)
    boundingBoxesForSignsWithCenters = []  # box contours and box centers

    # try for bounding boxes - a bit shity
    contoursForSignsWithCenters = []
    try:
        model_robust, inliers_for_boxes = ransac(box_centers, LineModelND, min_samples=2,
                                                 residual_threshold=20,
                                                 max_trials=100)

        index = 0
        for box_contour, is_inlier in zip(boxes, inliers_for_boxes):
            if is_inlier:
                box_center = box_centers[index]
                boundingBoxesForSignsWithCenters.append([box_contour, box_center])
            index += 1

        boundingBoxesForSignsWithCenters.sort(key=key_to_sort)

        startBox = boundingBoxesForSignsWithCenters[0]
        distance_prev = 0
        new = []
        # for index, boundingBoxeForSignWithCenter in enumerate(boundingBoxesForSignsWithCenters):
        #
        #     distance = ((startBox[1][0] - boundingBoxeForSignWithCenter[1][0]) ** 2
        #                 + (startBox[1][1] - boundingBoxeForSignWithCenter[1][1]) ** 2) ** 0.5
        #
        #     if abs(distance - distance_prev) > 20 or index == 0:
        #         new.append(boundingBoxeForSignWithCenter)
        #         distance_prev = distance
        #     else:
        #         prev_object = boundingBoxesForSignsWithCenters[index - 1]
        #         prev_contour_lenght = cv2.arcLength(prev_object[0], False)
        #         contour_lenght = cv2.arcLength(boundingBoxeForSignWithCenter[0], False)
        #         suma = (prev_contour_lenght + contour_lenght) / 2
        #         if contour_lenght < suma:
        #             new.append(boundingBoxesForSignsWithCenters[index - 1])
        #             boundingBoxesForSignsWithCenters.pop(index)
        #         else:
        #             new.append(boundingBoxeForSignWithCenter)
        #             boundingBoxesForSignsWithCenters.pop(index - 1)

        for boundingBoxIndex, boundingBox in enumerate(boundingBoxesForSignsWithCenters):
            boxesIn = 0
            for boundingBoxIndex_, boundingBox_ in enumerate(boundingBoxesForSignsWithCenters):
                if boundingBoxIndex != boundingBoxIndex_:
                    result = cv2.pointPolygonTest(boundingBox[0], (boundingBox_[1][0], boundingBox_[1][1]), False)
                    if result == 1:
                        boxesIn += 1
                        if boxesIn > 1:
                            perimeter = cv2.arcLength(boundingBox[0], False)
                            perimeter_ = cv2.arcLength(boundingBox_[0], False)
                            if perimeter > perimeter_:
                                boundingBoxesForSignsWithCenters.pop(boundingBoxIndex_)

        index = 0
        for sign_contour, is_inlier in zip(contours, inliers_for_boxes):
            if is_inlier:
                box_center = box_centers[index]
                contoursForSignsWithCenters.append([sign_contour, box_center])
            index += 1

        contoursForSignsWithCenters.sort(key=key_to_sort)

        contoursForSigns = [contour[0] for contour in contoursForSignsWithCenters]
        BoundingBoxesForSigns = [contour[0] for contour in boundingBoxesForSignsWithCenters]

        # make full description of sign, with inside contours
        for index_of_box, box in enumerate(BoundingBoxesForSigns):
            singsIn = 0
            for index_of_sign, sign in enumerate(contoursForSignsWithCenters):
                result = cv2.pointPolygonTest(box, (sign[1][0], sign[1][1]), False)
                if result == 1:
                    singsIn += 1
                    if singsIn > 1:
                        perimeter = cv2.arcLength(sign[0], False)
                        prev_perm = cv2.arcLength(contoursForSignsWithCenters[index_of_sign - 1][0], False)
                        if prev_perm > perimeter:
                            contoursForSignsWithCenters.pop(index_of_sign)

        print("                  len of signs : ", len(contoursForSignsWithCenters))
        print("                  len of boundgboxes: ", len(BoundingBoxesForSigns))
        cv2.drawContours(image, contoursForSigns, -1, (255, 255, 0), 2)
        cv2.imshow("", image)
        cv2.waitKey()

    except ValueError as e:
        print(e)
        pass

    ############################################### SAVING DESCRIPTORS ###################################3
    # contoursForSigns = [contour[0] for contour in contoursForSignsWithCenters]
    #
    # dictionary = {}
    # if len(contoursForSigns) == 7:
    #     for signContour, sign in zip(contoursForSigns, answers):
    #         descriptionOfSigns = []
    #         #################################### HU DESCRIPTOR ##############################
    #         moments = cv2.moments(signContour)
    #         huMoments = cv2.HuMoments(moments)
    #         for i in range(0, 7):
    #             if huMoments[i] == 0.:
    #                 huMoments[i] = 0.001
    #             huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
    #         huMoments = huMoments.tolist()
    #
    #         for hu in huMoments:
    #             descriptionOfSigns.append(hu[0])
    #
    #         dictionary[sign] = descriptionOfSigns
    #         ##################################################################################
    #         ################################ CONVEX HULL DESCRIPTOR ##########################
    #
    #         # hull = cv2.convexHull(signContour, returnPoints=False)
    #         # defects = cv2.convexityDefects(signContour, hull)
    #         # defects = defects.tolist()
    #         # for defect in defects:
    #         #     # print("defect",defect[0])
    #         #     descriptionOfSigns.append(defect[0])
    #         # dictionary[sign] = descriptionOfSigns
    #         # # print(defects)
    #
    #
    # # print(dictionary)
    # return dictionary
    # TODO: sprawdzanie poprawności dla convexhulla
    ##################################### MATCHING SHAPES #############################3
    signs = [contour[0] for contour in contoursForSignsWithCenters]
    readedSigns = []
    for index_of_sign, sign in enumerate(signs):
        ##################################### HU DESCRIPTOR ############################
        best_match = "?"
        moments = cv2.moments(sign)
        huMoments = cv2.HuMoments(moments)
        for i in range(0, 7):

            if huMoments[i] == 0.:
                huMoments[i] = 0.0001
            huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
        prev_fit = 100

        for dataSet in data:
            fit = 0
            dataSignHU = dataSet[1]
            dataSign = dataSet[0]
            if index_of_sign == 0 or index_of_sign == 1:
                if not dataSign.isalpha():
                    continue
            for i in range(0, 7):
                # TODO: wiout abs
                fit += abs((1 / huMoments[i]) - (1 / dataSignHU[i]))

            if fit < prev_fit:
                best_match = dataSign
                prev_fit = fit
        readedSigns.append(best_match)
        ###################################################################################

        ##################################### CONVEX HULL #################################
        # hull = cv2.convexHull(sign, returnPoints=False)
        # defects = cv2.convexityDefects(sign, hull)
        # prev_fit = 1000000
        #
        # for dataSet in data:
        #     fit = 0
        #     dataSign = dataSet[0]
        #     dataDefects = dataSet[1]
        #     # print(len(data))
        #     if len(dataDefects) == len(defects):
        #         for i in range(0,len(dataDefects)):
        #             # print(defects[i][0][0])
        #             # print(dataDefects[i])
        #             fit += abs(defects[i][0][0]-dataDefects[i][0])
        #             # print(fit)
        #         if fit < prev_fit:
        #             best_match = dataSign
        #             prev_fit = fit
        #
        # readedSigns.append(best_match)

    # count score
    score = 0
    maxScore = 0
    answerListOfChars = []
    for char in answers:
        answerListOfChars.append(char)

    if len(answerListOfChars) == len(readedSigns):
        for sign_1, sign_2 in zip(answerListOfChars, readedSigns):
            maxScore += 1
            if sign_1 == sign_2:
                score += 1

    # convert to string
    readedPlate = ""
    for rSign in readedSigns:
        readedPlate += str(rSign)

    return readedPlate, score, maxScore


# def perform_processing(image: np.ndarray, answer) -> str
def perform_processing(image: np.ndarray, answer):
    image = cv2.resize(image, (2560, 1920))
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    # createHist()
    # findSigns_test(image)
    # findBoxes(image)
    findPlate(image)
    # data = readDataSet()
    # data=None
    # conturs_boxes = findSigns(image)
    # describedSigns = extraxtSigns(conturs_boxes, image, answer, data)
    # describedSign, score, maxscore = describedSigns
    # print(describedSigns)

    return "PO12345"
    # return describedSign, score, maxscore
    # pathToReturn = "/home/jakub/PycharmProjects/LicensePlateRecognition/"
    # return describedSigns
