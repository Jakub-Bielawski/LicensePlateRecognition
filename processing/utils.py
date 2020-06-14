import numpy as np
import cv2
import math
from skimage.measure import LineModelND, ransac
import csv
import ast
HU_for_zero = np.array(
    [[0.7127819],
     [1.94480616],
     [6.13277474],
     [7.06535321],
     [13.77031787],
     [8.22354605],
     [14.02725957]])

HU_for_one = np.array(
    [[0.18651126],
     [0.4455705],
     [1.14463054],
     [1.29726841],
     [2.51904724],
     [1.52040395],
     [3.79461274]]
)


def show(image):
    while 1:
        if cv2.waitKey(20) & 0xFF == 27:
            break
        # image_resized = cv2.resize(image, (500, 375))
        cv2.imshow("", image)


def empty_callback(_):
    pass

def readDataSet():
    f = open('/home/jakub/PycharmProjects/Projekt_SW/signDescriptors/sign_HUMoments.csv', 'r')
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
            index+=1

    data =[]
    for sign, hu in zip(signs,huMoments):
        data.append([sign,hu])
    return data


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
    cv2.createTrackbar('area_min', 'image', 500, 10000, empty_callback)
    cv2.createTrackbar('len_min', 'image', 0, 20000, empty_callback)
    cv2.createTrackbar('len_max', 'image', 1500, 20000, empty_callback)

    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    print("shape after resize: ", image.shape)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_filtered = cv2.GaussianBlur(image_gray, (9, 9), 0.0)
    show(img_filtered)
    image_edge = cv2.Canny(img_filtered, 40, 110, apertureSize=3, L2gradient=True)
    show(image_edge)
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

            if area_min < area and len_min < perimeter < len_max:
                cv2.drawContours(image, image_contours, index, (0, 0, 255), 3)
                ilosc += 1
        img_edge = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow('image', img_edge)


def findSigns(image):
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_filtered = cv2.GaussianBlur(image_gray, (9, 9), 0.0)
    image_edge = cv2.Canny(img_filtered, 40, 110, apertureSize=3, L2gradient=True)
    # TODO ustawić progi tak aby znajdowało wszystkie znaki, wtedy boundingboxy beda mialy sens

    image_contours, image_he = cv2.findContours(image_edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    best_fitted_contours = []
    new_hierarchy = []
    for index, countour in enumerate(image_contours):
        area = cv2.contourArea(countour)
        perimeter = cv2.arcLength(countour, False)
        if 480 < area and 150 < perimeter < 800:
            best_fitted_contours.append(countour)
            new_hierarchy.append(image_he[0][index])


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

    print("Found ",len(none_duplicated_conturs)," countours that probabli are signs")


    countoursBoxes = []
    for index,countour in enumerate(best_fitted_contours):

        rectangle = cv2.minAreaRect(countour)
        box = cv2.boxPoints(rectangle)
        box = np.int0(box)
        countoursBoxes.append([countour, box])
    ################### TO DRAW BOUNDING BOXES UNCOMMENT THIS #############################
    #     cv2.drawContours(image, [box], -1, (155, 0, 0), 3)
    #
    # cv2.drawContours(image, none_duplicated_conturs, -1, (0, 255, 255), 2)
    # cv2.imshow("", image)
    # cv2.waitKey(1000)

    return countoursBoxes


def findZero(contours, image):
    fits = []
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    for contour in contours:
        moments = cv2.moments(contour)
        huMoments = cv2.HuMoments(moments)
        for i in range(0, 7):
            huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
        fit = 0
        for i in range(0, 7):
            fit += abs((1 / abs(huMoments[i])) - (1 / HU_for_zero[i]))
        fits.append(fit)
    find = False
    found_in = None
    for index, fit in enumerate(fits):
        if fit < 0.15:
            find = True
            found_in = index
    return find


def extraxtSigns(conturs_boxes, image,answers,data):
    def key_to_sort(signwithcenter):
        return signwithcenter[1][0]  # x value of contour center

    def key_to_boxes(box):
        perimeter = cv2.arcLength(box, False)
        return perimeter

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

    # TODO kryterium na odległosc

    contour_centers = np.array(contour_centers)
    box_centers = np.array(box_centers)
    # contours, boxes and their centers
    contoursWithCenters = []
    boundingBoxesForSignsWithCenters = []

    # try for bounding boxes - a bit shity

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
        new= []
        for index,boundingBoxeForSignWithCenter in enumerate(boundingBoxesForSignsWithCenters):

            distance =((startBox[1][0] - boundingBoxeForSignWithCenter[1][0])**2
                           +(startBox[1][1] - boundingBoxeForSignWithCenter[1][1])**2)**0.5


            if abs(distance-distance_prev) > 20 or index == 0:
                new.append(boundingBoxeForSignWithCenter)
                distance_prev = distance
            else:
                prev_object = boundingBoxesForSignsWithCenters[index-1]
                prev_contour_lenght = cv2.arcLength(prev_object[0], False)
                contour_lenght=cv2.arcLength(boundingBoxeForSignWithCenter[0],False)
                suma = (prev_contour_lenght+contour_lenght)/2
                # print(prev_contour_lenght)
                # print(contour_lenght)
                # print(suma)
                if contour_lenght<suma:
                    new.append(boundingBoxesForSignsWithCenters[index-1])
                    boundingBoxesForSignsWithCenters.pop(index)
                else:
                    new.append(boundingBoxeForSignWithCenter)
                    boundingBoxesForSignsWithCenters.pop(index-1)




        BoundingBoxesForSigns = [contour[0] for contour in boundingBoxesForSignsWithCenters]


        contoursForSignsWithCenters = []
        index = 0
        for sign_contour, is_inlier in zip(contours, inliers_for_boxes):
            if is_inlier:
                box_center = box_centers[index]
                contoursForSignsWithCenters.append([sign_contour, box_center])
            index += 1

        contoursForSignsWithCenters.sort(key=key_to_sort)

        contoursForSigns = [contour[0] for contour in contoursForSignsWithCenters]

        # cv2.drawContours(image, contoursForSigns, -1, (255, 255, 0), 2)
        # cv2.imshow("",image)
        # cv2.waitKey()

    except ValueError:
        pass







    ############################################### SAVING DESCRIPTORS ###################################3
    # signs = [sign[0] for sign in boundingBoxesForSignsWithCenters]
    # descriptionOfSigns = []
    # dictionary = {}
    # if len(signs) == 7:
    #     # cv2.drawContours(image,signs,-1,(0,0,255),2)
    #     # for i,sign in enumerate(boundingBoxesForSignsWithCenters):
    #     #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     #     bottomLeftCornerOfText = (sign[1][0],sign[1][1])
    #     #     fontScale = 1
    #     #     fontColor = (0, 255, 0)
    #     #     lineType = 2
    #     #
    #     #     cv2.putText(image, str(i),
    #     #                 bottomLeftCornerOfText,
    #     #                 font,
    #     #                 fontScale,
    #     #                 fontColor,
    #     #                 lineType)
    #
    #     for signContour,sign in zip(signs,answers):
    #         moments = cv2.moments(signContour)
    #         huMoments = cv2.HuMoments(moments)
    #         for i in range(0, 7):
    #
    #             if huMoments[i] == 0.:
    #                 huMoments[i] = 0.001
    #             huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
    #
    #
    #         huMoments=huMoments.tolist()
    #         newHu = []
    #         for hu in huMoments:
    #             newHu.append(hu[0])
    #
    #         dictionary[sign] = newHu
    #
    # print(dictionary)
    # return dictionary
    #








    ##################################### MATCHING SHAPES #############################3

    signs = [sign[0] for sign in boundingBoxesForSignsWithCenters]
    readedSigns = []
    for sign in signs:
        best_match = "?"
        moments = cv2.moments(sign)
        huMoments = cv2.HuMoments(moments)
        for i in range(0, 7):

            if huMoments[i] == 0.:
                huMoments[i] = 0.001
            huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
        prev_fit = 100
        for set in data:
            fit = 0
            # fits=[]
            dataSignHU = set[1]
            for i in range(0, 7):
                fit += abs((1 / huMoments[i]) - (1 / dataSignHU[i]))
                # fit += abs((huMoments[i]) - (dataSignHU[i]))
                # fit = abs(huMoments[i]-dataSignHU[i])/abs(huMoments[i])
                # fits.append(fit)
            # print("fit",fit)
            # print("prev fit",prev_fit)
            # fit = max(fits)
            if fit < prev_fit:
                best_match = set[0]
                prev_fit=fit
        readedSigns.append(best_match)
    # print(answers)
    answerListOfChars=[]
    for char in answers:
        answerListOfChars.append(char)

    score = 0
    maxScore=0
    if len(answerListOfChars) == len(readedSigns):
        for sign_1,sign_2 in zip(answerListOfChars,readedSigns):
            maxScore+=1
            if sign_1 == sign_2:
                score+=1


    # cv2.waitKey()
    return readedSigns, score,maxScore


# def perform_processing(image: np.ndarray, answer) -> str
def perform_processing(image: np.ndarray, answer):
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here

    # findSigns_test(image)
    # findBoxes(image)

    data = readDataSet()
    # data=None
    conturs_boxes = findSigns(image)
    describedSigns = extraxtSigns(conturs_boxes, image,answer,data)
    describedSigns,score,maxscore = describedSigns
    # print(describedSigns)
    return describedSigns,score, maxscore
