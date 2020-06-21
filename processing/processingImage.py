import numpy as np
import cv2
import math
import csv
import ast
from processing import platesHists


def XCoordinateSort(box):
    return box[0]


def readDataSet():
    """
    Read a date set ( sign description by HuMoments )
    :return: list of [Sign,HuMoments of sign]
    """
    f = open('signDescriptors/HuMomentsOfSigns_allPictures.csv', 'r')
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
    hist_2 = np.array(hist_2).astype(np.float32)
    hist_1 = np.array(hist_1).astype(np.float32)
    return cv2.compareHist(hist_1, hist_2, method=cv2.HISTCMP_CORREL)


errors = 0


def findPlate(image, th1, coefficient=1., SHOW=False):
    """
    :param image: Image in (1920,2560) size
    :param th1: If set, use binary thresh with given thresh value. Else do the best to return well threshed plate
    :param coefficient: Increases the limits of possible plates
    :param SHOW: To see returns, set to True
    :return: Grayscale image of plate and thresholded image of plate B&W
    """
    global errors
    IMAGE_COPY = np.copy(image)
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

        rect_min = cv2.minAreaRect(countour)
        box = cv2.boxPoints(rect_min)
        box = np.int0(box)
        aspect_ratio = float(w) / h
        arearec = w * h
        if 17_000 < arearec < 200_000:
            if 1.5 < aspect_ratio < 5.8 * coefficient:
                perimeter = 2 * w + 2 * h
                if perimeter < 2_000 * coefficient and 1200 * coefficient > w > 400 * (1 / coefficient):
                    area = cv2.contourArea(box)
                    if area < 50_000 * (1 / coefficient):
                        continue
                    mask = np.zeros(image_gray.shape, np.uint8)
                    cv2.drawContours(mask, [box], 0, 255, -1)

                    plate = cv2.bitwise_and(image_gray, mask)
                    plates.append(plate)
                    hist = cv2.calcHist([image_gray], [0], mask, [32], [0, 256])
                    hist_norm = np.linalg.norm(hist)
                    hist_n = ((hist / hist_norm) * 255).astype(np.uint8)

                    histOfImagePieces.append(hist_n)
    matches = []
    for histogram in histOfImagePieces:
        matches_ = []
        for histogram_ in platesHists.best:
            matches_.append(return_intersection(histogram, histogram_))
        matches.append(np.max(matches_))

    matches = np.array(matches)
    try:
        indexOfPlate = np.argmax(matches)
        plate = np.copy(plates[indexOfPlate])
        if th1 == 0:
            blur_plate = cv2.GaussianBlur(plate, (5, 5), 0)
            _, th3 = cv2.threshold(blur_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th3 = cv2.dilate(th3, (5, 5), iterations=2)
        else:
            _, th3 = cv2.threshold(plate, th1, 255, cv2.THRESH_BINARY)

        if SHOW:
            while (1):
                if cv2.waitKey(20) & 0xFF == 27:
                    break

                img_edge = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
                cv2.imshow("PLATE", plate)
                cv2.imshow("PLATE_THRESH", th3)

                cv2.imshow('image', img_edge)
        return plate, th3

    except ValueError:
        if errors == 0:
            errors += 1
            plate, th3 = findPlate(IMAGE_COPY, 0, 1.5)
            errors = 0
            return plate, th3
        print("ERROR IN PLATE SEARCHING")


def findBoxes(image, SHOW=False):
    """
    Find boxes that probably are signs on image of plate.
    :param image: Image of plate B&W
    :param SHOW: If you want to see bounding boxes of signs, set to True
    :return: Bounding boxes for signs
    """
    image_contours, image_he = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    signsBoxes = []
    allH = []
    for index, countour in enumerate(image_contours):
        x, y, w, h = cv2.boundingRect(countour)
        area = w * h
        apectRatio = h / w

        if 22_000 > area > 1_900 and apectRatio > 0.8:
            allH.append(h)
            signsBoxes.append((x, y, w, h))

    # eliminate inside boxes
    boxCenters = [(box[0] + (box[2] / 2), box[1] + (box[3] / 2)) for box in signsBoxes]
    # print(len(boxCenters))
    for boxID, box in enumerate(signsBoxes):
        x, y, w, h = box
        for boxCenterID, boxCenter in enumerate(boxCenters):
            if boxID != boxCenterID:
                # Check if is inside

                if x < boxCenter[0] < x + w and y < boxCenter[1] < y + h:
                    area = w * h
                    box_ = signsBoxes[boxCenterID]
                    area_ = box_[2] * box_[3]
                    if area < area_:
                        signsBoxes.pop(boxID)
                        boxCenters.pop(boxCenterID)

    # eliminate height outliers
    if len(allH) > 0:
        meanH = sum(allH) / len(allH)
        normalizedH = [h / meanH for h in allH]
        index_to_erase = 0
        for box, H_nn in zip(signsBoxes, normalizedH):
            #
            if H_nn < 0.50 or H_nn > 1.25:
                signsBoxes.pop(index_to_erase)
            index_to_erase += 1

    Boxes = []
    for box in signsBoxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 4)
        Boxes.append(box)

    # sort Boxes
    Boxes.sort(key=XCoordinateSort)

    if SHOW:
        while (1):
            if cv2.waitKey(20) & 0xFF == 27:
                break
            cv2.imshow("", image)

    return Boxes


def cutSigns(Plate, Boxes):
    """
    Cuts signs from image of plate given license plate, using a bit larger boxes than given in Boxes.
    :param Plate: Grayscale image of plate
    :param Boxes: List of contours for signs
    :return: List of signs image ( white sign, black background)
    """
    Signs = []
    for box in Boxes:
        # make box a bit larger
        x, y, w, h = box
        x -= 3
        y -= 3
        w += 6
        h += 6
        Signs.append(Plate[y:y + h, x:x + w])

    BwSigns = []
    if len(Signs) == 7:
        for sign in Signs:
            _, thresh = cv2.threshold(sign, 80, 255, cv2.THRESH_BINARY)
            dilation = cv2.dilate(cv2.bitwise_not(thresh), (3, 3), iterations=2)
            BwSigns.append(dilation)
    return BwSigns


def saveHuDescriptors(Signs, answers):
    """
    Create a dictionary sign -> HuMomenst
    :param Signs: List of sign images, sign = 255, background = 0
    :param answers: String with correct plate signs
    :return: Dict sign -> HuDescriptors
    """

    dictionary = {}
    for sign, AnswerSign in zip(Signs, answers):
        descriptionOfSigns = []
        #################################### HU DESCRIPTOR ##############################
        moments = cv2.moments(sign)
        huMoments = cv2.HuMoments(moments)
        for i in range(0, 7):
            # if huMoments[i] == 0.:
            #     huMoments[i] = 0.001
            huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
        huMoments = huMoments.tolist()

        for hu in huMoments:
            descriptionOfSigns.append(hu[0])
        dictionary[AnswerSign] = descriptionOfSigns
    # print(dictionary)
    return dictionary


def matchSigns(signs):
    """

    :param signs: Signs to describe
    :return: String with matched signs
    """
    data = readDataSet()
    readedSigns = []
    for index_of_sign, sign in enumerate(signs):
        ##################################### HU DESCRIPTOR ############################
        best_match = "?"
        moments = cv2.moments(sign)
        huMoments = cv2.HuMoments(moments)
        for i in range(0, 7):
            if huMoments[i] == 0.:
                huMoments[i] = 0.001
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
                if dataSignHU[i] == 0.:
                    dataSignHU[i] = 0.001
                fit += abs((1 / huMoments[i]) - (1 / dataSignHU[i]))

            if fit < prev_fit:
                best_match = dataSign
                prev_fit = fit
        readedSigns.append(best_match)

    # convert to string
    readedPlateString = ""
    if len(readedSigns) < 7:
        readedPlateString = "???????"
    for rSign in readedSigns:
        readedPlateString += str(rSign)
    return readedPlateString


def perform_processing(image: np.ndarray) -> str:
    """

    :param image:
    :return: String ( Plate signs )
    """

    image = cv2.resize(image, (2560, 1920))

    plate, thresholdedPlate = findPlate(image, 0)
    boxes = findBoxes(thresholdedPlate)

    signs = cutSigns(plate, boxes)

    readedPlate = matchSigns(signs)
    tryies = 7
    startThresh = 80
    for iteration in range(0, tryies):
        if readedPlate == "???????" and tryies != 0:
            plate, thresholdedPlate = findPlate(image, startThresh)
            boxes = findBoxes(thresholdedPlate)
            signs = cutSigns(plate, boxes)
            readedPlate = matchSigns(signs)
            startThresh += 10

    print(readedPlate)
    return readedPlate
