import argparse
import json
from pathlib import Path
import csv
import cv2
from matplotlib import pyplot as plt

from processing.processingImage import perform_processing

answers = ["PZ267NY", "PO9JW55", "PZ50891", "PZ206SP", "PO778SS", "PO389LJ", "PO4FK55", "PZ685TC", "PO8MG89", "CB107GP",
           "PO8FV07", "PLE57S2", "PO970CN", "DJ69026", "PZ199NK", "DW6K559", "WF9481T", "WE596RF", "PKOH518", "PK89752",
           "GD9305V", "PO028EG", "PZ25962", "PO434FY", "ELW05YC", "PKA91KX", "WP7285G", "FZG66FU", "FG52945", "PO9E342",
           "PO778SS", "PRAEF88", "POBGU41", "POBGU41", "PP31442", "PO868RL", "PGNPX52", "PZ185CW", "POZ40FE", "PZ185CW",
           "PO771VX", "PO073VR", "PO751SM", "PO5206T", "PO5206T", "PO751SM", "PSLRK05", "PO722VE", "POBGU41"
           ]
answersForNewData = ["PO692TY", "PO692TY", "PWACJ25", "PWACJ25", "PWACJ25", "WE828NJ", "WE828NJ", "PN09394", "PN09394",
                     "PP2376J", "PP2376J", "PO257RU", "PO257RU", "PO8M998", "PO5GN86", "PZ859HA", "PZ796TG", "PZ415RH",
                     "PO6NA38", "PO6NA38", "PO6NA38", "CG80588", "PO138VC", "PSZ71UY", "PSZ71UY", "FG7067F", "FG7067F",
                     "FG7067F", "PO7CU79", "PO7CU79", "FZ1787L", "PO47T82", "SC0615R", "SC0615R", "SC0615R", "PZ6947K",
                     "PO2T010", "PO2T010", "PO01078", "PO01078", "PL11677", "PO8AJ07", "PO8HT72", "FG2652C", "FG2652C",
                     "PO7CF11", "WE301MH", "WE301MH", "WE301MH", "PO3MU53", "PO3MU53", "ZWA50MS", "ZWA50MS", "PO5NC03",
                     "PO5NC03", "PO5S123", "PO5S123", "PN0546A", "PN0546A", "PO2JC41", "PO2JC41", "PO683SX", "PO4R823",
                     "PO4R823", "PO339LH", "PO339LH", "PO339LH", "CBYCG05", "PO4J901", "PO4J901", "PO4J901", "PO4M754",
                     "PO4M754", "PO302EH", "PO302EH", "PO8L319", "PO8L319", "PO8L319", "PO8L319"

                     ]

cropping = False
allPlates =  []
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}
    dataToSave = []
    score = 0
    maxScore = 0
    histToSave = []
    for index, image_path in enumerate(images_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue
        # image = cv2.resize(image, (2560, 1920))
        # image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #
        # x_start, y_start, x_end, y_end = 0, 0, 0, 0
        #
        # oriImage = image.copy()
        #
        # def mouse_crop(event, x, y, flags, param):
        #     # grab references to the global variables
        #     global x_start, y_start, x_end, y_end, cropping
        #
        #     # if the left mouse button was DOWN, start RECORDING
        #     # (x, y) coordinates and indicate that cropping is being
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         x_start, y_start, x_end, y_end = x, y, x, y
        #         cropping = True
        #
        #     # Mouse is Moving
        #     elif event == cv2.EVENT_MOUSEMOVE:
        #         if cropping == True:
        #             x_end, y_end = x, y
        #
        #     # if the left mouse button was released
        #     elif event == cv2.EVENT_LBUTTONUP:
        #         # record the ending (x, y) coordinates
        #         x_end, y_end = x, y
        #         cropping = False  # cropping is finished
        #
        #         refPoint = [(x_start, y_start), (x_end, y_end)]
        #
        #         if len(refPoint) == 2:  # when two points were found
        #             roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
        #             allPlates.append(roi)
        #             histr = cv2.calcHist([roi], [0], None, [256], [0, 256])
        #             # show the plotting graph of an image
        #             histToSave.append(histr)
        #
        #             cv2.imshow("Cropped", roi)
        #
        # f = open('/home/jakub/PycharmProjects/LicensePlateRecognition/signDescriptors/histPlates.csv', 'w')
        # with f:
        #     writer = csv.writer(f)
        #     for histr in histToSave:
        #         values = [value[0] for value in histr]
        #         writer.writerow((values))
        # plt.show()
        #
        #
        # cv2.namedWindow("image")
        # cv2.setMouseCallback("image", mouse_crop)
        #
        # while (1):
        #     if cv2.waitKey(20) & 0xFF == 27:
        #         break
        #     i = image.copy()
        #     if not cropping:
        #         cv2.imshow("image", image)
        #     elif cropping:
        #         cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        #         cv2.imshow("image", i)
        #     cv2.waitKey(1)
        # # close all open windows
        # cv2.destroyAllWindows()


        results[image_path.name] = perform_processing(image, answersForNewData[index])
        # dictionarywithSignDescriptor = perform_processing(image, answersForNewData[index])
        # dataToSave.append(dictionarywithSignDescriptor)
        # print(answersForNewData[index])
        # print(dictionarywithSignDescriptor[0])
        # score += dictionarywithSignDescriptor[1]
        # maxScore += dictionarywithSignDescriptor[2]
        # maxScore += 7
    # print(f"Results: {score}/{maxScore}. Acuraccy: {100 * (score / maxScore)}%")
    ########################## SAVING DATA ##############################'
    # csv.register_dialect("hashes", delimiter="#")
    # f = open('/home/jakub/PycharmProjects/LicensePlateRecognition/signDescriptors/HU_corrected_Learn.csv', 'w')
    # with f:
    #     for data in dataToSave:
    #         writer = csv.writer(f, dialect="hashes")
    #         for key in data:
    #             writer.writerow((key, data[key]))
    ###################################################################

    ###### CALC HISTS ######



    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
