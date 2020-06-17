import argparse
import json
from pathlib import Path
import csv
import cv2
from matplotlib import pyplot as plt

from processing.processingImage import perform_processing

answers = ["PZ267NY", "PO9JW55", "PZ50891", "PZ206SP", "PO778SS", "PO389LJ", "PO4FK55", "PZ685TC", "PO8MG89",
           # "CB107GP","PO8FV07", "PLE57S2", "PO970CN", "DJ69026", "PZ199NK", "DW6K559", "WF9481T", "WE596RF", "PKOH518",
           "PK89752","GD9305V", "PO028EG", "PZ25962", "PO434FY", "ELW05YC", "PKA91KX", "WP7285G", "FZG66FU", "FG52945",
           "PO9E342","PO778SS", "PRAEF88", "POBGU41", "POBGU41"
    # , "PP31442", "PO868RL", "PGNPX52", "PZ185CW", "POZ40FE",
    #        "PZ185CW","PO771VX", "PO073VR", "PO751SM", "PO5206T", "PO5206T", "PO751SM", "PSLRK05", "PO722VE", "POBGU41"
           ]
# answers = ["CB107GP","PO8FV07", "PLE57S2", "PO970CN", "DJ69026", "PZ199NK", "DW6K559", "WF9481T", "WE596RF", "PKOH518",
#            "PP31442", "PO868RL", "PGNPX52", "PZ185CW", "POZ40FE",
#            "PZ185CW","PO771VX", "PO073VR", "PO751SM", "PO5206T", "PO5206T", "PO751SM", "PSLRK05", "PO722VE", "POBGU41"]
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
allPlates = []
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
        print(image_path)
        # results[image_path.name] = perform_processing(image, answersForNewData[index])
        dictionarywithSignDescriptor,maSC = perform_processing(image, answers[index])
        # score += dictionarywithSignDescriptor
        # dataToSave.append(dictionarywithSignDescriptor)
        # print(answersForNewData[index])
        # print(dictionarywithSignDescriptor[0])
        score += dictionarywithSignDescriptor
        maxScore += maSC
        # maxScore += 7
    print(f"Results: {score}/{maxScore}. Acuraccy: {100 * (score / maxScore)}%")
    ########################## SAVING DATA ##############################'
    # csv.register_dialect("hashes", delimiter="#")
    # f = open('/home/jakub/PycharmProjects/LicensePlateRecognition/signDescriptors/HuMomentsOfSigns.csv', 'w')
    # with f:
    #     for data in dataToSave:
    #         writer = csv.writer(f, dialect="hashes")
    #         for key in data:
    #             writer.writerow((key, data[key]))
    ###################################################################



    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
