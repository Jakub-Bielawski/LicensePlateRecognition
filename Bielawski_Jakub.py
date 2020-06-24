import argparse
import json
from pathlib import Path
import csv
import cv2
from matplotlib import pyplot as plt
import numpy as np
from processing.processingImage import perform_processing

answers = ["List of corrects plates signs."]


cropping = False


def SaveHuDescriptors(data):
    """
    Save data to CSV file:
    @:param data - dict sign -> HuMoments
    """

    csv.register_dialect("hashes", delimiter="#")
    f = open('/home/jakub/PycharmProjects/LicensePlateRecognition/signDescriptors/HuMomentsOfSigns_allPictures.csv', 'w')
    with f:
        for data in data:
            writer = csv.writer(f, dialect="hashes")
            for key in data:
                writer.writerow((key, data[key]))

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
    allPlates = 0
    NotReadedPlates = 0
    for index, image_path in enumerate(images_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue
        print(image_path)
        results[image_path.name] = perform_processing(image)
        with results_file.open('w') as output_file:
            json.dump(results, output_file, indent=4)

    #     ########## Calculating scores #########
    #     point = 0
    #     for cPlate, cAnswer in zip(results[image_path.name], answers[index]):
    #         if cPlate == cAnswer:
    #             point += 1
    #     score += point
    #     maxScore += 7
    #     allPlates += 1
    #     if results[image_path.name] == "???????":
    #         NotReadedPlates += 1
    # print(f"Plates haven't been found on: {NotReadedPlates}/{allPlates} pictures")
    # print(f"Results: {score}/{maxScore}. Acuraccy: {100 * (score / maxScore)}%")


    # SaveHuDescriptors(dataToSave)




if __name__ == '__main__':
    main()
