import argparse
import json
from pathlib import Path
import csv
import cv2
from matplotlib import pyplot as plt
import numpy as np
from processing.processingImage import perform_processing

answers = ["PZ267NY", "PO9JW55", "PZ50891", "PZ206SP", "PO778SS", "PO389LJ", "PO4FK55", "PZ685TC", "PO8MG89",
           "CB107GP", "PO8FV07", "PLE57S2", "PO970CN", "DJ69026", "PZ199NK", "DW6K559", "WF9481T", "WE596RF", "PKOH518",
           "PK89752", "GD9305V", "PO028EG", "PZ25962", "PO434FY", "ELW05YC", "PKA91KX", "WP7285G", "FZG66FU", "FG52945",
           "PO9E342", "PO778SS", "PRAEF88", "POBGU41", "POBGU41", "PP31442", "PO868RL", "PGNPX52", "PZ185CW", "POZ40FE",
           "PZ185CW", "PO771VX", "PO073VR", "PO751SM", "PO5206T", "PO5206T", "PO751SM", "PSLRK05", "PO722VE", "POBGU41"
           ]


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

        results[image_path.name] = perform_processing(image)


        ########## Calculating scores #########
        point = 0
        for cPlate, cAnswer in zip(results[image_path.name], answers[index]):
            if cPlate == cAnswer:
                point += 1
        score += point
        maxScore += 7
        allPlates += 1
        if results[image_path.name] == "???????":
            NotReadedPlates += 1
    print(f"Plates haven't been found on: {NotReadedPlates}/{allPlates} pictures")
    print(f"Results: {score}/{maxScore}. Acuraccy: {100 * (score / maxScore)}%")


    # SaveHuDescriptors(dataToSave)

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
