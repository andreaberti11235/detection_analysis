import os
import argparse
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Analyse the prediction outputs to find TP and FP.')
    parser.add_argument('gt_txt', help='Absolute path of the txt file with the ground-truth annotations')
    parser.add_argument('pred_txt', help='Absolute path of the txt file with the annotations of the detected bboxes')
    args = parser.parse_args()

    gt_txt = args.gt_txt
    pred_txt = args.pred_txt

    gt_df = pd.read_csv(gt_txt, sep=' ', header=None)
    pred_df = pd.read_csv(pred_txt, sep=' ', header=None)

    gt_list = []
    pred_list = []

    for idx in gt_df.index:
        x_center = gt_df.iloc[idx][1]
        y_center = gt_df.iloc[idx][2]
        width = gt_df.iloc[idx][3]
        height = gt_df.iloc[idx][4]

        gt_element = [x_center, y_center, width, height]
        gt_list.append(gt_element) # è una lista di liste, ciascuna delle quali contiene gli elementi di un bbox

    for idx in pred_df.index:
        x_center = pred_df.iloc[idx][1]
        y_center = pred_df.iloc[idx][2]
        width = pred_df.iloc[idx][3]
        height = pred_df.iloc[idx][4]
        confidence = pred_df.iloc[idx][5]

        pred_element = [x_center, y_center, width, height, confidence]
        pred_list.append(pred_element)

    for detection in pred_list:
        distances = []
        # per ogni elemento trovato, inizializzo una lista, 
        # poi guardo quanto dista da tutti i bbox della gt e appendo alla lista
        for bbox in gt_list:
            pred_xy = np.array(detection[0:2])
            gt_xy = np.array(bbox[0:2])

            distance = linalg.norm(pred_xy - gt_xy)
            distances.append(distance)
        
        closest_item = np.argmin(distances)
        diagonal = np.hypot(gt_list[closest_item][2], gt_list[closest_item][3])

        if distances[closest_item] <= (1.5*diagonal)/2:
            print(f'è un vero Vero Positivo, massa più vicina riga {closest_item}')

        else:
            print('è un Falso Positivo')




    # leggo tutti i GT e li metto in una lista,
    # leggo tutte le predizioni e le metto in una lista
    # per ogni predizione dovrei calcolare le distanze da ogni GT


if __name__ == "__main__":
    main()
