import os
import argparse
from datetime import datetime
import glob
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
import pandas as pd

def iou(x1, y1, w1, h1, x2, y2, w2, h2):
  # convert the coordinates to the top-left and bottom-right corners
  x1_1 = x1 - w1 / 2
  y1_1 = y1 - h1 / 2
  x1_2 = x1 + w1 / 2
  y1_2 = y1 + h1 / 2
  x2_1 = x2 - w2 / 2
  y2_1 = y2 - h2 / 2
  x2_2 = x2 + w2 / 2
  y2_2 = y2 + h2 / 2

  # compute the area of the bounding boxes
  area1 = w1 * h1
  area2 = w2 * h2

  # compute the intersection area
  xi1 = max(x1_1, x2_1)
  yi1 = max(y1_1, y2_1)
  xi2 = min(x1_2, x2_2)
  yi2 = min(y1_2, y2_2)
  inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

  # compute the union area
  union_area = area1 + area2 - inter_area

  # compute the IoU
  iou = inter_area / union_area

  return iou


def main():
    parser = argparse.ArgumentParser(description='Analyse the prediction outputs to find TP and FP.')
    parser.add_argument('gt_dir', help='Absolute path of the folder containing the ground-truth txt files')
    parser.add_argument('pred_dir', help='Absolute path of the folder containing the prediction txt files')
    parser.add_argument('out_dir', help='Absolute path of the folder where the output file, with the resulting numbers, will be saved')
    parser.add_argument('-th', '--iou_threshold', type=float, default=0.1, help='Threshold for IOU (default = 0.1)')
    args = parser.parse_args()

    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    out_dir = args.out_dir
    iou_threshold = args.iou_threshold

    tot_n_masses = 0
    nr_TP = 0
    nr_FP = 0
    nr_FN = 0

    for element in glob.glob(os.path.join(gt_dir, '*')):
        gt_txt = element
        pred_txt = os.path.join(pred_dir, os.path.basename(element))

        gt_df = pd.read_csv(gt_txt, sep=' ', header=None)

        gt_list = []
        pred_list = []


        for idx in gt_df.index:
            x_center = gt_df.iloc[idx][1]
            y_center = gt_df.iloc[idx][2]
            width = gt_df.iloc[idx][3]
            height = gt_df.iloc[idx][4]

            gt_element = [x_center, y_center, width, height]
            gt_list.append(gt_element) # è una lista di liste, ciascuna delle quali contiene gli elementi di un bbox

        
        tot_n_masses += len(gt_list)
        found_masses = np.zeros(shape=len(gt_list))


        if os.path.exists(pred_txt):
            pred_df = pd.read_csv(pred_txt, sep=' ', header=None)


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
                gt_values = gt_list[closest_item]
                iou_value = iou(x1=detection[0], y1=detection[1], w1=detection[2], h1=detection[3],
                                x2=gt_values[0], y2=gt_values[1], w2=gt_values[2], h2=gt_values[3])

                if iou_value <= iou_threshold:
                    # print(f'è un vero Vero Positivo, massa più vicina riga {closest_item}')
                    found_masses[closest_item] = 1
                    

                # else:
                #     # print('è un Falso Positivo')
                #     nr_FP +=1

        nr_TP += np.sum(found_masses)
        nr_FP += len(pred_list) - np.sum(found_masses)
        nr_FN += len(found_masses) - np.sum(found_masses)


    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M")
    filename = f'count_{dt_string}.txt'
   
    while os.path.exists(os.path.join(out_dir, filename)):
        filename = f'{filename[:-4]}_new.txt'
    
    with open(os.path.join(out_dir, filename), 'w') as out_file:
        out_file.write(f'Total number of found masses TP={nr_TP}\n')
        out_file.write(f'Total number of wrong detections found FP={nr_FP}\n')
        out_file.write(f'Number of non-detected masses FN={nr_FN}\n')
        out_file.write(f'Total number of masses Tot={tot_n_masses}')

        # leggo tutti i GT e li metto in una lista,
        # leggo tutte le predizioni e le metto in una lista
        # per ogni predizione dovrei calcolare le distanze da ogni GT


if __name__ == "__main__":
    main()
