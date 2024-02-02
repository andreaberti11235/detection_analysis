import os
import argparse
from datetime import datetime
import glob
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
import pandas as pd

def iou(x1, y1, w1, h1, x2, y2, w2, h2):
  """Computes the intersection over union (IoU) of two bounding boxes.

  The IoU is a measure of how much two bounding boxes overlap. It is defined as the ratio of the area of the intersection to the area of the union of the two boxes.

  Parameters
  ----------
  x1 : float
    The x-coordinate of the center of the first box
  y1 : float
    The y-coordinate of the center of the first box
  w1 : float
    The width of the first box
  h1 : float
    The height of the first box
  x2 : float
    The x-coordinate of the center of the second box
  y2 : float
    The y-coordinate of the center of the second box
  w2 : float
    The width of the second box
  h2 : float
    The height of the second box

  Returns
  -------
  float
    The IoU of the two boxes, ranging from 0 to 1

  Raises
  ------
  ValueError
    If any of the parameters are negative or zero

  Examples
  --------
  >>> iou(0.5, 0.5, 1, 1, 0.6, 0.6, 0.8, 0.8)
  0.64
  >>> iou(0.5, 0.5, 1, 1, 1.5, 1.5, 1, 1)
  0.0
  """
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


def fill_in_pred_list(pred_list, pred_txt):
    """
    Fills in a list with the bounding box and confidence values from a text file.

    Args:
    pred_list (list): A list to store the predictions.
    pred_txt (str): The path to the text file with the predictions.

    Returns:
    list: The updated list with the predictions.
    """
    if os.path.exists(pred_txt):
        # Read the file as a pandas dataframe, skipping the first column
        predictions = pd.read_csv(pred_txt, sep=' ', header=None, usecols=[1, 2, 3, 4, 5])

        # Iterate over the rows of the dataframe
        for row in predictions.itertuples(index=False):
            # Extract the values from the row
            x_center, y_center, width, height, confidence = row

            # Create a list with the values
            prediction = [x_center, y_center, width, height, confidence]

            # Append the list to the pred_list
            pred_list.append(prediction)
    else:
        # Handle the case when the file does not exist
        pred_list = []

    # Return the pred_list
    return pred_list



def main():
    parser = argparse.ArgumentParser(description='Analyse the prediction outputs to find TP and FP.')
    parser.add_argument('gt_dir', help='Absolute path of the folder containing the ground-truth txt files')
    parser.add_argument('pred_v5_dir', help='Absolute path of the folder containing the prediction txt files from YOLO v5')
    parser.add_argument('pred_v8_dir', help='Absolute path of the folder containing the prediction txt files from YOLO v8')
    parser.add_argument('out_dir', help='Absolute path of the folder where the output file, with the resulting numbers, will be saved')
    parser.add_argument('-iou_th_gt', '--iou_threshold_gt', type=float, default=0.1, help='Threshold for IOU (default = 0.1)when comparing predictions to GT')
    parser.add_argument('-iou_th_p', '--iou_threshold_prediction', type=float, default=0.1, help='Threshold for IOU (default = 0.1), when comparing predictions from V5 to those from v8')
    parser.add_argument('-conf_th', '--conf_threshold', type=float, default=0.1, help='Threshold for confidence value (default = 0.1)')
    args = parser.parse_args()

    gt_dir = args.gt_dir
    pred_v5_dir = args.pred_v5_dir
    pred_v8_dir = args.pred_v8_dir
    out_dir = args.out_dir
    iou_threshold_gt = args.iou_threshold_gt
    iou_threshold_pred = args.iou_threshold_prediction
    conf_threshold = args.conf_threshold

    tot_n_masses = 0
    nr_TP = 0
    nr_FP = 0
    nr_FN = 0

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for element in glob.glob(os.path.join(gt_dir, '*')):
        gt_txt = element
        pred_v5_txt = os.path.join(pred_v5_dir, os.path.basename(element))
        pred_v8_txt = os.path.join(pred_v8_dir, os.path.basename(element))

        gt_df = pd.read_csv(gt_txt, sep=' ', header=None)

        gt_list = []
        pred_v5_list = []
        pred_v8_list = []


        for idx in gt_df.index:
            x_center = gt_df.iloc[idx][1]
            y_center = gt_df.iloc[idx][2]
            width = gt_df.iloc[idx][3]
            height = gt_df.iloc[idx][4]

            gt_element = [x_center, y_center, width, height]
            gt_list.append(gt_element) # è una lista di liste, ciascuna delle quali contiene gli elementi di un bbox

        
        tot_n_masses += len(gt_list)
        found_masses = np.zeros(shape=len(gt_list))

        # devo spezzarli: prima guardo se esiste il txt di v5, in caso sompilo la lista delle predizioni; poi altro if
        # allo stesso livello e guardo se esiste il txt di v8, nel caso compilo la lista per v8;
        # dopodiché, controllo IOU tra le due liste (se entrambe non vuote), eventualmente fondendo le predizioni
        # aggiungendo le pred in più, sommando la probabilità e prendendo il bbox più grande per quelle coincidenti;
        # per finire controllo IOU tra la lista risultante e la GT

        pred_v5_list = fill_in_pred_list(pred_v5_list, pred_v5_txt)
        pred_v8_list = fill_in_pred_list(pred_v8_list, pred_v8_txt)


        
        for detection in pred_v8_list:
            n_trovati = 0
            distances = []
            # per ogni elemento trovato, inizializzo una lista, 
            # poi guardo quanto dista da tutti i bbox della v5 e appendo alla lista
            for bbox in pred_v5_list:
                pred_v8_xy = np.array(detection[0:2])
                pred_v5_xy = np.array(bbox[0:2])

                distance = linalg.norm(pred_v8_xy - pred_v5_xy)
                distances.append(distance)
            
            # trovo a quale massa GT sta più vicino e per quella calcolo la IOU
            closest_item = np.argmin(distances)
            pred_v5_values = pred_v5_list[closest_item]
            iou_value = iou(x1=detection[0], y1=detection[1], w1=detection[2], h1=detection[3],
                            x2=pred_v5_values[0], y2=pred_v5_values[1], w2=pred_v5_values[2], h2=pred_v5_values[3])

            # se la IOU è minore di una certa soglia, le considero come coincidenti
            if iou_value >= iou_threshold_pred:
                # print(f'è un vero Vero Positivo, massa più vicina riga {closest_item}')
                n_trovati += 1
            if n_trovati >= 2:
                print(f'Paziente {element} troppi finding coincidono, n_trovati={n_trovati}')
                









    #     for detection in pred_list:
    #         distances = []
    #         # per ogni elemento trovato, inizializzo una lista, 
    #         # poi guardo quanto dista da tutti i bbox della gt e appendo alla lista
    #         for bbox in gt_list:
    #             pred_xy = np.array(detection[0:2])
    #             gt_xy = np.array(bbox[0:2])

    #             distance = linalg.norm(pred_xy - gt_xy)
    #             distances.append(distance)
            
    #         # trovo a quale massa GT sta più vicino e per quella calcolo la IOU
    #         closest_item = np.argmin(distances)
    #         gt_values = gt_list[closest_item]
    #         iou_value = iou(x1=detection[0], y1=detection[1], w1=detection[2], h1=detection[3],
    #                         x2=gt_values[0], y2=gt_values[1], w2=gt_values[2], h2=gt_values[3])

    #         # se la IOU è minore di una certa soglia, le considero come coincidenti
    #         if iou_value >= iou_threshold_gt:
    #             # print(f'è un vero Vero Positivo, massa più vicina riga {closest_item}')
    #             found_masses[closest_item] = 1
                

    #         # else:
    #         #     # print('è un Falso Positivo')
    #         #     nr_FP +=1

    #     nr_TP += np.sum(found_masses)
    #     nr_FP += len(pred_list) - np.sum(found_masses)
    #     nr_FN += len(found_masses) - np.sum(found_masses)


    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y__%H_%M")
    # filename = f'count_{dt_string}.txt'
   
    # while os.path.exists(os.path.join(out_dir, filename)):
    #     filename = f'{filename[:-4]}_new.txt'
    
    # with open(os.path.join(out_dir, filename), 'w') as out_file:
    #     out_file.write(f'Total number of found masses TP={nr_TP}\n')
    #     out_file.write(f'Total number of wrong detections found FP={nr_FP}\n')
    #     out_file.write(f'Number of non-detected masses FN={nr_FN}\n')
    #     out_file.write(f'Total number of masses Tot={tot_n_masses}\n\n')
    #     out_file.write(f'IOU threshold value = {iou_threshold}\n')
    #     out_file.write(f'Confidence threshold value = {conf_threshold}\n')

    #     # leggo tutti i GT e li metto in una lista,
    #     # leggo tutte le predizioni e le metto in una lista
    #     # per ogni predizione dovrei calcolare le distanze da ogni GT


if __name__ == "__main__":
    main()
