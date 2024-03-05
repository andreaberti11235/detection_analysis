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

def get_minimal_bounding_box(x1, y1, w1, h1, x2, y2, w2, h2):
    """
    Returns the minimal bounding box containing two bounding boxes.

    Args:
    x1 (float): The x coordinate of the center of the first bounding box.
    y1 (float): The y coordinate of the center of the first bounding box.
    w1 (float): The width of the first bounding box.
    h1 (float): The height of the first bounding box.
    x2 (float): The x coordinate of the center of the second bounding box.
    y2 (float): The y coordinate of the center of the second bounding box.
    w2 (float): The width of the second bounding box.
    h2 (float): The height of the second bounding box.

    Returns:
    tuple: A tuple of four floats representing the x coordinate, y coordinate, width, and height of the minimal bounding box.
    """
    # Get the corners of the first bounding box
    x1_min = x1 - w1 / 2
    x1_max = x1 + w1 / 2
    y1_min = y1 - h1 / 2
    y1_max = y1 + h1 / 2

    # Get the corners of the second bounding box
    x2_min = x2 - w2 / 2
    x2_max = x2 + w2 / 2
    y2_min = y2 - h2 / 2
    y2_max = y2 + h2 / 2

    # Find the minimum and maximum values of x and y coordinates
    x_min = min(x1_min, x2_min)
    x_max = max(x1_max, x2_max)
    y_min = min(y1_min, y2_min)
    y_max = max(y1_max, y2_max)

    # Calculate the width and height of the minimal bounding box
    w = x_max - x_min
    h = y_max - y_min

    # Calculate the x_center and y_center of the minimal bounding box
    x = x_min + w / 2
    y = y_min + h / 2

    # Return the x_center, y_center, width, and height of the minimal bounding box
    return (x, y, w, h)

def main():
    parser = argparse.ArgumentParser(description='Analyse the prediction outputs from the ensemble detections to find TP and FP. Keep only the first n detections with the highest confidence scores for each image.')
    parser.add_argument('gt_dir', help='Absolute path of the folder containing the ground-truth txt files')
    parser.add_argument('pred_v5_dir', help='Absolute path of the folder containing the prediction txt files from YOLO v5')
    parser.add_argument('pred_v8_dir', help='Absolute path of the folder containing the prediction txt files from YOLO v8')
    parser.add_argument('out_dir', help='Absolute path of the folder where the output file, with the resulting numbers, will be saved')
    parser.add_argument('-iou_th_gt', '--iou_threshold_gt', type=float, default=0.1, help='Threshold for IOU (default = 0.1)when comparing predictions to GT')
    parser.add_argument('-iou_th_p', '--iou_threshold_prediction', type=float, default=0.3, help='Threshold for IOU (default = 0.3), when comparing predictions from V5 to those from v8')
    parser.add_argument('-i', '--interval_percentage', type=float, default=0.2, help='Only the detections within a percentage from the one with the highest confidence. For ex., if the detection with the highest confidence has conf=1.0, and we give -i 0.2, we will consider only the events with confidence >= 0.8')
    parser.add_argument('-file', '--append_to_file', type=str, help='Path to the txt file, where the resulting metrics will be appended (for comparison analysis). If not given, only the file in the out_dir will be produced. The file MUST EXIST and have the following columns nr_TP nr_FP nr_FN (separated by a space)')
    args = parser.parse_args()

    gt_dir = args.gt_dir
    pred_v5_dir = args.pred_v5_dir
    pred_v8_dir = args.pred_v8_dir
    out_dir = args.out_dir
    iou_threshold_gt = args.iou_threshold_gt
    iou_threshold_pred = args.iou_threshold_prediction
    interval_percentage = args.interval_percentage
    append_to_file = args.append_to_file

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

        pred_fusion = pred_v5_list[:]
        # parto inizializzando pred_fusion a v5, poi scorro v8, se la predizione coincide con una di quelle di v5
        # levo da v5 e la sostituisco con la fusione, altrimenti appendo alla lista


        
        for detection in pred_v8_list:
            if len(pred_v5_list) == 0:
                pred_fusion = pred_v8_list[:]
            else:
                # n_trovati = 0
                distances = []
                # per ogni elemento trovato, inizializzo una lista, 
                # poi guardo quanto dista da tutti i bbox della v5 e appendo alla lista
                for bbox in pred_v5_list:
                    pred_v8_xy = np.array(detection[0:2])
                    pred_v5_xy = np.array(bbox[0:2])

                    distance = linalg.norm(pred_v8_xy - pred_v5_xy)
                    distances.append(distance)
            
                # trovo a quale v5 sta più vicina a quella v8 che sto considerando e per quella calcolo la IOU
                closest_item = np.argmin(distances)
                pred_v5_values = pred_v5_list[closest_item]
                iou_value = iou(x1=detection[0], y1=detection[1], w1=detection[2], h1=detection[3],
                                x2=pred_v5_values[0], y2=pred_v5_values[1], w2=pred_v5_values[2], h2=pred_v5_values[3])

                # se la IOU è maggiore di una certa soglia, le considero come coincidenti
                if iou_value >= iou_threshold_pred:
                    # print(f'è un vero Vero Positivo, massa più vicina riga {closest_item}')
                #     n_trovati += 1
                # if n_trovati >= 2:
                #     print(f'Paziente {element} troppi finding coincidono, n_trovati={n_trovati}')
                    x, y, w, h = get_minimal_bounding_box(x1=detection[0], y1=detection[1], w1=detection[2], h1=detection[3],
                                x2=pred_v5_values[0], y2=pred_v5_values[1], w2=pred_v5_values[2], h2=pred_v5_values[3])

                    conf = detection[4] + pred_v5_values[4]
                    # per il momento faccio banalmente una somma, da raffinare

                    # se coincide con una massa di v5, la levo dalla lista (sia di v5 che della fusione). Nella lista
                    # delle fusioni la sostituisco con quella nuova
                    pred_fusion.remove(pred_v5_values)
                    pred_v5_list.remove(pred_v5_values)
                    pred_fusion.append([x, y, w, h, conf])

                else:
                    # se le due detection non coincidono, allora la detection di v8 è una detection nuova e la aggiungo
                    pred_fusion.append(detection)
        
        # a questo punto ho completato la lista delle predizioni dell'ensemble, combinando le detection di v5 e di v8
        # ora gli n elementi con conf più alta:
        # Sort the list by the fourth element in descending order
        sorted_list = sorted(pred_fusion, key=lambda x: x[4], reverse=True)

        # get the threshold confidence
        if len(sorted_list) > 0:
            conf_th = 1 - (interval_percentage * sorted_list[0][4])
        else:
            conf_th = 1.0

        # take only the elements above threshold
        pred_fusion = [sublist for sublist in sorted_list if sublist[4] >= conf_th]

        for detection in pred_fusion:
            # ora confronto la GT con le detection fuse con l'ensemble
            distances = []
            # per ogni elemento trovato, inizializzo una lista, 
            # poi guardo quanto dista da tutti i bbox della gt e appendo alla lista
            for bbox in gt_list:
                pred_xy = np.array(detection[0:2])
                gt_xy = np.array(bbox[0:2])

                distance = linalg.norm(pred_xy - gt_xy)
                distances.append(distance)
            
            # trovo a quale massa GT sta più vicino e per quella calcolo la IOU
            closest_item = np.argmin(distances)
            gt_values = gt_list[closest_item]
            iou_value = iou(x1=detection[0], y1=detection[1], w1=detection[2], h1=detection[3],
                            x2=gt_values[0], y2=gt_values[1], w2=gt_values[2], h2=gt_values[3])
            
            # se la IOU è minore di una certa soglia, le considero come coincidenti
            if iou_value >= iou_threshold_gt:
                found_masses[closest_item] = 1

        nr_TP += np.sum(found_masses)
        nr_FP += len(pred_fusion) - np.sum(found_masses)
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
        out_file.write(f'Total number of masses Tot={tot_n_masses}\n\n')
        out_file.write(f'IOU threshold value = {iou_threshold_gt}\n')
        out_file.write(f'Percentage threshold = {interval_percentage}\n')

    if append_to_file is not None:
        if not os.path.exists(append_to_file):
            print('Error: the file must already exist!')
        else:
            with open(append_to_file, 'a') as general_file:
                general_file.write(f'{nr_TP} {nr_FP} {nr_FN}\n')

        # leggo tutti i GT e li metto in una lista,
        # leggo tutte le predizioni e le metto in una lista
        # per ogni predizione dovrei calcolare le distanze da ogni GT


if __name__ == "__main__":
    main()
