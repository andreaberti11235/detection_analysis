import os
import argparse
import numpy as np
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

    for idx in gt_df.index:
        x_center = gt_df.iloc[idx][1]
        y_center = gt_df.iloc[idx][2]
        width = gt_df.iloc[idx][3]
        height = gt_df.iloc[idx][4]

        


    # leggo tutti i GT e li metto in una lista,
    # leggo tutte le predizioni e le metto in una lista
    # per ogni predizione dovrei calcolare le distanze da ogni GT


if __name__ == "__main__":
    main()
