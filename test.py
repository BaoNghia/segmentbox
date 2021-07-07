import os
import torch
import argparse
import numpy as np
import pandas as pd

from utils.general import yaml_loader
from data_loader import dataloader, transforms
from utils import metrics



def test_result(model, test_loader, device, cfg):
    # testing the model by turning model "eval" mode
    model.eval()
    all_labels = []
    all_preds = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)

        all_labels.append(targets.cpu())
        all_preds.append(outputs.cpu())
        
    all_labels = np.stack(all_labels, axis=0)
    all_preds = np.stack(all_preds, axis=0)

    return {"iou": metrics.iou(all_preds, all_labels, threshold= 0.5),\
            "dice": metrics.dice(all_preds, all_labels, threshold= 0.5)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-cfg', '--configure', default='cfgs/tenes.cfg', help='JSON file')
    parser.add_argument('-cp', '--checkpoint', default=None, help = 'checkpoint path')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    print("Testing process beginning here....")
    # read configure file
    cfg = yaml_loader(args.configure)

    # load model
    print("Loading model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path)
    test_model = checkpoint['model']
    test_model.load_state_dict(checkpoint['state_dict'])
    test_model = test_model.to(device)

    print("Inference on the testing set")
    test_data = cfg["data"]["test_csv_name"]
    data_path = cfg["data"]["data_path"]
    test_df = pd.read_csv(test_data)

    # prepare the dataset
    train_loader, valid_loader, test_loader = dataloader.get_data_loader(cfg)
    print(test_result(test_model, test_loader, device,cfg))
    # print(tta_labels(model,test_loader,device,cfg))