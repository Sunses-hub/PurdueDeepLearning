import argparse
from Dataset import MyCOCODataset
from network import HW5Net
from train import train, evaluation
import pickle
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import random
import numpy
from utils import display_preds


if __name__ == "__main__":
    # for reproducible results 
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    # take the user defined variables 
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=15, type=int, help="numbere of epochs for training")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")

    args = parser.parse_args()

    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size

    # TRAININGN PART 
    # train HW5Net with CIoU
    model_cIoU = HW5Net()
    exp1 = train(model_cIoU, EPOCH, BATCH_SIZE, True, 'cIoU_modelR')
    
    # train HW5Net with MSE 
    model_mse = HW5Net()
    exp2 = train(model_mse, EPOCH, BATCH_SIZE, False, 'mse_model')
    
    # save the results 
    with open('cIoU_model_results.pkl', 'wb') as f:
        pickle.dump(exp1, f)
    with open('mse_model_results.pkl', 'wb') as f:
        pickle.dump(exp2, f)
    print("experiment results were saved.")

    # load train data, test data and label mapping 
    with open('train_data.pkl', 'rb') as f:
        train_dict = pickle.load(f)
    with open('test_data.pkl', 'rb') as f:
        test_dict = pickle.load(f)
    with open('inv_map.pkl', 'rb') as f:
        inv_map = pickle.load(f)

    # EVALUATION PART 
    # evaluate on test set 
    test_data = MyCOCODataset(test_dict, inv_map, train=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # display predictions of model trained with cIoU and mse model (over test data)
    display_preds(test_loader, "cIoU_model")
    display_preds(test_loader, "mse_model")
    
    # evaluate on train set 
    train_data = MyCOCODataset(train_dict, inv_map, train=True)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # display predictions of model trained with cIoU and mse model (over train data)
    display_preds(train_loader, "cIoU_model", train_eval=True)
    display_preds(train_loader, "mse_model", train_eval=True)


