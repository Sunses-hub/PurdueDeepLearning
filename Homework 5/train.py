
import argparse
from Dataset import MyCOCODataset
from network import HW5Net
from utils import update_cm, mIoU, evaluation
import pickle
from torchvision.ops import complete_box_iou_loss
import torch.nn as nn
import torch
from utils import CompleteIOULoss
from torch.utils.data import DataLoader
import random
import numpy as np

def train(net, num_epochs, batch_size, cIoU=True, model_name='best_model'):

    model = net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"training is runing on {device}")
    # load train, test and inv. map dictionaries
    with open('train_data.pkl', 'rb') as f:
        train_dict = pickle.load(f)
    with open('test_data.pkl', 'rb') as f:
        test_dict = pickle.load(f)
    with open('inv_map.pkl', 'rb') as f:
        inv_map = pickle.load(f)
    print("train, test, and inv. map are loaded.")

    # train and test dataset
    train_data = MyCOCODataset(train_dict, inv_map, train=True)
    test_data = MyCOCODataset(test_dict, inv_map, train=False)
    # train and test dataloaders 
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    train_size = len(train_loader)
    # train parameters
    criterion1 = nn.CrossEntropyLoss()
    if cIoU:
        criterion2 = CompleteIOULoss('mean')
        loss_name = "CIoU"
    else:
        criterion2 = nn.MSELoss()
        loss_name = "MSE"
    # choose Adam optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    # train history
    train_history = {
        "cross_loss": [],
        "bbox_loss" : [],
        "acc" : []
    }
    test_history = {
        "cross_loss": [],
        "bbox_loss": [],
        "acc": [] 
    }
        
    max_model_score = 0
    print(f"training for {model_name} loss is in progress...")
    for epoch in range(1, num_epochs+1):
        # TRAIN PART 
        run_cross_loss = 0.0
        run_reg_loss = 0.0
        run_iou_loss = 0.0
        run_acc = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            img, bbox, label = data['image'], data['bbox'], data['label']
            # load data to device
            img = img.to(device)
            bbox = bbox.to(device)
            label = label.to(device)

            # start train
            optimizer.zero_grad()
            output = model(img)
            pred_cat = output[0]
            pred_bbox = output[1]

            # calculate loss
            cross_loss = criterion1(pred_cat, label)
            cross_loss.backward(retain_graph=True)
            reg_loss = criterion2(pred_bbox, bbox)
            reg_loss.backward()
            optimizer.step()

            # accuracy
            _, pred = torch.max(pred_cat, 1)
            acc = torch.eq(pred, label).float().mean().item()

            # mIoU loss
            miou_loss = mIoU()(pred_bbox, bbox)

            run_cross_loss += cross_loss.item()
            run_reg_loss += reg_loss.item()
            run_iou_loss += miou_loss
            run_acc += acc
            #print(f"[Iteration {i+1}/{train_size}]")

        # calculate mean evaluations
        run_cross_loss /= train_size
        run_reg_loss /= train_size
        run_iou_loss /= train_size
        run_acc /= train_size
        # report results
        print("*"*12, "TRAIN", "*"*12)
        print(f"[epoch {epoch}/{num_epochs}] train cross entropy loss: {round(run_cross_loss,4)}")
        print(f"[epoch {epoch}/{num_epochs}] train {loss_name}: {round(run_reg_loss, 4)}")
        print(f"[epoch {epoch}/{num_epochs}] train mIoU loss: {round(run_iou_loss, 4)}")
        print(f"[epoch {epoch}/{num_epochs}] train classification accuracy {round(run_acc*100, 2)}")
        print("*"*30)
        # save the losses
        train_history["cross_loss"].append(run_cross_loss)
        train_history["bbox_loss"].append(run_reg_loss)
        train_history["acc"].append(run_acc)

        # TEST PART 
        _, results = evaluation(model, test_loader, criterion1, criterion2, device)
        # get the results 
        run_cross_loss = results["cross-entropy"]
        run_reg_loss = results["reg_loss"]
        run_iou_loss = results["miou_loss"]
        run_acc = results["accuracy"]
        # report results
        print("*"*12, "TEST", "*"*12)
        print(f"[epoch {epoch}/{num_epochs}] test cross entropy loss: {round(run_cross_loss, 4)}")
        print(f"[epoch {epoch}/{num_epochs}] test {loss_name}: {round(run_reg_loss, 4)}")
        print(f"[epoch {epoch}/{num_epochs}] test mIoU loss: {round(run_iou_loss, 4)}")
        print(f"[epoch {epoch}/{num_epochs}] test classification accuracy {round(run_acc*100, 2)}")
        print("*"*30)
        # save the losses
        test_history["cross_loss"].append(run_cross_loss)
        test_history["bbox_loss"].append(run_reg_loss)
        test_history["acc"].append(run_acc)
        
        model_score = round(run_acc, 3) + (1 - run_reg_loss)
        if model_score > max_model_score:
            torch.save(model.state_dict(), model_name)
            print("best model is saved...")
            max_model_score = model_score


    return {"model" : model,
            "train_history" : train_history,
            "test_history" : test_history}

if __name__ == '__main__':

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=15, type=int, help="numbere of epochs for training")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")

    args = parser.parse_args()

    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size

    #model = BeastNet()
    model = HW5Net()
    results = train(model, EPOCH, BATCH_SIZE)

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
        print("results are saved.")

    




