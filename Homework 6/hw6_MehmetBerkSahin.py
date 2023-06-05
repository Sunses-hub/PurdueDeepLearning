from dataset import YoloDataset
from model import YoloNet
import argparse 
from torch.utils.data import DataLoader
import random
import torch, gc
import numpy as np 
import os
from pycocotools.coco import COCO
import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs for training")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--yolo_interval", default=20, type=int, help="length of one yolo cell")
    parser.add_argument("--anchors", default=5, type=int, help="number of anchor boxes")
    parser.add_argument("--coco_dir", default="/Users/berksahin/Desktop",
                        help="parent directory of coco dataset")
    parser.add_argument("--batch_size", default=32, type=int, help="size of the minibatch")
    parser.add_argument("--img_size", default=256, type=int, help="size of the images for training")
    parser.add_argument("--threshold", default=0.9, type=float, help="threshold for predicted objectness")
    parser.add_argument("--iou_threshold", default=0.4, type=float, help="iou threshold for nonmax supression")
    parser.add_argument("--lambda1", default=5, type=float, help="lambda for objects (mse)") # lambda_coord 
    parser.add_argument("--lambda2", default=.5, type=float, help="lambda for no objects") # lambda_noobj
    parser.add_argument("--inference", default=False, type=bool, help="open inference mode")
    args = parser.parse_args()
    # take the inputs 
    EPOCH = args.epochs
    LR = args.lr
    YOLO_INT = args.yolo_interval
    ANC = args.anchors
    COCO_DIR = args.coco_dir
    BATCH = args.batch_size
    IMG_SIZE = args.img_size
    THRESHOLD = args.threshold
    IOU_THS = args.iou_threshold 
    LAMBDA1 = args.lambda1
    LAMBDA2 = args.lambda2
    INFERENCE = args.inference

    # for reproducible results
    seed = 3
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False

    # data file info
    class_list = ['bus', 'cat', 'pizza']
    train_json = 'instances_train2014.json'  
    test_json = "instances_val2014.json"
    # create data directories if doesn't exist
    if not os.path.exists("train_data"):
        os.mkdir("train_data")
    if not os.path.exists("test_data"):
        os.mkdir("test_data")

    # train and test COCOs
    coco_train = COCO(train_json)
    coco_test = COCO(test_json)
    # mapping from coco labels to my labels
    coco_inv_labels = {}
    catIds = coco_train.getCatIds(catNms=class_list)
    for idx, catId in enumerate(sorted(catIds)):
        coco_inv_labels[catId] = idx
    # save inverse maps 
    pickle.dump(coco_inv_labels, open('inv_map.pkl', 'wb'))
    print("Inverse map saved!")
    # create custom dataset for training and test/inference
    train_dataset = YoloDataset(coco=coco_train, catIds=catIds, data_path="",
                                coco_inv_labels=coco_inv_labels, train=True)
    print(f"length of the train dataset: {len(train_dataset)}")
    test_dataset = YoloDataset(coco=coco_test, catIds=catIds, data_path="",
                               coco_inv_labels=coco_inv_labels, train=False)
    print(f"length of the test dataset: {len(test_dataset)}")
    # initialize dataloaders 
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=True)

    # initialize model 
    model = YoloNet(threshold=THRESHOLD, iou_ths=IOU_THS, lamb_obj=LAMBDA1, lamb_noobj=LAMBDA2)

    if INFERENCE:
        # do inference and save figures 
        test_loader2 = DataLoader(test_dataset, batch_size=1, shuffle=True)
        model.inference("best_model", test_loader2)
    else:
        # start training
        loss = model.run_train(train_loader, test_loader, epochs=EPOCH, lr=LR)
        # save the loss history 
        pickle.dump(loss, open("loss_history.pkl", "wb"))
        print("training was completed succesfully and losses were saved!")
    

    
