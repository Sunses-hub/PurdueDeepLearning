import numpy as np
from torchvision.ops import box_iou
import torch.nn as nn 
from torchvision.ops import complete_box_iou_loss
from network import HW5Net
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch 
import cv2


class mIoU(nn.Module):
    def __init__(self):
        super(mIoU, self).__init__()
    
    def forward(self, output, target):
        # for single sample (inference)
        if output.shape[0] == 1:
            return box_iou(output, target.unsqueeze(0))
        # for batches (training)
        else:
            mean_loss = 0
            batch_size = output.shape[0]
            # calculate and return the mean loss (1 - mIoU)
            for i in range(batch_size):
                mean_loss += (1 - box_iou(output[i].unsqueeze(0), target[i].unsqueeze(0)).item()) / batch_size
            return mean_loss

class CompleteIOULoss(nn.Module):

    def __init__(self, reduction='none'):
        super(CompleteIOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        loss = complete_box_iou_loss(output, target, self.reduction)
        return loss

def update_cm(preds, target, cm):
    # detach the preds tensor from gpu and convert to cpu
    preds = preds.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    # update cm
    for i in range(len(preds)):
        cm[target[i], preds[i]] += 1 
    return cm 

def plot_cm(loss_type, cm, IoU, train_eval=False):
    # classes
    classes = ['buss', 'cat', 'pizza']
    acc = round(np.trace(cm) / np.sum(cm), 3)
    plt.figure()
    s = sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes,
                    yticklabels=classes, fmt='g')
    s.set(xlabel='Predicted label', ylabel='True label')
    word = "Train" if train_eval else "Test"
    plt.title(f"{loss_type} Loss, {word} Accuracy={round(acc*100, 2)}%, mIoU={round(1-IoU, 2)}")
    #plt.show()
    plt.savefig(f"{word}cm_{loss_type}.jpeg")
    print("confusion matrix figure is saved...")

def display_preds(test_loader, model_name="cIoU_model", num_examples=5, train_eval=False):

    class_list = ['bus', 'cat', 'pizza']

    loss_type = model_name.split("_")[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss() if "mse" in model_name else CompleteIOULoss('mean')

    model = HW5Net()
    model.load_state_dict(torch.load(model_name))
    model.to(device)

    cm, results = evaluation(model, test_loader, criterion1, criterion2, device)
    plot_cm(loss_type, cm, results["miou_loss"], train_eval)

    ce, reg, acc, miou = results["cross-entropy"], results["reg_loss"], results["accuracy"], 1-results["miou_loss"]
    # report the results 
    print(f"MODEL NAME: {model_name}")
    print("*"*40)
    print(f"cross-entropy loss: {ce:.2f}")
    print(f"{loss_type} loss: {reg:.2f}")
    print(f"classification accuracy: {acc*100:.2f}%")
    print(f"mean IoU: {miou:.2f}")
    print("*"*40)

    # initialize counter and image dictionaries
    counter = {ids: 0 for ids in [0, 1, 2]}
    imgs = {ids: [] for ids in [0, 1, 2]}
    # stop criteria 
    criteria = np.array([counter[i] >= num_examples for i in range(3)])
    stop = False
    # choose images 
    for data in test_loader:
        label = data["label"]
        for idx, cat in enumerate(label):
            ids = cat.item()
            if counter[ids] < num_examples:
                counter[ids] += 1
                # create sample dict
                sample = {"image" : data["image"][idx],
                          "bbox" : data["bbox"][idx],
                          "label": data["label"][idx]}
                imgs[ids].append(sample)

            if np.all(criteria):
                stop = True 
                break 
        if stop:
            break 

    model.to("cpu")
    model.eval()
    # display the predictions in a figure 
    fig, axs = plt.subplots(3, num_examples)
    fig.tight_layout()

    for row, cat in enumerate(list(imgs.keys())):
        for col, data in enumerate(imgs[cat]):
            img, bbox, label = data["image"], data["bbox"], data["label"]
            pred_label, pred_bbox = model(img.unsqueeze(0))
            # iou loss 
            iou = mIoU()(pred_bbox, bbox)
            # unwrap the tensors 
            pred_label = torch.argmax(pred_label).item()
            label = label.item()
            bbox, pred_bbox = bbox.numpy(force=True) * 255, pred_bbox.numpy(force=True) * 255
            img = np.uint8(img.numpy(force=True) * 255)
            img = img.transpose((1, 2, 0))
            # get the box locations 
            [x1, y1, x2, y2] = bbox 
            [x1p, y1p, x2p, y2p] = pred_bbox[0]
            # ground truth 
            img = np.ascontiguousarray(img)
            img = cv2.rectangle(img, (round(x1), round(y1)),
                                     (round(x2), round(y2)),
                                     (36, 256, 12), 2)

            img = cv2.putText(img, class_list[cat], (round(x1), round(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.8, (36, 256, 12), 2)
            # prediction
            img = cv2.rectangle(img, (round(x1p), round(y1p)),
                                     (round(x2p), round(y2p)),
                                     (256, 36, 12), 2)

            img = cv2.putText(img, class_list[cat], (round(x1p), round(y1p - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.8, (256, 36, 12), 2)
            # plot the image
            axs[row, col].imshow(img)
            axs[row, col].axis('off')
            axs[row, col].set_title(f"{class_list[cat]} {col + 1}\nClassification: {label == pred_label}\nIoU: {round(iou[0].item(), 2)}", size=7)
    
    word = "Train" if train_eval else "Test"
    plt.savefig(f"{model_name}_{word}preds.jpeg")
    print("Predictions are plotted and figure is saved...")


def evaluation(model, test_loader, criterion1, criterion2, device):

    cm = np.zeros((3,3))
    # TEST PART 
    # initalize the losses with 0 
    run_cross_loss = 0.0
    run_reg_loss = 0.0
    run_iou_loss = 0.0
    run_acc = 0.0
    # size of the dataset
    test_size = len(test_loader)
    model.eval() # evaluation mode 
    for i, data in enumerate(test_loader):
        img, bbox, label = data['image'], data['bbox'], data['label']
        # load data to device
        img = img.to(device)
        bbox = bbox.to(device)
        label = label.to(device)

        # test
        output = model(img)
        pred_cat = output[0]
        pred_bbox = output[1]

        # calculate loss
        cross_loss = criterion1(pred_cat, label)
        reg_loss = criterion2(pred_bbox, bbox)

        # accuracy
        _, pred = torch.max(pred_cat, 1)
        acc = torch.eq(pred, label).float().mean().item()

        # calculate mIoU loss 
        miou_loss = mIoU()(pred_bbox, bbox)
        
        # update confusion matrix
        cm = update_cm(pred, label, cm)
        # update the losses with batch mean loss 
        run_cross_loss += cross_loss.item()
        run_reg_loss += reg_loss.item()
        run_iou_loss += miou_loss
        run_acc += acc
    
    # calculate mean evaluations
    run_cross_loss /= test_size
    run_reg_loss /= test_size
    run_iou_loss /= test_size
    run_acc /= test_size
    # return the mean losses 
    return cm, {"cross-entropy": run_cross_loss,
                "reg_loss": run_reg_loss,
                "accuracy": run_acc,
                "miou_loss": run_iou_loss}

