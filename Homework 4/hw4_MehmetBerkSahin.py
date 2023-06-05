# author: Mehmet Berk Sahin
# import necessary packages
import argparse
import random
import torch
from pycocotools.coco import COCO
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvt
import os
import pickle
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

class MyCOCODataset(Dataset):
    def __init__(self, folder_name, aug=False):
        super().__init__()
        self.folder_name = folder_name # dataset path
        self.files = os.listdir(folder_name) # file names in list
        random.shuffle(self.files) # shuffle the list
        # Later one can add augmentations to this pipeline easily
        self.transforms = tvt.Compose([
            tvt.ToTensor()
        ])
        # class names
        self.classes = ["airplane", "bus", "cat", "dog", "pizza"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file_name = self.files[item]
        img = Image.open(os.path.join(self.folder_name, file_name)) # load image
        img = self.transforms(img) # convert it to tensor with normalization
        class_name = file_name.split("_")[0] # get the index of label
        label = self.classes.index(class_name) # get label
        return img, label

class HW4Net(nn.Module):
    def __init__(self):
        super(HW4Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) # 58x58
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(5408, 64) # XXXX = 5408
        self.fc2 = nn.Linear(64, 5) # XX = 5

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net2(HW4Net):
    def __init__(self):
        super(Net2, self).__init__()
        # padding is added
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(7200, 64)
        self.fc2 = nn.Linear(64, 5)

class Net3(HW4Net):
    def __init__(self):
        super(Net3, self).__init__()
        # consecutive convolutions
        self.conv_chain = nn.ModuleList([nn.Conv2d(32, 32, 3, padding=1) for conv in range(10)])
        # fully connected classifier layers
        self.fc1 = nn.Linear(5408, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # pass x through chain of convolutions
        for layer in self.conv_chain:
            x = F.relu(layer(x)) # activation function
        x = x.view(x.shape[0], -1) # flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def update_confusion_matrix(matrix, pred, label):
    for i in range(pred.size(dim=0)):
        matrix[label[i].item(), pred[i].item()] += 1
    return matrix

def set_datasets(data_dir="/Users/berksahin/Desktop"):
    # create folders
    if not os.path.exists("train_data"):
        os.mkdir("train_data")
    if not os.path.exists("val_data"):
        os.mkdir("val_data")

    annFile = os.path.join(data_dir, "coco/annotations/instances_train2014.json")
    coco = COCO(annFile)
    # classes for the problem in hw4
    classes = ['airplane', 'bus', 'cat', 'dog', 'pizza']
    catIds = coco.getCatIds(catNms=classes)  # get class indices
    # keys: class names, values: list of files names
    train_data = dict(zip(classes, [[] for i in range(len(classes))]))
    val_data = dict(zip(classes, [[] for i in range(len(classes))]))
    print("dataset generation has started...")
    for i, idx in enumerate(catIds):
        imgIds = coco.getImgIds(catIds=idx)
        imgIds = np.random.choice(imgIds, size=2000, replace=False)

        for counter, img_idx in enumerate(imgIds):
            # get the file name of the image
            file_name = coco.loadImgs(int(img_idx))[0]['file_name']
            img_path = os.path.join(data_dir, f"coco/images/{file_name}")
            # open the image as PIL image in RGB format
            img = Image.open(img_path).convert("RGB")
            img = img.resize((64, 64)) # resize

            class_name = classes[i]
            save_name = class_name + "_" + file_name
            if counter < 1500:
                save_dir = "train_data"
                train_data[class_name].append(save_name)
            else:
                save_dir = "val_data"
                val_data[class_name].append(save_name)
            # save the image in new dataset folder
            img.save(os.path.join(save_dir, save_name))

    print("train and validation datasets are ready!")

    return {"train_data" : train_data, "val_data" : val_data}

def train(model, train_data_loader, val_data_loader, epochs):
    # set device to gpu if available else cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training is running on {device}...")
    # evaluation metrics for each 100 steps
    train_cross_hist, train_class_hist = [], []
    val_cross_hist, val_class_hist = [], []
    # evaluation metrics for each epoch
    etrain_cross_hist, etrain_class_hist = [], []
    eval_cross_hist, eval_class_hist = [], []
    # set device, loss and optimization method
    net = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99))
    max_acc_train, max_acc_val = 0, 0
    best_tconfusion, best_vconfusion = np.zeros((5, 5)), np.zeros((5, 5))
    for epoch in range(epochs):
        # initialize confusion matrices
        train_confusion, val_confusion = np.zeros((5, 5)), np.zeros((5, 5))
        # running evaluations for each 100 steps
        cross_running_loss = 0.0
        class_running_acc = 0.0
        # running evaluations for each epoch
        ecross_running_loss = 0.0
        eclass_running_acc = 0.0
        net.train() # open train mode
        for i, data in enumerate(train_data_loader):
            # put model and data to gpu if available
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # set gradients 0
            outputs = net(inputs) # forward pass
            # calculate cross-entropy loss
            loss = criterion(outputs, labels)
            loss.backward() # backpropagation
            optimizer.step() # update weights
            cross_running_loss += loss.item()
            ecross_running_loss += loss.item()
            # classification accuracy
            outputs = F.softmax(outputs, dim=1) # softmax layer
            preds = torch.argmax(outputs, dim=1) # choose the highest prob. class
            results = torch.eq(labels, preds) # classification accuracy (ca)
            # save the accuracy
            acc = torch.sum(results).item() / results.size(dim=0) # mean ca
            class_running_acc += acc
            eclass_running_acc += acc
            # update confusion matrix for train
            update_confusion_matrix(train_confusion, preds, labels)

            if (i+1) % 100 == 0:
                print(f"TRAIN: [epoch {epoch + 1}, batch: {i + 1}] cross-entropy" +
                      f" loss: {round(cross_running_loss / 100, 3)}")
                print(f"TRAIN: [epoch {epoch + 1}, batch: {i + 1}] classification" +
                      f" accuracy: {round(class_running_acc / 100, 3)}")
                # save the results for each 100 steps
                train_cross_hist.append(cross_running_loss / 100)
                train_class_hist.append(class_running_acc / 100)
                cross_running_loss = 0.0
                class_running_acc = 0.0

        mean_acc = eclass_running_acc / len(train_data_loader)
        # save the results for each epoch
        etrain_cross_hist.append(ecross_running_loss / len(train_data_loader))
        etrain_class_hist.append(mean_acc)
        # save the best performance
        if mean_acc > max_acc_train:
            best_tconfusion = train_confusion
            max_acc_train = mean_acc

        # reset epoch evaluations
        ecross_running_loss = 0.0
        eclass_running_acc = 0.0
        # reset evaluations for each 100 step
        cross_running_loss = 0.0
        class_running_acc = 0.0
        net.eval() # open eval mode
        for i, data in enumerate(val_data_loader):
            # put model and data to gpu if available
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs) # forward pass
            # calculate cross-entropy loss
            loss = criterion(outputs, labels)
            cross_running_loss += loss.item()
            ecross_running_loss += loss.item()
            # classification accuracy
            outputs = F.softmax(outputs, dim=1) # softmax layer
            preds = torch.argmax(outputs, dim=1) # choose class with highest prob.
            results = torch.eq(labels, preds) # calculate ca
            # save the accuracy
            acc = torch.sum(results).item() / results.size(dim=0) # mean ca
            class_running_acc += acc
            eclass_running_acc += acc
            # update confusion matrix for validation
            update_confusion_matrix(val_confusion, preds, labels)

            if (i+1) % 100 == 0:
                print(f"VAL: [epoch {epoch + 1}, batch: {i + 1}] cross-entropy" +
                      f" loss: {round(cross_running_loss / 100, 3)}")
                print(f"VAL: [epoch {epoch + 1}, batch: {i + 1}] classification" +
                      f" accuracy: {round(class_running_acc / 100, 3)}")
                # save the results for each 100 steps
                val_cross_hist.append(cross_running_loss / 100)
                val_class_hist.append(class_running_acc / 100)
                cross_running_loss = 0.0
                class_running_acc = 0.0

        mean_acc = eclass_running_acc / len(val_data_loader)
        # save the results for each epoch
        eval_cross_hist.append(ecross_running_loss / len(val_data_loader))
        eval_class_hist.append(mean_acc)
        # save the best performance
        if mean_acc > max_acc_val:
            best_vconfusion = val_confusion
            max_acc_val = mean_acc

    # all evaluation metrics put into dictionaries for easy usage
    results_100step = {"train_cross_entropy" : train_cross_hist,
                       "train_class_acc" : train_class_hist,
                       "val_cross_entropy" : val_cross_hist,
                       "val_class_acc" : val_class_hist}

    results_epoch = {"train_cross_entropy" : etrain_cross_hist,
                     "train_class_acc" : etrain_class_hist,
                     "val_cross_entropy" : eval_cross_hist,
                     "val_class_acc" : eval_class_hist}

    confusion = {"train_confusion" : best_tconfusion,
                 "val_confusion" : best_vconfusion}

    print("training is completed!")

    return {"100steps" : results_100step,
            "epoch" : results_epoch,
            "confusion matrix" : confusion}

def plot_confusion_matrix(network_name, results, aug=False):
    title_aug = ""
    if aug:
        title_aug = "(Aug.)"
    # classes
    classes = ['airplane', 'bus', 'cat', 'dog', 'pizza']
    # plot the train confusion
    conf_matrix = results['confusion matrix']['train_confusion'].astype('i')
    acc = round(np.trace(conf_matrix) / np.sum(conf_matrix), 3)
    plt.figure()
    s = sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=classes,
                    yticklabels=classes, fmt='g')

    s.set(xlabel='Predicted label', ylabel='True label')
    plt.title(f"{network_name}{title_aug} Train Accuracy=%{acc*100}")
    plt.show()
    # plot the validation confusion
    conf_matrix2 = results['confusion matrix']['val_confusion'].astype('i')
    acc2 = round(np.trace(conf_matrix2) / np.sum(conf_matrix2), 3)
    plt.figure()
    s2 = sns.heatmap(conf_matrix2, annot=True, cmap='Blues', xticklabels=classes,
                    yticklabels=classes, fmt='g')

    s2.set(xlabel='Predicted label', ylabel='True label')
    plt.title(f"{network_name}{title_aug} Validation Accuracy=%{acc2*100}")
    plt.show()

def plot_images(train_data):
    # class names
    classes = ["airplane", "bus", "cat", "dog", "pizza"]
    # plot random 5 images for each class
    fig, axs = plt.subplots(5, 5)
    fig.tight_layout()
    for i, cat in enumerate(classes):
        files = train_data[cat]
        chosen = random.sample(files, 5) # sample 5 random images
        for j, file_name in enumerate(chosen):
            img = Image.open(os.path.join('train_data', file_name)) # load image
            axs[i, j].imshow(img) # plot image
            axs[i, j].axis('off')
            axs[i, j].set_title(f"{cat} {j + 1}", size=7)
    plt.show()

def plot_evaluations(results):
    train_ce = [result['epoch']['train_cross_entropy'] for result in results]
    # plot train cross-entropy loss for epochs
    plt.title("Cross-entropy v.s. Epochs")
    plt.plot(train_ce[0])
    plt.plot(train_ce[1])
    plt.plot(train_ce[2])
    plt.xlabel("Epochs")
    plt.ylabel("Train Cross-entropy")
    plt.legend(["Net1", "Net2", "Net3"])
    plt.grid("ON", linewidth=0.5)
    plt.show()

    # plot classification accucarcy for epochs
    train_ca = [result['epoch']['train_class_acc'] for result in results]

    plt.title("Classification Accuracy v.s. Epochs")
    plt.plot(train_ca[0])
    plt.plot(train_ca[1])
    plt.plot(train_ca[2])
    plt.xlabel("Epochs")
    plt.ylabel("Train Classification Accuracy")
    plt.legend(["Net1", "Net2", "Net3"])
    plt.grid("ON", linewidth=0.5)
    plt.show()

    # plot cross-entropy ploss for each 100 iteration
    train_ce = [result['100steps']['train_cross_entropy'] for result in results]

    plt.title("Cross-entropy v.s. Steps")
    plt.plot(train_ce[0])
    plt.plot(train_ce[1])
    plt.plot(train_ce[2])
    plt.xlabel("Steps")
    plt.ylabel("Train Cross-entropy")
    plt.legend(["Net1", "Net2", "Net3"])
    plt.grid("ON", linewidth=0.5)
    plt.show()

    # plot classification accuracy for each 100 iteration
    train_ca = [result['100steps']['train_class_acc'] for result in results]

    plt.title("Classification Accuracy v.s. Steps")
    plt.plot(train_ca[0])
    plt.plot(train_ca[1])
    plt.plot(train_ca[2])
    plt.xlabel("Steps")
    plt.ylabel("Train Classification Accuracy")
    plt.legend(["Net1", "Net2", "Net3"])
    plt.grid("ON", linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    # take the data directory as input
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/Users/berksahin/Desktop", help="Current directory path of the coco folder")
    args = parser.parse_args()

    np.random.seed(5)
    random.seed(5)

    # prepare the dataset
    data = set_datasets(args.data_dir)
    plot_images(data["train_data"])
    # construct datasets
    train_dataset = MyCOCODataset(folder_name='train_data')
    val_dataset = MyCOCODataset(folder_name='val_data')
    # construct dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=4,
                              num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=4,
                            num_workers=2)

    # CNN Task 1
    model = HW4Net()
    result1 = train(model, train_loader, val_loader, epochs=15)

    # save the dictionary to the current directory
    with open('results1.pkl', 'wb') as f:
        pickle.dump(result1, f)
        print("results were saved succesfully to file results1.pkl")

    plot_confusion_matrix("Net 1", result1)
    
    # CNN Task 2
    model = Net2()
    result2 = train(model, train_loader, val_loader, epochs=15)

    # save the dictionary to the current directory
    with open('results2.pkl', 'wb') as f:
        pickle.dump(result2, f)
        print("results were saved succesfully to file results2.pkl")

    plot_confusion_matrix("Net 2", result2)
    
    # CNN Task 3
    model = Net3()
    result3 = train(model, train_loader, val_loader, epochs=15)

    # save the dictionary to the current directory
    with open('results3.pkl', 'wb') as f:
        pickle.dump(result3, f)
        print("results were saved succesfully to file results3.pkl")

    plot_confusion_matrix("Net 3", result3)

    # plot learning curves
    results = [result1, result2, result3]
    print("plots are generated!")
    plot_evaluations(results)








