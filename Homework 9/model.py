from ViTHelper import MasterEncoder
import argparse
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch

class ProjectionLayer(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=100, img_size=64):
        super(ProjectionLayer, self).__init__()
        self.in_channels = in_channels # image channels 
        self.patch_size = patch_size # height and width of patch which are equal
        self.emb_size = emb_size # emebedding size 
        # projection layer = patching + linear layer 
        self.projection_layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.emb_size,
                                          kernel_size=self.patch_size, stride=self.patch_size)

        self.class_token = nn.Parameter(torch.randn(1, 1, self.emb_size)) # for classification 
        self.pos_emb = nn.Parameter(torch.randn(1, (img_size//patch_size)**2 + 1, self.emb_size)) # positional embedding

    def forward(self, x):
        out = self.projection_layer(x) # fully-connected layer with patching 
        out = out.reshape(out.shape[0], out.shape[1], -1) # sequence format
        out = torch.transpose(out, dim0=1, dim1=2) 
        # expand class tokens
        cls_tokens = self.class_token.expand(out.shape[0], 1, self.emb_size)
        out = torch.cat((cls_tokens, out), dim=1) # add class token to the beginning
        # expand positional embeddings
        pos_emb = self.pos_emb.expand(out.shape[0], out.shape[1], out.shape[2])
        # add positional embeddings to cls_token + image embeddings
        out += pos_emb # add the positional embeddings to patch embeddings 
        return out

class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        # initial layer 
        self.project_layer = ProjectionLayer(emb_size=args.emb_size, img_size=args.img_size)
        # encoder part of the transformer
        self.encoder = MasterEncoder(max_seq_length=args.max_seq_length, embedding_size=args.emb_size,
                                     how_many_basic_encoders=args.num_encoders, num_atten_heads=args.num_heads, 
                                     myAttention=args.my_attention) # myAttention is added by me 
        # classifier 
        self.fc = nn.Linear(in_features=args.emb_size, out_features=args.class_num)

    def forward(self, x):
        out = self.project_layer(x)
        out = self.encoder(out)
        out = self.fc(out[:,0,:]) # take the class token for class prediction 
        return out

def train(model, train_loader, val_loader, args):
    device = args.device
    print(f"training is running on {device}...")
    # evaluation metrics for each 100 steps
    train_cross_hist, train_class_hist = [], []
    # set device, loss and optimization method16
    net = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    for epoch in range(args.epochs):
        # running evaluations for each 100 steps
        cross_running_loss = 0.0
        class_running_acc = 0.0
        for i, data in enumerate(train_loader):
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
            # classification accuracy
            preds = torch.argmax(outputs, dim=1) # choose the highest prob. class
            results = torch.eq(labels, preds) # classification accuracy (ca)
            # save the accuracy
            acc = torch.sum(results).item() / results.size(dim=0) # mean ca
            class_running_acc += acc

            if (i+1) % 100 == 0:
                print(f"TRAIN: [epoch {epoch + 1}, batch: {i + 1}] cross-entropy" +
                      f" loss: {round(cross_running_loss / 100, 3)}")
                print(f"TRAIN: [epoch {epoch + 1}, batch: {i + 1}] classification" +
                      f" accuracy: {round(class_running_acc / 100, 3)}")
                print("*"*40)
                # save the results for each 100 steps
                train_cross_hist.append(cross_running_loss / 100)
                train_class_hist.append(class_running_acc / 100)
                cross_running_loss = 0.0
                class_running_acc = 0.0
        
        #inference(net, val_loader, args)
    # save the last model
    print("training is completed!")
    torch.save(net.state_dict(), args.model_name)
    print("model is saved!")
    # return the loss and accuracy 
    return {"cross_entropy" : train_cross_hist,
            "class_acc" : train_class_hist}

def inference(model, val_loader, args):
    model.load_state_dict(torch.load(args.model_name)) # load the saved model 
    device = args.device
    print(f"inference is running on {device}...")
    y_preds, y_trues = [], []
    # set device, loss and optimization method16
    net = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    net.eval() # inference mode 
    class_acc, cross_loss = 0, 0 
    for i, data in enumerate(val_loader):
        # put model and data to gpu if available
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs) # forward pass
        # calculate cross-entropy loss
        loss = criterion(outputs, labels)
        cross_loss += loss.item()
        # classification accuracy
        preds = torch.argmax(outputs, dim=1) # choose the highest prob. class
        results = torch.eq(labels, preds) # classification accuracy (ca)
        y_trues.extend(labels.cpu())
        y_preds.extend(preds.cpu())
        # save the accuracy
        acc = torch.sum(results).item() / results.size(dim=0) # mean ca
        class_acc += acc
    
    class_acc /= len(val_loader)
    cross_loss /= len(val_loader)
    print("inference is completed!")
    print("*"*45)
    print(f"validation cross-entropy loss: {cross_loss:.4f}")
    print(f"validation classification accuracy: {class_acc:.4f}")
    # plot confusion matrix
    classes = ["airplane", "bus", "cat", "dog", "pizza"]
    cm = confusion_matrix(y_trues, y_preds)
    df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
    acc = (np.trace(cm) / np.sum(cm)) * 100
    sns.heatmap(df_cm, annot=True, fmt='g', cmap="Blues")
    plt.title(f"Confusion matrix for {args.model_name}, acc: {acc:.2f}%")
    plt.xlabel("predicted classes")
    plt.ylabel("ground truth classes")
    plt.savefig(f'{args.model_name}_confusion_matrix.png')



# test code
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # attention parameters
    parser.add_argument("--num_heads", default=10, type=int, help="number of attention heads")
    parser.add_argument("--num_encoders", default=5, type=int, help="number of encoders in transformer")
    # model parameters
    parser.add_argument("--emb_size", default=100, type=int, help="number of embedding dimension")
    parser.add_argument("--img_size", default=64, type=int, help="heigh=width of an image")
    parser.add_argument("--max_seq_length", default=16+1, type=int, help="length of the patch sequences")
    parser.add_argument("--patch_size", default=16, type=int, help="patch size")
    parser.add_argument("--class_num", default=5, type=int, help="number of classes to classify")
    

    args = parser.parse_args()
    # test projection layer
    model = ProjectionLayer()
    # (B, C, H, W) = (4, 3, 64, 64)
    x = torch.randn(4, 3, 64, 64)
    y = model(x)
    print("ProjectionLayer works successfully!")
    print("output shape:", y.shape)

    # test transformer encoder
    model = VisionTransformer(args)
    x = torch.randn(4, 3, 64, 64)
    y = model(x)
    print("VisionTransformer works successfully!")
    print("output shape:", y.shape)


