import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import argparse
from dataset import SentimentDataset
import random
# for confusion matrix 
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import time 


class GRU_cell(nn.Module):

    def __init__(self, args):
        super(GRU_cell, self).__init__()
        # vector sizes 
        self.hidden_size = args.hidden_size
        self.input_size = args.input_size 
        self.batch_size = args.batch_size
        self.pm = args.pm # switch for poor man's GRU
        # poor man's GRU (only one forget gate)
        if args.pm:
            self.WU_f = nn.Sequential(nn.Linear(self.input_size + self.hidden_size, self.hidden_size), nn.Sigmoid())
        # real GRU (with update and reset gate)
        else:
            self.WU_z = nn.Sequential(nn.Linear(self.input_size + self.hidden_size, self.hidden_size), nn.Sigmoid())
            self.WU_r = nn.Sequential(nn.Linear(self.input_size + self.hidden_size, self.hidden_size), nn.Sigmoid())
        # layer for determining candidate hidden state
        self.WU_h = nn.Sequential(nn.Linear(self.input_size + self.hidden_size, self.hidden_size), nn.Tanh())

    def forward(self, x, h):
        inp_hid = torch.cat((x, h), dim=1)
        if self.pm:
            forget = self.WU_f(inp_hid) # forget gate 
            inp_fhid = torch.cat((x, forget * h), dim=1) # input for cand. hid
            h_cand = self.WU_h(inp_fhid) # candidate hidden state
            new_h = (1 - forget) * h + forget * h_cand # next hidden state
        else: 
            update = self.WU_z(inp_hid) # update gate
            reset = self.WU_r(inp_hid) # reset gate 
            inp_rhid = torch.cat((x, reset * h), dim=1) # input for cand. hid 
            h_cand = self.WU_h(inp_rhid) # candidate hidden state 
            new_h = (1 - update) * h + update * h_cand # next hidden state
        return new_h, new_h

class GRUNet(nn.Module):

    def __init__(self, args):
        super(GRUNet, self).__init__()
        """
        Some important tensor shapes (assuming batch_first=True):
        GRU input tensor: ( batch_size, sequence length, input_size ) 
        GRU output tensor: ( batch_size, sequence length, D*hidden_size )
        hidden tensor: ( D * num_layers, batch_size, hidden_size ) 
        """
        # GRU parameters 
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size 
        self.D = 2 if args.bidirectional else 1 
        # GRU layers 
        if args.torch_gru:
            self.gru = nn.GRU(args.input_size, args.hidden_size, num_layers=args.num_layers, 
                              bidirectional=args.bidirectional, batch_first=True, dropout=args.dropout)
        else:
            self.gru_cell = GRU_cell(args)
        # output layer 
        self.fc = nn.Linear(self.D * args.hidden_size, args.output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1) # logsoftmax layer for cross-entropy 

    def zero_hidden(self, batch_size):
        # return a zero vector with the same data type as weights 
        weight = next(self.parameters())
        hidden = weight.new(self.D * self.num_layers, batch_size, self.hidden_size).zero_()
        return hidden
    
    def forward(self, x, h):
        if args.torch_gru:
            # torch's GRU
            out, new_h = self.gru(x, h)
            if self.D == 1:
                # pick the last output 
                out = self.fc(F.relu(out[:,-1]))
            elif self.D == 2:
                # pick the last output of the forward direction 
                out_forward = out[:,-1,:self.hidden_size]
                # pick the last output of the backward direction 
                out_backward = out[:,0,self.hidden_size:]
                # concat them 
                out = torch.cat((out_forward, out_backward), dim=1)
                out = self.fc(F.relu(out))
        else:
            # iterate through sequences for hand-crafted GRU
            seq_length, hidden = x.shape[1], h.squeeze(dim=1)
            # forward pass through time/words
            for word_idx in range(seq_length):
                out, hidden = self.gru_cell(x[:,word_idx], hidden)      
            out = self.fc(F.relu(out))
        return self.logsoftmax(out)
    
    def run_train(self, args, train_loader):
        # load model to device 
        device = f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu"
        self = self.to(device)
        print(f"model name: {args.model_name}")
        print(f"training is running on {device}!")
        # Adam optimizer with Negative Log-Likelihood Loss 
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()
        ce_hist = [] # save cross-entropy losses 
        start_time = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            ce_loss = 0.0
            for i, data in enumerate(train_loader, 1):
                # get the data 
                embeddings, category, sentiment = data["review"], data["category"], data["sentiment"]
                # reset the optimizer 
                optimizer.zero_grad()
                # load the tensors to device
                embeddings, sentiment = embeddings.to(device), sentiment.to(device)
                # set the hidden vector to zero 
                hidden = model.zero_hidden(batch_size=1).to(device)
                preds = model(embeddings, hidden)
                # calculate cross-entropy  
                loss = criterion(preds, torch.argmax(sentiment, dim=1))
                loss.backward() # back-propagation
                optimizer.step()
                ce_loss += loss.item()
                
                if i % 100 == 0:
                    mean_loss = ce_loss / 100
                    print(f"[EPOCH: {epoch}/{args.epochs}, ITER: {i}/{len(train_loader)}] cross-entropy loss: {round(mean_loss, 4)}")
                    ce_hist.append(mean_loss)
                    ce_loss = 0.0

        end_time = time.perf_counter()
        print(f"time elapsed for training: {round(end_time - start_time, 4)}")
        # save the model 
        torch.save(self.state_dict(), args.model_name)
        # save the loss history 
        np.save(f"{args.model_name}_train_hist", np.array(ce_hist))
        return np.array(ce_hist)

    def inference(self, args, test_loader):
        device = f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu"
        self.load_state_dict(torch.load(args.model_name)) # load the saved model 
        self = self.to(device)
        print(f"model name: {args.model_name}")
        print(f"inference is running on {device}")
        # evaluation mode 
        self.eval()
        criterion = nn.NLLLoss()
        ce_hist, y_preds, y_trues = [], [], []
        ce_loss = 0
        for i, data in enumerate(test_loader, 1):
            # get the data 
            embeddings, category, sentiment = data["review"], data["category"], data["sentiment"]
            # load the tensors to device
            embeddings, sentiment = embeddings.to(device), sentiment.to(device)
            # set the hidden vector to zero 
            hidden = model.zero_hidden(batch_size=1).to(device)
            preds = model(embeddings, hidden)
            # save the predictions 
            y_pred, y_true = torch.argmax(preds, dim=1), torch.argmax(sentiment, dim=1)
            y_preds.extend(y_pred.cpu())
            y_trues.extend(y_true.cpu())
            # calculate cross-entropy  
            loss = criterion(preds, y_true)
            ce_loss += loss.item()
            if i % 100 == 0:
                mean_loss = ce_loss / i
                print(f"[ITER: {i}/{len(test_loader)}] cross-entropy loss: {round(mean_loss, 4)}")
                ce_hist.append(mean_loss)
                ce_loss = 0

        # draw & save confusion matrix 
        classes = ["negative", "positive"]
        cm = confusion_matrix(y_trues, y_preds)
        df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
        acc = (np.trace(cm) / np.sum(cm)) * 100
        plt.figure()
        sns.heatmap(df_cm, annot=True, fmt='g', cmap="Blues")
        plt.title(f"Confusion matrix for {args.model_name}, acc: {acc:.2f}%")
        plt.xlabel("predicted sentiments")
        plt.ylabel("ground truth sentiments")
        plt.savefig(f'{args.model_name}_confusion_matrix.png')
        # save the loss history 
        np.save(f"{args.model_name}_test_hist", np.array(ce_hist))
        return np.array(ce_hist)


# test code 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # training details
    parser.add_argument("--batch_size", default=1, type=int, help="batch size for training")
    parser.add_argument("--epochs", default=6, type=int, help="number of epochs for training")
    parser.add_argument("--bidirectional", default=False, type=eval, help="switch for bidirectional GRU")
    parser.add_argument("--torch_gru", default=True, type=eval, help="switch for using torch's GRU")
    parser.add_argument("--cuda_idx", default=0, type=int, help="cuda idx for GPU training")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0, type=float, help="dropout probability")

    # other
    parser.add_argument("--model_name", default="GRUb", type=str, help="model will be saved & loaded with this name")
    parser.add_argument("--train", default=True, type=eval, help="switch for train or test")

    # GRU parameters 
    parser.add_argument("--input_size", default=300, type=int, help="input size for GRU")
    parser.add_argument("--hidden_size", default=100, type=int, help="dimension of hidden vector")
    parser.add_argument("--output_size", default=2, type=int, help="dimension of output vector")
    parser.add_argument("--num_layers", default=1, type=int, help="number of layers for GRU")
    parser.add_argument("--pm", default=False, type=eval, help="poor man's GRU")
    args = parser.parse_args()

    # for reproducible results 
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    torch.autograd.set_detect_anomaly(True)

    """
    Recommended parameters for GRU models: 

    Torch's GRU
    lr              :1e-3
    epochs          :6
    batch_size      :1
    bidirectional   :False 
    input_size      :300
    """

    # train GRUNet without bidirection
    model = GRUNet(args)
    # load dataset 
    train_data = SentimentDataset(mode="train")
    test_data = SentimentDataset(mode="test")
    train_loader = DataLoader(train_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)
    if args.train:
        model.run_train(args, train_loader)
        print("training is succesfull!")
    else:
        model.inference(args, test_loader)
        print("inference is succesfull!")
    


