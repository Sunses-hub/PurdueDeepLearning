 # my code
from dataset import MyCOCODataset
from model import VisionTransformer, train, inference
# torch 
import torch 
from torch.utils.data import DataLoader
# other 
import argparse 
import pickle 
import random
import numpy as np



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # attention parameters
    parser.add_argument("--num_heads", default=5, type=int, help="number of attention heads")
    parser.add_argument("--num_encoders", default=3, type=int, help="number of encoders in transformer")
    parser.add_argument("--my_attention", default=False, type=eval, help="switch for using my attention")
    # model parameters
    parser.add_argument("--emb_size", default=100, type=int, help="number of embedding dimension")
    parser.add_argument("--img_size", default=64, type=int, help="heigh=width of an image")
    parser.add_argument("--max_seq_length", default=16+1, type=int, help="length of the patch sequences")
    parser.add_argument("--patch_size", default=16, type=int, help="patch size")
    parser.add_argument("--class_num", default=5, type=int, help="number of classes to classify")
    parser.add_argument("--model_name", default="transformer", type=str, help="name of the model")
    # train parameters 
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument("--epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta1 parameter in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta2 parameter in Adam")
    parser.add_argument("--device", default="cuda:0", type=str, help="select a device")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size for SGD")
    # train & test 
    parser.add_argument("--train", default=True, type=eval, help="run code for training or inference")
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

    # initialize the dataset
    train_data = MyCOCODataset(folder_name='train_data')
    val_data = MyCOCODataset(folder_name='val_data', train=False)
    train_loader = DataLoader(dataset=train_data, batch_size=16, num_workers=2)
    val_loader = DataLoader(dataset=val_data, batch_size=16, num_workers=2)
    # initialize model 
    model = VisionTransformer(args)
    if args.train:
        results = train(model, train_loader, val_loader, args)
        pickle.dump(results, open("train_results.pkl", "wb"))
    else:
        inference(model, val_loader, args)

