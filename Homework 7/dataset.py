
import torch.nn as nn 
import os 
from skimage import io 
import torchvision.transforms as tvt

class PizzaDataset(nn.Module):

    def __init__(self, path="pizza", train=True):
        super(PizzaDataset, self).__init__()
        self.data_dir = os.path.join(path, "train") if train else os.path.join(path, "eval")
        self.file_list = os.listdir(self.data_dir) # list of file names 
        # copy image -> tensor -> normalize
        self.transform = tvt.Compose([tvt.Lambda(lambda img: img.copy()),
                                      tvt.ToTensor(),
                                      tvt.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]) # normalization 
    
    def __len__(self):
        return len(self.file_list) # data size 

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_dir, self.file_list[idx])
        I = io.imread(file_name)
        img = self.transform(I)
        return img, file_name

# test code 
if __name__ == "__main__":
    
    print("dataset is being generated...")
    train_data = PizzaDataset(path="pizzas", train=True)
    test_data = PizzaDataset(path="pizzas", train=False)
    print("train and test data were constructed!")
    print(f"train length: {len(train_data)}")
    print(f"test length: {len(test_data)}")

    a = 5
    print(len(train_data[0]))




