
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
import time
import numpy as np
from dataset import YoloDataset
import matplotlib.pyplot as plt
from utils import IoU
import cv2
from torchvision.ops import nms, remove_small_boxes
from torchvision.ops import box_iou
from skimage import io
from pytorch_model_summary import summary 


# RESIDUAL BLOCK
class Block(nn.Module):

    def __init__(self, width, learnable_res=False):
        super(Block, self).__init__()
        # 2 convolutional layers with batchnorm
        self.conv = nn.Sequential(nn.Conv2d(width, width, 3, 1, 1),
                                  nn.BatchNorm2d(width),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(width, width, 3, 1, 1),
                                  nn.BatchNorm2d(width))
        # for learnable skip-connections/residuals
        self.learnable_res = learnable_res
        if self.learnable_res:
            self.res_conv = nn.Sequential(
                nn.Conv2d(width, width, 3, 1, 1),
                nn.BatchNorm2d(width)
            )

    def forward(self, x):
        out = self.conv(x)  # pass through CNN
        if self.learnable_res:
            out += self.res_conv(x)  # pass through learnable res
        else:
            out += x  # skip-connection
        return F.relu(out)  # ReLU


# ENTIRE NETWORK
class YoloNet(nn.Module):
    def __init__(self, in_channels=3, width=8, n_blocks=5, learnable_res=False,
                       max_col_cell=12, max_row_cell=12, anchor_num=5, yolo_interval=20, 
                       threshold=0.2, iou_ths=0.5, lamb_obj=5, lamb_noobj=.5):
        assert (n_blocks >= 0)
        super(YoloNet, self).__init__()
        # hyperparamters 
        self.threshold = threshold
        self.lamb_obj = lamb_obj
        self.lamb_noobj = lamb_noobj
        self.iou_ths = iou_ths
        # output size
        self.out_dim = max_col_cell * max_row_cell * anchor_num * 8
        self.max_row_cell = max_row_cell
        self.max_col_cell = max_col_cell
        self.yolo_interval = yolo_interval
        self.anchor_num = anchor_num
        # base model
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, width, kernel_size=7,
                           padding=0),
                 nn.BatchNorm2d(width),
                 nn.ReLU(True)] 

        # downsampling layers 
        n_down = 4
        mult = 0
        for k in range(n_down):
            expansion = 2 ** k
            model += [nn.Conv2d(width * expansion, width * expansion * 2,
                                kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(width * expansion * 2),
                      nn.ReLU()] # relu added 
            mult = width * expansion * 2
        # add residual blocks
        for i in range(n_blocks):
            model += [Block(mult, learnable_res)]
        # put the objects in list to nn.Sequential
        self.model = nn.Sequential(*model)
        # classifier head
        self.class_head = nn.Sequential(
            nn.Linear(32768, 11520),
            nn.BatchNorm1d(11520),
            nn.ReLU(True),
            nn.Linear(11520, self.out_dim)
        )

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, 1)
        out = self.class_head(out)
        return out

    def run_train(self, train_loader, test_loader, epochs=50, lr=1e-3, betas=(0.9, 0.99)):
        # load model to device (cpu or gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        # loss functions for binary class, mse, multi class 
        criterion1 = nn.BCELoss()
        criterion2 = nn.MSELoss()
        criterion3 = nn.CrossEntropyLoss()
        # choose optimization method
        optimizer = optim.Adam(self.parameters(), lr, betas)
        start_time = time.perf_counter()
        train_loss_hist = {"bce" : [], "mse" : [], "ce" : []}
        test_loss_hist = {"bce" : [], "mse" : [], "ce" : []}
        min_loss = 100000000000
        print("Training is running...")
        # training loop
        for epoch in range(epochs):
            # losses that will be reported at each epoch 
            run_loss_bce = 0.0
            run_loss_ce = 0.0 
            run_loss_mse = 0.0
            self.train()
            for data in train_loader:
                # extract data
                imgs, gts = data
                yolo_tensor, label = gts['yolo_tensor'], gts['label']
                # load them to gpu or cpu
                imgs = imgs.to(device)
                yolo_tensor = yolo_tensor.float().to(device)
                # forward pass 
                optimizer.zero_grad()
                output = self(imgs)
                output = output.view(yolo_tensor.shape)
                
                # CALCULATE TOTAL LOSS 
                loss = torch.tensor(0.0, requires_grad=True).float().to(device)
                objectness = yolo_tensor[:,:,:,0] # objectness ground truths 
                yolo_idx = torch.nonzero(objectness) # consider cell&anch that have objects

                # CALCULATE BCE LOSS
                # calculate binary cross entropy for no object cases 
                tmp = torch.nonzero(objectness == 0)
                no_obj_pred = output[tmp[:,0], tmp[:,1], tmp[:,2]]
                no_obj_gt = yolo_tensor[tmp[:,0], tmp[:,1], tmp[:,2]]
                loss_no_obj = criterion1(nn.Sigmoid()(no_obj_pred[:,0]), no_obj_gt[:,0]) 
                loss += self.lamb_noobj * loss_no_obj

                # calculate binary cross entropy for objects cases 
                obj_pred = output[yolo_idx[:,0], yolo_idx[:,1], yolo_idx[:,2]]
                obj_gt = yolo_tensor[yolo_idx[:,0], yolo_idx[:,1], yolo_idx[:,2]]
                loss_obj = criterion1(nn.Sigmoid()(obj_pred[:,0]), obj_gt[:,0]) 
                loss += loss_obj

                # save the loss for bce
                run_loss_bce += loss_no_obj.item() + loss_obj.item()

                # CALCULATE MSE LOSS
                yolo_tensor = yolo_tensor.view(-1, self.max_row_cell, self.max_col_cell, self.anchor_num, 8)
                output = output.view(yolo_tensor.shape)
                # pull cell numbers of objects 
                objects = torch.nonzero(label != 13) # only include real objects 
                cell_nos = gts['cell_idx'][objects[:,0], objects[:,1]].type(torch.long)
                anchor_nos = gts['anchor_idx'][objects[:,0], objects[:,1]].type(torch.long)
                #anchor_nos = gts['anchor_idx'][objects[:,0], objects[:,1]]
                output = output[objects[:,0], cell_nos[:,0], cell_nos[:,1], anchor_nos]
                yolo_tensor = yolo_tensor[objects[:,0], cell_nos[:,0], cell_nos[:,1], anchor_nos]
                # calculate mse loss
                mse_loss_obj = criterion2(output[:,1:5], yolo_tensor[:,1:5])
                loss += self.lamb_obj * mse_loss_obj
                run_loss_mse += mse_loss_obj.item()
                # calculate multi class cross entropy loss
                labels = torch.argmax(yolo_tensor[:,5:], dim=1)              
                tmp = criterion3(output[:,5:], labels)
                loss += tmp
                run_loss_ce += tmp.item()
                # backward pass
                loss.backward()
                optimizer.step()

            # calculate mean losses
            run_loss_bce /= len(train_loader)
            run_loss_mse /= len(train_loader)
            run_loss_ce /= len(train_loader)
            total_loss = (run_loss_bce + run_loss_mse + run_loss_ce) / 3 # equal weights
            # save mean losses
            train_loss_hist["bce"].append(run_loss_bce)
            train_loss_hist["mse"].append(run_loss_mse)
            train_loss_hist["ce"].append(run_loss_ce)
            # report the results
            print("*"*15 + "TRAIN" + "*"*15)
            print(f"[EPOCH {epoch+1}/{epochs}] Total Mean Loss: {round(total_loss, 5)}")
            print(f"[EPOCH {epoch+1}/{epochs}] BCE Mean Loss: {round(run_loss_bce, 5)}")
            print(f"[EPOCH {epoch+1}/{epochs}] MSE Loss: {round(run_loss_mse, 5)}")
            print(f"[EPOCH {epoch+1}/{epochs}] CE Mean Loss: {round(run_loss_ce, 5)}")

            # evaluation of the model (to check overfitting)
            run_loss_bce = 0.0
            run_loss_ce = 0.0 
            run_loss_mse = 0.0
            self.eval()
            for data in test_loader:
                # extract data
                imgs, gts = data
                yolo_tensor, label = gts['yolo_tensor'], gts['label']
                # load them to gpu or cpu
                imgs = imgs.to(device)
                yolo_tensor = yolo_tensor.float().to(device)
                # forward pass 
                output = self(imgs)
                output = output.view(yolo_tensor.shape)
                # total loss 
                loss = torch.tensor(0.0, requires_grad=True).float().to(device)
                objectness = yolo_tensor[:,:,:,0] # consider cell&anch that have no objects
                yolo_idx = torch.nonzero(objectness) # consider cell&anch that have objects
                # CALCULATE BCE LOSS
                # calculate binary cross entropy for no object cases 
                tmp = torch.nonzero(objectness == 0)
                no_obj_pred = output[tmp[:,0], tmp[:,1], tmp[:,2]]
                no_obj_gt = yolo_tensor[tmp[:,0], tmp[:,1], tmp[:,2]]
                loss_no_obj = criterion1(nn.Sigmoid()(no_obj_pred[:,0]), no_obj_gt[:,0]) 
                # calculate binary cross entropy for objects cases 
                obj_pred = output[yolo_idx[:,0], yolo_idx[:,1], yolo_idx[:,2]]
                obj_gt = yolo_tensor[yolo_idx[:,0], yolo_idx[:,1], yolo_idx[:,2]]
                loss_obj = criterion1(nn.Sigmoid()(obj_pred[:,0]), obj_gt[:,0]) 
                # save the loss for bce
                run_loss_bce += loss_no_obj.item() + loss_obj.item()
                # CALCULATE MSE LOSS
                yolo_tensor = yolo_tensor.view(-1, self.max_row_cell, self.max_col_cell, self.anchor_num, 8)
                output = output.view(yolo_tensor.shape)
                # pull cell numbers of objects 
                objects = torch.nonzero(label != 13) # only include real objects 
                cell_nos = gts['cell_idx'][objects[:,0], objects[:,1]].type(torch.long)
                anchor_nos = gts['anchor_idx'][objects[:,0], objects[:,1]].type(torch.long)
                #anchor_nos = gts['anchor_idx'][objects[:,0], objects[:,1]]
                output = output[objects[:,0], cell_nos[:,0], cell_nos[:,1], anchor_nos]
                yolo_tensor = yolo_tensor[objects[:,0], cell_nos[:,0], cell_nos[:,1], anchor_nos]
                # calculate mse loss
                mse_loss_obj = criterion2(output[:,1:5], yolo_tensor[:,1:5])
                run_loss_mse += mse_loss_obj.item()
                # calculate multi class cross entropy loss
                labels = torch.argmax(yolo_tensor[:,5:], dim=1)              
                tmp = criterion3(output[:,5:], labels)
                run_loss_ce += tmp.item()

            # calculate mean losses
            run_loss_bce /= len(test_loader)
            run_loss_mse /= len(test_loader)
            run_loss_ce /= len(test_loader)
            total_loss = (run_loss_bce + run_loss_mse + run_loss_ce) / 3 # equal weights
            # save mean losses
            test_loss_hist["bce"].append(run_loss_bce)
            test_loss_hist["mse"].append(run_loss_mse)
            test_loss_hist["ce"].append(run_loss_ce)
            # report the results
            print("*"*15 + "TEST" + "*"*15)
            print(f"[EPOCH {epoch+1}/{epochs}] Total Mean Loss: {round(total_loss, 5)}")
            print(f"[EPOCH {epoch+1}/{epochs}] BCE Mean Loss: {round(run_loss_bce, 5)}")
            print(f"[EPOCH {epoch+1}/{epochs}] MSE Loss: {round(run_loss_mse, 5)}")
            print(f"[EPOCH {epoch+1}/{epochs}] CE Mean Loss: {round(run_loss_ce, 5)}")

            if total_loss < min_loss:
                min_loss = total_loss
                # save the best model
                torch.save(self.state_dict(), "best_model")
                print("Best model has been saved!")
        
        # save the last model Y
        torch.save(self.state_dict(), "last_model")
        print("last model has been saved!")
             

        return {"train": train_loss_hist,
                "test" : test_loss_hist}

    def bbox_to_corners(self, indices, yolo_tensor, device):

        [dx, dy, h, w] = [yolo_tensor[:,i] for i in range(1,5)]
        
        h *= self.yolo_interval
        w *= self.yolo_interval

        row_cell_idx = indices // self.max_col_cell
        col_cell_idx = indices % self.max_col_cell

        cell_i_center = row_cell_idx * self.yolo_interval + self.yolo_interval/2
        cell_j_center = col_cell_idx * self.yolo_interval + self.yolo_interval/2

        # broadcast 
        x_center = cell_i_center + dx * self.yolo_interval
        y_center = cell_j_center + dy * self.yolo_interval

        x1 = (y_center - w/2).unsqueeze(dim=1)
        y1 = (x_center - h/2).unsqueeze(dim=1)
        x2 = (y_center + w/2).unsqueeze(dim=1)
        y2 = (x_center + h/2).unsqueeze(dim=1)

        bbox = torch.cat((yolo_tensor[:,0].unsqueeze(dim=1), x1, y1, x2, y2, yolo_tensor[:,5:]), dim=1).to(device)

        return bbox 

    def inference(self, model_path, test_loader, sample_num=2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device 
        class_list = ['bus', 'cat', 'pizza'] # class list
        # load model
        self.load_state_dict(torch.load(model_path)) 
        self.to(device)
        self.eval()
        # set the figure and axes for display 
        fig, axs = plt.subplots(3, sample_num)
        fig.tight_layout()
        counter = 0
        # iterate through data loader, batch size = 1
        for data in test_loader:
            # display only sample_num number of images per 
            if counter > 3 * sample_num - 1:
                break
            # extract the annotations and images 
            imgs, gts = data
            yolo_tensor, anchor_idx = gts['yolo_tensor'], gts['anchor_idx']
            cell_idx, labels = gts['cell_idx'], gts['label']

            # prepare image for display
            img = np.uint8(imgs[0].numpy() * 255)
            img = img.transpose((1, 2, 0))
            img = np.ascontiguousarray(img)

            # load them to gpu or cpu 
            imgs = imgs.to(device)
            yolo_tensor = yolo_tensor.float().to(device)
            output = self(imgs)
            # reshape the prediction and labels to (num. row cell, num. col cell, anchor id, 8)
            output = output.view(yolo_tensor.shape)
            yolo_tensor = yolo_tensor.view(self.max_row_cell, self.max_col_cell, self.anchor_num, 8)
            # forward pass for inference
            yolo_idx = torch.nonzero(anchor_idx[0] != 13)
            if len(yolo_idx) == 1:
                continue
            # iterate through objects to display ground truths 
            for obj_idx in range(len(yolo_idx)):
                # extract ground truths for each object 
                label = int(labels[0, obj_idx].item())
                row_cell_idx, col_cell_idx = cell_idx[0, obj_idx]
                anch_idx = anchor_idx[0, obj_idx]
                [row_cell_idx, col_cell_idx, anch_idx] = list(map(lambda x: int(x.item()), [row_cell_idx, col_cell_idx, anch_idx]))
                # relevant yolo vector
                yolo_vector = yolo_tensor[row_cell_idx, col_cell_idx, anch_idx]
                # calculate ground truth bbox size 
                h = yolo_vector[3].item() * self.yolo_interval
                w = yolo_vector[4].item() * self.yolo_interval
                # calculate cell centers 
                cell_i_center = row_cell_idx * self.yolo_interval + self.yolo_interval / 2
                cell_j_center = col_cell_idx * self.yolo_interval + self.yolo_interval / 2
                # calculate the center of gt bbox 
                x_center = yolo_vector[1].item() * self.yolo_interval + cell_i_center
                y_center = yolo_vector[2].item() * self.yolo_interval + cell_j_center
                [x1, y1, x2, y2] = [round(y_center-w/2), round(x_center-h/2), round(y_center+w/2), round(x_center+h/2)]
                # draw the bounding box
                img = cv2.rectangle(img, (round(x1), round(y1)),
                                     (round(x2), round(y2)),
                                     (36, 256, 12), 2)
                # draw its class name 
                img = cv2.putText(img, class_list[label], (round(x1), round(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.8, (36, 256, 12), 2)
            # objectness prediction
            output[0,:,:,0] = nn.Sigmoid()(output[0,:,:,0])
            cobj_idx = torch.nonzero(output[0,:,:,0] > self.threshold)
            box_cand = output[0, cobj_idx[:,0], cobj_idx[:,1]]
            # extract corners of the predicted bounding box 
            pred_boxes = self.bbox_to_corners(cobj_idx[:,0], box_cand, device)
            bboxes = []
            # predicted categories 
            pred_cats = torch.argmax(pred_boxes[:,5:], dim=1)
            # perform non-max supression for each class 
            for i in range(3):
                idxs = torch.nonzero(pred_cats == i)
                if idxs.shape[0] != 0:
                    # non-max supression for objectness = 1
                    class_tensors = pred_boxes[idxs[:,0]]
                    indices = nms(boxes=class_tensors[:,1:5], scores=class_tensors[:,0], iou_threshold=self.iou_ths)
                    bboxes.append(class_tensors[indices,:])   
        

            # display predicted bounding boxes 
            for obj_idx in range(len(bboxes)):  
                yolo_pred = bboxes[obj_idx][0]   
                # predicted class
                label = torch.argmax(yolo_pred[5:]).item()
                # bounding box corners 
                [x1, y1, x2, y2] = [round(yolo_pred[i].item()) for i in range(1,5)]
                # draw the bounding box 
                img = cv2.rectangle(img, (x1, y1),
                                    (x2, y2),
                                    (256, 36, 12), 2)
                # draw predicted class name 
                img = cv2.putText(img, class_list[label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (256, 36, 12), 2)
            
            # plot the image to the corresponding cell 
            row, col = counter // sample_num, counter % sample_num
            axs[row, col].imshow(img)
            axs[row, col].axis('off')
            axs[row, col].set_title(f"Prediction {counter+1}", size=12)
            counter += 1
        # save figure 
        plt.savefig("test_pred.jpeg")
        print("Predictions are plotted and figure is saved!")
    
# test code for CNN backbone of YoloNet 
if __name__ == "__main__":
    x = torch.rand((8, 3, 256, 256))
    model = YoloNet()
    y = model(x)
    print("Input shape: ", x.shape)
    print("Output shape: ", y.shape)
    # print model summary 
    model_sum = summary(model, torch.zeros((1, 3, 256, 256)), show_input=False, show_hierarchical=True)

    file_obj = open("model_sum.txt", "w")
    file_obj.write(model_sum)
    print("Model summary was saved...")

    print("Learnable layers:",  len(list(model.parameters())))
