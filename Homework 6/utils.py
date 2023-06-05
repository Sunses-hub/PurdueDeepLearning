
from torchvision.ops import box_iou
import torch 
import torch.nn as nn 

class IoU(nn.Module):

    def __init__(self, yolo_interval, device, reduction='none', image_size=256):
        super(IoU, self).__init__()
        self.reduction = reduction
        self.yolo_interval = yolo_interval
        self.device = device
        self.image_size = image_size

    def bbox_to_corners(self, cell_nos, dx, dy, h, w):

        h *= self.yolo_interval
        w *= self.yolo_interval
        
        row_cell_idx = cell_nos[:,0].unsqueeze(dim=1)
        col_cell_idx = cell_nos[:,1].unsqueeze(dim=1)

        cell_i_center = row_cell_idx * self.yolo_interval + self.yolo_interval/2
        cell_j_center = col_cell_idx * self.yolo_interval + self.yolo_interval/2

        # broadcast 
        x_center = cell_i_center.repeat(1, dx.shape[1]).to(self.device) + dx * self.yolo_interval
        y_center = cell_j_center.repeat(1, dy.shape[1]).to(self.device) + dy * self.yolo_interval
    
        #CHECK THIS TOMORROW 
        x1 = (y_center - w/2).unsqueeze(dim=2)
        y1 = (x_center - h/2).unsqueeze(dim=2)
        x2 = (y_center + w/2).unsqueeze(dim=2)
        y2 = (x_center + h/2).unsqueeze(dim=2)

        bbox = torch.cat((x1, y1, x2, y2), dim=2).to(self.device)
        return bbox

    def forward(self, cell_nos, output, target):

        [dx, dy, h, w] = [output[:,:,i] for i in range(1,5)]
        bbox_pred = self.bbox_to_corners(cell_nos, dx, dy, h, w)
        [dx, dy, h, w] = [target[:,i].unsqueeze(dim=1) for i in range(1,5)]
        bbox_gt = self.bbox_to_corners(cell_nos, dx, dy, h, w).squeeze(dim=1)
        # calculate iou score
        results = torch.zeros(bbox_pred.shape[:-1], device=self.device, requires_grad=True)
        idx = torch.arange(0, results.shape[0])
        for i in range(results.shape[1]):
            tmp = box_iou(bbox_pred[:,i,:], bbox_gt)
            results[:,i] = tmp[idx, idx]
        return results



