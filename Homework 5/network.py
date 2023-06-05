
import torch.nn as nn
import torch.nn.functional as F
import torch
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
        out = self.conv(x) # pass through CNN
        if self.learnable_res:
            out += self.res_conv(x) # pass through learnable res 
        else: 
            out += x # skip-connection 
        return F.relu(out) # ReLU
# ENTIRE NETWORK 
class HW5Net(nn.Module):

    def __init__(self, in_channels=3, width=8, n_blocks=5, learnable_res=False):
        assert (n_blocks >= 0)
        super(HW5Net, self).__init__()
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
                      ]
            mult = width * expansion * 2
        # add residual blocks 
        for i in range(n_blocks):
            model += [Block(mult, learnable_res)]
        # put the objects in list to nn.Sequential
        self.model = nn.Sequential(*model)
        # classifier head
        self.class_head = nn.Sequential(
            nn.Linear(32768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 3),
        )
        # bounding box detector head
        self.bbox_head = nn.Sequential(
            nn.Linear(32768, 5196), 
            nn.BatchNorm1d(5196),
            nn.ReLU(True),
            nn.Linear(5196, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 4))
        
    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, 1)
        cat = self.class_head(out)
        bbox = self.bbox_head(out)
        return cat, bbox

# test code 
if __name__ == "__main__":
    """
    # test the Bottleneck block
    x = torch.randn((4,3,256,256))
    block1 = BottleNeck(in_channels=3, out_channels=3, width=3, downsample=False)
    y = block1(x)
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    

    x = torch.randn((4, 3, 224, 224))
    resnet = ResNet50()
    y = resnet(x)
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print("Parameters:", len(list(resnet.parameters())))
    """
    """
    print(torch.cuda.is_available())
    x2 = torch.randn((4, 3, 256, 256))
    beast = BeastNet()
    y1, y2 = beast(x2)
    print("y1 shape:", y1.shape)
    print("y2 shape:", y2.shape)

    # test localizer 
    x3 = torch.randn((4, 64, 256, 256))
    localizer = Localizer()
    y = localizer(x3)
    print("Localizer output shape:", y.shape)

    x = torch.randn((4,3,256,256))
    model = HW5Net()
    cat, bbox = model(x)
    print(cat.shape)
    print(bbox.shape)
    """
    x = torch.randn((4, 3, 256, 256))
    model = Block(3, learnable_res=True)
    y = model(x)
    print("Output shape:", y.shape)
    print("Block works sucessfully")

    x = torch.randn((4,3,256,256))
    model = HW5Net(learnable_res=False)
    cat, bbox = model(x)
    print("Whole network works sucessfully")

    # print model summary
    model = HW5Net()
    # show output shape and hierarchical view of net
    model_sum = summary(model, torch.zeros((1, 3, 256, 256)), show_input=False, show_hierarchical=True)

    file_obj = open("model_sum.txt", "w")
    file_obj.write(model_sum)
    print("Model summary was saved...")

    print("Learnable layers:",  len(list(model.parameters())))

