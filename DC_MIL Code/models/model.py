import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import copy

####################### Curriculum 1 #########################

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion: int = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MainBlock(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MainBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class AugBlock(nn.Module):    
    
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: int,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(AugBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 256
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer = self._make_layer(block, 128, layers, stride=1,
                                       dilate=replace_stride_with_dilation[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

class BasicNetwork(nn.Module):
    def __init__(self, blocks, channel, scale_factor=4, multi_gpu=False):
        super(BasicNetwork, self).__init__()
        
        self.bnum = len(blocks)

        if self.bnum == 1:
            self.b1 = blocks[0]
            self.blocks = [self.b1]
        elif self.bnum == 2:
            self.b1, self.b2 = blocks
            self.blocks = [self.b1, self.b2]
        elif self.bnum == 3:
            self.b1, self.b2, self.b3 = blocks
            self.blocks = [self.b1, self.b2, self.b3]

        self.channel = channel

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.upsample = nn.Sequential(nn.Upsample(scale_factor=scale_factor,mode='bilinear',align_corners=True),
                                        nn.Sigmoid())

        self.attention = nn.Sequential(
            nn.Linear(self.channel, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )
        self.fc = nn.Sequential(
            nn.Linear(self.channel*self.bnum,2)
        )

        if multi_gpu:
            self.blocks = [nn.DataParallel(i) for i in self.blocks]
            self.attention = nn.DataParallel(self.attention)
            self.fc = nn.DataParallel(self.fc)

    def forward(self, input, exist_MF=False, last_M=None, last_F=None):
        x = input
        if exist_MF: 
            temp = self.upsample(last_M)
            x = x * temp

        for i, block in enumerate(self.blocks):
            x = block(x)
        xg = self.avgpool(x)
        xg = xg.view(-1,self.channel)
        A = self.attention(xg)
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)
        Fea = torch.mm(A, xg)  # KxL

        if exist_MF:
            Fea = torch.cat((last_F, Fea),1)
        Y_pred = self.fc(Fea)
        Y_softmax = F.softmax(Y_pred, dim=1)
        Y_hat = torch.argmax(Y_softmax, 1)
        return Y_pred, Y_hat

    def _predict(self, input, weight_map=None, mask_exist=False, last_M=None, extract_F=False):
        x = input
        with torch.no_grad():
            if mask_exist: 
                temp = self.upsample(last_M)
                x = x * temp            
            for i, block in enumerate(self.blocks):
                x = block(x)

            xg = self.avgpool(x)
            xg = xg.view(-1,self.channel)
            A = self.attention(xg)
            A_raw = A.view(-1)
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)
            Fea = torch.mm(A, xg)
            
            if extract_F:
                return Fea
            else:
                mask = torch.transpose(x, 1, 3)
                mask = torch.matmul(mask, weight_map.view(-1,1))
                mask = torch.transpose(mask, 1, 3)

                return Fea, mask, A_raw

class ThirdBranch(nn.Module):
    def __init__(self, model, multi_gpu):
        super(ThirdBranch, self).__init__()                            
        for param in model.parameters():
            param.requires_grad=False        
        self.first_branch = model.first_branch
        self.weight_map1 = list(self.first_branch.fc.parameters())[-2][1][-256:]

        self.second_branch = model.second_branch
        self.weight_map2 = list(self.second_branch.fc.parameters())[-2][1][-256:]

        self.block1 =  copy.deepcopy(model.block1)
        self.block2 =  copy.deepcopy(model.block2)
        
        self.FT_param = []
        for layer in self.block2.layer:
            self.FT_param.append(list(layer.conv1.parameters())[0])
            self.FT_param.append(list(layer.conv2.parameters())[0])
            self.FT_param.append(list(layer.conv3.parameters())[0])

        self.FT_param_ori =  copy.deepcopy(self.FT_param)

        for param in self.block2.layer.parameters():
            param.requires_grad=True

        self.block3 = AugBlock(Bottleneck, 2)
        self.third_branch = BasicNetwork([self.block1,self.block2,self.block3], 256, multi_gpu=multi_gpu,scale_factor=32)

    def forward(self, input):
        x_1, x_2, x_3 = input

        with torch.no_grad():
            F_first, M_first, _ = self.first_branch._predict(x_1, self.weight_map1)
            F_second, M_second, _ = self.second_branch._predict(x_2, self.weight_map2, mask_exist=True, last_M=M_first)

        risk_third, hat_third = self.third_branch(x_3, exist_MF=True, last_M=M_second, last_F=torch.cat((F_first, F_second),1))
        
        return risk_third, hat_third
    def _fea_map(self, input):
        x_1, x_2, x_3 = input
        with torch.no_grad():
            F_first, M_first, _ = self.first_branch._predict(x_1, self.weight_map1)
            F_second, M_second, _ = self.second_branch._predict(x_2, self.weight_map2, mask_exist=True, last_M=M_first)   
            weight_map3 = list(self.third_branch.fc.parameters())[-2][1][-256:]     
            F_third, M_third, _ = self.third_branch._predict(x_3, weight_map3, mask_exist=True, last_M=M_second, extract_F=False) 
            F_total = torch.cat((F_first, F_second, F_third),1).detach().cpu()
            risk_third, hat_third = self.third_branch(x_3, exist_MF=True, last_M=M_second, last_F=torch.cat((F_first, F_second),1))
        return F_total, risk_third.detach().cpu(), hat_third.detach().cpu()
    
class SecondBranch(nn.Module):
    def __init__(self, model, multi_gpu):
        super(SecondBranch, self).__init__()
        self.first_branch = model.first_branch
        for param in self.first_branch.parameters():
            param.requires_grad=False
        self.weight_map = list(self.first_branch.fc.parameters())[-2][1][-256:]

        self.block1 =  copy.deepcopy(model.block)
        self.FT_param = []
        for layer in self.block1.layer3:
            self.FT_param.append(list(layer.conv1.parameters())[0])
            self.FT_param.append(list(layer.conv2.parameters())[0])
            try:
                self.FT_param.append(list(layer.downsample[0].parameters())[0])
            except:
                pass 

        self.FT_param_ori =  copy.deepcopy(self.FT_param)

        for param in self.block1.layer3.parameters():
            param.requires_grad=True

        self.block2 = AugBlock(Bottleneck, 2)
        self.second_branch = BasicNetwork([self.block1,self.block2], 256, multi_gpu=multi_gpu,scale_factor=32)

    def forward(self, input):
        x_1, x_2 = input
        with torch.no_grad():
            F_first, M_first, _ = self.first_branch._predict(x_1, self.weight_map)
        risk_second, hat_second = self.second_branch(x_2, exist_MF=True, last_M=M_first, last_F=F_first)
        return risk_second, hat_second

class FirstBranch(nn.Module):
    def __init__(self, multi_gpu):
        super(FirstBranch, self).__init__()
        self.block = MainBlock(BasicBlock, [2,2,2])
        self.first_branch = BasicNetwork([self.block], 256, multi_gpu=multi_gpu)
    def forward(self, input):
        x_1 = input
        risk_first, hat_first = self.first_branch(x_1)
        
        return risk_first, hat_first

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
    return None

####################### Curriculum 2 #########################

class SurvModel(nn.Module):
    def __init__(self, k):
        super(SurvModel, self).__init__()
        self.k = k

        self.linear = nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.linear_z = nn.Sequential(
            nn.Linear(32*self.k, 128),
            nn.ReLU(),
        )

        self.linear_c = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.attention_c = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(128,1)
        )
        self.Wa  = nn.ModuleList([nn.Linear(128, 32, bias=False) for i in range(3)])
        self.softmax  = nn.Softmax(dim=1)

    def forward(self, input, training=False):
        x = input.squeeze(0)
        x = self.linear(x)
        A = self.attention(x)
        A = F.softmax(A, dim=0).squeeze(1)

        A, ind = torch.sort(A, descending=True)

        zs = x[ind[:self.k]] * F.softmax(A[:self.k], dim=0).unsqueeze(1)
        zq = self.Wa[0](zs)
        zk = self.Wa[1](zs)
        zv = self.Wa[2](zs)
        zw = self.softmax(zq @ zk.T) @ zv

        zw = self.linear_z(zw.view(1,-1))
        Y_pred = self.fc(zw)

        if training:  
            c = x[ind[self.k:]] * F.softmax(A[self.k:], dim=0).unsqueeze(1)
            c = self.linear_c(c)
            Ac = self.attention_c(c)
            Ac = torch.transpose(Ac, 1, 0)  
            Ac = F.softmax(Ac, dim=1)
            c = torch.mm(Ac, c)
            return Y_pred, zw, c
        
        return Y_pred

class CPCModel(nn.Module):
    def __init__(self):
        super(CPCModel, self).__init__()
        self.Wk  = nn.Linear(64, 128)
        self.Ww = nn.ModuleList([nn.Linear(64, 128) for i in range(2)])
        self.linear_0 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.linear_1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.attention_0 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.attention_1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.softmax  = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, **kwargs):
        zw_0 = kwargs['zw_0']
        zw_1 = kwargs['zw_1']
        c = kwargs['c']
        batch_0 = zw_0.shape[0]
        batch_1 = zw_1.shape[0]
        batch = batch_0 + batch_1
        pred = torch.empty((batch,128)).float().to(kwargs['device'])

        for i in range(batch_0):
            zt = self.linear_0(zw_0[torch.arange(batch_0)!=i])
            Ac = self.attention_0(zt)
            Ac = torch.transpose(Ac, 1, 0)  
            Ac = F.softmax(Ac, dim=1)
            zt = torch.mm(Ac, zt)
            pred[i,] = self.Ww[0](zt).view(-1)+self.Wk(c[i]).view(-1)
        for i in range(batch_1):
            zt = self.linear_1(zw_1[torch.arange(batch_1)!=i])
            Ac = self.attention_1(zt)
            Ac = torch.transpose(Ac, 1, 0)  
            Ac = F.softmax(Ac, dim=1)
            zt = torch.mm(Ac, zt)
            pred[batch_0+i,] = self.Ww[1](zt).view(-1)+self.Wk(c[batch_0+i]).view(-1)

        zw = torch.cat([zw_0,zw_1]) # b*128
        total = torch.mm(zw, torch.transpose(pred,0,1)) # b*b
        correct = torch.sum(torch.eq(torch.argmax(self.softmax(total).detach().cpu(), dim=0), torch.arange(0, batch)))
        nce_w = torch.sum(torch.diag(self.lsoftmax(total)))
        nce_w /= -1.*batch
        accuracy_w = 1.*correct.item()/batch

        # print(f"accuracy_s/w: {accuracy_s} and {accuracy_w}")

        return nce_w




