import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
from einops.einops import rearrange
import os 
root_dir = os.path.abspath(os.path.dirname(__file__))
# root_dir = "/".join(base_dir.split("/")[:-1])
# print(root_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "sp2"))
sys.path.append(os.path.join(root_dir, "sp2/src/loftr"))
sys.path.append(os.path.join(root_dir, "sp2/src/loftr/utils"))

from Model.resnet import resnet34, resnet50
from src.loftr.resnet_new import resnet18
from Model.e2p import E2P
from Model.projection import cp2
import config as cf
from position_encoding import PositionEncodingSine
from loftr_module import LocalFeatureTransformer
from src.config.default import get_cfg_defaults
from src.loftr.loftr import LoFTR, lower_config
from yacs.config import CfgNode as CN

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.se = SELayer(planes, reduction)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.se(out)

        out = self.relu(out)

        return out



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_relu(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))

class DulaNet_Branch_Equi(nn.Module):
    def __init__(self, backbone, encoder, config):
        super(DulaNet_Branch_Equi, self).__init__()

        bb_dict = {'resnet18':resnet18,
                    'resnet34':resnet34,
                    'resnet50':resnet50}

        self.encoder = encoder

        feat_dim = 512 if backbone != 'resnet50' else 2048

        self.decoder = nn.ModuleList([
            conv3x3_relu(feat_dim, 256),
            conv3x3_relu(256, 128),
            conv3x3_relu(128, 64),
            conv3x3_relu(64, 32),
            conv3x3_relu(32, 16),
        ])
        self.last = conv3x3(16, 2)    # Additional boundary seg output

    def forward_get_feats(self, x):
        _, x = self.encoder(x)

        feats = [x]
        for conv in self.decoder:
            x = F.interpolate(x, scale_factor=(2,2), mode='nearest')
            x = conv(x)
            feats.append(x)
        out = self.last(x)
        return out, feats


class DulaNet_Branch_Persp(nn.Module):
    def __init__(self, backbone, encoder, config):
        super(DulaNet_Branch_Persp, self).__init__()

        bb_dict = {'resnet18':resnet18,
                    'resnet34':resnet34,
                    'resnet50':resnet50}
        self.encoder = encoder


        feat_dim = 1024 if backbone != 'resnet50' else 4096
        self.decoder = nn.ModuleList([
            conv3x3_relu(feat_dim, 256),
            conv3x3_relu(512, 128),
            conv3x3_relu(256, 64),
            conv3x3_relu(128, 32),
            conv3x3_relu(64, 16),
        ])
        if backbone != 'resnet50':
            self.persp_fusion = SEBasicBlock(1024, 1024)
            self.persp_downsample = conv3x3_relu(1024, 512)
        else:
            self.persp_fusion = SEBasicBlock(4096, 4096)
            self.persp_downsample = conv3x3_relu(4096, 2048)
        if config["corner"]:
            self.last = conv3x3(16, 3)    # Additional boundary seg output
        else:
            self.last = conv3x3(16, 2)

    def forward_from_feats(self, x1, x2, feats):
        _, x1 = self.encoder(x1)
        _, x2 = self.encoder(x2)
        persp_concat = torch.cat([x1, x2], axis=1)

        x = self.persp_fusion(persp_concat)
        x = self.persp_downsample(x)
        #x = (x1 + x2) / 2.0
        for i, conv in enumerate(self.decoder):
            x = torch.cat([x, feats[i]], axis=1)
            x = F.interpolate(x, scale_factor=(2,2), mode='nearest')
            x = conv(x)
        out = self.last(x)
        return out




class PSMNet(nn.Module):
    def __init__(self, backbone, config_layout, config_pose):
        super(PSMNet, self).__init__()



        self.backbone = resnet18(pretrained=False)
        self.pos_encoding = PositionEncodingSine(config_pose['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config_pose['coarse'])

        num_input_channel = config_pose['predict']['input_c']
        self.layer1 = nn.Conv2d(num_input_channel, num_input_channel // 4, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(num_input_channel // 4)
        self.layer2 = nn.Conv1d(num_input_channel // 4, num_input_channel // 16, kernel_size=3, stride=3, padding=0)
        self.bn2 = nn.BatchNorm1d(num_input_channel // 16)
        self.layer3 = nn.Conv1d(num_input_channel // 16, 1, kernel_size=3, stride=3, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(config_pose['predict']['fc_in'], 3)


        self.model_equi = DulaNet_Branch_Equi(backbone, self.backbone, config_layout)
        self.model_up = DulaNet_Branch_Persp(backbone, self.backbone, config_layout)

        self.model_h = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.Dropout(inplace=True),
                    nn.Linear(256, 64),
                    nn.Dropout(inplace=True),
                    nn.Linear(64, 1)
                )

        self.e2p = E2P(cf.pano_size, cf.fp_size, cf.fp_fov)

        self.e2pm = cp2(
            cf.pano_size, cf.fp_size, cf.fp_fov
        )

        fuse_dim = [int((cf.pano_size[0]/32)*2**i) for i in range(6)]
        self.e2ps_f = torch.nn.ModuleList([E2P((n, n*2), n, cf.fp_fov) for n in fuse_dim])

        self.e2ps_fm = torch.nn.ModuleList([
            cp2(
                (n, n*2), n, cf.fp_fov
            )
            for n in fuse_dim
        ])
        if backbone == 'resnet50':
            equi_fusion_channels = [4096, 512, 256, 128, 64, 32]
        else:
            equi_fusion_channels = [1024, 512, 256, 128, 64, 32]
        equi_fusion = []
        for num_channels in equi_fusion_channels:
            equi_fusion.append(SEBasicBlock(num_channels, num_channels))
        self.equi_fusion = torch.nn.ModuleList(equi_fusion)

        equi_downsample = []
        for num_channels in equi_fusion_channels:
            equi_downsample.append(conv3x3_relu(num_channels, num_channels//2))
        self.equi_downsample = torch.nn.ModuleList(equi_downsample)


    # @autocast()
    def forward(self, inputs, data):

        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['image0'].dim() != 4:
            data['image0'] = data['image0'].view(data['image0'].size(0), data['image0'].shape[2], data['image0'].shape[3], data['image0'].shape[4])
            data['image1'] = data['image1'].view(data['image1'].size(0), data['image1'].shape[2], data['image1'].shape[3], data['image1'].shape[4])


        feats_c, _ = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
        feat_c0, feat_c1 = feats_c.split(data['bs'])
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_c01 = torch.stack((feat_c0, feat_c1), dim=1)
        feat_c01 = feat_c01.permute(0, 3, 1, 2)
        output = self.layer1(feat_c01)
        output = self.relu(self.bn1(output))
        output = output.view(output.shape[0], output.shape[1], output.shape[3])
        output = self.layer2(output)
        output = self.relu(self.bn2(output))
        output = self.relu(self.layer3(output))
        output = self.flatten(output)
        output = self.fc(output)
        data.update({
            'prediction': output,
        })

        p1, p2 = inputs
        pred_delta_x = data['coarse_delta_x_fm'] + output[:, 0]
        pred_delta_y = data['coarse_delta_y_fm'] + output[:, 1]
        pred_rotation = data['coarse_rotation'] + output[:, 2]
        layout_input = [p1, p2, pred_delta_x.data.cpu().numpy(), pred_delta_y.data.cpu().numpy(), pred_rotation.data.cpu().numpy()]

        fcmap1_list, fcmap2_list, fpmap_list = [], [], []
        for idx in range(len(inputs[0])):
            pano_view1, pano_view2, delta_x_fm, delta_y_fm, rotation = [val[idx] for val in layout_input]



            [up_view1, down_view1] = self.e2p(pano_view1)
            [up_view2, down_view2] = self.e2pm(pano_view2,
                                               delta_x_fm=delta_x_fm,
                                               delta_y_fm=delta_y_fm,
                                               rotation=rotation)

            fcmap1, feats_equi1 = self.model_equi.forward_get_feats(pano_view1)
            fcmap2, feats_equi2 = self.model_equi.forward_get_feats(pano_view2)
            fcmap1_list.append(fcmap1)
            fcmap2_list.append(fcmap2)

            feats_fuse = []
            for i, feat1 in enumerate(feats_equi1):

                [_, feat_down1] = self.e2ps_f[i](feat1)
                [_, feat_down2] = self.e2ps_fm[i](feats_equi2[i],
                                                  delta_x_fm=delta_x_fm,
                                                  delta_y_fm=delta_y_fm,
                                                  rotation=rotation)
                feat_concat = torch.cat([feat_down1, feat_down2], axis=1)

                feats_se = self.equi_fusion[i](feat_concat)
                feats_se = self.equi_downsample[i](feats_se)
                feats_fuse.append((feats_se)* 0.3 * (1/3)**i)

            fpmap = self.model_up.forward_from_feats(down_view1, down_view2, feats_fuse)
            fpmap_list.append(fpmap)

        fpmap = torch.cat(fpmap_list)
        fcmap1 = torch.cat(fcmap1_list)
        fcmap2 = torch.cat(fcmap2_list)
        for val in [fpmap, fcmap1, fcmap2]:
            if torch.any(torch.isnan(val)):
                import pdb; pdb.set_trace()
                print('fpmap output is NaN')

        return fpmap, fcmap1, fcmap2

