import torch
import torch.nn as nn
from einops.einops import rearrange
import sys
import os 
base_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = "/".join(base_dir.split("/")[:-2])
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src/loftr"))
sys.path.append(os.path.join(root_dir, "src/loftr/utils"))
from resnet_new import resnet50, resnet18
from backbone import build_backbone
from position_encoding import PositionEncodingSine
from loftr_module import LocalFeatureTransformer
from src.config.default import get_cfg_defaults
from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        
        self.backbone = resnet18(pretrained=True)
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])

        num_input_channel = config['predict']['input_c']
        self.layer1 = nn.Conv2d(num_input_channel, num_input_channel // 4, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(num_input_channel // 4)
        self.layer2 = nn.Conv1d(num_input_channel // 4, num_input_channel // 16, kernel_size=3, stride=3, padding=0)
        self.bn2 = nn.BatchNorm1d(num_input_channel // 16)
        self.layer3 = nn.Conv1d(num_input_channel // 16, 1, kernel_size=3, stride=3, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(config['predict']['fc_in'], 4)

    def forward(self, data):
        # 1. Local Feature CNN
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

