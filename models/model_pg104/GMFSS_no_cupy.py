import torch
import torch.nn.functional as F

from models.gmflow.gmflow import GMFlow
from models.model_pg104.IFNet_HDv3 import IFNet
# from models.model_pg104.MetricNet import MetricNet
from models.model_pg104.FeatureNet import FeatureNet
from models.model_pg104.FusionNet import GridNet
# from models.model_pg104.softsplat import softsplat as warp
from models.model_pg104.forward_warp2 import ForwardWarp

device = torch.device("cuda")


class Model:
    def __init__(self):
        self.flownet = GMFlow()
        self.ifnet = IFNet()
        # self.metricnet = MetricNet()
        self.fwarp = ForwardWarp()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet(9, 64 * 2, 128 * 2, 192 * 2, 3)
        self.version = 3.9

    def half(self):
        self.flownet.half()
        self.ifnet.half()
        # self.metricnet.half()
        self.fwarp.half()
        self.feat_ext.half()
        self.fusionnet.half()

    def eval(self):
        self.flownet.eval()
        self.ifnet.eval()
        # self.metricnet.eval()
        self.feat_ext.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.ifnet.to(device)
        # self.metricnet.to(device)
        self.feat_ext.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))
        self.ifnet.load_state_dict(torch.load('{}/rife.pkl'.format(path)))
        # self.metricnet.load_state_dict(torch.load('{}/metric.pkl'.format(path)))
        self.feat_ext.load_state_dict(torch.load('{}/feat.pkl'.format(path)))
        self.fusionnet.load_state_dict(torch.load('{}/fusionnet.pkl'.format(path)))

    def reuse(self, img0, img1, scale):
        feat11, feat12, feat13 = self.feat_ext(img0)
        feat21, feat22, feat23 = self.feat_ext(img1)
        feat_ext0 = [feat11, feat12, feat13]
        feat_ext1 = [feat21, feat22, feat23]

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)

        if scale != 1.0:
            imgf0 = F.interpolate(img0, scale_factor=scale, mode="bilinear", align_corners=False)
            imgf1 = F.interpolate(img1, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            imgf0 = img0
            imgf1 = img1
        flow01 = self.flownet(imgf0, imgf1)
        flow10 = self.flownet(imgf1, imgf0)
        if scale != 1.0:
            flow01 = F.interpolate(flow01, scale_factor=1. / scale, mode="bilinear", align_corners=False) / scale
            flow10 = F.interpolate(flow10, scale_factor=1. / scale, mode="bilinear", align_corners=False) / scale

        # metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)

        # return flow01, flow10, metric0, metric1, feat_ext0, feat_ext1
        return flow01, flow10, feat_ext0, feat_ext1

    def inference(self, img0, img1, reuse_things, timestep, last=None):
        # flow01, metric0, feat11, feat12, feat13 = reuse_things[0], reuse_things[2], reuse_things[4][0], reuse_things[4][
        #     1], reuse_things[4][2]
        # flow10, metric1, feat21, feat22, feat23 = reuse_things[1], reuse_things[3], reuse_things[5][0], reuse_things[5][
        #     1], reuse_things[5][2]

        flow01, feat11, feat12, feat13 = reuse_things[0], reuse_things[2][0], reuse_things[2][1], reuse_things[2][2]
        flow10, feat21, feat22, feat23 = reuse_things[1], reuse_things[3][0], reuse_things[3][1], reuse_things[3][2]

        F1t = timestep * flow01
        F2t = (1 - timestep) * flow10

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        # I1t = warp(img0, F1t, Z1t, strMode='soft')
        I1t = self.fwarp(img0, F1t)
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
        # I2t = warp(img1, F2t, Z2t, strMode='soft')
        I2t = self.fwarp(img1, F2t)

        imgs = torch.cat((img0, img1), 1)
        rife = self.ifnet(imgs, timestep, scale_list=[8, 4, 2, 1])

        # feat1t1 = warp(feat11, F1t, Z1t, strMode='soft')
        # feat2t1 = warp(feat21, F2t, Z2t, strMode='soft')
        feat1t1 = self.fwarp(feat11, F1t)
        feat2t1 = self.fwarp(feat21, F2t)

        F1td = F.interpolate(F1t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        # Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear", align_corners=False)
        # feat1t2 = warp(feat12, F1td, Z1d, strMode='soft')
        feat1t2 = self.fwarp(feat12, F1td)
        F2td = F.interpolate(F2t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        # Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear", align_corners=False)
        # feat2t2 = warp(feat22, F2td, Z2d, strMode='soft')
        feat2t2 = self.fwarp(feat22, F2td)

        F1tdd = F.interpolate(F1t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        # Z1dd = F.interpolate(Z1t, scale_factor=0.25, mode="bilinear", align_corners=False)
        # feat1t3 = warp(feat13, F1tdd, Z1dd, strMode='soft')
        feat1t3 = self.fwarp(feat13, F1tdd)
        F2tdd = F.interpolate(F2t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        # Z2dd = F.interpolate(Z2t, scale_factor=0.25, mode="bilinear", align_corners=False)
        # feat2t3 = warp(feat23, F2tdd, Z2dd, strMode='soft')
        feat2t3 = self.fwarp(feat23, F2tdd)

        out = self.fusionnet(torch.cat([I1t, rife, I2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1),
                             torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

        return torch.clamp(out, 0, 1)
