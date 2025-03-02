import torch
import torch.nn as nn
from .clip import clip_feature_surgery
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, InterpolationMode


img_resize = Compose([
    Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
])


def get_clip_score_from_feature(model, image, text_features, temp=100.):
    # size of image: [b, 3, 224, 224]
    image = img_resize(image)
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    probs = temp * clip_feature_surgery(image_features, text_features)[:, 1:, :]
    similarity = torch.mean(probs.softmax(dim=-1), dim=1, keepdim=False)
    loss = 1. - similarity[:, 0]
    loss = torch.sum(loss) / len(loss)
    return loss


class L_clip_from_feature(nn.Module):
    def __init__(self, temp=100.):
        super(L_clip_from_feature, self).__init__()
        self.temp = temp
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, model, x, text_features):
        k1 = get_clip_score_from_feature(model, x, text_features, self.temp)
        return k1


def get_clip_score_MSE(res_model, pred, inp, weight):
    stack = img_resize(torch.cat([pred, inp], dim=1))
    pred_image_features = res_model.encode_image(stack[:, :3, :, :])
    inp_image_features = res_model.encode_image(stack[:, 3:, :, :])

    MSE_loss = 0
    for feature_index in range(len(weight)):
        MSE_loss = MSE_loss + weight[feature_index] * F.mse_loss(pred_image_features[1][feature_index], inp_image_features[1][feature_index])

    return MSE_loss


class L_clip_MSE(nn.Module):
    def __init__(self):
        super(L_clip_MSE, self).__init__()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, model, pred, inp, weight=None):
        if weight is None:
            weight = [1.0, 1.0, 0.8, 0.5, 0.3]
        res = get_clip_score_MSE(model, pred, inp, weight)
        return res
