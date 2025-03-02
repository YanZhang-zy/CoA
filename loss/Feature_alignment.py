import torch
import torch.nn as nn
import torch.nn.functional as F


def simplify_feature_map(feature_map, pool_size=2):
    return feature_map


def convert_to_distribution(feature_map):
    N, C, H, W = feature_map.size()
    feature_map_reshaped = feature_map.view(N, C, -1)
    distribution = F.softmax(feature_map_reshaped, dim=-1)
    return distribution.view(N, C, H, W)


def compute_emd_loss(teacher_dist, student_dist):
    emd_loss = torch.mean((teacher_dist - student_dist) ** 2)
    return emd_loss


class GaussianLoss(nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, teacher_feat, student_feat):
        teacher_mean = teacher_feat.mean(dim=(2, 3), keepdim=True)
        student_mean = student_feat.mean(dim=(2, 3), keepdim=True)
        teacher_var = teacher_feat.var(dim=(2, 3), keepdim=True, unbiased=False)
        student_var = student_feat.var(dim=(2, 3), keepdim=True, unbiased=False)
        gaussian_loss = ((teacher_mean - student_mean) ** 2).mean() + ((teacher_var - student_var) ** 2).mean()
        return gaussian_loss


def compute_attention_weights(teacher_feats, student_feats):
    attention_weights = []
    num_layers = len(teacher_feats)

    for i in range(num_layers):
        teacher_feat_flat = teacher_feats[i].view(teacher_feats[i].size(0), -1)
        student_feat_flat = student_feats[i].view(student_feats[i].size(0), -1)

        similarity = F.cosine_similarity(teacher_feat_flat, student_feat_flat, dim=-1).mean()
        attention_weights.append(similarity)

    attention_weights = torch.stack(attention_weights)
    attention_weights = F.softmax(attention_weights, dim=0)
    return attention_weights


def compute_local_emd_loss(teacher_feat, student_feat, patch_size=4):
    teacher_patches = teacher_feat.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    student_patches = student_feat.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patch_emd_loss = compute_emd_loss(teacher_patches, student_patches)
    return patch_emd_loss


class FA(nn.Module):
    def __init__(self, patch_size=4, use_local_emd=True, pool_size=2):
        super(FA, self).__init__()
        self.patch_size = patch_size
        self.use_local_emd = use_local_emd
        self.pool_size = pool_size
        self.gaussian_loss_fn = GaussianLoss()

    def forward(self, teacher_feats, student_feats):
        num_layers = len(teacher_feats)
        attention_weights = compute_attention_weights(teacher_feats, student_feats)
        total_loss = 0.0

        for i in range(num_layers):

            simplified_teacher_feat = simplify_feature_map(teacher_feats[i], self.pool_size)
            simplified_student_feat = simplify_feature_map(student_feats[i], self.pool_size)

            teacher_dist = convert_to_distribution(simplified_teacher_feat)
            student_dist = convert_to_distribution(simplified_student_feat)

            global_emd_loss = compute_emd_loss(teacher_dist, student_dist)

            if self.use_local_emd:
                local_emd_loss = compute_local_emd_loss(simplified_teacher_feat, simplified_student_feat, self.patch_size)
                emd_loss = global_emd_loss + local_emd_loss
            else:
                emd_loss = global_emd_loss

            gaussian_loss = self.gaussian_loss_fn(teacher_feats[i], student_feats[i])

            total_loss += attention_weights[i] * (emd_loss + gaussian_loss)

        return total_loss


class FeatureAlignmentLoss(nn.Module):
    def __init__(self):
        super(FeatureAlignmentLoss, self).__init__()

    def forward(self, student_features, teacher_features):

        loss = torch.mean((student_features - teacher_features) ** 2)
        return loss