import torch
import torch.nn as nn
import torch.nn.functional as F



def simplify_feature_map(feature_map, output_size=2):
    return feature_map



def convert_to_distribution(feature_map, epsilon=1e-8):
    N, C, H, W = feature_map.size()
    feature_map_reshaped = feature_map.view(N, C, -1)


    distribution_log = F.log_softmax(feature_map_reshaped, dim=-1)
    return distribution_log.view(N, C, H, W)



def compute_kl_loss(teacher_dist_log, student_dist_log):
    kl_loss = F.kl_div(student_dist_log, teacher_dist_log.exp(), reduction='batchmean')
    return kl_loss



class GaussianLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(GaussianLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, teacher_feat, student_feat):
        teacher_mean = teacher_feat.mean(dim=(2, 3), keepdim=True)
        student_mean = student_feat.mean(dim=(2, 3), keepdim=True)

        teacher_var = teacher_feat.var(dim=(2, 3), keepdim=True, unbiased=False) + self.epsilon
        student_var = student_feat.var(dim=(2, 3), keepdim=True, unbiased=False) + self.epsilon


        mean_loss = F.mse_loss(teacher_mean, student_mean)
        var_loss = F.mse_loss(teacher_var, student_var)
        gaussian_loss = mean_loss + var_loss
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



class FAK(nn.Module):
    def __init__(self, output_size=2, epsilon=1e-8):
        super(FAK, self).__init__()
        self.output_size = output_size
        self.epsilon = epsilon
        self.gaussian_loss_fn = GaussianLoss(epsilon=self.epsilon)

    def forward(self, teacher_feats, student_feats):
        num_layers = len(teacher_feats)
        attention_weights = compute_attention_weights(teacher_feats, student_feats)
        total_loss = 0.0

        for i in range(num_layers):

            simplified_teacher_feat = simplify_feature_map(teacher_feats[i], self.output_size)
            simplified_student_feat = simplify_feature_map(student_feats[i], self.output_size)


            teacher_dist_log = convert_to_distribution(simplified_teacher_feat, self.epsilon)
            student_dist_log = convert_to_distribution(simplified_student_feat, self.epsilon)


            kl_loss = compute_kl_loss(teacher_dist_log, student_dist_log)


            gaussian_loss = self.gaussian_loss_fn(teacher_feats[i], student_feats[i])


            total_loss += attention_weights[i] * (kl_loss + gaussian_loss)

        return total_loss
