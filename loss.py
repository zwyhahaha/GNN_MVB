import torch
import torch.nn.functional as F
from torchmetrics import F1Score, Precision, Recall, AveragePrecision,  Accuracy
import numpy as np
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS = torch.tensor(1e-6).to(DEVICE)
zero = torch.tensor(0).to(DEVICE)
one = torch.tensor(1).to(DEVICE)
f1 = F1Score(average="macro", task='binary').to(DEVICE)
pr = Precision(average="macro", task='binary').to(DEVICE)
re = Recall(average="macro", task='binary').to(DEVICE)
avg_pr = AveragePrecision(task="binary").to(DEVICE)
acc = Accuracy(task='binary').to(DEVICE)

def one_hot_embedding(labels, num_classes=2):
    y = torch.eye(num_classes, device=DEVICE)
    return y[labels]

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)# + 1.0

evidence_funcs = {'relu': relu_evidence, 'exp': exp_evidence, 'softplus':softplus_evidence} 
 
def kl_divergence(alpha, num_classes, device=DEVICE):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=DEVICE)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, lambda_, device=DEVICE):
    y = y.to(DEVICE)
    alpha = alpha.to(DEVICE)
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div =  kl_divergence(kl_alpha, num_classes, device=DEVICE)

    return A + lambda_ * kl_div, A, kl_div

def loglikelihood_loss(y, alpha, device=DEVICE):

    S = torch.sum(alpha, dim=1, keepdim=True)
   
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var

    return loglikelihood

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, lambda_, device=DEVICE):
  
    loglikelihood = loglikelihood_loss(y, alpha, device=DEVICE)
    
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div =  kl_divergence(kl_alpha, num_classes, device=DEVICE)

    return loglikelihood + lambda_ * kl_div, loglikelihood, kl_div

def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, lambda_, evidence_func=softplus_evidence, device=DEVICE):

    evidence = evidence_func(output)
    alpha = evidence + 1

    loss, A, kl_div = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, lambda_, DEVICE)

    return loss.mean(), A.mean(), kl_div.mean()

def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, lambda_, evidence_func=softplus_evidence, device=DEVICE):
    
    evidence = evidence_func(output)
    alpha = evidence + 1

    loss, A, kl_div = edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, lambda_, DEVICE)

    return loss.mean(), A.mean(), kl_div.mean()

def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, lambda_, evidence_func=softplus_evidence, device=DEVICE):
    
    evidence = evidence_func(output)
    alpha = evidence + 1

    loss, logll, kl_div = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, lambda_, device=DEVICE)
    
    return loss.mean(), logll.mean(), kl_div.mean()

loss_funcs = {'edl_digamma': edl_digamma_loss, 
              'edl_log':edl_log_loss,
              'edl_mse':edl_mse_loss,
              'bce': torch.nn.CrossEntropyLoss(),
              'mse':torch.nn.MSELoss()}

def get_evidence_and_uncertainty(outputs, preds, labels, num_classes=2, evidence_func=softplus_evidence):
    
    match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
    evidence = evidence_func(outputs)
    alpha = evidence + 1
    uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    mean_uncertainy_succ = torch.mean(uncertainty[match==1.0]) if torch.sum(match==1.0) > 0 else torch.tensor(0.5)
    mean_uncertainy_fail = torch.mean(uncertainty[match==0.0]) if torch.sum(match==0.0) > 0 else torch.tensor(0.5)

    total_evidence = torch.sum(evidence, 1, keepdim=True)
    mean_evidence = torch.mean(total_evidence)

    mean_evidence_succ = torch.sum(total_evidence * match) / (torch.sum(match) + EPS)
    mean_evidence_fail = torch.sum(total_evidence * (1 - match)) / (torch.sum(1 - match) + EPS)

    return [mean_evidence_succ.item(), mean_evidence_fail.item(), mean_uncertainy_succ.mean().item(), mean_uncertainy_fail.mean().item()], uncertainty

def scoring(batch_idx, loss, bias_tuples, evidence_tuples, uncertainty, pred, y, lr, bias_threshold, step_type, print_log):
   
    pred = torch.where(pred.view(-1) <= bias_threshold, zero, one)
    y = torch.where(y.view(-1) <= bias_threshold, zero, one)

    bias_tuple = torch.mean(torch.tensor(bias_tuples), dim=0)
    evidence_tuple = torch.mean(torch.tensor(evidence_tuples), dim=0)
    
    metric_names = [step_type + "_" + metric 
                    for metric in 
                    ["loss", "acc", "f1", "precision", "recall",
                    "evidence_succ",  "evidence_fail", "uncertainty_succ", "uncertainty_fail", \
                    "true_bias", 'pred_bias', 'soft_pred_bias', 'bias_error', 'lr']]
    
    metrics = torch.tensor([(loss/(batch_idx+1)), acc(pred, y), f1(pred, y), pr(pred, y), re(pred, y), \
                            *evidence_tuple, *bias_tuple, lr]).cpu().numpy()
    
    assert len(metric_names) == len(metrics)
    
    log = dict(zip(metric_names, metrics))
    
    if print_log:
        print(pd.Series(log).round(8))

    return log

class LossHandler(torch.nn.Module):
    def __init__(self, edl_lambda, evidence_func, loss_type, n_steps):
        super(LossHandler, self).__init__()
        self.edl_lambda = edl_lambda
        self.evidence_func = evidence_funcs[evidence_func] if evidence_func else exp_evidence
        self.loss_type = loss_type
        self.loss_func = loss_funcs[loss_type]
        self.n_steps = n_steps

    def forward(self, global_step, graph_idx, batch, output, y, binary_pred, step_type):
    
        target_output = output[batch.is_binary] if binary_pred else output
        target_y = y[batch.is_binary] if binary_pred else y

        if self.loss_type == "bce":
            loss = self.loss_func(target_output, target_y)
        else:
            loss, A, kl_div = self.loss_func(target_output, one_hot_embedding(target_y), global_step, 2, global_step/self.n_steps, self.edl_lambda, self.evidence_func)

        discrete_pred = torch.argmax(target_output, -1)
        evidence_tuple, uncertainty = get_evidence_and_uncertainty(target_output, discrete_pred.view(-1,1), target_y.view(-1,1), 2, self.evidence_func)
        
        return loss, evidence_tuple, uncertainty