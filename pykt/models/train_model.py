import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config import que_type_models,needs_uid_models
import pandas as pd
import torch.nn.functional as F
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pykt.models.long_dkt import StudentHiddenStateManager
#成对排序损失函数（近似AUC计算）
def pairwise_ranking_loss(predictions, targets):
    """
    计算近似AUC的成对排序损失
    
    参数:
        predictions: 模型预测值 (形状: [N])
        targets: 真实标签 (形状: [N])
        
    返回:
        成对排序损失值
    """
    # 获取正负样本索引
    pos_indices = torch.where(targets == 1)[0]
    neg_indices = torch.where(targets == 0)[0]
    
    # 如果没有正样本或负样本，返回0损失
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    # 创建所有正负样本对
    pos_preds = predictions[pos_indices]
    neg_preds = predictions[neg_indices]
    
    # 计算所有正负样本对之间的差异
    diff = neg_preds.view(-1, 1) - pos_preds.view(1, -1)  # shape: (num_neg, num_pos)
    
    # 计算损失
    loss = F.softplus(diff).mean()
    return loss

# 学生AUC差异计算函数
def student_auc_discrepancy(predictions, targets, uids, weights=None):
    """
    计算学生间的AUC差异
    
    参数:
        predictions: 模型预测值 (形状: [B, L])
        targets: 真实标签 (形状: [B, L])
        uids: 学生ID (形状: [B])
        weights: 不同学生对的权重 (可选)
        
    返回:
        学生AUC差异的加权平均值
    """
    # 获取唯一的学生ID
    unique_uids = torch.unique(uids)
    student_aucs = []
    
    # 如果批次中只有一个学生，直接返回0
    if len(unique_uids) <= 1:
        return torch.tensor(0.0, device=predictions.device)
    
    # 为每个学生计算近似AUC（成对排序损失）
    for uid in unique_uids:
        # 获取当前学生的预测和标签
        mask = (uids == uid)
        stud_preds = predictions[mask].view(-1)
        stud_targets = targets[mask].view(-1)
        
        # 计算该学生的成对排序损失（近似负AUC）
        auc_loss = pairwise_ranking_loss(stud_preds, stud_targets)
        student_aucs.append(auc_loss)
    
    student_aucs = torch.stack(student_aucs)
    
    # 计算两两学生之间的AUC差异
    diff_tensor = torch.abs(student_aucs.view(-1, 1) - student_aucs.view(1, -1))
    
    # 提取上三角矩阵（不包括对角线）
    triu_mask = torch.triu(torch.ones_like(diff_tensor), diagonal=1).bool()
    pair_diffs = diff_tensor[triu_mask]
    
    # 如果没有有效的差异对，返回0
    if len(pair_diffs) == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    # 使用权重（如果提供）
    if weights is not None and weights.shape == diff_tensor.shape:
        pair_weights = weights[triu_mask]
        return (pair_diffs * pair_weights).sum() / pair_weights.sum()
    
    # 计算平均差异
    return pair_diffs.mean()

def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    model_name = model.model_name
    # print(f"[DEBUG] ys.shape: {ys} )")
    if model_name in ["atdkt", "simplekt", "stablekt", "bakt_time", "sparsekt", "cskt", "hcgkt", "dbakt", "abqr"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # print(f"loss1: {y.shape}")
        loss1 = binary_cross_entropy(y.double(), t.double())

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1
    elif model_name in ["rekt"]:
        # print("ys shape:", ys[0].shape)
        # print("sm shape:", sm.shape)
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())
    
    elif model_name in ["ukt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss1 = binary_cross_entropy(y.double(), t.double())
        if model.use_CL:
            loss2 = ys[1]
            loss1 = loss1 + model.cl_weight * loss2
        loss =loss1

    elif model_name in ["rkt","dimkt","dkt", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes", "mamba_atakt", "mamba_atakt", "long_dkt", "at_dkt", "TransformerKT", "mult_dataset_dkt", "hawkes_lstm", "hawkes_mamba", "mamba_dkt", "mamba_hawkes_dkt"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())
    elif model_name in ["balance_dkt"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        c_loss = binary_cross_entropy(y.double(), t.double())
        print(f"c_loss{c_loss}+preloss{preloss[0]}")
        loss = c_loss+ preloss[0]
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y_next.double(), r_next.double())

        loss_r = binary_cross_entropy(y_curr.double(), r_curr.double()) # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
    elif model_name in ["akt","extrakt","folibikt", "robustkt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","lefokt_akt", "dtransformer", "fluckt",  "Transformer_template", "balance_akt", "qwen", "multi_dataset_akt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        criterion = nn.BCELoss(reduction='none')        
        loss = criterion(y, t).sum()
    
    return loss


def model_forward(model, data, rel=None):
    model_name = model.model_name
    
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget", "bakt_time", "dbakt"]:
        dcur, dgaps = data
    else:
        dcur = data
    if model_name in ["dimkt"]:
        q, c, r, t,sd,qd = dcur["qseqs"].to(device, non_blocking=True), dcur["cseqs"].to(device, non_blocking=True), dcur["rseqs"].to(device, non_blocking=True), dcur["tseqs"].to(device, non_blocking=True),dcur["sdseqs"].to(device, non_blocking=True),dcur["qdseqs"].to(device, non_blocking=True)
        qshft, cshft, rshft, tshft,sdshft,qdshft = dcur["shft_qseqs"].to(device, non_blocking=True), dcur["shft_cseqs"].to(device, non_blocking=True), dcur["shft_rseqs"].to(device, non_blocking=True), dcur["shft_tseqs"].to(device, non_blocking=True),dcur["shft_sdseqs"].to(device, non_blocking=True),dcur["shft_qdseqs"].to(device, non_blocking=True)
    else:
        q, c, r, t = dcur["qseqs"].to(device, non_blocking=True), dcur["cseqs"].to(device, non_blocking=True), dcur["rseqs"].to(device, non_blocking=True), dcur["tseqs"].to(device, non_blocking=True)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device, non_blocking=True), dcur["shft_cseqs"].to(device, non_blocking=True), dcur["shft_rseqs"].to(device, non_blocking=True), dcur["shft_tseqs"].to(device, non_blocking=True)
        
    m, sm = dcur["masks"].to(device, non_blocking=True), dcur["smasks"].to(device, non_blocking=True)
    # if model_name in needs_uid_models:
    #     uid = dcur["uid"].to(device, non_blocking=True)  # 提取 uid 并送到设备
    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    if model_name in ["hawkes", "hawkes_lstm", "hawkes_mamba", "mamba_hawkes_dkt"]:
        ct = torch.cat((t[:,0:1], tshft), dim=1)
    elif model_name in ["rkt"]:
        y, attn = model(dcur, rel, train=True)
        ys.append(y[:,1:])
    if model_name in ["atdkt"]:
        # is_repeat = dcur["is_repeat"]
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3] # first: yshft
    elif model_name in ["simplekt", "stablekt", "sparsekt", "cskt"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["rekt"]:
        y = model(dcur, train=True)
        ys = [y]
    elif model_name in ["ukt"]:
        if model.use_CL != 0 :
            y, sim, y2, y3, temp = model(dcur, train=True)
            ys = [y[:,1:],sim,y2, y3]
        else:
            y, y2, y3 = model(dcur, train=True)
            ys = [y[:,1:], y2, y3]
    elif model_name in ["hcgkt"]:
        
        step_size = model.step_size
        step_m = model.step_m
        grad_clip = model.grad_clip
        mm = model.mm

        # the xxx.pt file of pre_load_gcn can be found in :
        # https://drive.google.com/drive/folders/1JWstsquI3TzbUlqB1EyCbjem4qPyRLCh?usp=drive_link
        matrix = None
        if dataset_name == 'assist2009':
            pre_load_gcn = "../data/assist2009/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'algebra2005':
            pre_load_gcn = "../data/algebra2005/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'bridge2algebra2006':
            pre_load_gcn = "../data/bridge2algebra2006/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'peiyou':
            pre_load_gcn = "../data/peiyou/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'nips_task34':
            pre_load_gcn = "../data/nips_task34/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        perturb_shape = (matrix.shape[0], emb_size)
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device, non_blocking=True)
        perturb.requires_grad_()
        y, y2, y3, contrast_loss = model(dcur, train=True, perb=perturb)
        ys = [y[:,1:], y2, y3]
        loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
        loss /= step_m
        opt.zero_grad()
        for _ in range(step_m - 1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            y, y2, y3, contrast_loss = model(dcur, train=True, perb=perturb)
            ys = [y[:,1:], y2, y3]
            loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
            loss /= step_m
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        model.sfm_cl.gcl.update_target_network(mm)  
        return loss
    elif model_name in ["abqr"]:
        opt = rel
        step_size = model.step_size
        step_m = model.step_m
        grad_clip = model.grad_clip
        mm = model.mm

        # the xxx.pt file of pre_load_gcn can be found in :
        # https://drive.google.com/drive/folders/1JWstsquI3TzbUlqB1EyCbjem4qPyRLCh?usp=drive_link
        
        perturb_shape = (model.matrix.shape[0], model.emb_size)
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device, non_blocking=True)
        perturb.requires_grad_()
        y, y2, y3, contrast_loss = model(dcur, train=True, perb=perturb)
        
        
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y]
        
        loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
        loss /= step_m
        opt.zero_grad()
        for _ in range(step_m - 1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            y, y2, y3, contrast_loss = model(dcur, train=True, perb=perturb)
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            ys = [y]
            loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
            loss /= step_m
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        model.gcl.update_target_network(mm)  
        return loss
    elif model_name in ["dtransformer"]:
        if model.emb_type == "qid_cl":
            y, loss = model.get_cl_loss(cc.long(), cr.long(), cq.long())  # with cl loss
        else:
            y, loss = model.get_loss(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(loss)
    elif model_name in ["bakt_time", "dbakt"]:
        y, y2, y3 = model(dcur, dgaps, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_itseqs"]), dim=1)
    if model_name in ["dkt", "mult_dataset_dkt", "mamba_dkt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y) # first: yshft
    elif model_name in ["long_dkt"]:
        uids = dcur["uid"].to(device, non_blocking=True)
        # print(f"uids：{uids}")
        
        # 如果提供了学生状态管理器，则使用持久化的隐藏状态
        student_state_manager = rel
        if student_state_manager is not None:
            # 获取当前批次学生的隐藏状态
            initial_states = student_state_manager.get_student_states(uids)
            
            # 调用模型时传入初始隐藏状态
            y, final_states = model(c.long(), r.long(), initial_states)
            
            # 更新学生的隐藏状态
            student_state_manager.update_student_states(uids, final_states[0], final_states[1])
        else:
            # 如果没有状态管理器，使用原来的方式
            y = model(c.long(), r.long())
        
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y) # first: yshft
    elif model_name in ["balance_akt"]:
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        # 提取真实标签和学生ID
        targets = rshft
        # 应用sigmoid获取概率值
        probas = y[:,1:]  # 已经通过Sigmoid
        
        # print(dcur.keys()) 
        uids = dcur["uid"].to(device, non_blocking=True)
        masks = sm
        # 获取当前批次的学生权重（假设权重存储在dcur中）
        weights = None
        # if "student_weights" in dcur:
        #     weights = dcur["student_weights"].to(device, non_blocking=True)
        
        # 计算学生间AUC差异
        auc_discrepancy = student_auc_discrepancy(
            probas, 
            targets, 
            uids.unsqueeze(-1).expand(-1, probas.size(1)).contiguous(),
            weights
        )
        
        # 计算整体批次AUC（使用成对排序损失作为负AUC的近似）
        batch_preds = torch.masked_select(probas, masks)
        batch_targets = torch.masked_select(targets, masks)
        batch_auc_neg = pairwise_ranking_loss(batch_preds, batch_targets)
        
        # 综合差异和整体AUC
        # 公式: λ * |差异 - (整体AUC + ε)|
        # 鼓励学生间的AUC差异接近整体批次AUC
        regularization = model.reg_lambda * torch.abs(
            auc_discrepancy - (batch_auc_neg + model.reg_epsilon)
        )
        
        # 打印调试信息（可选）
        if model.training:
            print(
                  f"AUC discrepancy: {auc_discrepancy.item():.4f}, "
                  f"Batch AUC neg: {batch_auc_neg.item():.4f}, "
                  f"Regularization: {regularization.item():.4f}")
        
        # 总损失 = 原始损失 + 正则化项
        preloss.append(reg_loss+regularization)

    elif model_name in ["balance_dkt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)  # yshft存储模型预测结果
        
        # 应用sigmoid获取概率值
        probas = torch.sigmoid(y)
        
        # 提取真实标签和学生ID
        targets = rshft
        # print(dcur.keys()) 
        uids = dcur["uid"].to(device, non_blocking=True)
        masks = sm
        
        # 计算原始损失
        
        
        # 获取当前批次的学生权重（假设权重存储在dcur中）
        weights = None
        # if "student_weights" in dcur:
        #     weights = dcur["student_weights"].to(device, non_blocking=True)
        
        # 计算学生间AUC差异
        auc_discrepancy = student_auc_discrepancy(
            probas, 
            targets, 
            uids.unsqueeze(-1).expand(-1, probas.size(1)).contiguous(),
            weights
        )
        
        # 计算整体批次AUC（使用成对排序损失作为负AUC的近似）
        batch_preds = torch.masked_select(probas, masks)
        batch_targets = torch.masked_select(targets, masks)
        batch_auc_neg = pairwise_ranking_loss(batch_preds, batch_targets)
        
        # 综合差异和整体AUC
        # 公式: λ * |差异 - (整体AUC + ε)|
        # 鼓励学生间的AUC差异接近整体批次AUC
        regularization = model.reg_lambda * torch.abs(
            auc_discrepancy - (batch_auc_neg + model.reg_epsilon)
        )
        
        # 打印调试信息（可选）
        if model.training:
            print(
                  f"AUC discrepancy: {auc_discrepancy.item():.4f}, "
                  f"Batch AUC neg: {batch_auc_neg.item():.4f}, "
                  f"reg_lamda: {model.reg_lambda},"
                  f"Regularization: {regularization.item():.4f}")
        
        # 总损失 = 原始损失 + 正则化项
        preloss.append(regularization)
        # return loss
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn","deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:,1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["akt","extrakt","folibikt", "robustkt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx", "lefokt_akt", "fluckt",  "Transformer_template", "qwen", "multi_dataset_akt"]:               
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name in ["TransformerKT"]:               
        y = model(cc.long(), cr.long(), cq.long())
        print(f"[DEBUG] y.shape: {y.shape} (type: {type(y.shape)})")
        ys.append(y[:,1:])
    elif model_name in ["atkt", "atktfix", "mamba_atakt", "mamba_atakt", "at_dkt"]:
        # 常规前向计算
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)
        
        # 对抗训练部分
        if (model.emb_type.startswith("at") and model_name in ["at_dkt"]) or model_name not in ["at_dkt"]:
            if not model.emb_type.endswith("pgd"):
                # FGSM对抗训练
                features_grad = grad(loss, features, retain_graph=True)
                p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
                p_adv = Variable(p_adv).to(device, non_blocking=True)
                pred_res, _ = model(c.long(), r.long(), p_adv)
                pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
                adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
                loss = loss + model.beta * adv_loss
            else:
                # PGD对抗训练
                perturbation = model(c.long(), r.long(), pgd_attack=True, 
                                loss_fn=lambda pred: cal_loss(model, [pred], r, rshft, sm),cshft=cshft)
                final_adv_output, _ = model(c.long(), r.long(), perturbation)
                final_adv_output = (final_adv_output * one_hot(cshft.long(), model.num_c)).sum(-1)
                final_adv_loss = cal_loss(model, [final_adv_output], r, rshft, sm)
                loss = loss + model.beta * final_adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)  
    # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])  
    elif model_name in ["hawkes", "hawkes_lstm", "hawkes_mamba", "mamba_hawkes_dkt"]:
        # ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
        # y = model(cc[0:1,0:5].long(), cq[0:1,0:5].long(), ct[0:1,0:5].long(), cr[0:1,0:5].long(), csm[0:1,0:5].long())
        y = model(cc.long(), cq.long(), ct.long(), cr.long())#, csm.long())
        ys.append(y[:, 1:])
    elif model_name in que_type_models and model_name not in ["lpkt", "rkt"]:
        y,loss = model.train_one_step(data)
    elif model_name == "dimkt":
        y = model(q.long(),c.long(),sd.long(),qd.long(),r.long(),qshft.long(),cshft.long(),sdshft.long(),qdshft.long())
        ys.append(y) 

    if model_name not in ["atkt", "atktfix","mamba_atakt", "at_dkt","abqr"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    if model_name in ["ukt"] and model.use_CL != 0:
        return loss,temp
    return loss


def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None,use_trained=0, accumulation_steps=1):
    start_train_time = time.time()

    max_auc, best_epoch = 0, -1
    train_step = 0

    model_path = os.path.join(ckpt_path, model.emb_type + "_model.ckpt")
    if os.path.exists(model_path) and use_trained==1:
        print(f"检测到预训练模型文件，正在从 {model_path} 加载...")
        model.load_state_dict(torch.load(model_path))
        print("模型加载成功！")
    elif os.path.exists(model_path) and use_trained==0:
        print(f"检测到预训练模型文件，但use_trained：{use_trained}，将从头开始训练。")
    else:
        print("未找到预训练模型文件，将从头开始训练。")
    # 为long_dkt创建学生隐藏状态管理器
    student_state_manager = None
    if model.model_name == "long_dkt":
    
        hidden_size = getattr(model, 'emb_size')  # 默认256
        num_layers = getattr(model, 'num_layers',1)      # 默认1层
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]  # 需要在data_config中添加dataset_name
        
        student_state_manager = StudentHiddenStateManager(
            data_config_path="../configs/data_config.json",

            hidden_size=hidden_size,
            num_layers=num_layers,
            mode = "train"
        )
    rel = None
    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))

    if model.model_name=='lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)

    for i in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        loss_mean = []
        train_phase_start = time.time()

        for batch_idx, data in enumerate(train_loader):
            train_step+=1
            if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                model.model.train()
            else:
                model.train()
            if model.model_name=='rkt':
                loss = model_forward(model, data, rel)
            elif model.model_name in ["ukt"] and model.use_CL != 0:
                loss,temp = model_forward(model, data)
            elif model.model_name == "long_dkt":
                loss = model_forward(model, data,student_state_manager)
            elif model.model_name == "abqr":
                loss = model_forward(model, data,opt)
            else:
                loss = model_forward(model, data)
            if model.model_name not in ["hcgkt","abqr"]:

                opt.zero_grad()
                loss_scaled = loss / accumulation_steps
                loss_scaled.backward()

                #debug 强化学习验证
                # if model.model_name.startswith("qikt_iekt_dual_gae") and (batch_idx % 100 == 0):
                #     print(f"\n[Gradient Check] Batch: {batch_idx}")
                #     print(f"Model keys: {model.model.__dict__.keys()}")
                #     # 这里的 model 实际上是 QIKT_IEKT_DUAL_GAE，里面的 RL 组件在 model.model 属性中
                #     # 1. 检查 Question 分支的 Actor
                #     if hasattr(model.model, 'policy_net_q'):
                #         for name, param in model.model.policy_net_q.named_parameters():
                #             if param.grad is not None:
                #                 print(f"  Q-Actor Grad ({name}): {param.grad.abs().mean().item():.8f}")
                #             else:
                #                 print(f"  !!! Q-Actor Grad is NONE ({name}) !!!")
                                
                #     # 2. 检查 Question 分支的 Critic
                #     if hasattr(model.model, 'critic_net_q'):
                #         for name, param in model.model.critic_net_q.named_parameters():
                #             if param.grad is not None:
                #                 print(f"  Q-Critic Grad ({name}): {param.grad.abs().mean().item():.8f}")
                                
                #     # 3. 检查 Concept 分支
                #     if hasattr(model.model, 'policy_net_c'):
                #         p_grad = model.model.policy_net_c[0].weight.grad # 取第一层示例
                #         if p_grad is not None:
                #             print(f"  C-Actor Grad: {p_grad.abs().mean().item():.8f}")
                # --- END: 梯度检查排查位置 ---


                if (batch_idx + 1) % accumulation_steps == 0:
                    # 梯度裁剪 (针对不同模型)
                    if model.model_name == "rkt":
                        clip_grad_norm_(model.parameters(), model.grad_clip)
                    if model.model_name == "dtransformer":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()      # 更新参数
                # # loss.backward()#compute gradients
                # if model.model_name == "rkt":
                #     clip_grad_norm_(model.parameters(), model.grad_clip)
                # if model.model_name == "dtransformer":
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # opt.step()#update model’s parameters
                
            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name == "gkt" and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")


        train_phase_end = time.time()
        if model.model_name not in ["hcgkt", "abqr"] and (batch_idx + 1) % accumulation_steps != 0:
            if model.model_name == "rkt":
                clip_grad_norm_(model.parameters(), model.grad_clip)
            if model.model_name == "dtransformer":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        
        loss_mean = np.mean(loss_mean)
        
        val_phase_start = time.time()


        if model.model_name=='rkt':
            auc, acc = evaluate(model, valid_loader, model.model_name, rel)
        else:
            auc, acc = evaluate(model, valid_loader, model.model_name)
        if model.model_name=='lpkt':
            scheduler.step()#update each epoch

        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)
        val_phase_end = time.time()
        if auc > max_auc+1e-3:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            validauc, validacc = auc, acc
        print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        train_duration = train_phase_end - train_phase_start
        val_duration = val_phase_end - val_phase_start
        
        print(f"")
        print(f"Total Epoch Time: {epoch_duration:.2f}s | Train Phase: {train_duration:.2f}s | Valid Phase: {val_duration:.2f}s")
        print(f"Avg Batch Time: {train_duration / len(train_loader):.4f}s")
        print("-" * 30)

        if i - best_epoch >= 10:
            print(f"Early stopping at epoch {i}")
            break
    total_duration = time.time() - start_train_time # 5. 任务总耗时
    print(f"✅ Training Finished! Total Time Cost: {total_duration:.2f} s")
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
