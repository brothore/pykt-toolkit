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
from pykt.config import que_type_models
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if model_name in ["atdkt", "simplekt", "stablekt", "bakt_time", "sparsekt", "cskt", "hcgkt", "dbakt"]:
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

    elif model_name in ["rkt","dimkt","dkt", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes", "lstm_template", "lstm_template", "lstm_template", "lstm_template", "lstm_template", "lstm_template", "mambakt", "mambakt", "mamba_atakt", "mamba_atakt"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())
    elif model_name in ["balance_dkt"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())+ preloss[0]
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
    elif model_name in ["akt","extrakt","folibikt", "robustkt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","lefokt_akt", "dtransformer", "fluckt",  "Transformer_template", "mamba_akt", "mamba_akt"]:
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
        q, c, r, t,sd,qd = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device),dcur["sdseqs"].to(device),dcur["qdseqs"].to(device)
        qshft, cshft, rshft, tshft,sdshft,qdshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device),dcur["shft_sdseqs"].to(device),dcur["shft_qdseqs"].to(device)
    else:
        q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
        
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)
    if model_name in needs_uid_models:
        uid = dcur["uids"].to(device)  # 提取 uid 并送到设备
    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
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
        
        step_size = step_size
        step_m = step_m
        grad_clip = grad_clip
        mm = mm

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
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
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
    if model_name in ["dkt", "lstm_template", "lstm_template", "lstm_template", "lstm_template", "lstm_template", "lstm_template", "mambakt", "mambakt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y) # first: yshft
        
    elif model_name in ["balance_dkt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)  # yshft存储模型预测结果
        
        # 应用sigmoid获取概率值
        probas = torch.sigmoid(y)
        
        # 提取真实标签和学生ID
        targets = rshft
        uids = dcur["uids"].to(device)
        masks = sm
        
        # 计算原始损失
        
        
        # 获取当前批次的学生权重（假设权重存储在dcur中）
        weights = None
        # if "student_weights" in dcur:
        #     weights = dcur["student_weights"].to(device)
        
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
        preloss.append(regularization)
        return loss
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
    elif model_name in ["akt","extrakt","folibikt", "robustkt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx", "lefokt_akt", "fluckt",  "Transformer_template", "mamba_akt", "mamba_akt"]:               
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix", "mamba_atakt", "mamba_atakt"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
        loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)  
    # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])  
    elif model_name == "hawkes":
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

    if model_name not in ["atkt", "atktfix","mamba_atakt"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    if model_name in ["ukt"] and model.use_CL != 0:
        return loss,temp
    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None):
    max_auc, best_epoch = 0, -1
    train_step = 0

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
        loss_mean = []
        for data in train_loader:
            train_step+=1
            if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                model.model.train()
            else:
                model.train()
            if model.model_name=='rkt':
                loss = model_forward(model, data, rel)
            elif model.model_name in ["ukt"] and model.use_CL != 0:
                loss,temp = model_forward(model, data)
            else:
                loss = model_forward(model, data)
            opt.zero_grad()
            loss.backward()#compute gradients
            if model.model_name == "rkt":
                clip_grad_norm_(model.parameters(), model.grad_clip)
            if model.model_name == "dtransformer":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()#update model’s parameters
                
            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name == "gkt" and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")
        if model.model_name=='lpkt':
            scheduler.step()#update each epoch
        loss_mean = np.mean(loss_mean)
        
        if model.model_name=='rkt':
            auc, acc = evaluate(model, valid_loader, model.model_name, rel)
        else:
            auc, acc = evaluate(model, valid_loader, model.model_name)
        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)

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


        if i - best_epoch >= 10:
            break
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
