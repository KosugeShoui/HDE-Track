# Modified by Peize Sun, Rufeng Zhang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
import time
sys.path.append('.')
from ultralytics import YOLO, RTDETR



def multiply_loss_giou_values(weight_dict, factor):
    for key in weight_dict:
        if 'loss_giou' in key:
            weight_dict[key] = factor
    return weight_dict

# Sigmoid Base Scheduler : proposed method
def sigmoid(x):
    return  1 / (1 + np.exp(-x))

def sigmoid_base_sche(initial_weight,final_weight,num_epochs):
    
    x = np.linspace(0, num_epochs, num_epochs)  
    scaled_x = 12 * (x / num_epochs) - 6
    y = sigmoid(scaled_x)
    scaled_y = (initial_weight-final_weight) * (1 - y) + final_weight 
    return scaled_y

def normalize_to_rgb(array):
        array = np.asarray(array)
        array = np.clip(array, 0, 1)
        rgb_array = (array * 255).astype(np.uint8)
        
    
        return rgb_array

def nest2tensor(samples,tensor_type):
        samples.tensors = samples.tensors.type(tensor_type)
        return samples.tensors

def save_img(samples,tensor_type,name):
        samples.tensors = samples.tensors.type(tensor_type)
        samples_sub = samples.tensors
        unnormalize = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    
        
        for i,img in enumerate(samples_sub):
            img = unnormalize(img)
            img = img.to('cpu').detach().numpy().copy()
            img = normalize_to_rgb(img.transpose(1,2,0))
            #print('img_shape = ',img.shape)
            plt.imshow(img)
            plt.savefig(name + '_{}.png'.format(i))  

def bbox_norm(bbox_tensor,image_width,image_height):
    bbox_normalized = bbox_tensor[:, :6].clone()  # 前4列を取得
    bbox_normalized[:, 0] /= image_width  # x座標を正規化
    bbox_normalized[:, 1] /= image_height  # y座標を正規化
    bbox_normalized[:, 2] /= image_width  # wを正規化
    bbox_normalized[:, 3] /= image_height  # hを正規化
    
    return bbox_normalized



def yolo_pad(input_tensor,device):
    fixed_num = 100
    input_tensor = input_tensor.unsqueeze(0).to(device)
    bbox_num = input_tensor.shape[1]
    #print(input_tensor.shape)
    if bbox_num < fixed_num:
        padding = torch.zeros(1, fixed_num - bbox_num, 6)  # [1, 足りない分, 6]
        #print(padding.shape)
        padding = padding.to(device)
        adjusted_tensor = torch.cat([input_tensor, padding], dim=1)  # [1, 100, 6]
    else:
        adjusted_tensor = input_tensor[:,:fixed_num,:]
        
    return adjusted_tensor

def resize_to_multiple_of_32(tensor):
    # 現在の高さと幅を取得
    _, height, width = tensor.shape
    
    # 高さと幅を32の倍数に調整
    new_height = math.ceil(height / 32) * 32
    new_width = math.ceil(width / 32) * 32
    
    # Tensorをリサイズ
    resized_tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), 
        size=(new_height, new_width), 
        mode='bilinear', 
        align_corners=False
    )
    
    return resized_tensor
        


def train_one_epoch(model: torch.nn.Module, yolo_model:torch.nn.Module,criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, scaler: torch.cuda.amp.GradScaler,
                    epoch: int, new_weight_dict : dict , max_norm: float = 0,fp16=False):
    fp16 = False
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
    
    model.train()
    criterion.train()
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    #metric_logger.add_meter('grad_norm_decoder_detect', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    #cdmetric_logger.add_meter('grad_norm_decoder_track', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = '\n --------- Epoch : [{}] ----------\n'.format(epoch + 1)
    print_freq = 200

    
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets, pre_samples = prefetcher.next()
    #print(pre_samples[0].shape,pre_samples[1].shape)
    #pre_samples = torch.stack(pre_samples,dim=0)
    #print(pre_samples.shape)
    #print('engine.py = ',type(samples),type(pre_samples))
    
    # --------------- YOLOモデルの定義 --------------
    #yolo_model = yolo_model.to('cpu')

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        samples.tensors = samples.tensors.type(tensor_type)
        #print(samples.tensors.shape)
        samples.mask = samples.mask.type(tensor_type)
        # データ確認
        #save_img(samples,tensor_type,'w_input_frame/sample')
        #pre_samples_n = normalize_to_rgb(pre_samples[0].permute(1, 2, 0).numpy())
        #plt.imshow(pre_samples_n)
        #plt.savefig('w_input_frame/presample.png')
        
        #print(targets[0]['frame_id'])

        with torch.cuda.amp.autocast(enabled=fp16):
            
            # -------yolo detect phase ------
            
            yolo_sample = unnorm(samples,tensor_type)
            #b,c,h,w = yolo_sample.shape
    
            
            for i , yolo_input in enumerate(yolo_sample):
                #print('DETR Input Shape = ',yolo_input.shape)
                yolo_input = resize_to_multiple_of_32(yolo_input)
                _,_,h,w = yolo_input.shape
                #print('resize DETR Input Shape = ',yolo_input.shape)
                if i == 0: 
                    #yolo_input = yolo_input.to('cpu').detach().numpy().copy()
                    #print(yolo_input.shape)
                    yolo_output_list = yolo_model(yolo_input,imgsz = w,verbose=False)
                    #detr_tgt = yolo_output_list[-4]
                    #detr_pos = yolo_output_list[-3]
                    #detr_ref = yolo_output_list[-2]
                    detr_output = yolo_output_list[-1]
                    #detr_space = yolo_output_list[-3]
                    #detr_valid = yolo_output_list[-2]
                    #detr_mask = yolo_output_list[-1]
                    
                    
                else:
                    yolo_output_list_n = yolo_model(yolo_input,imgsz = w,verbose=False)
                    detr_tgt_n = yolo_output_list_n[-4]
                    detr_pos_n = yolo_output_list_n[-3]
                    detr_ref_n = yolo_output_list_n[-2]
                    detr_output_n = yolo_output_list_n[-1]
                    #detr_space_n = yolo_output_list_n[-3]
                    #detr_valid_n = yolo_output_list_n[-2]
                    #detr_mask_n = yolo_output_list_n[-1]
                    
                    #detr_tgt = torch.cat([detr_tgt,detr_tgt_n],dim = 0)
                    #detr_pos = torch.cat([detr_pos,detr_pos_n],dim = 0)
                    #detr_ref = torch.cat([detr_ref,detr_ref_n],dim = 0)
                    detr_output = torch.cat([detr_output,detr_output_n],dim = 0)
                    #detr_space = torch.cat([detr_space,detr_space_n],dim = 0)
                    #detr_valid = torch.cat([detr_valid,detr_valid_n],dim = 0)
                    #detr_mask = torch.cat([detr_mask,detr_mask_n],dim = 0)
                    
                
            #detr_tgt.to(device=device)
            #detr_ref.to(device=device)
            detr_output.to(device=device)
            #detr_space.to(device=device)
            #detr_valid.to(device=device)
            #detr_mask.to(device=device)
            #print(detr_output.shape)
           
            detr_dict = {'output':detr_output}

            outputs, pre_outputs, pre_targets = model([samples, targets, pre_samples],detr_dict)
            loss_dict = criterion(outputs, targets, pre_outputs, pre_targets)
            
            #weight_dict = criterion.weight_dict
            #schedule weight
            weight_dict = new_weight_dict
            
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            #print(weight_dict)
            #print(losses)
            #print(loss_dict)

        #print('loss update')
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        
        #print('loss value = ',loss_value)
        #print('losses = ',losses)
        #32.31くらい

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        
        # Decoderの勾配計算
        """
        decoder_norms = {
                name: torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
                for name, module in model.named_modules() if "decoder.layers" in name}
        decoder_track_norms = {
                name: torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
                for name, module in model.named_modules() if "decoder_track.layers" in name}

        print('sum decoder_detect_norms = ',sum(decoder_norms.values()))
        print('sum decoder_track_norms = ',sum(decoder_track_norms.values()))
        
        lambda_reg = 0.001  # 正則化項の重みを調整
        regularization_loss = lambda_reg * abs(sum(decoder_norms.values()) - sum(decoder_track_norms.values()))
        #print(tmp)
        #regularization_loss_value = lambda_reg * abs(sum(decoder_norms.values()).item() - sum(decoder_track_norms.values()).item())
        print('reg loss = ',regularization_loss)
        
        
        #lossの追加
        loss_value = loss_value + regularization_loss.item()   
        reg_losses = losses + regularization_loss
        #print('new loss = ',reg_losses)
        """
        
        
        # ここで勾配ノルムの合計を計算している??
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            
            
        scaler.step(optimizer)
        scaler.update()
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        #metric_logger.update(grad_norm_decoder_detect=sum(decoder_norms.values()))
        #metric_logger.update(grad_norm_decoder_track=sum(decoder_track_norms.values()))

        samples, targets, presamples = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('\n')
    print("-------------- Averaged stats ------------ \n ", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def nest2tensor(samples,tensor_type):
        samples.tensors = samples.tensors.type(tensor_type)
        return samples.tensors

def save_combined_image(tensor1, tensor2, filename='combined_image.png'):
    fp16 = False
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
        # training

    # GPUからCPUに移す
    tensor1 = nest2tensor(tensor1,tensor_type)
    tensor2 = nest2tensor(tensor2,tensor_type)
    tensor1_cpu = tensor1.cpu()
    tensor2_cpu = tensor2.cpu()

    # テンソルをnumpy配列に変換
    np_tensor1 = tensor1_cpu.squeeze(0).permute(1, 2, 0).numpy()  # (高さ, 幅, チャンネル)の形に変換
    np_tensor2 = tensor2_cpu.squeeze(0).permute(1, 2, 0).numpy()  # (高さ, 幅, チャンネル)の形に変換

    # 画像データを正規化
    np_tensor1 = (np_tensor1 - np_tensor1.min()) / (np_tensor1.max() - np_tensor1.min())
    np_tensor2 = (np_tensor2 - np_tensor2.min()) / (np_tensor2.max() - np_tensor2.min())

    # matplotlibを使って2つの画像を横に並べて表示
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np_tensor1)
    ax[1].imshow(np_tensor2)

    # 画像を保存
    plt.savefig(filename)

#imgの逆正規化する
def unnorm(sample,tensor_type):
        sample = nest2tensor(sample,tensor_type)
        #samples_sub = sample.tensors
        unnormalize = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        img = unnormalize(sample)
        #img = img.to('cpu').detach().numpy().copy()
        #img = normalize_to_rgb(img.transpose(1,2,0))
        return img
        


@torch.no_grad()
def evaluate(model,yolo_model, criterion, postprocessors, data_loader, base_ds, device, output_dir, tracker=None, 
             phase='train', det_val=False, fp16=False):
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
    # YOLOモデルの定義
    yolo_model = yolo_model
    #yolo_model.eval()
    
    model.eval()
#   criterion.eval()
       
    metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    res_tracks = dict()
    pre_embed = None
    #count = 1
    for samples, targets , past_samples in metric_logger.log_every(data_loader, 10, header):
        # pre process for track.
        if tracker is not None:
            if phase != 'train':
                assert samples.tensors.shape[0] == 1, "Now only support inference of batchsize 1." 
            frame_id = targets[0].get("frame_id", None)
            assert frame_id is not None
            frame_id = frame_id.item()
            if frame_id == 1:
                tracker.reset_all()
                pre_embed = None
                
        samples = samples.to(device)
        #print(nest2tensor(samples,tensor_type).shape)
        samples.tensors = samples.tensors.type(tensor_type)
        samples.mask = samples.mask.type(tensor_type)
        
        #yolo用のデータ容易
        #yolo_sample = nest2tensor(samples,tensor_type)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=fp16):
            if det_val:
                outputs = model(samples)
            else:
                # ------ YOLOv9の検出結果を導入する -------
                # 推論モードにする
            
                yolo_sample = unnorm(samples,tensor_type)
                
                yolo_output_list = yolo_model(yolo_sample,verbose=False)
                detr_tgt = yolo_output_list[-4]
                detr_output = yolo_output_list[-1]
                
                detr_dict = {'tgt':detr_tgt,'output':detr_output}
                
                # Time Frame Input 
                #start_time = time.time()
                outputs, pre_embed = model(samples, past_samples,detr_dict,pre_embed)
                #print('output = ',outputs)
                #end_time = time.time()
                #inference_time = end_time - start_time
                #print(f"画像1枚あたりの推論速度: {inference_time*1000:.4f}ms")
                    
                    
            
#             loss_dict = criterion(outputs, targets)
            
#         weight_dict = criterion.weight_dict

#         reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # 検出と追尾の結果が全部入ってる
        #print(results)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        #print(res.keys())
        


        # post process for track.
        if tracker is not None:
            #初期フレームにおける処理
            if frame_id == 1:
                #res_track = tracker.init_track(yolo_output[0])
                res_track = tracker.init_track(results[0])
                #print('res track = ',res_track)
            #2フレーム目以降の追尾処理
            else:
                res_track = tracker.step(results[0])
            
            #追尾結果
            res_tracks[targets[0]['image_id'].item()] = res_track

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator, res_tracks
