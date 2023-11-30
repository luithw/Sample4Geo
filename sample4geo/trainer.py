import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    # for loop over one epoch
    for query, reference, ids in bar:

        if scaler:
            # data (batches) to device
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)
            # Forward pass
            features1, features2 = model(query, reference)

            if train_config.coming2earth:
                loss = loss_function(features1, features2, model.logit_scale.exp())
            else:
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp())

                scaler.scale(loss).backward()

                # Gradient clipping
                if train_config.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
                scaler.step(optimizer)
                scaler.update()

                # Zero gradients for next step
                optimizer.zero_grad()

                # Scheduler
                if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                    scheduler.step()

            losses.update(loss.item())

   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            # if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
            #     loss = loss_function(features1, features2, model.module.logit_scale.exp())
            # else:
            loss = loss_function(features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            # # Calculate gradient using backward pass
            # loss.backward()
            #
            # # Gradient clipping
            # if train_config.clip_grad:
            #     torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)
            #
            # # Update model parameters (weights)
            # optimizer.step()
            # # Zero gradients for next step
            # optimizer.zero_grad()
            #
            # # Scheduler
            # if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
            #     scheduler.step()
            #
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):

    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []
    with torch.no_grad():

        for img, ids in bar:

            ids_list.append(ids)

            with autocast():

                img = img.to(train_config.device)
                img_feature = model(img)

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

    if train_config.verbose:
        bar.close()

    return img_features, ids_list


def predict_duo(train_config, model, query_dataloader, reference_dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(zip(query_dataloader, reference_dataloader), total=len(query_dataloader))
    else:
        bar = zip(query_dataloader, reference_dataloader)

    ref_features_list = []
    query_features_list = []

    ref_ids_list = []
    query_ids_list = []

    with torch.no_grad():

        for i, ((query, query_ids), (reference, ref_ids)) in enumerate(bar):
            query_ids_list.append(query_ids)
            ref_ids_list.append(ref_ids)
            with autocast():

                query = query.to(train_config.device)
                reference = reference.to(train_config.device)

                query_feature, reference_feature = model.predict(query, reference)

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    query_feature = F.normalize(query_feature, dim=-1)
                    reference_feature = F.normalize(reference_feature, dim=-1)

            # save features in fp32 for sim calculation
            query_features_list.append(query_feature.to(torch.float32))
            ref_features_list.append(reference_feature.to(torch.float32))

        # keep Features on GPU
        query_features_list = torch.cat(query_features_list, dim=0)
        ref_features_list = torch.cat(ref_features_list, dim=0)

        ref_ids_list = torch.cat(ref_ids_list, dim=0).to(train_config.device)
        query_ids_list = torch.cat(query_ids_list, dim=0).to(train_config.device)

    if train_config.verbose:
        bar.close()

    return query_features_list, query_ids_list, ref_features_list, ref_ids_list