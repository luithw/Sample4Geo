import os
import time
import math
import shutil
import sys
from multiprocessing.dummy import Namespace

import torch
import pickle
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from comingdowntoearth.networks.c_gan import define_G, define_D, define_R
from comingdowntoearth.utils import rgan_wrapper_cvact
from sample4geo.dataset.cvact import CVACTDatasetTrain, CVACTDatasetEval, CVACTDatasetTest
from sample4geo.transforms import get_transforms_train, get_transforms_val
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.cvusa_and_cvact import evaluate, calc_sim
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Training 
    mixed_precision: bool = True
    # mixed_precision: bool = False

    seed = 1
    epochs: int = 40
    batch_size: int = 128          # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,1)     # GPU ids for training

    # Similarity Sampling
    custom_sampling: bool = True   # use custom sampling instead of random
    gps_sample: bool = True        # use gps sampling
    sim_sample: bool = True        # use similarity sampling
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128     # pool size for selection
    gps_dict_path: str = "./data/CVACT/gps_dict.pkl"   # path to pre-computed distances
 
    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 4        # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                   # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False   # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.001                  # 1 * 10^-4 for ViT | 1 * 10^-3 for CNN
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    
    # Dataset
    data_folder = "./data/CVACT"     
    
    # Augment Images
    # prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_rotate: float = 0          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously

    # Savepath for model checkpoints
    model_path: str = "./cvact"
    
    # Eval before training
    zero_shot: bool = False
    
    # Checkpoint to start from
    checkpoint_start = None
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 10

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False

#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':


    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#

    infoNCE = InfoNCE(loss_function=torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing), device=config.device)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        

    #define Perspective Trnasform networks
    opt = Namespace(results_dir='./CVACT_results', name='', seed=10, gpu_ids=config.gpu_ids, isTrain=True, resume=True, start_epoch=0,
              data_root='./data/CVACT/ANU_data_small/', data_list='./data/CVACT/ACT_data.mat', polar=True,
              save_step=10, rgan_checkpoint=None, n_epochs=200, batch_size=config.batch_size, lr_g=0.0001, lr_d=0.0001, lr_r=0.0001,
              weight_decay=0.0, b1=0.5, b2=0.999, lambda_gp=10, lambda_l1=100, lambda_ret1=1000, lambda_sm=10,
              hard_topk_ratio=1.0, hard_decay1_topk_ratio=0.1, hard_decay2_topk_ratio=0.05, hard_decay3_topk_ratio=0.01,
              n_critic=1, input_c=3, segout_c=3, realout_c=3, n_layers=3, feature_c=64, g_model='unet-skip',
              d_model='basic', r_model='SAFA', condition=1, is_Train=True, gan_loss='vanilla', device=config.device)

    generator = define_G(netG=opt.g_model, gpu_ids=config.gpu_ids)
    print('Init {} as generator model'.format(opt.g_model))

    discriminator = define_D(input_c=opt.input_c, output_c=opt.realout_c, ndf=opt.feature_c, netD=opt.d_model,
                             condition=opt.condition, n_layers_D=opt.n_layers, gpu_ids=config.gpu_ids)
    print('Init {} as discriminator model'.format(opt.d_model))

    retrieval = define_R(ret_method=opt.r_model, polar=opt.polar, gpu_ids=config.gpu_ids)
    print('Init {} as retrieval model'.format(opt.r_model))
    model = rgan_wrapper_cvact.RGANWrapper(opt, sys.stdout.file, generator, discriminator, retrieval, infoNCE)

    image_size_sat = (112, 616) if opt.polar else (256, 256)
    img_size_ground = (112, 616)

    # print("\nModel: {}".format(config.model))
    # model = TimmModel(config.model,
    #                       pretrained=True,
    #                       img_size=config.img_size)
    #
    # data_config = model.get_config()
    # print(data_config)
    # mean = data_config["mean"]
    # std = data_config["std"]
    # img_size = config.img_size
    #
    # image_size_sat = (img_size, img_size)
    #
    # new_width = config.img_size * 2
    # new_hight = round((224 / 1232) * new_width)
    # img_size_ground = (new_hight, new_width)
    #
    # # Activate gradient checkpointing
    # if config.grad_checkpointing:
    #     model.set_grad_checkpointing(True)
    #
    # # Load pretrained Checkpoint
    # if config.checkpoint_start is not None:
    #     print("Start from:", config.checkpoint_start)
    #     model_state_dict = torch.load(config.checkpoint_start)
    #     model.load_state_dict(model_state_dict, strict=False)
    #
    # # Data parallel
    # print("GPUs available:", torch.cuda.device_count())
    # if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    #
    # # Model to device
    # model = model.to(config.device)
    #
    # print("\nImage Size Sat:", image_size_sat)
    # print("Image Size Ground:", img_size_ground)
    # print("Mean: {}".format(mean))
    # print("Std:  {}\n".format(std))
    #
    # # original size
    # # Image Size Sat: (384, 384)
    # # Image Size Ground: (140, 768)

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                   img_size_ground,
                                                                   # mean=mean,
                                                                   # std=std,
                                                                   )
                                                                   
                                                                   
    # Train
    train_dataset = CVACTDatasetTrain(data_folder=config.data_folder ,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               # mean=mean,
                                                               # std=std,
                                                               )


    # Reference Satellite Images
    reference_dataset_val = CVACTDatasetEval(data_folder=config.data_folder ,
                                             split="val",
                                             img_type="reference",
                                             transforms=sat_transforms_val,
                                             )
    
    reference_dataloader_val = DataLoader(reference_dataset_val,
                                          batch_size=config.batch_size_eval,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_val = CVACTDatasetEval(data_folder=config.data_folder ,
                                         split="val",
                                         img_type="query",    
                                         transforms=ground_transforms_val,
                                         )
    
    query_dataloader_val = DataLoader(query_dataset_val,
                                      batch_size=config.batch_size_eval,
                                      num_workers=config.num_workers,
                                      shuffle=False,
                                      pin_memory=True)
    
    
    print("Reference Images Val:", len(reference_dataset_val))
    print("Query Images Val:", len(query_dataset_val))
    
    
    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:
    
        # Query Ground Images Train for simsampling
        query_dataset_train = CVACTDatasetEval(data_folder=config.data_folder ,
                                               split="train",
                                               img_type="query",   
                                               transforms=ground_transforms_val,
                                               )
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        
        
        reference_dataset_train = CVACTDatasetEval(data_folder=config.data_folder ,
                                                   split="train",
                                                   img_type="reference", 
                                                   transforms=sat_transforms_val,
                                                   )
        
        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)


        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    def create_scheduler(optimizer):
        if config.scheduler == "polynomial":
            print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                                  num_training_steps=train_steps,
                                                                  lr_end = config.lr_end,
                                                                  power=1.5,
                                                                  num_warmup_steps=warmup_steps)

        elif config.scheduler == "cosine":
            print("\nScheduler: cosine - max LR: {}".format(config.lr))
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_training_steps=train_steps,
                                                        num_warmup_steps=warmup_steps)

        elif config.scheduler == "constant":
            print("\nScheduler: constant - max LR: {}".format(config.lr))
            scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                           num_warmup_steps=warmup_steps)

        else:
            scheduler = None

        print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
        print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        return scheduler
    scheduler = create_scheduler(optimizer)
    model.scheduler_G = create_scheduler(model.optimizer_G)
    model.scheduler_R = create_scheduler(model.optimizer_R)
    model.scheduler_D = create_scheduler(model.optimizer_D)
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

      
        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_val,
                           query_dataloader=query_dataloader_val, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
        
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train, 
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)
                
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    

    for epoch in range(1, config.epochs+1):
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=infoNCE,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_val,
                               query_dataloader=query_dataloader_val, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
            
            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train, 
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)
                
            if r1_test > best_score:

                best_score = r1_test

                # if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                #     torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                # else:
                # torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                model.save_networks(epoch, model_path, best_acc=r1_test, is_best=True)


        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)
                
    # if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    #     torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    # else:
    #     torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))

    model.save_networks(epoch, model_path, last_ckpt=True)

    #-----------------------------------------------------------------------------#
    # Test                                                                        #
    #-----------------------------------------------------------------------------#
    
    # Reference Satellite Images
    reference_dataset_test = CVACTDatasetTest(data_folder=config.data_folder ,
                                              img_type="reference",
                                              transforms=sat_transforms_val)
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_test = CVACTDatasetTest(data_folder=config.data_folder ,
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))          


    print("\n{}[{}]{}".format(30*"-", "Test", 30*"-"))  

  
    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=reference_dataloader_test,
                       query_dataloader=query_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
