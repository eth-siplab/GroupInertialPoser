import argparse
import configargparse
import os
from Trainer import Trainer
import numpy as np
import random
import torch

def get_args():
    p = configargparse.ArgumentParser()
    
    p.add_argument("--config_file",required=False,is_config_file=True,help="path to argument config file")
    
    #training config
    p.add_argument("--batch_size",default=1,type=int,help="training batch size")
    p.add_argument("--lr",default=1e-3,type=float,help="training leatning rate")
    p.add_argument("--scheduler_step",default=50,type=int,help="learning rate scheduler step")
    p.add_argument("--epochs",default=100,type=int,help="training epochs")
    p.add_argument("--grad_clip",default=5,type=float,help="cliping gradient, <0 means no clipping")
    p.add_argument('--finetune',action="store_true",help='whether to start the finetune process')
    p.add_argument('--training_phase', default=None, type=str, nargs='+')
    p.add_argument('--eval',action="store_true",help='whether to automatically evaluate after training')
    p.add_argument("--weight_decay",default=1e-5,type=float,help="weight decay for training contact loss")
    p.add_argument("--early_stop_delt",default=0,type=float,help="Minimum change in the monitored quantity to qualify as an improvement. -1 - no early stop")
    
    #misc
    p.add_argument("--device",default="cuda",help="device for training")
    p.add_argument("--seed", default=17, type=int, help="set the random seed")
    p.add_argument('--dry_run',action="store_true",help='start dry run')
    
    #dataset
    p.add_argument("--downsample_rate", default=60, type=int, help="downsample_rate for sampling the data sequence")
    p.add_argument("--resample_interval", default=-1, type=int, help="resample dataset after how many epochs, negative value means no resampling")
    p.add_argument("--static_uwb_noise",default=-1,type=float,help="UWB noise sigma under static assumption, <0 means using dynamic noise")
    p.add_argument("--eval_dataset",default="uwb-mixed", choices=['amass', 'dip-imu','tc-imu', 'uwb-imu','uwb-mixed','uwb-syn', 'multi-uwb', 'interhuman'], help='whether to test model on real data, otherwise tested on synthetic data')
    p.add_argument("--data_seq_len", default=200, type=int, help="sample raw sequence into same length chunks for training")
    p.add_argument("--normalize_uwb",action="store_true",help='whether to normalize uwb measurement by its head to pelvis distance')
    p.add_argument("--use_dataset_cached",action="store_true",help='whether to load cached preprocessed dataset')
    p.add_argument("--uwb_timesample_ratio",default=3.0,type=float,help="Syn data only, the ratio between time stamp of UWB and that of IMU(as ref)")
    p.add_argument("--flatten_uwb",action="store_true",help='whether to flatten uwb from dim 6 by 6 dist matrix to dim 15')
    p.add_argument("--use_virtual_uwb",action="store_true",help='whether to use no noise virtual uwb')
    p.add_argument("--exclude_tc_amass",action="store_true",help='whether to exclude the amass totalcapture data')
    p.add_argument("--remove_node",type=int, default=-1 ,help='node index to remove uwb')

    #network config
    p.add_argument('--network',default='PIP',help="Network name for training")
    p.add_argument("--pretrain_model",required=True,default='./data/weights.pt',help="Path to the pretrained model")
    p.add_argument('--add_guassian_noise',action="store_true",help='whether to add gaussian noise to predicted joint positions during training')
    p.add_argument('--no_rnn_init',action="store_true",help='whether to remove rnn initialization')
    p.add_argument('--lgd_step_size', type=float, default=0.1, help='Step size for IEF update.')
    p.add_argument('--lgd_use_gradient', action='store_true', help='Feed gradient to the network.')
    p.add_argument('--lgd_use_imu', action='store_true', help='Feed imu mesasure to the network.')
    p.add_argument('--lgd_use_lj_pos', action='store_true', help='Feed leaf joint position to the network.')
    p.add_argument('--augment_imu', action='store_true', help='Whether to agument imu input with uwb.')
    p.add_argument("--joint_loss_w",default=0.1,type=float,help="weight for joint loss")
    p.add_argument("--recon_loss_w",default=0.01,type=float,help="weight for uwb recon loss")
    p.add_argument('--pred_full_joint', action='store_true', help='Whether to map uwb to full joint')
    
    #pip transformer
    p.add_argument("--tf_layer_num",default=4,type=int,help="Number of the layer of transformer encoder")
    p.add_argument("--rnn_branch_layer",default=1,type=int,help="Number of the layer of rnn branch")

    #log
    p.add_argument("--log_dir",default='./output/log/',help="Path to the log dir")
    p.add_argument("--save_interval",default=50,type=int,help="Epochs intervals to save ckpt")
    
    #my config
    p.add_argument('--use_uwb',action="store_true",default=None,help='whether to use UWB data')
    p.add_argument('--wandb',action="store_true",default=None,help='whether to use wandb')
   
    opt = p.parse_known_args()
    return opt[0], p

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main():
    opt,parser = get_args()
    
    set_seed(opt.seed)
    
    trainer = Trainer(opt,parser)
    
    args = parser.parse_known_args()[0]
    parser.write_config_file(args,[os.path.join(opt.log_dir,"config.ini")])
    
    trainer.train()


if __name__=="__main__":
    main()