import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from prettytable import PrettyTable
from modules.dataset.data_utils import Batch,D_Batch
from config.config import uwb_collision_thr
__all__ = ["mapping_loss_func", "registered_loss","Loss_Func","Easy_dict","compute_uwb_reconstruction_error"]

mapping_loss_func = {"rnn1":["leaf_joint_loss"],
                     "rnn2":["full_joint_loss"],
                     "rnn3":["rotation_loss"],
                     "rnn4":["joint_vel_loss"],
                     "rnn5":["contact_loss"],
                     "rnn_uwb_b":["uwb_b_regress_loss"],
                     "lgd":["full_joint_loss","lgd_reconstruction_loss"], #validation only for lgd
                     "full_jointmapper":["full_joint_loss"],
                     "leaf_jointmapper":["leaf_joint_loss"],
                     "tf":["rotation_loss","joint_vel_loss","contact_loss"],
                     "gconv":["imu_node_joint_loss"],
                     "rnn_jp_mapper":["leaf_joint_loss_rnn"],
                     "gnn_jp_mapper":["leaf_joint_loss_gnn"],
                     "rnn_leaf_jp_mapper":["leaf_joint_loss"],
                     "graphomer":["leaf_joint_loss_gnn"]}
mapping_loss_weight = {
				     "tf":[100.0, 0.0005, 5]
                      }	

registered_loss = {}



def register_loss_func(func):
    registered_loss[func.__name__] = func
    return func

@register_loss_func
def leaf_joint_loss(batch: Batch, pred: D_Batch, **kwargs):
	y_leaf_joint_pred = pred.leaf_joint
	y_leaf_joint_gt = batch.lfj_gt
	return nn.MSELoss()(y_leaf_joint_gt, torch.stack(y_leaf_joint_pred))

@register_loss_func
def imu_node_joint_loss(batch: Batch, pred: D_Batch, **kwargs):
	y_leaf_joint_pred = pred.imu_node_position
	y_leaf_joint_gt = batch.lfj_gt
	return nn.MSELoss()(y_leaf_joint_gt, torch.stack(y_leaf_joint_pred))

@register_loss_func
def leaf_joint_loss_rnn(batch: Batch, pred: D_Batch, uwb_L_lambda=1.0, **kwargs):
	y_leaf_joint_pred = pred.imu_node_position_rnn
	y_leaf_joint_gt = batch.lfj_gt
	y_uwb_gt = batch.uwb_gt
	flag_uwb_normalized = batch.uwb_normalized
	return compute_gnn_uwb_recon_loss_cossim(y_leaf_joint_gt, torch.stack(y_leaf_joint_pred),y_uwb_gt,normalized=flag_uwb_normalized,k = uwb_L_lambda,**kwargs)

@register_loss_func
def leaf_joint_resi_loss_rnn(batch: Batch, pred: D_Batch, uwb_L_lambda=1.0, **kwargs):
	y_leaf_joint_pred = pred.imu_node_position_rnn
	y_leaf_joint_gt = batch.lfj_gt
	y_uwb_gt = batch.uwb_gt
	flag_uwb_normalized = batch.uwb_normalized
	return compute_gnn_uwb_recon_loss_cossim(y_leaf_joint_gt, torch.stack(y_leaf_joint_pred),y_uwb_gt,normalized=flag_uwb_normalized,k = uwb_L_lambda,**kwargs)

@register_loss_func
def leaf_joint_loss_gnn(batch: Batch, pred: D_Batch, uwb_L_lambda=1.0, **kwargs):
	y_leaf_joint_pred = pred.imu_node_position_gnn
	y_leaf_joint_gt = batch.lfj_gt
	y_uwb_gt = batch.uwb_gt
	flag_uwb_normalized = batch.uwb_normalized
	return compute_gnn_uwb_recon_loss_cossim(y_leaf_joint_gt, torch.stack(y_leaf_joint_pred),y_uwb_gt,normalized=flag_uwb_normalized,k = uwb_L_lambda,**kwargs)

@register_loss_func
def leaf_joint_loss_gnn_mse(batch: Batch, pred: D_Batch, **kwargs):
	y_leaf_joint_pred = pred.imu_node_position_gnn
	y_leaf_joint_gt = batch.lfj_gt
	y_uwb_gt = batch.vuwb
	return nn.MSELoss()(y_leaf_joint_gt, torch.stack(y_leaf_joint_pred))

@register_loss_func
def full_joint_loss(batch: Batch, pred: D_Batch, **kwargs):
	y_full_joint_pred = pred.full_joint
	y_full_joint_gt = batch.joint_gt
	return nn.MSELoss()(y_full_joint_gt, torch.stack(y_full_joint_pred))

@register_loss_func
def rotation_loss(batch: Batch, pred: D_Batch, **kwargs):
	y_6drot_pred = pred.global_6d_pose
	y_6drot_gt = batch.jrot_6d
	return nn.MSELoss()(y_6drot_gt, torch.stack(y_6drot_pred))
	# return - torch.mean(nn.CosineSimilarity(dim=2,eps=1e-6)(y_6drot_gt,torch.stack(y_6drot_pred)))

@register_loss_func
def joint_vel_loss(batch: Batch, pred: D_Batch, **kwargs):
	y_jvel_pred = pred.joint_velocity
	y_jvel_gt = batch.jvel_gt
	# return accumulate_loss(y_jvel_gt, torch.stack(y_jvel_pred))
	return nn.MSELoss()(y_jvel_gt, torch.stack(y_jvel_pred))

@register_loss_func
def contact_loss(batch: Batch, pred: D_Batch, **kwargs):
	y_contact = pred.contact
	y_contact_gt = batch.contact_p
	return contact_loss(y_contact_gt, torch.stack(y_contact)) 

@register_loss_func
def uwb_b_regress_loss(batch: Batch, pred: D_Batch, **kwargs):
	y_uwbb_pred = pred.uwb_c
	y_uwbb_gt = batch.uwb_c
	return nn.MSELoss()(y_uwbb_gt, torch.stack(y_uwbb_pred))

@register_loss_func
def uwb_b_BCE_loss(batch: Batch, pred: D_Batch, **kwargs):
	y_uwbb_pred = pred.uwb_c
	y_uwbb_gt = batch.uwb_c
	return uwb_b_bce_loss(y_uwbb_gt, torch.stack(y_uwbb_pred))

@register_loss_func
def lgd_reconstruction_loss(batch: Batch, pred: D_Batch, **kwargs):
	# Expect all input with size (B, F, 36).
	y_uwb_gt = batch.vuwb
	uwb_c_pred = pred.uwb_c
	y_uwb_pred = pred.uwb_h
	return compute_uwb_reconstruction_error(y_uwb_gt, torch.stack(y_uwb_pred), torch.stack(uwb_c_pred))

def compute_gnn_uwb_recon_loss(lj_gt,lj_pred,uwb_gt,normalized=False,k=1.0,uwb_init_norm=None):
	lj_pred = lj_pred.view(-1,5,3) #[lk,rk,head,lw,rw]
	pad = torch.zeros((lj_pred.size(0), 1, 3),dtype=lj_pred.dtype,device=lj_pred.device)
	jp = torch.cat([lj_pred[:,[3,4,0,1,2]],pad],dim=1)
	rec_uwb = torch.cdist(jp,jp) #[lw,rw,lk,rk,head,root]
	if normalized:
		if uwb_init_norm is None:
			d_root2head = torch.norm(lj_gt.view(-1,5,3)[0,2,:]) 
		else:
			d_root2head = uwb_init_norm
		rec_uwb = rec_uwb / d_root2head
	loss = k * nn.L1Loss()(rec_uwb,uwb_gt.view(-1,6,6)) + nn.MSELoss()(lj_pred,lj_gt.view_as(lj_pred)) 
	return loss

def compute_gnn_uwb_recon_loss_cossim(lj_gt,lj_pred,uwb_gt,normalized=False,k=1.0,uwb_init_norm=None,disable_recon=False):
	if disable_recon:
		loss = - torch.mean(nn.CosineSimilarity(dim=2,eps=1e-6)(lj_pred,lj_gt.view_as(lj_pred)))
	else:
		lj_pred = lj_pred.view(-1,5,3) #[lk,rk,head,lw,rw]
		pad = torch.zeros((lj_pred.size(0), 1, 3),dtype=lj_pred.dtype,device=lj_pred.device)
		jp = torch.cat([lj_pred[:,[3,4,0,1,2]],pad],dim=1)
		rec_uwb = torch.cdist(jp,jp) #[lw,rw,lk,rk,head,root]
		if normalized:
			pass
			# if uwb_init_norm is None:
			# 	d_root2head = torch.norm(lj_gt.view(-1,5,3)[0,2,:]) 
			# else:
			# 	d_root2head = uwb_init_norm
			# import ipdb;ipdb.set_trace()
			# rec_uwb = rec_uwb / d_root2head
		loss = k * nn.L1Loss()(rec_uwb,uwb_gt.view(-1,6,6)) - torch.mean(nn.CosineSimilarity(dim=2,eps=1e-6)(lj_pred,lj_gt.view_as(lj_pred)))
	return loss

def compute_uwb_reconstruction_error(uwb_gt, uwb_pred, uwb_c):
	if uwb_c is not None:
		diff = torch.abs(uwb_gt - uwb_pred) * (uwb_c < uwb_collision_thr) #only collision-free uwb measure are taken into account
	else:
		diff = torch.abs(uwb_gt - uwb_pred)
  
	return diff.sum(dim=-1).mean()

def vel_loss(vel_gt, vel_pred, n = 1, frame_T=100):
	num_samples = frame_T//n
	batch_size,frame_num,_ = vel_gt.size()
	res = torch.zeros(num_samples,batch_size,24)
	for m in range(num_samples):
		frame_s = m
		frame_e = m * n + n
		norm_p_batch_p_joint = torch.linalg.norm(torch.sum(vel_pred[:,frame_s:frame_e,:] - vel_gt[:,frame_s:frame_e,:],dim=1).view(-1,24,3), dim=2)**2
		res[m] = norm_p_batch_p_joint
	return torch.sum(res,dim=0).mean() #average over the accumulate vel error per batch per joint
	
def accumulate_loss(vel_gt,vel_pred,component=[1,3,9,27]):
	T = vel_pred.size(1)
	k_vel_loss = partial(vel_loss,vel_gt,vel_pred,frame_T=T)
	return sum([k_vel_loss(n=k) for k in component])

def uwb_b_bce_loss(b_gt,b_pred):
    return F.binary_cross_entropy(b_pred, (b_gt > uwb_collision_thr).float())

def contact_loss(c_gt,c_pred):
	c_l = F.binary_cross_entropy(torch.sigmoid(c_pred), c_gt.float())
	return c_l 

class Easy_dict(dict):
	
	def _div(self,deno=1):
		for key,value in self.items():
			self[key] = value/deno
			
	def div(self,deno=1):
		out_dict = Easy_dict()
		for key,value in self.items():
			out_dict[key] = value/deno
		return out_dict
	
	def sum(self):
		sum = 0
		for value in self.values():
			sum += value
		return sum
	
	def _add(self,a:dict):
		for key,value in self.items():
			self[key] = value + a[key]
			
	def _add_item(self,a:dict):
		for key,value in self.items():
			self[key] = value + a[key].item()
	
class Loss_Func:
	@staticmethod
	def add_args(parser):
		group = parser.add_argument_group("Loss Function Config")
		group.add_argument('--use_gnn_mse_loss', action='store_true', help='Whether to use the same MSE loss for gnn')
		group.add_argument('--uwb_L_lambda', type=float, default=0.01, help='Whether to use the same MSE loss for gnn')
		group.add_argument('--disable_recon',action='store_true')

	def __init__(self,opt) -> None:
		self.mapping_loss_func = mapping_loss_func
		if opt.use_gnn_mse_loss:
			print("Use loss func leaf_joint_loss_gnn_mse")
			self.mapping_loss_func["gnn_jp_mapper"] = ["leaf_joint_loss_gnn_mse"]
		self.loss_func_kwargs = {
			"uwb_L_lambda":opt.uwb_L_lambda,
   			"disable_recon":opt.disable_recon
		}
	
	def set_training_phase(self,phase):
		mode,module = phase.split("_",1)
		if mode in ['baseline', 'finetune']:
			self.loss_func = [registered_loss[k] for k in self.mapping_loss_func[module]]
		else:
			raise KeyError(f"Invalid phase mode{mode}")

		if module in mapping_loss_weight:
			self.loss_weight = mapping_loss_weight[module]
		else:
			self.loss_weight = [1] * len(self.loss_func)

		assert len(self.loss_func)==len(self.loss_weight)
		print(f"Update loss function {self.loss_func[0].__name__} for phase {phase}")
 
	def compute_total_loss(self, y_pred: D_Batch, batch: Batch):
		# loss_dict = Easy_dict({"leaf_joint_loss":0,"full_joint_loss":0,"rotation_loss":0,"joint_vel_loss":0,"contact_loss":0})
		kwargs = self.loss_func_kwargs
		loss_dict = Easy_dict({loss_func.__name__:loss_func(batch, y_pred, **kwargs) * weight for weight,loss_func in zip(self.loss_weight,self.loss_func)})
		loss_dict["total_loss"] = loss_dict.sum()
		return loss_dict

