import os
import torch
from tqdm import tqdm
import torch.optim as optim
from functools import partial
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from config.config import *
from modules.dataset.dataset import *
from modules.model import get_model
from modules.loss.loss_utils import *
from modules.utils import *
from modules.evaluate.eval_utils import PoseEvaluator
from modules.evaluate.evaluator import evaluate_model,print_eval_result
from modules.evaluate import evaluator_interhuman
import wandb

# make sure it won't take too much of cpu usage.CPU% < 800, 
torch.set_num_threads(8)

class Trainer:
	def __init__(self,opt,parser,debug=False) -> None:
		self.batch_size = opt.batch_size
		self.device = opt.device
		self.w_eval = opt.eval
		self.dry_run = opt.dry_run
		self.eval_dataset_name = opt.eval_dataset
		self.debug = debug
		model_cls = get_model(opt, parser)
		args = parser.parse_known_args()[0]
		self.model = model_cls(args=args).to(device=self.device)
		self.model_name = opt.network
		self.use_uwb = "vuwb" in self.model.imu_m
		self.save_interval = opt.save_interval
		self.flatten_uwb = opt.flatten_uwb
		self.use_virtual_uwb = opt.use_virtual_uwb
		self.exclude_tc_amass = opt.exclude_tc_amass
		if self.dry_run:
			print("###########You are in dry-run, which is only for quick testing!!!###############")
		self.wandb = opt.wandb
		
		#Training_phase
		if opt.finetune:
			self.training_phase = ["finetune_rnn1","finetune_rnn2","finetune_rnn3"]
		elif opt.training_phase is None:
			self.training_phase = ["baseline_rnn1","baseline_rnn2","baseline_rnn3","baseline_rnn4","baseline_rnn5"]
		else:
			self.training_phase = opt.training_phase

		#Load pretrain Model
		if opt.pretrain_model:
			weight_loaded = torch.load(opt.pretrain_model, weights_only=True)
			strict= True if opt.network not in ["UWB_B_Net","Iterative_Fitting_UWB"] else False
			if "net" in weight_loaded:
				main_ckpt = weight_loaded["net"]
				
				self.model.load_state_dict(main_ckpt,strict=strict)
			else:
				self.model.load_state_dict(weight_loaded,strict=strict)

			print(f"Load model weight from {opt.pretrain_model}")
	
		#Initialize optimizer
		self.epochs = opt.epochs if not self.dry_run else 1
		self.lr = opt.lr
		self.lr_scalar = {"baseline_rnn1": 1, "baseline_rnn2": 1, "baseline_rnn3":1, "baseline_rnn4": 0.5, "baseline_rnn5": 0.1, "finetune_rnn1":0.1,"finetune_rnn2":0.1,"finetune_rnn3":0.1}
		self.grad_clip = opt.grad_clip
		self.scheduler_step = opt.scheduler_step
		self.weight_decay = opt.weight_decay
		self.early_stop_delt = opt.early_stop_delt
		
		#Dataset
		self.dataset = None
		self.downsample_rate = opt.downsample_rate
		self.batch_size = opt.batch_size
		self.resampling_interval = opt.resample_interval
		self.normalize_uwb = opt.normalize_uwb
		self.remove_node = opt.remove_node
		self.dataset_common_kwargs = {"official_model_file":paths.smpl_file,
										"seq_length":opt.data_seq_len,
										"device":"cpu",
										"add_uwb":self.use_uwb,
										"imu_m":self.model.imu_m,
		  								"static_uwb_noise":opt.static_uwb_noise,
										"normalize_uwb":self.normalize_uwb,
										"use_cached":opt.use_dataset_cached,
										"uwb_timesample_ratio":opt.uwb_timesample_ratio,
										"flatten_uwb":opt.flatten_uwb,
				  						"dry_run": opt.dry_run,
										"remove_node": self.remove_node}
		print(f"Dataset Config: {self.dataset_common_kwargs}")
  
		#loss func
		Loss_Func.add_args(parser)
		args = parser.parse_known_args()[0]
		self.loss_func = Loss_Func(args)
		
		#initialize writer
		if self.dry_run:
			opt.log_dir = opt.log_dir + "_dry_run"
		tb_log = os.path.join(opt.log_dir,"runs")
		self.ckpt_dir = os.path.join(opt.log_dir,"ckpt")
		self.eval_dir = os.path.join(opt.log_dir,"eval")
		os.makedirs(tb_log,exist_ok=True)
		os.makedirs(self.ckpt_dir,exist_ok=True)
		os.makedirs(self.eval_dir,exist_ok=True)
		self.writer = SummaryWriter(log_dir=tb_log)
		self.writer_log = Easy_dict()
		if self.wandb:
			wandb.init(project="UIP_", name=os.path.basename(opt.log_dir), config=vars(args))
		else:
			wandb.init(mode="disabled")
  
		self.model.save_config(args, os.path.join(opt.log_dir,"model_args.json"))
		self.log_dir = os.path.basename(opt.log_dir)

	def _init_dataloader(self,phase):
		train_phase,module_name = phase.split("_",1)
		if train_phase == "finetune":
			if self.eval_dataset_name in ["uwb-imu",'uwb-mixed']:
				#Finetune on UWB-IMU data
				dataset_path = paths.uwbimu_dir if not self.use_virtual_uwb else os.path.join(paths.uwbimu_dir,"sigma0")
				dataset_real = UWBIMU_real_data_train(dataset_path,down_sample_rate=20,**self.dataset_common_kwargs)
				self.dataset = dataset_real

				self.data_loader = DataLoader(self.dataset,shuffle=True,
										pin_memory=True,
										batch_size=self.batch_size,
										num_workers=1)

				dataset_val = UWBIMU_real_data_val(dataset_path,down_sample_rate=100,train_split=False,**self.dataset_common_kwargs)
				self.val_dataloader = DataLoader(dataset_val,shuffle=True,pin_memory=True,batch_size=self.batch_size,num_workers=1)
			
			elif self.eval_dataset_name in ['multi-uwb']:
				# fine-tune on Multi-UWB dataset
				dataset_path = paths.multiuwb_dir
				dataset_real = Multi_UWB_real_data_train(dataset_path,down_sample_rate=20,**self.dataset_common_kwargs)
				self.dataset = dataset_real

				self.data_loader = DataLoader(self.dataset,shuffle=True,
										pin_memory=True,
										batch_size=self.batch_size,
										num_workers=1)

				dataset_val = Multi_UWB_real_data_val(dataset_path,down_sample_rate=100,train_split=False,**self.dataset_common_kwargs)
				self.val_dataloader = DataLoader(dataset_val,shuffle=True,pin_memory=True,batch_size=self.batch_size,num_workers=1)
			
			else:
				#Finetune on DIP-IMU data
				self.dataset = DIPIMU_real_data_train(paths.dipimu_dir,down_sample_rate=self.downsample_rate,**self.dataset_common_kwargs)
				self.data_loader = DataLoader(self.dataset,shuffle=True,
										pin_memory=True,
										batch_size=self.batch_size,
										num_workers=1)

				dataset_val = DIPIMU_real_data_val(paths.dipimu_dir,down_sample_rate=100,train_split=False,**self.dataset_common_kwargs)
				self.val_dataloader = DataLoader(dataset_val,shuffle=True,pin_memory=True,batch_size=self.batch_size,num_workers=1)
   
		elif train_phase == "baseline":
			if isinstance(self.dataset,AMASS_syn_data):
				return
			dataset_path = paths.amass_dir if not self.exclude_tc_amass else os.path.join(paths.amass_dir,"no_tc")
			self.dataset = AMASS_syn_data(dataset_path,down_sample_rate=self.downsample_rate,**self.dataset_common_kwargs)
			#AMASS_DATA = AMASS_syn_data(dataset_path,down_sample_rate=self.downsample_rate,**self.dataset_common_kwargs)
			#DIP_IMU_dataset = DIPIMU_real_data_test(paths.dipimu_dir,down_sample_rate=self.downsample_rate,**self.dataset_common_kwargs)
			#self.dataset = torch.utils.data.ConcatDataset([AMASS_DATA,DIP_IMU_dataset])
			self.data_loader = DataLoader(self.dataset,shuffle=True,
									pin_memory=True,
									batch_size=self.batch_size,
									num_workers=1)
			if self.eval_dataset_name in ['uwb-mixed','uwb_imu']:
				dataset_val = UWBIMU_real_data_val(paths.uwbimu_dir,down_sample_rate=100,train_split=False,**self.dataset_common_kwargs)
				self.val_dataloader = DataLoader(dataset_val,shuffle=True,pin_memory=True,batch_size=self.batch_size,num_workers=1)
			elif self.eval_dataset_name in ['multi-uwb']:
				# dataset_val = Multi_UWB_real_data_val(paths.multiuwb_dir,down_sample_rate=100,train_split=False,**self.dataset_common_kwargs)
				dataset_val = Multi_UWB_real_data_val(paths.interhuman_dir,down_sample_rate=100,train_split=False,**self.dataset_common_kwargs)
				self.val_dataloader = DataLoader(dataset_val,shuffle=True,pin_memory=True,batch_size=self.batch_size,num_workers=1)
			else:
				dataset_val = AMASS_syn_data_val(paths.amass_dir,down_sample_rate=100,train_split=False,**self.dataset_common_kwargs)
				self.val_dataloader = DataLoader(dataset_val,shuffle=True,pin_memory=True,batch_size=self.batch_size,num_workers=1)
		else:
			raise KeyError(f"Invalid training phase {train_phase}")
		return 
  
	def _init_optimizer(self,phase):
		train_phase,module_name = phase.split("_",1)
		if train_phase in ["finetune", "baseline"]:
			for name,param in self.model.named_parameters():
				if name.startswith(module_name):
					param.requires_grad = True
				else:
					param.requires_grad = False
		else:
			raise NotImplementedError(f"Invalid training phase {phase}")

		non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
		num_param = sum(p.numel() for p in non_frozen_parameters)
		print(f"Initialize Training phase {phase} -- Number of Parameter: {num_param}")
  
		self.optimizer = optim.Adam(non_frozen_parameters,lr=self.lr * self.lr_scalar.setdefault(phase,1.0),weight_decay=self.weight_decay)
		self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=self.scheduler_step,gamma=0.33)
		
	def tb_logging(self,epoch,phase,**log_data):
		assert "train_loss" in log_data
		assert "val_loss" in log_data
  
		for key in log_data["val_loss"].keys():
			self.writer.add_scalar(f"{phase}/val_{key}",log_data["val_loss"][key], epoch)

		for key in log_data["train_loss"].keys():
			self.writer.add_scalar(f"{phase}/train_{key}",log_data["train_loss"][key], epoch)

		for key in log_data.keys():
			if key in ["train_loss", "val_loss"]:
				continue
			self.writer.add_scalar(f"{phase}/{key}",log_data[key],epoch)
	
	def wandb_logging(self, epoch, phase, **log_data):
		assert "train_loss" in log_data
		assert "val_loss" in log_data

		# Initialize a dictionary to hold all metrics
		log_dict = {}

		# Log validation losses
		for key in log_data["val_loss"].keys():
			log_dict[f"{phase}/val_{key}"] = log_data["val_loss"][key]

		# Log training losses
		for key in log_data["train_loss"].keys():
			log_dict[f"{phase}/train_{key}"] = log_data["train_loss"][key]

		# Log other metrics
		for key in log_data.keys():
			if key in ["train_loss", "val_loss"]:
				continue
			log_dict[f"{phase}/{key}"] = log_data[key]
		
		log_dict["epoch"] = epoch  # Add epoch to the log
		wandb.log(log_dict)  # Log everything at once

		 
		
	def train(self):
		self.model.train()
		for phase in self.training_phase:
			print()
			self.early_stop_check = EarlyStop(delta=self.early_stop_delt)
			self._init_dataloader(phase)
			self._init_optimizer(phase)
			self.loss_func.set_training_phase(phase)
			for epoch in range(self.epochs):
				self.train_one_epoch(epoch, phase=phase)
				self.lr_scheduler.step()
				wandb.log({"lr": self.lr_scheduler.get_last_lr()[0], "epoch": epoch}, commit=False)
				if self.early_stop_check.early_stop:
					if phase in ["baseline_rnn5","baseline_tf"] or phase == self.training_phase[-1]:
						self.eval(epoch,phase)
						# self.evaluate_model(epoch,phase)
					print(f"Early stop {phase} @ Epoch {epoch}")
					break
				if (epoch != 0 and epoch % self.save_interval == 0) or epoch == self.epochs-1:
					self.eval(epoch,phase)

	 		
	def eval(self,epoch,phase):
		#Evaluating the current model on full dataset
		self.model.eval()
		if epoch == self.epochs-1:
			file_name = os.path.join(self.ckpt_dir, f"{phase}_best_model.pt")
		else:
			file_name = os.path.join(self.ckpt_dir, f"{phase}_ckpt_{str(epoch).zfill(3)}.pt")
		self.save_checkpoint(file_name=file_name,epoch=epoch)
		#To save time only evaluate in last epoch of finetuning phase and last epoch of baseline
		## !!!TODO eval using our data in the last training phase
		if (phase in ["baseline_rnn5","baseline_tf"] or phase == self.training_phase[-1]) and epoch == self.epochs-1:
			self.evaluate_model(epoch,phase)
   
		print(f"Model saved at {file_name}")
  
		self.model.train()
	
	def evaluate_model(self,epoch,phase):
		if self.w_eval:
			net = self.model.to("cpu")
			best_model = torch.load(os.path.join(self.ckpt_dir, f"{phase}_best_model.pt"), weights_only=True)
			net.load_state_dict(best_model["net"])
			epoch_save = best_model["epoch"]
			print(f"Eval Best Model {net.name} @ Epoch {epoch_save} ...")
			if self.eval_dataset_name == "dip-imu":
				seq_ids = [0] if self.dry_run else None
				eval_dip = evaluate_model(net, paths.dipimu_dir, pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=False, \
			 							flush_cache=True,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node)
				eval_tc = evaluate_model(net, paths.totalcapture_dir, pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True, \
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node)
				table = print_eval_result([eval_dip,eval_tc],filter_keys=set(eval_dip) & set(eval_tc))
				eval_trans = eval_tc
			if self.eval_dataset_name == "tc-imu":
				seq_ids = [0] if self.dry_run else None
				eval_tc = evaluate_model(net, paths.totalcapture_dir, pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True, \
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node)
				table = print_eval_result([eval_tc],filter_keys=set(eval_tc),title=["    ","TotalCapture"])
				eval_trans = eval_tc
			elif self.eval_dataset_name == "amass":
				seq_ids = [0] if self.dry_run else list(range(30))
				eval_amass = evaluate_model(net, os.path.join(paths.amass_dir,"test_split"), pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True,\
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node)
				table = print_eval_result([eval_amass],filter_keys=set(eval_amass),title=["    ", "AMASS Dance-DB"])
				eval_trans = eval_amass
			elif self.eval_dataset_name in ["multi-uwb"]:
				'''
				evaluate on our test and interhuman test
				'''
				seq_ids = [0] if self.dry_run else [0,1,2,3]
				if 'MAM' in net.name: net = net.to("cuda")
				
				eval_multi_uwb = evaluator_interhuman.evaluate_model(net, os.path.join(paths.multiuwb_dir,"test"), pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True, \
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node, device = self.device, \
											eval_save_dir=self.log_dir+"/mu")
				
				eval_interhuman = evaluator_interhuman.evaluate_model(net, os.path.join(paths.interhuman_dir,"test"), pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True,\
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node, device = self.device, \
											eval_save_dir="_interhuman")
				
				table = print_eval_result([eval_multi_uwb, eval_interhuman],filter_keys=set(eval_multi_uwb)&set(eval_interhuman),title=["    ", "Multi-UWB Test", "interhuman Test"])
				eval_trans = eval_multi_uwb
			elif self.eval_dataset_name in ["interhuman"]:
				'''
				evaluate on interhuman test
				'''
				seq_ids = [0] if self.dry_run else [0,1,2,3]

				if 'MAM' in net.name: net = net.to("cuda")
				
				eval_interhuman = evaluator_interhuman.evaluate_model(net, os.path.join(paths.interhuman_dir,"test"), pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True,\
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node, device = self.device, \
											eval_save_dir="_interhuman")
				
				table = print_eval_result([eval_interhuman],filter_keys=set(eval_interhuman),title=["    ", "interhuman Test"])
				eval_trans = eval_interhuman

			elif self.eval_dataset_name == "uwb-imu":
				seq_ids = [0] if self.dry_run else None
				dataset_path = paths.uwbimu_dir if not self.use_virtual_uwb else os.path.join(paths.uwbimu_dir,"sigma0")
				eval_uwb_imu = evaluate_model(net, dataset_path, pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True,\
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node)
				table = print_eval_result([eval_uwb_imu],filter_keys=set(eval_uwb_imu),title=["    ", "UWB-IMU Test"])
				eval_trans = eval_uwb_imu
			elif self.eval_dataset_name == "uwb-mixed":
				seq_ids = [0] if self.dry_run else list(range(30))
				eval_amass = evaluate_model(net, os.path.join(paths.amass_dir,"test_split"), pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True,\
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node)
				seq_ids = [0] if self.dry_run else None
				dataset_path = paths.uwbimu_dir if not self.use_virtual_uwb else os.path.join(paths.uwbimu_dir,"sigma0")
				eval_uwb_imu = evaluate_model(net, dataset_path, pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True,\
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node)
				table = print_eval_result([eval_amass,eval_uwb_imu],filter_keys=set(eval_uwb_imu)& set(eval_amass),title=["    ", "AMASS Dance-DB", "UWB-IMU Test"])
				eval_trans = eval_uwb_imu
			elif self.eval_dataset_name == "uwb-syn":
				seq_ids = [0] if self.dry_run else list(range(30))
				eval_amass = evaluate_model(net, os.path.join(paths.amass_dir,"test_split"), pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True,\
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node)
				seq_ids = [0] if self.dry_run else None
				eval_dip = evaluate_model(net, paths.dipimu_dir, pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=False, \
			 							flush_cache=True,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb)
				eval_tc = evaluate_model(net, paths.totalcapture_dir, pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True, \
										flush_cache=True, evaluate_tran=True, plt_tran=False,sequence_ids=seq_ids,normalize_uwb=self.normalize_uwb,flatten_uwb=self.flatten_uwb,remove_node=self.remove_node)
				table = print_eval_result([eval_amass,eval_dip,eval_tc],filter_keys=set(eval_dip) & set(eval_tc)&set(eval_amass),title=["    ", "AMASS Dance-DB", "DIP-IMU Test","TotalCapture"])
				eval_trans = eval_tc
			else:
				raise KeyError("Invalid eval dataset name")
   
			with open(os.path.join(self.eval_dir,f"{phase}_e{epoch}_error_table.csv"), 'w', newline='') as fid:
				fid.write(table.get_csv_string())

			plt.plot([0] + [_ for _ in eval_trans["trans_error"].keys()], [0] + [torch.tensor(_).mean() for _ in eval_trans["trans_error"].values()], label=self.model.name)
			plt.legend(fontsize=15)
			plt.savefig(os.path.join(self.eval_dir,f"{phase}_e{epoch}_translation_error.png"))
			plt.close("all")
 
	def save_checkpoint(self,file_name,epoch):
		state = {'epoch':epoch,
				 'net':self.model.state_dict(),
				 'optim':self.optimizer.state_dict()
	 			}
		torch.save(state, file_name)
	
	def preprocess(self,data:Batch):
		return data.get_listed_batch(keys=["x_imu","lj_init","jvel_init"])
	

	def forward_model(self,x_input):
		y_pred = self.model(x_input)
		assert len(y_pred) == len(self.model.model_output)
		tmp = {k:v for k,v in zip(self.model.model_output, y_pred)}
		return D_Batch(tmp)
	
	def train_one_epoch(self,epoch, phase=''):
		total_time = 0
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		self.model.train()
		if self.resampling_interval > 0 and epoch % self.resampling_interval==0 and epoch!=0:
			self.dataset.resubsampling()
			self.data_loader = DataLoader(self.dataset,shuffle=True,
										pin_memory=True,
										batch_size=self.batch_size,
										num_workers=1)
   
   
		batch_idx = 0
		loss_train = Easy_dict({l.__name__:0 for l in self.loss_func.loss_func})
		grad_norm = 0
		loop_bar = tqdm(self.data_loader)
		for data in loop_bar:
			data = Batch(**data).to_device(self.device)
			data.uwb_normalized = self.normalize_uwb
			x_input = self.preprocess(data)
			
			y_pred = self.forward_model(x_input)
			if 'lgd' in phase:
				# learnable gradient descent need history data in model
				loss_dict = self.model.backward(y_pred,data)
			else:
				loss_dict = self.loss_func.compute_total_loss(y_pred,data)

			self.optimizer.zero_grad()
			loss_dict["total_loss"].backward()
   
			total_grad_norm = None
			if self.grad_clip > 0:
				total_grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.grad_clip)
			else:
				total_grad_norm = torch.linalg.norm(torch.cat([param.grad.view(-1) for param in self.optimizer.param_groups[0]["params"]]))
			
			self.optimizer.step()
			
			batch_idx += 1
   
			#logging 
			loss_train._add_item(loss_dict)
			grad_norm += total_grad_norm.item()
			total_loss = loss_dict["total_loss"].item()
			loop_bar.set_description(f"Phase:{phase}||Epoch:{epoch}/{self.epochs}||lr: {self.lr_scheduler.get_last_lr()[0]:.2E}||Loss: {total_loss:.4f}")
		
		loss_train._div(batch_idx)
		grad_norm /= batch_idx

		val_loss = self.validation(epoch,phase)
  
		logging_dict = {
			"train_loss":loss_train,
			"grad_norm":grad_norm,
			"val_loss":val_loss
		}

		if self.early_stop_check(val_loss.sum()):
			file_name = os.path.join(self.ckpt_dir, f"{phase}_best_model.pt")
			self.save_checkpoint(file_name=file_name,epoch=epoch)
   
		if wandb.run is not None:
			self.wandb_logging(epoch, phase, **logging_dict)
		self.tb_logging(epoch,phase,**logging_dict)

	def validation(self,epoch,phase):
		#self.model.eval()
		train_phase,module_name = phase.split("_",1)
		validation_loss = [l for l in self.loss_func.loss_func]
		val_loss_weight = [w for w in self.loss_func.loss_weight]
		batch_idx = 0
		loss_val = Easy_dict({loss_func.__name__:0 for loss_func in validation_loss})
		loop_bar = tqdm(self.val_dataloader)
		for data in loop_bar:
			data = Batch(**data).to_device(self.device)
			x_input = self.preprocess(data)
			
			y_pred = self.forward_model(x_input)

			loss = {loss_func.__name__:loss_func(data, y_pred) * w for w,loss_func in zip(val_loss_weight,validation_loss)}

			loss_val._add_item(loss)
			loop_bar.set_description(f"[Validation]Phase:{phase}||Epoch:{epoch}||")

			batch_idx += 1
		
		loss_val._div(deno=batch_idx)
		return loss_val