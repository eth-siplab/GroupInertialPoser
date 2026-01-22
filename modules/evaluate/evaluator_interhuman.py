'''
# --------------------------------------------
# Group Inertial Poser Network
# --------------------------------------------
# Group Inertial Poser: Multi-Person Pose and Global Translation from Sparse Inertial Sensors and Ultra-Wideband Ranging (ICCV 2025)
# https://github.com/eth-siplab/GroupInertialPoser
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import argparse
import os
from typing import Any
from config.config import *
import torch
import articulate as art
import tqdm
from articulate import math as M

from modules.evaluate.eval_utils import *
from modules.model import MODELS
from collections import OrderedDict
from prettytable import PrettyTable
import time

import math
import itertools


new_vi_mask = torch.tensor([1961, 5424, 1067, 4553, 457, 3021]) # "lh", "rh", "lk", "rk", "h", "r"
vi_mask = new_vi_mask

# constrain the cpu usage
torch.set_num_threads(8)

def bfgs_optimizer_2p_trans(pose, tran, pose2, tran2, uwb_2p_, num_steps, body_model, model_kwargs=None):
	"""
	optimize the two-person trajectory based on uwb distances.
	"""

 
	###### add velocity and acceleration regularization ######
	# velo_ori1 = tran[1:] - tran[:-1]
	# acc_ori1 = velo_ori1[1:] - velo_ori1[:-1]
 
	# velo_ori2 = tran[1:] - tran[:-1]
	# acc_ori2 = velo_ori2[1:] - velo_ori2[:-1]
	############
	tran_origin = tran.clone()
	tran2_origin = tran2.clone()
	
	with torch.enable_grad():
		tran = tran.clone().detach().contiguous().requires_grad_(True)
		tran2 = tran2.clone().detach().contiguous().requires_grad_(True)
  
		def closure():

			lbfgs.zero_grad()

			grot, joint, vert = body_model.forward_kinematics(pose = pose, tran = tran, calc_mesh=True)
			grot2, joint2, vert2 = body_model.forward_kinematics(pose = pose2, tran = tran2, calc_mesh=True)

			verts = torch.cat((vert[:, vi_mask], vert2[:, vi_mask]), dim=1)
			uwb_p = torch.cdist(verts,verts)# [bs, 6,6]

			uwb_loss =  torch.nn.MSELoss()(uwb_p[:, :6, -6:], uwb_2p_[:, :6, -6:]) 

			loss_smooth_tran1 = torch.nn.MSELoss()(tran[:, 1:]-tran[:, :-1], tran_origin[:, 1:]-tran_origin[:, :-1])  # Person 1 smoothness
			loss_smooth_tran2 = torch.nn.MSELoss()(tran2[:, 1:]-tran2[:, :-1], tran2_origin[:, 1:]-tran2_origin[:, :-1])  # Person 2 smoothness


 			# Acceleration loss (smooth motion over time)
			loss_acc = torch.nn.MSELoss()(tran[:, 2:] - 2 * tran[:, 1:-1] + tran[:, :-2], tran_origin[:, 2:] - 2 * tran_origin[:, 1:-1] + tran_origin[:, :-2])
			loss_acc2 = torch.nn.MSELoss()(tran2[:, 2:] - 2 * tran2[:, 1:-1] + tran2[:, :-2], tran2_origin[:, 2:] - 2 * tran2_origin[:, 1:-1] + tran2_origin[:, :-2])

			# print("uwb: ", uwb_loss, ", vel: ", loss_smooth_tran1+loss_smooth_tran2, ", acc: ", loss_acc + loss_acc2 )

			objective = uwb_loss + 0.05 * (loss_smooth_tran1+loss_smooth_tran2) + 0.05*(loss_acc +loss_acc2)
		
			objective.backward()

			return objective
		
		lbfgs = torch.optim.LBFGS([tran, tran2],
				history_size=10, 
				max_iter=4, 
				line_search_fn="strong_wolfe")
		
		for _ in range(num_steps):
			lbfgs.step(closure)

	return pose, tran.clone().detach(), pose2, tran2.clone().detach()

def bfgs_optimizer_init(pose, tran, pose2, tran2, uwb_2p_, num_steps, body_model, model_kwargs=None):
    """
	Refine the initial position (x,y,z) of the second person relative to the first
    """
    
    with torch.enable_grad():
        
        # Initialize x and z as the optimizable variables (initial root position), the second person to the first
        x = torch.zeros(1, device=pose.device, requires_grad=True)
        y = torch.zeros(1, device=pose.device, requires_grad=True)
        z = torch.zeros(1, device=pose.device, requires_grad=True)

        # Define the closure function for LBFGS optimizer
        def closure():
            lbfgs.zero_grad()
            
            # Construct new translation matrices tran2 with x and z updated
            translation_second = torch.cat([x, y, z]).unsqueeze(0)  # Shape (1, 3)
            new_tran2 = tran2 + translation_second

            # Forward kinematics with updated translations
            grot, joint, vert = body_model.forward_kinematics(pose=pose, tran=tran, calc_mesh=True)
            grot2, joint2, vert2 = body_model.forward_kinematics(pose=pose2, tran=new_tran2, calc_mesh=True)
            
            verts = torch.cat((vert[:, vi_mask], vert2[:, vi_mask]), dim=1)
            # Compute the pairwise distances between vertices
            uwb_p = torch.cdist(verts, verts)  # [bs, 12, 12]
            
            # Compute the objective (MSE loss between predicted and target UWB distances)
            objective = torch.nn.MSELoss()(uwb_p, uwb_2p_)  # self+inter
            
            # print(objective)
            objective.backward()
            return objective
        
        # Initialize the LBFGS optimizer with the optimizable variables x and z
        lbfgs = torch.optim.LBFGS([x, y, z],
                                  history_size=10,
                                  max_iter=4,
                                  line_search_fn="strong_wolfe")
        
        # Perform the optimization for the given number of steps
        for _ in range(num_steps):
            lbfgs.step(closure)
    
    # After optimization, update the translation matrices with the final x and z values
    optimized_translation = torch.cat([x, y, z]).unsqueeze(0).detach()
    tran2 = tran2.clone() + optimized_translation  

    return pose, tran, pose2, tran2


# Define the smoothing function
import torch.nn.functional as F
def smooth_positions_moving_average(data, window_size=5):
    # Ensure window size is odd to have a symmetric window around each point
    if window_size % 2 == 0:
        window_size += 1

    # Create a uniform averaging kernel
    kernel = torch.ones((3, 1, window_size)) / window_size  # Shape (1, 1, window_size)

    # Prepare data for 1D convolution
    # Add batch dimension and transpose to shape (1, 3, N)
    data = data.unsqueeze(0).transpose(1, 2)

    # Apply padding to the data along the time axis to handle edge effects
    padded_data = F.pad(data, (window_size // 2, window_size // 2), mode='replicate')

    # Apply the moving average smoothing using convolution
    smoothed_data = F.conv1d(padded_data, kernel, groups=3)  # groups=3 to apply independently on each 3D channel

    # Remove the extra dimensions to return to shape (N, 3)
    return smoothed_data.squeeze(0).transpose(0, 1)


def evaluate_model(net, data_dir, sequence_ids=None, flush_cache=False, pose_evaluator=PoseEvaluator(),
			 evaluate_pose=True, evaluate_tran=False, evaluate_zmp=False, plt_tran=False, device='cpu', eval_save_dir='interhuman', **kwargs):
	r"""
	Evaluate poses and translations of `net` on all sequences in `sequence_ids` from `data_dir`.
	`net` should implement `net.name` and `net.predict(glb_acc, glb_rot)`.
	"""
	if 'MAM' in net.name: net = net.to(device)

	start_time = time.time()
	paths.result_dir = os.path.join(paths.result_dir, eval_save_dir)
	data_name = os.path.basename(data_dir)
	result_dir1 = os.path.join(paths.result_dir, "person1", net.name)
	result_dir2 = os.path.join(paths.result_dir, "person2", net.name)
	print_title('Evaluating "%s" on "%s"' % (net.name, data_name))

	_, _, pose_t_all, tran_t_all, *res = torch.load(os.path.join(data_dir, "person1", 'test.pt'), weights_only=True).values() #[45,Size([4113, 24, 3])],[45,torch.Size([4113, 3])]
	_, _, pose_t_all2, tran_t_all2, *res2 = torch.load(os.path.join(data_dir, "person2", 'test.pt'), weights_only=True).values() #[45,Size([4113, 24, 3])],[45,torch.Size([4113, 3])]
	
	vuwb_2p = torch.load(os.path.join(data_dir, 'vuwb_2human12.pt'), weights_only=True)
	print(f"Total data duration: {sum(pose_.shape[0] for pose_ in pose_t_all if pose_.shape[0]>200)/ 3600:.2f} minutes")
	
	if res is not None and res[-1] is not None and isinstance(res[-1][0],str):
		file_path = res[-1]
	else:
		file_path = None
  
	if sequence_ids is None:
		sequence_ids = list(range(len(pose_t_all))) #[0~44]
	if flush_cache and os.path.exists(result_dir1):
		shutil.rmtree(result_dir1)
	if flush_cache and os.path.exists(result_dir2):
		shutil.rmtree(result_dir2)


	missing_ids = [i for i in sequence_ids if not os.path.exists(os.path.join(result_dir1, '%d.pt' % i)) or not os.path.exists(os.path.join(result_dir2, '%d.pt' % i))]
	cached_ids = [i for i in sequence_ids if os.path.exists(os.path.join(result_dir1, '%d.pt' % i)) and os.path.exists(os.path.join(result_dir2, '%d.pt' % i))]
	print('Cached ids: %s\nMissing ids: %s' % (cached_ids, missing_ids))
	if len(missing_ids) > 0:
		run_pipeline(net, os.path.join(data_dir, "person1"), missing_ids,**kwargs)
		run_pipeline(net, os.path.join(data_dir, "person2"), missing_ids,**kwargs)

	output_eval_res = OrderedDict()
	pose_errors = []
	tran_errors = {window_size: [] for window_size in list(range(1, 14))}
	zmp_errors = []
	tran_rmse = []
	tran_mse = []
	tran_mae = []
	
	GT_disatances = []
	Pred_distances = []

	body_model = art.ParametricModel(paths.smpl_file, device = device)
	for i in tqdm.tqdm(sequence_ids):
		try:
			result1 = torch.load(os.path.join(result_dir1, '%d.pt' % i),weights_only=True)
			result2 = torch.load(os.path.join(result_dir2, '%d.pt' % i),weights_only=True)
		except:
			print('Failed to load %d.pt' % i)
			continue

		pose_p, tran_p = result1[0].to(device), result1[1].to(device)   # torch.Size([4113, 24, 3, 3]) torch.Size([4113, 3])
		pose_p2, tran_p2 = result2[0].to(device), result2[1].to(device)
		pose_t, tran_t = pose_t_all[i].to(device), tran_t_all[i].to(device) # torch.Size([429, 24, 3]
		pose_t2, tran_t2 = pose_t_all2[i].to(device), tran_t_all2[i].to(device)
		
		vuwb_2p_ = vuwb_2p[i].to(device)

		
		############## the initial position using GT:###############
		tran_p[:] -= tran_p[0] - tran_t[0]
		tran_p2[:] -= tran_p2[0] - tran_t2[0]

		#### if we use optimization:
		if True:
			pose_p, tran_p, pose_p2, tran_p2 = bfgs_optimizer_init(pose_p, tran_p, pose_p2, tran_p2, vuwb_2p_, 10, body_model)
			pose_p, tran_p, pose_p2, tran_p2 = bfgs_optimizer_2p_trans(pose_p, tran_p, pose_p2, tran_p2, vuwb_2p_, 10, body_model)
			
  

		# match the position of the first person. for visualization compare
		discrepancy = tran_p[0] - tran_t[0]
		tran_p -= discrepancy
		tran_p2 -= discrepancy


		pose_p, tran_p = pose_p.cpu(), tran_p.cpu()
		pose_p2, tran_p2 = pose_p2.cpu(), tran_p2.cpu()
		pose_t, tran_t = pose_t.cpu(), tran_t.cpu()
		pose_t2, tran_t2 = pose_t2.cpu(), tran_t2.cpu()
		vuwb_2p_ = vuwb_2p_.cpu()

		# first calculate the predicted distance, and then the GT distance.
		pred_d = torch.linalg.norm(tran_p-tran_p2, dim=1)
		gt_d = torch.linalg.norm(tran_t- tran_t2, dim=1)
		
		Pred_distances.append(pred_d)
		GT_disatances.append(gt_d)

		if evaluate_pose:
			pose_t = art.math.axis_angle_to_rotation_matrix(pose_t).view_as(pose_p) #torch.Size([4113, 24, 3, 3])
			pose_errors.append(pose_evaluator(pose_p, pose_t, tran_p, tran_t))
			pose_t2 = art.math.axis_angle_to_rotation_matrix(pose_t2).view_as(pose_p2) #torch.Size([4113, 24, 3, 3])
			pose_errors.append(pose_evaluator(pose_p2, pose_t2, tran_p2, tran_t2))
		if evaluate_tran:
			# compute gt move distance at every frame
			move_distance_t = torch.zeros(tran_t.shape[0])
			v = (tran_t[1:] - tran_t[:-1]).norm(dim=1)
			for j in range(len(v)):
				move_distance_t[j + 1] = move_distance_t[j] + v[j]

			for window_size in tran_errors.keys():
				# find all pairs of start/end frames where gt moves `window_size` meters
				frame_pairs = []
				start, end = 0, 1
				while end < len(move_distance_t):
					if move_distance_t[end] - move_distance_t[start] < window_size:
						end += 1
					else:
						if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
							frame_pairs.append((start, end))
						start += 1

				# calculate mean distance error
				errs = []
				for start, end in frame_pairs:
					vel_p = tran_p[end] - tran_p[start] 
					vel_t = tran_t[end] - tran_t[start]
					errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
				if len(errs) > 0:
					tran_errors[window_size].append(sum(errs) / len(errs))
		if evaluate_tran:
			# compute gt move distance at every frame
			move_distance_t = torch.zeros(tran_t2.shape[0])
			v = (tran_t2[1:] - tran_t2[:-1]).norm(dim=1)
			for j in range(len(v)):
				move_distance_t[j + 1] = move_distance_t[j] + v[j]

			for window_size in tran_errors.keys():
				# find all pairs of start/end frames where gt moves `window_size` meters
				frame_pairs = []
				start, end = 0, 1
				while end < len(move_distance_t):
					if move_distance_t[end] - move_distance_t[start] < window_size:
						end += 1
					else:
						if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
							frame_pairs.append((start, end))
						start += 1

				# calculate mean distance error
				errs = []
				for start, end in frame_pairs:
					vel_p = tran_p2[end] - tran_p2[start] 
					vel_t = tran_t2[end] - tran_t2[start]
					errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
				if len(errs) > 0:
					tran_errors[window_size].append(sum(errs) / len(errs))
		if evaluate_tran:
			######### for the case of estimated init position#########
			# make sure the inite height descrepancy is not counted
			# also put the first person origin to be matched:
			discrepancy = tran_p[0] - tran_t[0]
			tran_p -= discrepancy
			tran_p2 -= discrepancy
			#############################################
			distances1 = torch.linalg.norm(tran_p-tran_t, dim=1)
			distances2 = torch.linalg.norm(tran_p2-tran_t2, dim=1)

			tran_rmse.append(torch.sqrt(torch.mean(distances1**2)))
			tran_rmse.append(torch.sqrt(torch.mean(distances2**2)))
   
			tran_mae.append(torch.mean(distances1))
			tran_mae.append(torch.mean(distances2))

		if evaluate_zmp:
			zmp_errors.append(evaluate_zmp_distance(pose_p, tran_p))
			zmp_errors.append(evaluate_zmp_distance(pose_p2, tran_p2))


	# torch.save(Pred_distances, f"tmp_pth/{result_dir1.split('/')[-3]}_{result_dir1.split('/')[-1]}_Pred.pth")
	# torch.save(GT_disatances, f"tmp_pth/{result_dir1.split('/')[-3]}_{result_dir1.split('/')[-1]}_GT.pth")


	time_indx =  np.arange(0, 60*70, 60) # 0,1,2,...13 s
	results = {x:[] for x in time_indx}
	for i in range(len(Pred_distances)):
		distance_error = np.abs(GT_disatances[i] - Pred_distances[i]) # [#frames]
		for ti in time_indx:
			if ti >= len(distance_error): break
			results[ti].append(distance_error[ti])
	# result_mean1 = {ti:np.mean(results[ti]) for ti in time_indx}
	result_mean = {i:round(np.mean(results[ti])*100, 2)  for i, ti in enumerate(time_indx)} # s: dist. error  *100 m->cm
	result = dict(itertools.takewhile(lambda item: not math.isnan(item[1]), result_mean.items()))
	print("==========================================")
	print("distances error between-human(cm), from the second 0, 1, 2...")
	print(result)
	print()
	print([result[i] for i in result])
	print("==========================================")


	if evaluate_pose:
		pose_errors_mean = torch.stack(pose_errors).mean(dim=0)
		pose_errors_max = torch.stack(pose_errors).max(dim=0)
		pose_worse_idx = torch.mode(pose_errors_max.indices[:,0]).values.item()
		pose_errors_min = torch.stack(pose_errors).min(dim=0)
		pose_best_idx = torch.mode(pose_errors_min.indices[:,0]).values.item()
		
		for name, error in zip(pose_evaluator.names, pose_errors_mean):#['SIP Error (deg)', 'Angle Error (deg)', 'Joint Error (cm)', 'Vertex Error (cm)', 'Jitter Error (km/s^3)']
			#print('%s: %.4f' % (name, error[0]))
			output_eval_res[name] = error[0]
   
		if file_path is None:
			output_eval_res['best'] = (os.path.join(result_dir1, '%d.pt' % pose_best_idx), pose_errors[pose_best_idx][:,0])
			output_eval_res['worst'] = (os.path.join(result_dir1, '%d.pt' % pose_worse_idx), pose_errors[pose_worse_idx][:,0])
		else:
			output_eval_res['best'] = (file_path[pose_best_idx], pose_errors[pose_best_idx][:,0])
			output_eval_res['worst'] = (file_path[pose_worse_idx], pose_errors[pose_worse_idx][:,0])
   
		# print([(str(seq_id),e) for seq_id,e in enumerate(pose_errors)])
			
	if evaluate_zmp:
		print('ZMP Distance (m): %.4f' % (sum(zmp_errors) / len(zmp_errors)))
		output_eval_res["ZMP Distance"] = sum(zmp_errors) / len(zmp_errors)
		
	if evaluate_tran:
		if plt_tran:
			plt.plot([0] + [_ for _ in tran_errors.keys()], [0] + [torch.tensor(_).mean() for _ in tran_errors.values()], label=net.name)
			plt.legend(fontsize=15)
			plt.show()
		for i in tran_errors.keys():
			print(i+1, torch.tensor(tran_errors[i]).mean().item())
		output_eval_res["trans_error"] = tran_errors
		output_eval_res["trans_error_2m"] = torch.tensor(tran_errors[1]).mean().item()
		output_eval_res["trans_error_5m"] = torch.tensor(tran_errors[4]).mean().item()
		output_eval_res["trans_error_8m"] = torch.tensor(tran_errors[7]).mean().item()
		output_eval_res["trans_error_11m"] = torch.tensor(tran_errors[10]).mean().item()
		output_eval_res["trans_rmse"] = torch.tensor(tran_rmse).mean().item()
		output_eval_res["tran_mse"] = torch.tensor(tran_mse).mean().item()
		output_eval_res["tran_mae"] = torch.tensor(tran_mae).mean().item()
  
		print('%s: %.4f' % ("trans error at 2 m", torch.tensor(tran_errors[1]).mean().item()))
		print('%s: %.4f' % ("trans error at 5 m", torch.tensor(tran_errors[4]).mean().item()))
		end_time = time.time()
		print(f"Evaluation time: {(end_time - start_time) / 60:.2f} minutes")
		output_eval_res["Evaluation time (min): "] = f'{(end_time - start_time) / 60:.2f}'
		
		# torch.save(tran_errors, f"tmp_pth/{result_dir1.split('/')[-3]}_{result_dir1.split('/')[-1]}___tran_erros.pth")
		tran_error_list = {i:torch.tensor(tran_errors[i]).mean().item() for i in tran_errors}
		# print(tran_error_list)
		print("=============trans_error_*m=============================")
		tran_error_list_round2 = {k: round(v*100, 2) if not isinstance(v, float) or not str(v).lower() == 'nan' else v for k, v in tran_error_list.items()}
		print(tran_error_list_round2)
		print()
		print([tran_error_list_round2[i] for i in tran_error_list_round2])
		print("==========================================")
		print()

		
		
	return output_eval_res
 

def print_eval_result(eval_dicts: list, filter_keys: set, title = ["   ", "DIP-IMU","TotalCapture"]):
	tab = PrettyTable(title)
	key_list = [key for key in eval_dicts[0].keys() if key in filter_keys]
	for key in key_list:
		if key != "trans_error":
			tab.add_row([key] + [f"{d[key]}" for d in eval_dicts])
	print(tab)
	return tab

def get_args():
	parser = argparse.ArgumentParser(description='Evaluation process')
	parser.add_argument('--network', type=str, default="UIP",
						help='network name for evaluating')
	parser.add_argument('--ckpt_path', type=str, default="",
						help='model weight for evaluating')
	parser.add_argument('--data_dir', type=str, default=0.0,
						help='test data directory for evaluation')
	parser.add_argument('--seq_id', nargs='+',
						help="Specify the sequence id for test dataset")
	parser.add_argument('--render', action='store_true',
						help='whether to use pybullet to render test results')
	parser.add_argument('--flush_cache', action='store_true',
						help='whether to flush cached results')
	parser.add_argument('--eval_trans', action='store_true',
						help='whether to evaluate global translation')
	parser.add_argument('--eval_save_dir', type=str, default="",
						help='dir to save evaluation results')
	parser.add_argument('--normalize_uwb', action='store_true',
						help='whether to normalize uwb value by head-pelvis distance')
	parser.add_argument('--flatten_uwb', action='store_true',
						help='whether to flatten uwb into vector of size 15')
	parser.add_argument('--add_guassian_noise', action='store_true',
						help='whether to add gaussian noise')
	parser.add_argument('--no_rnn_init', action='store_true',
						help='whether to remove rnn initial')
	parser.add_argument('--model_args_file', type=str, default="",
						help='Config file for the model .ini')
	parser.add_argument('--exp_name', type=str, default="",
						help='name add after the network name')
	parser.add_argument("--device",default="cuda",help="device for training")
	
	args = parser.parse_known_args()
	return args[0], parser

def main():
	args, parser = get_args()
	sequence_ids = [int(i) for i in args.seq_id] if args.seq_id else None
	print(f"data dir  : {args.data_dir}")
	if args.network in MODELS:
		model_cls = MODELS[args.network]
		print(model_cls)
		if args.model_args_file:
			net = model_cls.load_model_with_args(args.model_args_file)
		else:
			model_cls.add_args(parser)
			model_args = parser.parse_args()
			net = model_cls(model_args)
		missing_keys, unexpected_keys = net.load_state_dict(load_ckpt(args.ckpt_path),strict=False)


		net.name = net.name + args.exp_name
		print("network name:",net.name)
		eval_res = evaluate_model(net,data_dir=args.data_dir,sequence_ids=sequence_ids,flush_cache=args.flush_cache,evaluate_tran=args.eval_trans,pose_evaluator=PoseEvaluator(),
                          			normalize_uwb=args.normalize_uwb,flatten_uwb=args.flatten_uwb, device = args.device, eval_save_dir=args.eval_save_dir)
		y_axis_up = True
		gt_dir = args.data_dir
	else:
		raise KeyError("Invalid network name")

	if args.eval_save_dir:
		save_to = os.path.join("output", "eval", args.eval_save_dir)
		os.makedirs(save_to,exist_ok=True)
     
		tab = print_eval_result([eval_res],filter_keys=set(eval_res),title=["Dataset",args.data_dir])

		with open(os.path.join(save_to,f"[Eval_tab]{args.network}.csv"), 'w', newline='') as fid:
			fid.write(tab.get_csv_string())

		if "trans_error" in eval_res:
			plt.plot([0] + [_ for _ in eval_res["trans_error"].keys()], [0] + [torch.tensor(_).mean() for _ in eval_res["trans_error"].values()], label=args.network)
			plt.legend(fontsize=15)
			plt.savefig(os.path.join(save_to,f"Translation_error_{args.network}.png"))
			plt.close("all")

		
if __name__ == "__main__":
	main()
