import numpy as np
import random 
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from config.config import *
from articulate.model import ParametricModel
from articulate.math import *
from articulate.utils.bullet import *
from articulate.utils.rbdl import *
from modules.dataset.data_utils import *
from scipy.spatial.transform import Rotation as R

class _Dataset(Dataset):
    def __init__(self,
                 official_model_file: str,
                 rep = RotationRepresentation.AXIS_ANGLE,
                 device = torch.device('cpu'),
                 seq_length = 200,
                 down_sample_rate = 60,
                 use_cached = True,
                 add_uwb = False,
                 train_split = True,
                 imu_m = [""],
                 static_uwb_noise = -1,
                 normalize_uwb = False,
                 uwb_timesample_ratio = None,
                 flatten_uwb = False,
                 dry_run = False,
                 remove_node = -1,
                 **kwargs) -> None:
        super().__init__()
        self.model = ParametricModel(official_model_file, use_pose_blendshape=False, device=device)
        self.seq_length = seq_length
        self.down_sample_rate = down_sample_rate
        self.rep = rep
        self.use_cached = use_cached
        self.device = device
        self.add_uwb = add_uwb
        self.normalize_uwb = normalize_uwb
        self.train_split = train_split
        self.imu_m = imu_m
        self.static_uwb_noise = static_uwb_noise
        self.with_acc_sum = "acc_sum" in imu_m
        self.uwb_timesample_ratio = uwb_timesample_ratio
        self.flatten_uwb = flatten_uwb
        self.remove_node = remove_node
        self.dry_run = dry_run

    def _preprocess(self, pose, shape=None, tran=None):
        pose = to_rotation_matrix(pose.to(self.device), self.rep).view(pose.shape[0], -1)
        shape = shape.to(self.device) if shape is not None else shape
        tran = tran.to(self.device) if tran is not None else tran
        return pose, shape, tran
    
    def __len__(self):
        return self.seq_num
    
    
    def _get_subsample_seq_range(self,N_frames):
        frame_range = range(self.seq_length,N_frames)
        num_samples = np.maximum(round(len(frame_range) / self.down_sample_rate), 1)
        for t_end in random.sample(frame_range, k=num_samples):
            yield t_end
            
    def _check_cached(self,dir,file):
        file_path = os.path.join(dir,file)
        if os.path.exists(file_path) and self.use_cached:
            print("Use cached dataset from ",file_path)
            return True
        return False
    
    @staticmethod
    def add_syn_uwb_noise(uwb,sigma=-1,uwb_collision=None,normalize=False):
        if sigma >= 0:
            std = sigma
            noise = torch.normal(0, std=std, size=uwb.size()).to(uwb.device)
        else:
            #sigma = 0.021 + 0.062 * (uwb_collision > 0.2)
            #sigma = 0.051 + 0.062 * torch.max(torch.zeros_like(uwb_collision),(uwb_collision-0.3)/(1-0.3))
            #sigma from 0.04 to 0.18 assume uncertainty grows linearly with percentage of collision
            sigma = 0.051 + 0.032 * (uwb_collision > uwb_collision_thr)
            mean = torch.zeros_like(sigma)
            noise = torch.normal(mean=mean, std=sigma).to(uwb.device)
            
        #make it symmetric and diag zero
        noise = (noise + noise.transpose(1,2))/np.sqrt(2)
        torch.diagonal(noise, dim1=1, dim2=2).zero_()
        
        if normalize:
            d_root2head = uwb[0,4,5].clone() #normalized by distance between head and root, hard code
            return (uwb + noise) / d_root2head
        else:
            return uwb + noise
    
    @staticmethod
    def preprocess(pose, shape=None, tran=None, device="cpu", rep=RotationRepresentation.AXIS_ANGLE):
        pose = to_rotation_matrix(pose.to(device), rep).view(pose.shape[0], -1)
        shape = shape.to(device) if shape is not None else shape
        tran = tran.to(device) if tran is not None else tran
        return pose, shape, tran
    
    @staticmethod
    def get_non_root_joint_pos(glb_joint_pos,glb_root_ori):
        _r = glb_joint_pos[:, 1:, :] - glb_joint_pos[:, joint_set.root, :]
        return torch.einsum('bij,bjk->bik', _r, glb_root_ori.view(-1,3,3)) 
       
    @staticmethod
    def get_leaf_joint(glb_joint_pos,glb_root_ori):
        # get leaf joint position relative to root
        _r =  (glb_joint_pos[:, joint_set.leaf,:] - glb_joint_pos[:, joint_set.root])
        return torch.einsum('bij,bjk->bik', _r, glb_root_ori.view(-1,3,3))
    
    @staticmethod
    def  get_6d_rotation(global_rotmat,glb_root_ori):
        #get 6d rotation relative to root
        N_f = global_rotmat.size(0)
        global_rotmat = torch.einsum('abij,abjk->abik', glb_root_ori.view(-1,1,3,3).transpose(2,3), global_rotmat)
        return global_rotmat.view(N_f, -1, 3, 3)[:, :, :, :2].transpose(2, 3).contiguous().view(N_f,-1 ,6)
    
    @staticmethod
    def get_joint_vel(glb_joint_pos, glb_root_ori):
        N_f = glb_joint_pos.size(0)
        joint_vel = torch.zeros_like(glb_joint_pos)
        joint_vel[1:] = (glb_joint_pos[1:,:,:] - glb_joint_pos[:-1,:,:]) * 20
        joint_vel = torch.einsum('bij,bjk->bik', joint_vel, glb_root_ori.view(N_f,3,3))
        joint_vel[0] = joint_vel[1]
        return joint_vel 
    
    @staticmethod
    def get_contact_points(glb_joint_pos, th=0.0125):
        feet_contact = torch.zeros(glb_joint_pos.size(0),2)
        feet_contact[1:,:] = torch.linalg.norm(glb_joint_pos[1:,10:12,:] - glb_joint_pos[:-1,10:12,:], dim=2) < th
        feet_contact[0] = feet_contact[1]
        feet_contact = feet_contact.to(torch.int)
        return feet_contact
    
    @staticmethod
    def get_acc_sum(acc,window_size=40,down_scale=15):
        b = torch.cumsum(acc.view(-1,18),dim=0)
        b[window_size:, :] = b[window_size:, :] - b[:-window_size, :]
        b = b / down_scale #down scale to acc scale
        return b.view_as(acc)
    
    @staticmethod
    def get_local_imu(glb_acc, glb_rot):
        glb_acc = glb_acc.view(-1, 6, 3)
        glb_rot = glb_rot.view(-1, 6, 3, 3)
        acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_rot[:, -1])
        ori = torch.cat((glb_rot[:, 5:].transpose(2, 3).matmul(glb_rot[:, :5]), glb_rot[:, 5:]), dim=1)
        return acc, ori
    
    def _data_sampling(self, seq_info):
        """
        Subsample data sequence of different length into chunks of the same length(self.seq_length) for training.

        Args:
            seq_info seq_id:(leaf_joint,joint_vel,pose_global_6d,feet_contact,non_root_joint,imu_ori,imu_acc,*res)

        Returns:
            data_info,data_idx
        """
        seq_list = sorted(list(seq_info.keys())) #list of seq_ids
                
        data_info = {"lfj":[],"jv":[],"jrot_6d":[],"contact":[],"vrot":[],"vacc":[],"joint":[]}
        #if self.add_uwb:
        data_info["vuwb"] = []
        data_info["uwb_gt"] = []
        data_info["offset"] = []
        #if self.with_acc_sum:
        data_info["acc_sum"] = []
        
        data_idx = 0
        print(f"Sampling Dataset {self.__class__.__name__}")
        for seq_id in tqdm(seq_list):
            
            #leaf_joint,joint_vel,pose_global_6d,feet_contact,non_root_joint,imu_ori,imu_acc,*res = seq_info[seq_id]
            seq_d = seq_info[seq_id]
            assert isinstance(seq_d, SeqInfo)
            for t_end in self._get_subsample_seq_range(N_frames=seq_d.frame_size):
                assert t_end >= self.seq_length
                data_info["lfj"].append(seq_d.leaf_joint[t_end-self.seq_length : t_end].to(device=self.device))  #N,5,3
                data_info["jv"].append(seq_d.joint_vel[t_end-self.seq_length : t_end].to(device=self.device))    #N,J,3
                data_info["jrot_6d"].append(seq_d.pose_global_6d[t_end-self.seq_length : t_end, joint_set.reduced, :].to(device=self.device)) #N,J,6
                data_info["contact"].append(seq_d.feet_contact[t_end-self.seq_length : t_end].to(device=self.device))   #N,2
                data_info["joint"].append(seq_d.non_root_joint[t_end-self.seq_length : t_end, : , : ].to(device=self.device))
                data_info["vrot"].append(seq_d.imu_ori[t_end-self.seq_length : t_end].to(device=self.device))
                data_info["vacc"].append(seq_d.imu_acc[t_end-self.seq_length : t_end].to(device=self.device))
                if seq_d.with_uwb():
                    data_info["vuwb"].append(seq_d.uwb_m[t_end-self.seq_length : t_end].to(device=self.device))
                    data_info["uwb_gt"].append(seq_d.uwb_gt[t_end-self.seq_length : t_end].to(device=self.device))
                    data_info["offset"].append(seq_d.offset.to(device=self.device))
                if seq_d.with_acc_sum():
                    data_info["acc_sum"].append(seq_d.acc_sum[t_end-self.seq_length : t_end].to(device=self.device))
                data_idx += 1 
        return data_info, data_idx
    
    def resubsampling(self):
        """
        Resubsampling the data sequence with the cached preprocessed data

        Returns:
            None
        """
        #seq_info = torch.load(os.path.join(self.dataset_dir,"dataset_cached.pt"))
                                  
        data_info,data_idx = self._data_sampling(self.seq_info)
        
        self.seq_num = data_idx
        self.data_info = data_info
        
    
    def __getitem__(self, index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            X: IMU, lj_init, jvel_init (tensor [num_frames, 72], tensor [15], tensor [72])
            Y: Joint_position, Joint_velocity, Joint_leaf_position
            ([num_frames, 24,3]) ([num_frames, 24,3]) ([num_frames, 5,3])
        """
        kwargs_data = {}
        sensor_measure = []
        # if "vrot" and "vacc" in self.imu_m:
        #     x_rot = self.data_info["vrot"][index]
        #     x_acc = self.data_info["vacc"][index]
        #     #x_imu = normalize_and_concat(x_acc,x_rot) 
        #     x_imu = torch.cat((x_acc.flatten(1), x_rot.flatten(1)), dim=1)
        #     sensor_measure.append(x_imu)
        
        if self.imu_m:
            sensor_measure += [self.data_info[k][index].flatten(1) for k in self.imu_m]
            
        kwargs_data["x_imu"] = torch.cat(sensor_measure, dim=1)
        kwargs_data["lj_init"] = self.data_info["lfj"][index][0].view(-1)
        kwargs_data["jvel_init"] = self.data_info["jv"][index][0].view(-1)
        
        
        #Make it adjustable given the training phase for it
        kwargs_data["lfj_gt"] = self.data_info["lfj"][index].view(self.seq_length,15)
        kwargs_data["joint_gt"] = self.data_info["joint"][index].view(self.seq_length,69)
        kwargs_data["jrot_6d"] = self.data_info["jrot_6d"][index].view(self.seq_length,90)
        kwargs_data["jvel_gt"] = self.data_info["jv"][index].view(self.seq_length,72)
        kwargs_data["contact_p"] = self.data_info["contact"][index].view(self.seq_length,2)
        
        if "uwb_gt" in self.data_info:
            kwargs_data["uwb_gt"] = self.data_info["uwb_gt"][index].view(self.seq_length,-1)
            kwargs_data["vuwb"] = self.data_info["vuwb"][index].view(self.seq_length,-1)
            kwargs_data["uwb_offset"] = self.data_info["offset"][index].view(6,3)
            
        return kwargs_data

'''
AMASS Dataset
'''
class AMASS_syn_data(_Dataset):
    def __init__(self,amass_dir: str, **superkwargs) -> None:
        super().__init__(**superkwargs)
        self.dataset_dir = amass_dir

        seq_info = self._data_preprocess(self.dataset_dir)

        data_info,data_idx = self._data_sampling(seq_info)
    
        self.seq_num = data_idx
        
        self.data_info = data_info
        
        self.seq_info = seq_info
        
    def _data_preprocess(self,amass_dir): 

        if self._check_cached(amass_dir,"dataset_cached.pt") and not self.dry_run:
            return torch.load(os.path.join(amass_dir,"dataset_cached.pt"), weights_only=False)

        seq_info = {}
        
        keys = ["pose", "shape", "tran", "joint", "vrot", "vacc","vuwb","uwb_collision","offset"]
        
        raw_data = {
                        k:torch.load(os.path.join(amass_dir,f"{k}.pt"), weights_only=False)
                        for k in keys if os.path.exists(os.path.join(amass_dir,f"{k}.pt"))
                     }
        
        N_seq = len(raw_data["pose"]) if not self.dry_run else 1
        print(f"Preprocessing Dataset {self.__class__.__name__}")
        # sum_len = 0
        for seq_id in tqdm(range(N_seq)):
            joint_pos = raw_data["joint"][seq_id]
            N_frame = joint_pos.size(0)
            
            if N_frame <= self.seq_length:
                #print(f"Ignore seq{seq_id} with frame_num {N_frame}")
                continue
            # sum_len += raw_data["pose"][seq_id].shape[0]
            seq_kwargs = {}
            seq_kwargs["imu_acc"],seq_kwargs["imu_ori"] = \
                                        self.get_local_imu(self.add_syn_noise(raw_data["vacc"][seq_id]), raw_data["vrot"][seq_id])
            # seq_kwargs["imu_ori"] = raw_data["vrot"][seq_id]
            # seq_kwargs["imu_acc"] = self.add_syn_noise(raw_data["vacc"][seq_id])
            #Forward Ka
            pose_p, shape_p, tran_p = self._preprocess(raw_data["pose"][seq_id],
                                                     raw_data["shape"][seq_id],
                                                     raw_data["tran"][seq_id])
            
            #Compute pose/joint in world coordinate
            pose_global_p, joint_p = self.model.forward_kinematics(pose_p, shape_p, tran_p, calc_mesh=False)
            
            #TODO: replace FK root_pos/root_ori with gt_pos/gt_ori
            glb_root_ori = raw_data["vrot"][seq_id][:, [-1], ...]
            #glb_root_ori = pose_global_p[:, joint_set.root, ...]
            
            #get leaf joint position
            seq_kwargs["leaf_joint"] = self.get_leaf_joint(joint_pos, glb_root_ori)
            
            #non-root joint position wrt root position
            seq_kwargs["non_root_joint"] = self.get_non_root_joint_pos(joint_pos, glb_root_ori)
            
            #get 6D joint rotation
            seq_kwargs["pose_global_6d"] = self.get_6d_rotation(pose_global_p, glb_root_ori)
                     
            #get joint velocity 
            seq_kwargs["joint_vel"] = self.get_joint_vel(joint_pos, glb_root_ori)
            
            #contact points
            seq_kwargs["feet_contact"] = self.get_contact_points(joint_pos)
            
            #uwb
            uwb_c = raw_data["uwb_collision"][seq_id] if "uwb_collision" in raw_data else torch.zeros_like(raw_data["vuwb"][seq_id])
            uwb = self.add_syn_uwb_noise(raw_data["vuwb"][seq_id], sigma=self.static_uwb_noise, uwb_collision = uwb_c,normalize=self.normalize_uwb)
            if self.remove_node == -1:
                seq_kwargs["uwb_m"] = uwb
            else:
                uwb[:,self.remove_node,:] = 0
                uwb[:,:,self.remove_node] = 0
                seq_kwargs["uwb_m"] = uwb
            seq_kwargs["uwb_gt"] = raw_data["vuwb"][seq_id]
            seq_kwargs["uwb_c"] = uwb_c
            seq_kwargs["offset"] = raw_data["offset"][seq_id]
            #acceleration sum
            seq_kwargs["acc_sum"] = self.get_acc_sum(seq_kwargs["imu_acc"])

            seq_info_ = SeqInfo(seq_id=seq_id,frame_size=N_frame,**seq_kwargs)
            if self.uwb_timesample_ratio is not None and self.uwb_timesample_ratio != 1:
                seq_info_.syc_measurement(sampling_ratio=self.uwb_timesample_ratio)

            if self.flatten_uwb:
                seq_info_.flatten_uwb_measurement()
                
            seq_info[seq_id] = seq_info_
        
        # print(f'==== {amass_dir} has data >200 in total (min): { sum_len/3600:.2f}' )
        # breakpoint()
        if not self.dry_run:
            torch.save(seq_info,os.path.join(amass_dir,"dataset_cached.pt"))

        return seq_info

    def add_syn_noise(self,acc,bias_noise=0.1):
        noise = 2 * torch.rand_like(acc) * bias_noise - bias_noise
        return acc + noise
  
'''
AMASS Val Data
'''
class AMASS_syn_data_val(AMASS_syn_data):
    def __init__(self, amass_dir: str, **superkwargs) -> None:
        amass_dir = os.path.join(amass_dir,"test_split")
        super().__init__(amass_dir, **superkwargs)
        
class AMASS_syn_data_train_tc(AMASS_syn_data):
    def __init__(self, amass_dir: str, **superkwargs) -> None:
        amass_dir = os.path.join(amass_dir,"syn_tc_1.0")
        super().__init__(amass_dir, **superkwargs)


class UWBIMU_real_data(_Dataset):
    def __init__(self,data_dir: str, split,**superkwargs) -> None:
        super().__init__(**superkwargs)
        self.dataset_dir = data_dir
        
        seq_info = self._data_preprocess(self.dataset_dir,split=split)
        
        data_info,data_idx = self._data_sampling(seq_info)
        
        self.seq_num = data_idx
        self.data_info = data_info
        self.seq_info = seq_info
        print(f"Loading data size of {self.seq_num}..")

    def _data_preprocess(self,uwb_dir,split="train"):
        
        if self._check_cached(uwb_dir,"dataset_cached.pt") and self.train_split and not self.dry_run:
            return torch.load(os.path.join(uwb_dir,"dataset_cached.pt"))
        
        seq_info = {}
        
        print("Load raw data from ",os.path.join(uwb_dir,f"{split}.pt"))
        raw_data = torch.load(os.path.join(uwb_dir,f"{split}.pt"), weights_only=True)

        N_seq = len(raw_data["pose"]) if not self.dry_run else 1
        print(f"Preprocessing Dataset {self.__class__.__name__}")
        for seq_id in tqdm(range(N_seq)):
            N_frame = raw_data["pose"][seq_id].size(0)
            
            if N_frame <= self.seq_length:
                continue
            
            seq_kwargs = {}
            seq_kwargs["imu_acc"],seq_kwargs["imu_ori"] = \
                                        self.get_local_imu(raw_data["acc"][seq_id], raw_data["ori"][seq_id])
            #Forward kinematics
            pose_p, shape_p, tran_p = self._preprocess(raw_data["pose"][seq_id],
                                                     shape = None,
                                                     tran = raw_data["tran"][seq_id])  #DIP IMU does not contain translations
            pose_global_p, joint_p = self.model.forward_kinematics(pose_p, shape_p, tran_p, calc_mesh=False)
            
            #TODO: replace FK root_pos/root_ori with gt_pos/gt_ori

            glb_root_ori = raw_data["ori"][seq_id][:, [-1], ...]
            #glb_root_ori = pose_global_p[:, joint_set.root, ...]
            
            #get leaf joint position
            seq_kwargs["leaf_joint"] = self.get_leaf_joint(joint_p, glb_root_ori)
            
            #non-root joint position wrt root position
            seq_kwargs["non_root_joint"] = self.get_non_root_joint_pos(joint_p, glb_root_ori)
            
            #get 6D joint rotation
            # ji_mask = torch.tensor([18, 19, 4, 5, 15, 0]) 
            # pose_global_p[:,ji_mask,...] = raw_data["ori"][seq_id] #replace globl pose with IMU data
            seq_kwargs["pose_global_6d"] = self.get_6d_rotation(pose_global_p, glb_root_ori)
                     
            #get joint velocity (for DIP joint vel label is not used)
            seq_kwargs["joint_vel"] =self.get_joint_vel(joint_p, glb_root_ori)
            
            #contact points (for DIP contact label is not used)
            seq_kwargs["feet_contact"] = self.get_contact_points(joint_p)

            #uwb
            uwb = raw_data["vuwb"][seq_id]
            if self.normalize_uwb:
                d_root2head = uwb[0,4,5].clone() #normalized by distance between head and root, hard code
                uwb = uwb / d_root2head
            
            if self.flatten_uwb:
                index = torch.triu_indices(IMU_NUM, IMU_NUM, 1)
                uwb = uwb.view(-1,IMU_NUM,IMU_NUM)[:,index[0],index[1]]
            
            if self.remove_node == -1:
                seq_kwargs["uwb_m"] = uwb
            else:
                uwb[:,self.remove_node,:] = 0
                uwb[:,:,self.remove_node] = 0
                seq_kwargs["uwb_m"] = uwb

            seq_kwargs["uwb_c"] = torch.zeros_like(uwb)
            seq_kwargs["uwb_gt"] = raw_data["uwb_gt"][seq_id] if "uwb_gt" in raw_data else uwb
            seq_kwargs["offset"] = torch.zeros(6,3)

            seq_kwargs["acc_sum"] = self.get_acc_sum(seq_kwargs["imu_acc"])
            
            seq_info_ = SeqInfo(seq_id=seq_id,frame_size=N_frame,**seq_kwargs)
            
            seq_info[seq_id] = seq_info_

        if self.train_split and not self.dry_run:       
            torch.save(seq_info,os.path.join(uwb_dir,"dataset_cached.pt"))
        
        return seq_info

class UWBIMU_real_data_train(UWBIMU_real_data):
    def __init__(self, dip_dir: str, split="train", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = True
        
class UWBIMU_real_data_test(UWBIMU_real_data):
    def __init__(self, dip_dir: str, split="test", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = False

class UWBIMU_real_data_val(UWBIMU_real_data):
    def __init__(self, dip_dir: str, split="test", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = False 
        
class Multi_UWB_real_data(_Dataset):
    def __init__(self,data_dir: str, split,**superkwargs) -> None:
        super().__init__(**superkwargs)
        self.dataset_dir = data_dir
        
        seq_info = self._data_preprocess(self.dataset_dir,split=split)
        
        data_info,data_idx = self._data_sampling(seq_info)
        
        self.seq_num = data_idx
        self.data_info = data_info
        self.seq_info = seq_info
        print(f"Loading data size of {self.seq_num}..")

    def _data_preprocess(self,uwb_dir,split="train"):
        
        if self._check_cached(uwb_dir,"dataset_cached.pt") and self.train_split and not self.dry_run:
            return torch.load(os.path.join(uwb_dir,"dataset_cached.pt"))
        
        uwb_dir = os.path.join(uwb_dir, split)

        seq_info = {}
        keys = ["pose", "shape", "tran", "joint", "vrot", "vacc","vuwb","uwb_collision","offset"]
        raw_data = {
                        k:torch.load(os.path.join(uwb_dir, "person1", f"{k}.pt"), weights_only=True)
                        for k in keys if os.path.exists(os.path.join(uwb_dir, "person1", f"{k}.pt"))
                     }
        raw_data['vuwb_2p'] = torch.load(os.path.join(uwb_dir, 'vuwb_2human12.pt'), weights_only=True)

        if split in ["test", "train"]:
            print("Load raw data from ",os.path.join(uwb_dir,"person1", f"{split}.pt"))
            raw_data = torch.load(os.path.join(uwb_dir,"person1",f"{split}.pt"), weights_only=True)
            raw_data["vrot"] = raw_data['ori']
            raw_data["vacc"] = raw_data['acc']

        print("Load raw data from ", os.path.join(uwb_dir, "person1") )

        N_seq = len(raw_data["pose"]) if not self.dry_run else 1
        print(f"Preprocessing Dataset {self.__class__.__name__}")
        for seq_id in tqdm(range(N_seq)):
            N_frame = raw_data["pose"][seq_id].size(0)
            
            if N_frame <= self.seq_length:
                continue
            
            seq_kwargs = {}
            seq_kwargs["imu_acc"],seq_kwargs["imu_ori"] = \
                                        self.get_local_imu(raw_data["vacc"][seq_id], raw_data["vrot"][seq_id])
            #Forward kinematics
            pose_p, shape_p, tran_p = self._preprocess(raw_data["pose"][seq_id],
                                                     shape = None,
                                                     tran = raw_data["tran"][seq_id])  #DIP IMU does not contain translations
            pose_global_p, joint_p = self.model.forward_kinematics(pose_p, shape_p, tran_p, calc_mesh=False)
            
            #TODO: replace FK root_pos/root_ori with gt_pos/gt_ori

            glb_root_ori = raw_data["vrot"][seq_id][:, [-1], ...]
            #glb_root_ori = pose_global_p[:, joint_set.root, ...]
            
            #get leaf joint position
            seq_kwargs["leaf_joint"] = self.get_leaf_joint(joint_p, glb_root_ori)
            
            #non-root joint position wrt root position
            seq_kwargs["non_root_joint"] = self.get_non_root_joint_pos(joint_p, glb_root_ori)
            
            #get 6D joint rotation
            # ji_mask = torch.tensor([18, 19, 4, 5, 15, 0]) 
            # pose_global_p[:,ji_mask,...] = raw_data["ori"][seq_id] #replace globl pose with IMU data
            seq_kwargs["pose_global_6d"] = self.get_6d_rotation(pose_global_p, glb_root_ori)
                     
            #get joint velocity (for DIP joint vel label is not used)
            seq_kwargs["joint_vel"] =self.get_joint_vel(joint_p, glb_root_ori)
            
            #contact points (for DIP contact label is not used)
            seq_kwargs["feet_contact"] = self.get_contact_points(joint_p)

            #uwb
            uwb = raw_data["vuwb"][seq_id]
            if self.normalize_uwb:
                d_root2head = uwb[0,4,5].clone() #normalized by distance between head and root, hard code
                uwb = uwb / d_root2head
            
            if self.flatten_uwb:
                index = torch.triu_indices(IMU_NUM, IMU_NUM, 1)
                uwb = uwb.view(-1,IMU_NUM,IMU_NUM)[:,index[0],index[1]]
            
            if self.remove_node == -1:
                seq_kwargs["uwb_m"] = uwb
            else:
                uwb[:,self.remove_node,:] = 0
                uwb[:,:,self.remove_node] = 0
                seq_kwargs["uwb_m"] = uwb

            seq_kwargs["uwb_c"] = torch.zeros_like(uwb)
            seq_kwargs["uwb_gt"] = raw_data["vuwb"][seq_id]
            # seq_kwargs["uwb_gt"] = raw_data["uwb_gt"][seq_id] if "uwb_gt" in raw_data else uwb
            seq_kwargs["offset"] = torch.zeros(6,3)

            seq_kwargs["acc_sum"] = self.get_acc_sum(seq_kwargs["imu_acc"])
            
            seq_info_ = SeqInfo(seq_id=seq_id,frame_size=N_frame,**seq_kwargs)
            
            seq_info[seq_id] = seq_info_

        if self.train_split and not self.dry_run:       
            torch.save(seq_info,os.path.join(uwb_dir,"dataset_cached.pt"))
        
        return seq_info

class Multi_UWB_real_data_train(Multi_UWB_real_data):
    def __init__(self, data_dir: str, split="train", **superkwargs) -> None:
        super().__init__(data_dir, split, **superkwargs)
        self.train_split = True
        
class Multi_UWB_real_data_test(Multi_UWB_real_data):
    def __init__(self, data_dir: str, split="test", **superkwargs) -> None:
        super().__init__(data_dir, split, **superkwargs)
        self.train_split = False

class Multi_UWB_real_data_val(Multi_UWB_real_data):
    def __init__(self, data_dir: str, split="test", **superkwargs) -> None:
        super().__init__(data_dir, split, **superkwargs)
        self.train_split = False 
'''
DIP IMU Dataset
'''

class DIPIMU_real_data(_Dataset):
    def __init__(self,dip_dir: str, split,**superkwargs) -> None:
        super().__init__(**superkwargs)
        self.dataset_dir = dip_dir
        
        seq_info = self._data_preprocess(self.dataset_dir,split=split)

        data_info,data_idx = self._data_sampling(seq_info)
        
        self.seq_num = data_idx
        self.data_info = data_info
        self.seq_info = seq_info
        print(f"Loading data size of {self.seq_num}..")
        
    def _get_uwb_(self,joint_p):
        ji_mask = torch.tensor([18, 19, 4, 5, 15, 0]) 
        imu_joint = joint_p[:,ji_mask,...]
        return torch.cdist(imu_joint,imu_joint)
      
    def _data_preprocess(self,dip_dir,split="train"):
        
        if self._check_cached(dip_dir,"dataset_cached.pt") and self.train_split and not self.dry_run:
            return torch.load(os.path.join(dip_dir,"dataset_cached.pt"), weights_only=False)
        
        seq_info = {}
        
        # if self.add_uwb:
        #     uwb_collision = torch.load(os.path.join(dip_dir,f"{split}_uwb_collision.pt"))

        print("Load raw data from ",os.path.join(dip_dir,f"{split}.pt"))
        raw_data = torch.load(os.path.join(dip_dir,f"{split}.pt"), weights_only=False)

        N_seq = len(raw_data["pose"]) if not self.dry_run else 1
        print(f"Preprocessing Dataset {self.__class__.__name__}")
        for seq_id in tqdm(range(N_seq)):
            N_frame = raw_data["pose"][seq_id].size(0)
            
            if N_frame <= self.seq_length:
                continue
            
            seq_kwargs = {}
            seq_kwargs["imu_acc"],seq_kwargs["imu_ori"] = \
                                        self.get_local_imu(raw_data["acc"][seq_id], raw_data["ori"][seq_id])
            #Forward kinematics
            pose_p, shape_p, tran_p = self._preprocess(raw_data["pose"][seq_id],
                                                     shape = None,
                                                     tran = None)  #DIP IMU does not contain translations
            pose_global_p, joint_p = self.model.forward_kinematics(pose_p, shape_p, tran_p, calc_mesh=False)
            
            #TODO: replace FK root_pos/root_ori with gt_pos/gt_ori

            glb_root_ori = raw_data["ori"][seq_id][:, [-1], ...]
            #glb_root_ori = pose_global_p[:, joint_set.root, ...]
            
            #get leaf joint position
            seq_kwargs["leaf_joint"] = self.get_leaf_joint(joint_p, glb_root_ori)
            
            #non-root joint position wrt root position
            seq_kwargs["non_root_joint"] = self.get_non_root_joint_pos(joint_p, glb_root_ori)
            
            #get 6D joint rotation
            # ji_mask = torch.tensor([18, 19, 4, 5, 15, 0]) 
            # pose_global_p[:,ji_mask,...] = raw_data["ori"][seq_id] #replace globl pose with IMU data
            seq_kwargs["pose_global_6d"] = self.get_6d_rotation(pose_global_p, glb_root_ori)
                     
            #get joint velocity (for DIP joint vel label is not used)
            seq_kwargs["joint_vel"] =self.get_joint_vel(joint_p, glb_root_ori)
            
            #contact points (for DIP contact label is not used)
            seq_kwargs["feet_contact"] = self.get_contact_points(joint_p)

            #uwb
            v_uwb = raw_data["vuwb"][seq_id]
            seq_kwargs["uwb_gt"] = v_uwb.clone()
            uwb_collision = torch.zeros_like(v_uwb).to(v_uwb.device)
            uwb = self.add_syn_uwb_noise(v_uwb, sigma=self.static_uwb_noise, uwb_collision = uwb_collision[seq_id],normalize=self.normalize_uwb)
            
            if self.remove_node == -1:
                seq_kwargs["uwb_m"] = uwb
            else:
                uwb[:,self.remove_node,:] = 0
                uwb[:,:,self.remove_node] = 0
                seq_kwargs["uwb_m"] = uwb
                
            seq_kwargs["uwb_c"] = uwb_collision[seq_id]
            seq_kwargs["offset"] = raw_data["offset"][seq_id]
            #acc_sum
            seq_kwargs["acc_sum"] = self.get_acc_sum(seq_kwargs["imu_acc"])
            
            seq_info_ = SeqInfo(seq_id=seq_id,frame_size=N_frame,**seq_kwargs)
            
            if self.uwb_timesample_ratio is not None and self.uwb_timesample_ratio != 1:
                seq_info_.syc_measurement(sampling_ratio=self.uwb_timesample_ratio)
                
            if self.flatten_uwb:
                seq_info_.flatten_uwb_measurement()
            
            seq_info[seq_id] = seq_info_

        if self.train_split and not self.dry_run:       
            torch.save(seq_info,os.path.join(dip_dir,"dataset_cached.pt"))
        
        return seq_info

class DIPIMU_real_data_train(DIPIMU_real_data):
    def __init__(self, dip_dir: str, split="train", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = True
        
class DIPIMU_real_data_test(DIPIMU_real_data):
    def __init__(self, dip_dir: str, split="test", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = False

class DIPIMU_real_data_val(DIPIMU_real_data):
    def __init__(self, dip_dir: str, split="validation", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = False
        
    

          
