import articulate as art
from config.config import *
from modules.utils import *
from modules.dynamics import PhysicsOptimizer
from modules.model.model_base import BaseModel
from articulate.utils.torch import *

class PIP(BaseModel):
    name = "Physical Inertial Poser"
    imu_m = ["vacc","vrot", "vuwb"]
    model_output = ["leaf_joint", 
                    "full_joint",
                    "global_6d_pose", 
                    "joint_velocity", 
                    "contact"]
    @staticmethod
    def add_args(parser):
        BaseModel.add_args(parser)
        group = parser.add_argument_group("PIP Config")
        group.add_argument('--add_gaussian', action='store_true', help='Whether to add gaussian to leaf joints estimation')
        group.add_argument('--n_hidden', type=int, default=256, help='Hidden layer width')

    @staticmethod
    def get_config(args):
        config_str = f"{args.network}-input{PIP.imu_m},-output{PIP.model_output}"
        config_str += f"-input_sensor{PIP.imu_m}-hidden_dim{args.hidden_dim}"
        return config_str
    
    def _get_input_size(self):
        return sum([INPUT_DATA_SIZE[k] for k in self.imu_m])

    def __init__(self, args):  
        super().__init__(args)
        add_gaussian = args.add_guassian_noise
        with_rnn_init = not args.no_rnn_init
        self.n_hidden = args.n_hidden
        self.input_size = self._get_input_size()
        self.with_rnn_init = with_rnn_init
        if self.with_rnn_init:
            self.rnn1 = RNNWithInit(input_size=self.input_size,
                                    output_size=joint_set.n_leaf * 3,
                                    hidden_size=self.n_hidden,
                                    num_rnn_layer=2,
                                    dropout=0.4)
            self.rnn4 = RNNWithInit(input_size=self.input_size + joint_set.n_full * 3,
                                    output_size=24 * 3,
                                    hidden_size=self.n_hidden,
                                    num_rnn_layer=2,
                                    dropout=0.4)
        else:
            self.rnn1 = RNN(input_size=self.input_size,
                                    output_size=joint_set.n_leaf * 3,
                                    hidden_size=self.n_hidden,
                                    num_rnn_layer=2,
                                    dropout=0.4)
            self.rnn4 = RNN(input_size=self.input_size + joint_set.n_full * 3,
                                    output_size=24 * 3,
                                    hidden_size=self.n_hidden,
                                    num_rnn_layer=2,
                                    dropout=0.4)
            
        self.rnn2 = RNN(input_size=self.input_size + joint_set.n_leaf * 3,
                        output_size=joint_set.n_full * 3,
                        hidden_size=self.n_hidden,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn3 = RNN(input_size=self.input_size + joint_set.n_full * 3,
                        output_size=joint_set.n_reduced * 6,
                        hidden_size=self.n_hidden,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn5 = RNN(input_size=self.input_size + joint_set.n_full * 3,
                        output_size=2,
                        hidden_size=64,
                        num_rnn_layer=2,
                        dropout=0.4)

        body_model = art.ParametricModel(paths.smpl_file)
        self.inverse_kinematics_R = body_model.inverse_kinematics_R
        self.forward_kinematics = body_model.forward_kinematics
        self.dynamics_optimizer = PhysicsOptimizer(debug=False)
        self.rnn_states = [None for _ in range(5)]
        
        self.add_guassian = add_gaussian

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def forward(self, x):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 3-tuple
                  (tensor [num_frames, 72], tensor [15], tensor [72]).
        :return: [torch.Size([num_frames, 15]), 
                  torch.Size([num_frames, 69]), 
                  torch.Size([num_frames, 90]), 
                  torch.Size([num_frames, 72]), 
                  torch.Size([num_frames, 2])]
        """
        x, lj_init, jvel_init = list(zip(*x))
        leaf_joint = self.rnn1(list(zip(x, lj_init))) if self.with_rnn_init \
                else self.rnn1(list(x))
                
        leaf_joint_input = [leaf_joint[i] + torch.normal(0,0.04,size=leaf_joint[0].size()).to(leaf_joint[i].device) for i in range(len(leaf_joint))] if self.training and self.add_guassian else leaf_joint
        
        full_joint = self.rnn2([torch.cat(_, dim=-1) for _ in zip(leaf_joint_input, x)])
        full_joint_input = [full_joint[i] + torch.normal(0,0.025,size=full_joint[0].size()).to(full_joint[i].device) for i in range(len(full_joint))] if self.training and self.add_guassian else full_joint

        global_6d_pose = self.rnn3([torch.cat(_, dim=-1) for _ in zip(full_joint_input, x)]) 
        joint_velocity = self.rnn4(list(zip([torch.cat(_, dim=-1) for _ in zip(full_joint_input, x)], jvel_init))) if self.with_rnn_init \
                    else self.rnn4(list([torch.cat(_, dim=-1) for _ in zip(full_joint_input, x)]))
                    
        contact = self.rnn5([torch.cat(_, dim=-1) for _ in zip(full_joint_input, x)])
        
        return leaf_joint,full_joint,global_6d_pose,joint_velocity,contact
    
    def _process_input_imu(self,glb_acc,glb_rot,glb_uwb):
        return torch.cat([normalize_and_concat(glb_acc, glb_rot), glb_uwb.flatten(1)], dim=1)
        
    @torch.no_grad()
    def predict(self, glb_acc, glb_rot, init_pose, **kwargs):
        r"""
        Predict the results for evaluation.

        :param glb_acc: A tensor that can reshape to [num_frames, 6, 3].
        :param glb_rot: A tensor that can reshape to [num_frames, 6, 3, 3].
        :param init_pose: A tensor that can reshape to [1, 24, 3, 3].
        :param glb_uwb: A tensor that can reshape to [num_frames, 6, 6].
        :return: Pose tensor in shape [num_frames, 24, 3, 3] and
                 translation tensor in shape [num_frames, 3].
        """
        if "vuwb" in self.imu_m:
            glb_uwb = kwargs["glb_uwb"]
            offset = kwargs["offset"]
        else:
            glb_uwb = None
            
        self.dynamics_optimizer.reset_states()
        init_pose = init_pose.view(1, 24, 3, 3)
        init_pose[0, 0] = torch.eye(3)
        lj_init = self.forward_kinematics(init_pose)[1][0, joint_set.leaf].view(-1)
        jvel_init = torch.zeros(24 * 3)
        x = (self._process_input_imu(glb_acc,glb_rot,glb_uwb), lj_init, jvel_init)
        leaf_joint, full_joint, global_6d_pose, joint_velocity, contact = [_[0] for _ in self.forward([x])]
        pose = self._reduced_glb_6d_to_full_local_mat(glb_rot.view(-1, 6, 3, 3)[:, -1], global_6d_pose)
        joint_velocity = joint_velocity.view(-1, 24, 3).bmm(glb_rot[:, -1].transpose(1, 2)) #* vel_scale
        pose_opt, tran_opt = [], []
        for p, v, c, a in zip(pose, joint_velocity, contact, glb_acc):
            p, t = self.dynamics_optimizer.optimize_frame(p, v, c, a)
            pose_opt.append(p)
            tran_opt.append(t)
        pose_opt, tran_opt = torch.stack(pose_opt), torch.stack(tran_opt)

        return pose_opt, tran_opt


    @torch.no_grad()
    def predict_replace_with_gt(self, glb_acc, glb_rot, init_pose, glb_pose_gt = None, jvel_gt = None, contact_gt = None):
        r"""
        Predict the results for evaluation.

        :param glb_acc: A tensor that can reshape to [num_frames, 6, 3].
        :param glb_rot: A tensor that can reshape to [num_frames, 6, 3, 3].
        :param init_pose: A tensor that can reshape to [1, 24, 3, 3].
        :return: Pose tensor in shape [num_frames, 24, 3, 3] and
                 translation tensor in shape [num_frames, 3].
        """
        self.dynamics_optimizer.reset_states()
        init_pose = init_pose.view(1, 24, 3, 3)
        init_pose[0, 0] = torch.eye(3)
        lj_init = self.forward_kinematics(init_pose)[1][0, joint_set.leaf].view(-1)
        jvel_init = torch.zeros(24 * 3)
        x = (normalize_and_concat(glb_acc, glb_rot), lj_init, jvel_init)
        leaf_joint, full_joint, global_6d_pose, joint_velocity, contact = [_[0] for _ in self.forward([x])]
        global_6d_pose = global_6d_pose if glb_pose_gt is None else glb_pose_gt
        joint_velocity = joint_velocity if jvel_gt is None else jvel_gt
        contact = contact if contact_gt is None else contact_gt
        pose = self._reduced_glb_6d_to_full_local_mat(glb_rot.view(-1, 6, 3, 3)[:, -1], global_6d_pose)
        joint_velocity = joint_velocity.view(-1, 24, 3).bmm(glb_rot[:, -1].transpose(1, 2)) * vel_scale
        pose_opt, tran_opt = [], []
        for p, v, c, a in zip(pose, joint_velocity, contact, glb_acc):
            p, t = self.dynamics_optimizer.optimize_frame(p, v, c, a)
            pose_opt.append(p)
            tran_opt.append(t)
        pose_opt, tran_opt = torch.stack(pose_opt), torch.stack(tran_opt)   
        return pose_opt, tran_opt
        
    