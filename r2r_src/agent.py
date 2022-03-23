# R2R-EnvDrop, 2019, haotan@cs.unc.edu
# Modified in Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
import utils
from utils import padding_idx, print_progress
import model_OSCAR, model_PREVALENT
import param
from param import args
from collections import defaultdict


class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # Models
        if args.vlnbert == 'oscar':
            self.vln_bert = model_OSCAR.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda()
            self.critic = model_OSCAR.Critic().cuda()
        elif args.vlnbert == 'prevalent':
            self.vln_bert = model_PREVALENT.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda()
            self.critic = model_PREVALENT.Critic().cuda()
        self.models = (self.vln_bert, self.critic)

        # Optimizers
        self.vln_bert_optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        self.ndtw_criterion = utils.ndtw_initialize()
        self.device_id = 0

        if args.aug_path is not None:
            if args.con_path_loss_type == 'nce':
                self.queue_traj = torch.randn(args.multi_feat_dim, args.nce_k).cuda(self.device_id)
                self.queue_traj = nn.functional.normalize(self.queue_traj, dim=0).cuda(self.device_id)
                self.queue_ptr_traj = torch.zeros(1, dtype=torch.long).cuda(self.device_id)
                self.criterion_path = nn.CrossEntropyLoss().cuda(self.device_id)
            elif args.con_path_loss_type == 'circle':
                if args.circle_queue:
                    self.queue_traj = torch.randn(args.multi_feat_dim, args.nce_k).cuda(self.device_id)
                    self.queue_traj = nn.functional.normalize(self.queue_traj, dim=0).cuda(self.device_id)
                    self.queue_ptr_traj = torch.zeros(1, dtype=torch.long).cuda(self.device_id)
                self.soft_plus = nn.Softplus().cuda(self.device_id)


        if args.lang_loss_weight > 0:
            self.queue_lang = torch.randn(args.multi_feat_dim, args.nce_k).cuda(self.device_id)
            self.queue_lang = nn.functional.normalize(self.queue_lang, dim=0).cuda(self.device_id)
            self.queue_ptr_lang = torch.zeros(1, dtype=torch.long).cuda(self.device_id)
            if args.con_path_loss_type == 'nce':
                self.criterion_path = nn.CrossEntropyLoss().cuda(self.device_id)
            else:
                self.soft_plus_lang = nn.Softplus().cuda(self.device_id)
        if args.lang_local_loss_weight > 0:
            self.queue_lang_loc = torch.randn(args.multi_feat_dim, args.nce_k).cuda(self.device_id)
            self.queue_lang_loc = nn.functional.normalize(self.queue_lang_loc, dim=0).cuda(self.device_id)
            self.queue_ptr_lang_loc = torch.zeros(1, dtype=torch.long).cuda(self.device_id)
            if args.con_path_loss_type == 'nce':
                self.criterion_path = nn.CrossEntropyLoss().cuda(self.device_id)
            else:
                self.soft_plus_lang_loc = nn.Softplus().cuda(self.device_id)

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    @torch.no_grad()
    def _dequeue_and_enqueue_traj(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_traj)
        # assert args.nce_k % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > args.nce_k:
            batch_size = args.nce_k - ptr
        self.queue_traj[:, ptr:ptr + batch_size] = keys.T[:, :batch_size]
        ptr = (ptr + batch_size) % args.nce_k  # move pointer

        self.queue_ptr_traj[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_lang(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_lang)

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > args.nce_k:
            batch_size = args.nce_k - ptr
        self.queue_lang[:, ptr:ptr + batch_size] = keys.T[:, :batch_size]
        ptr = (ptr + batch_size) % args.nce_k  # move pointer

        self.queue_ptr_lang[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_lang_loc(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_lang_loc)

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > args.nce_k:
            batch_size = args.nce_k - ptr
        self.queue_lang_loc[:, ptr:ptr + batch_size] = keys.T[:, :batch_size]
        ptr = (ptr + batch_size) % args.nce_k  # move pointer

        self.queue_ptr_lang_loc[0] = ptr

    def _sort_batch(self, obs):
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor != padding_idx)

        token_type_ids = torch.zeros_like(mask)

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.long().cuda(), token_type_ids.long().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _glo_aug_sub_instr_sort_batch(self, obs, ind, positive=True):
        valid_idx = []
        if positive:
            seq_tensor = []
            for ob_ind, ob in enumerate(obs):
                new_enc = None
                if 'aug_new_instr_enc' not in ob or len(ob['aug_new_instr_enc']) == 0:
                    continue
                if args.multi_aug_instr:
                    aug_new_instr_enc = random.sample(ob['aug_new_instr_enc'], 1)[0]
                else:
                    aug_new_instr_enc = ob['aug_new_instr_enc']
                if len(aug_new_instr_enc) == 0:
                    continue
                valid_idx.append(ob_ind)
                for sub_ind, sub_enc in enumerate(aug_new_instr_enc):
                    if sub_ind == 0:
                        new_enc = np.zeros(args.maxInput)
                        if len(sub_enc[ind]) > args.maxInput:
                            new_enc = np.array(sub_enc[ind][:args.maxInput])
                        else:
                            new_enc[:len(sub_enc[ind])] = np.array(sub_enc[ind])
                    else:
                        if new_enc[-1] != 0:
                            break
                        new_ind = np.where(new_enc == 0)[0][0] - 1  # remove end token
                        enc_i = np.array(sub_enc[ind])
                        enc_i = enc_i[np.where(enc_i != 0)[0]]
                        if len(enc_i) + new_ind < len(new_enc):
                            new_enc[new_ind:new_ind+len(enc_i)] = enc_i
                        else:
                            len_w = len(new_enc) - new_ind
                            new_enc[new_ind:new_ind + len_w] = enc_i[:len_w]
                            new_enc[-1] = enc_i[-1]  # end token
                            break
                seq_tensor.append(new_enc.copy())
        else:
            seq_tensor = []
            for ob_ind, ob in enumerate(obs):
                if 'aug_new_instr_enc' not in ob or len(ob['aug_new_instr_enc']) == 0:
                    continue
                if args.multi_aug_instr:
                    aug_new_instr_enc = random.sample(ob['aug_new_instr_enc'], 1)[0]
                else:
                    aug_new_instr_enc = ob['aug_new_instr_enc']
                if len(aug_new_instr_enc) == 0:
                    continue
                valid_idx.append(ob_ind)
                if len(aug_new_instr_enc) == 1:
                    aug_type = 'repeat'
                else:
                    aug_type = random.choice(['shuffle', 'repeat'])
                if aug_type == 'repeat':
                    re_ind = random.randint(0, len(aug_new_instr_enc) - 1)
                    re_num = random.randint(0, 3)
                    new_enc = None
                    for sub_ind, sub_enc in enumerate(aug_new_instr_enc):
                        if sub_ind == 0:
                            new_enc = np.zeros(args.maxInput)
                            if len(sub_enc[ind]) > args.maxInput:
                                new_enc = np.array(sub_enc[ind][:args.maxInput])
                            else:
                                new_enc[:len(sub_enc[ind])] = np.array(sub_enc[ind])
                        else:
                            if new_enc[-1] != 0:
                                break
                            new_ind = np.where(new_enc == 0)[0][0] - 1  # remove end token
                            enc_i = np.array(sub_enc[ind])
                            enc_i = enc_i[np.where(enc_i != 0)[0]]
                            if len(enc_i) + new_ind < len(new_enc):
                                new_enc[new_ind:new_ind + len(enc_i)] = enc_i
                            else:
                                len_w = len(new_enc) - new_ind
                                new_enc[new_ind:new_ind + len_w] = enc_i[:len_w]
                                new_enc[-1] = enc_i[-1]  # end token
                                break
                        if sub_ind == re_ind:
                            for indi in range(re_num):
                                if new_enc[-1] != 0:
                                    break
                                new_ind = np.where(new_enc == 0)[0][0] - 1  # remove end token
                                enc_i = np.array(sub_enc[ind])
                                enc_i = enc_i[np.where(enc_i != 0)[0]]
                                if len(enc_i) + new_ind < len(new_enc):
                                    new_enc[new_ind:new_ind + len(enc_i)] = enc_i
                                else:
                                    len_w = len(new_enc) - new_ind
                                    new_enc[new_ind:new_ind + len_w] = enc_i[:len_w]
                                    new_enc[-1] = enc_i[-1]  # end token
                                    break
                    seq_tensor.append(new_enc.copy())
                else:
                    s_ind = list(range(len(aug_new_instr_enc)))
                    random.shuffle(s_ind)
                    new_enc = None
                    for indi, sub_ind in enumerate(s_ind):
                        sub_enc = aug_new_instr_enc[sub_ind]
                        if indi == 0:
                            new_enc = np.zeros(args.maxInput)
                            if len(sub_enc[ind]) > args.maxInput:
                                new_enc = np.array(sub_enc[ind][:args.maxInput])
                            else:
                                new_enc[:len(sub_enc[ind])] = np.array(sub_enc[ind])
                        else:
                            if new_enc[-1] != 0:
                                break
                            new_ind = np.where(new_enc == 0)[0][0] - 1  # remove end token
                            enc_i = np.array(sub_enc[ind])
                            enc_i = enc_i[np.where(enc_i != 0)[0]]
                            if len(enc_i) + new_ind < len(new_enc):
                                new_enc[new_ind:new_ind + len(enc_i)] = enc_i
                            else:
                                len_w = len(new_enc) - new_ind
                                new_enc[new_ind:new_ind + len_w] = enc_i[:len_w]
                                new_enc[-1] = enc_i[-1]  # end token
                                break
                    seq_tensor.append(new_enc.copy())
        seq_tensor = np.array(seq_tensor)
        seq_tensor = torch.from_numpy(seq_tensor)
        mask = (seq_tensor != padding_idx)
        token_type_ids = torch.zeros_like(mask)
        return Variable(seq_tensor, requires_grad=False).long().cuda(self.device_id), \
               mask.long().cuda(self.device_id), token_type_ids.long().cuda(self.device_id), valid_idx

    def _loc_aug_sub_instr_sort_batch(self, obs, ind=None, anchor=True, valid_idx=None):
        if anchor:
            valid_idx = []
            ind = []
            seq_tensor = []
            for ob_ind, ob in enumerate(obs):
                if 'new_instr_enc' not in ob or len(ob['new_instr_enc']) == 1:
                    continue
                valid_idx.append(ob_ind)
                indi = random.randint(0, len(ob['new_instr_enc']) - 1)
                seq_tensor.append(np.array(ob['new_instr_enc'][indi]))
                ind.append(indi)
        else:
            seq_tensor = []
            for indi, ob_ind in enumerate(valid_idx):
                ob = obs[ob_ind]
                ind_i = ind[indi]
                sel_ind_n = []
                if ind_i - 1 >= 0:
                    sel_ind_n.append(ind_i - 1)
                if ind_i + 1 < len(ob['new_instr_enc']):
                    sel_ind_n.append(ind_i + 1)
                sel_ind_ni = random.choice(sel_ind_n)
                seq_tensor.append(np.array(ob['new_instr_enc'][sel_ind_ni]))
        seq_tensor = np.array(seq_tensor)
        seq_tensor = torch.from_numpy(seq_tensor)
        mask = (seq_tensor != padding_idx)
        token_type_ids = torch.zeros_like(mask)
        return Variable(seq_tensor, requires_grad=False).long().cuda(self.device_id), \
               mask.long().cuda(self.device_id), token_type_ids.long().cuda(self.device_id), valid_idx, ind

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature']

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()
        # f_t = self._feature_variable(obs)      # Pano image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, candidate_feat, candidate_leng

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None, valid_idx=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])

        if perm_idx is None:
            perm_idx = range(len(perm_obs))

        if valid_idx is not None:
            perm_idx = [perm_idx[idx] for idx in valid_idx]
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

                state = self.env.env.sims[idx].getState()
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def pos_neg_path_feat(self, perm_idx, valid_idx, language_features, language_attention_mask, token_type_ids,
                          h_t=None, aug_path=None):
        language_features = language_features[valid_idx, :, :]
        language_attention_mask = language_attention_mask[valid_idx, :]
        token_type_ids = token_type_ids[valid_idx, :]
        if h_t is not None:
            h_t_ori = h_t[valid_idx, :]
        else:
            h_t_ori = None
        if aug_path is None:
            repeat_num = 1
        else:
            repeat_num = args.aug_path_num
            if len(valid_idx) < len(aug_path):
                aug_path = [aug_path[idx] for idx in valid_idx]
        h_t_list = []

        for re_ind in range(repeat_num):
            batch = self.env.batch.copy()
            obs = np.array(self.env.reset(batch))
            perm_obs = obs[perm_idx]
            perm_obs = perm_obs[valid_idx]
            traj = [{
                'instr_id': ob['instr_id'],
                'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            } for ob in perm_obs]
            ended = np.array([False] * len(valid_idx))
            if h_t_ori is not None:
                h_t = h_t_ori.clone()

            for t in range(self.episode_len):

                input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

                # the first [CLS] token, initialized by the language BERT, serves
                # as the agent's state passing through time steps
                if (t >= 1) or (args.vlnbert == 'prevalent'):
                    language_features = torch.cat((h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1)

                visual_temp_mask = (utils.length2mask(candidate_leng, gpu_id=self.device_id) == 0).long()
                visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

                self.vln_bert.vln_bert.config.directions = max(candidate_leng)
                ''' Visual BERT '''
                visual_inputs = {'mode': 'visual',
                                 'sentence': language_features,
                                 'attention_mask': visual_attention_mask,
                                 'lang_mask': language_attention_mask,
                                 'vis_mask': visual_temp_mask,
                                 'token_type_ids': token_type_ids,
                                 'action_feats': input_a_t,
                                 # 'pano_feats':         f_t,
                                 'cand_feats': candidate_feat}
                h_t, logit = self.vln_bert(**visual_inputs)

                if aug_path is None:
                    target = self._teacher_action(perm_obs, ended)
                    a_t = target
                else:
                    a_t = np.zeros(len(perm_obs), dtype=np.int64)
                    for i, ob in enumerate(perm_obs):
                        if ended[i]:  # Just ignore this index
                            a_t[i] = args.ignoreid
                        else:
                            if len(aug_path[i][re_ind]) > t + 1:
                                teacher = aug_path[i][re_ind][t + 1]
                            else:
                                teacher = aug_path[i][re_ind][-1]
                            for k, candidate in enumerate(ob['candidate']):
                                if candidate['viewpointId'] == teacher:  # Next view point
                                    a_t[i] = k
                                    break
                            else:  # Stop here
                                # if teacher != ob['viewpoint']:
                                #     import pdb;pdb.set_trace()
                                assert teacher == ob['viewpoint']  # The teacher action should be "STAY HERE"
                                a_t[i] = len(ob['candidate'])
                    a_t = torch.from_numpy(a_t).cuda(self.device_id)

                cpu_a_t = a_t.cpu().numpy()
                for i, next_id in enumerate(cpu_a_t):
                    if next_id == (candidate_leng[i] - 1) or next_id == args.ignoreid or ended[
                        i]:  # The last action is <end>
                        cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

                # Make action and get the new state
                self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj, valid_idx)
                obs = np.array(self.env._get_obs())
                perm_obs = obs[perm_idx]
                perm_obs = perm_obs[valid_idx]

                # Update the finished actions
                # -1 means ended or ignored (already ended)
                ended[:] = np.logical_or(ended, (cpu_a_t == -1))

                # Early exit if all ended
                if ended.all():
                    break
            h_t_list.append(h_t)
        return h_t_list

    def nce_loss(self, anchor, positive, negative, type='path'):
        pos = torch.einsum('nc,nc->n', [anchor, positive.detach()]).unsqueeze(-1)
        if type == 'path':
            neg = torch.einsum('nc,ck->nk',
                               [anchor, torch.cat([self.queue_traj.clone().detach(), negative.detach()], dim=1)])
        elif type == 'lang':
            neg = torch.einsum('nc,ck->nk',
                               [anchor, torch.cat([self.queue_lang.clone().detach(), negative.detach()], dim=1)])
        elif type == 'lang_loc':
            neg = torch.einsum('nc,ck->nk',
                               [anchor, self.queue_lang_loc.clone().detach()])
        if args.circle_mining or args.circle_fna:
            epsilon = 1e-5
            loss = 0
            for i in range(pos.shape[0]):
                pos_i = pos[i, :]
                neg_i = neg[i, :]
                try:
                    if args.circle_fna:
                        false_neg_index = torch.nonzero(neg_i > 0.7)
                        if false_neg_index.size(0) > 0:
                            false_neg_index = false_neg_index[
                                                  torch.sort(neg_i[false_neg_index.squeeze(1)], descending=True)[1]][
                                              :10].squeeze(1)
                            neg_index = [i for i in range(negative.shape[0]) if i not in false_neg_index]
                            neg_i = neg_i[neg_index]
                            if neg_i.shape[0] < 1:
                                continue

                    if args.circle_mining:
                        neg_i = neg_i[neg_i + args.circle_m > pos_i[0]]
                        if neg_i.shape[0] < 1:
                            continue
                except Exception as e:
                    print(e)
                    continue
                logits = torch.cat([pos_i, neg_i], dim=0).unsqueeze(0)
                logits /= args.nce_t
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.device_id)
                loss += self.criterion_path(logits, labels)
            loss /= pos.shape[0]
            self._dequeue_and_enqueue_traj(positive)
        else:
            logits = torch.cat([pos, neg], dim=1)
            logits /= args.nce_t
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.device_id)
            if type == 'path':
                self._dequeue_and_enqueue_traj(positive)
            elif type == 'lang':
                self._dequeue_and_enqueue_lang(positive)
            elif type == 'lang_loc':
                self._dequeue_and_enqueue_lang_loc(positive)
            loss = self.criterion_path(logits, labels)
        return loss

    def circle_loss(self, anchor, positive, negative=None, type='path'):
        positive = positive.detach()
        sp = torch.nn.functional.cosine_similarity(anchor.unsqueeze(1), positive, dim=-1)
        if args.circle_queue:
            if type == 'path':
                negative = torch.cat([self.queue_traj.clone().transpose(1, 0), negative], dim=0).detach()
            elif type == 'lang':
                if not args.no_intra_negative:
                    negative = torch.cat([self.queue_lang.clone().transpose(1, 0), negative], dim=0).detach()
                else:
                    negative = self.queue_lang.clone().transpose(1, 0).detach()
            elif type == 'lang_loc':
                negative = self.queue_lang_loc.clone().transpose(1, 0).detach()
            sn = torch.nn.functional.cosine_similarity(anchor.unsqueeze(1),
                                                       negative.unsqueeze(0).repeat(anchor.shape[0], 1, 1), dim=-1)
        else:
            negative = negative.view(args.aug_path_num, anchor.shape[0], -1).permute(1, 0, 2).detach()
            sn = torch.nn.functional.cosine_similarity(anchor.unsqueeze(1), negative, dim=-1)

        delta_p = 1 - args.circle_m
        delta_n = args.circle_m

        loss = 0
        if args.circle_mining or args.circle_fna:
            # epsilon = 1e-5
            # sp = sp[sp < 1 - epsilon]
            # sn = sn[sn + args.circle_m > sp.min()]
            # sp = sp[sp - args.circle_m < sn.max()]
            # if sn.shape[0] < 1 or sp.shape[0] < 1:
            #     return 0

            epsilon = 1e-5
            for i in range(anchor.shape[0]):
                sp_i = sp[i, :]
                sn_i = sn[i, :]
                try:
                    if args.circle_fna:
                        np_sim = torch.nn.functional.cosine_similarity(positive[i, :, :].unsqueeze(1),
                                                                       negative.unsqueeze(0).repeat(positive.shape[1], 1, 1),
                                                                       dim=-1)
                        # np_sim = torch.max(np_sim, dim=0)[0]
                        np_sim = torch.max(np_sim, dim=1)[0]
                        false_neg_index = torch.nonzero(np_sim > args.fna_thr)
                        if false_neg_index.size(0) > 0:
                            false_neg_index = false_neg_index[torch.sort(np_sim[false_neg_index.squeeze(1)], descending=True)[1]][:10].squeeze(1)
                            sp_i = torch.cat((sp_i, sn_i[false_neg_index]), dim=0)
                            neg_index = [i for i in range(negative.shape[0]) if i not in false_neg_index]
                            sn_i = sn_i[neg_index]
                            if sn_i.shape[0] < 1:
                                continue

                    if args.circle_mining:
                        sp_i = sp_i[sp_i < 1 - epsilon]
                        sn_i = sn_i[sn_i + args.circle_m > sp_i.min()]
                        sp_i = sp_i[sp_i - args.circle_m < sn_i.max()]
                        if sn_i.shape[0] < 1 or sp_i.shape[0] < 1:
                            continue
                except Exception as e:
                    print(e)
                    continue
                ap = torch.clamp_min(- sp_i.detach() + 1 + args.circle_m, min=0.)
                an = torch.clamp_min(sn_i.detach() + args.circle_m, min=0.)
                logit_p = - ap * (sp_i - delta_p) * args.circle_gamma
                logit_n = an * (sn_i - delta_n) * args.circle_gamma
                if type == 'path':
                    loss += self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
                elif type == 'lang':
                    loss += self.soft_plus_lang(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
                elif type == 'lang_loc':
                    loss += self.soft_plus_lang_loc(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
            loss = 1.0 * loss / anchor.shape[0]
            if type == 'path':
                self._dequeue_and_enqueue_traj(positive.contiguous().view(-1, anchor.shape[-1]))
                # self._dequeue_and_enqueue_traj(anchor.detach())
            elif type == 'lang':
                self._dequeue_and_enqueue_lang(positive.contiguous().view(-1, anchor.shape[-1]))
            elif type == 'lang_loc':
                self._dequeue_and_enqueue_lang_loc(positive.contiguous().view(-1, anchor.shape[-1]))
        else:
            ap = torch.clamp_min(- sp.detach() + 1 + args.circle_m, min=0.)
            an = torch.clamp_min(sn.detach() + args.circle_m, min=0.)

            logit_p = - ap * (sp - delta_p) * args.circle_gamma
            logit_n = an * (sn - delta_n) * args.circle_gamma

            if not args.circle_mining:
                loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)).mean()
            else:
                loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        # Language input
        sentence, language_attention_mask, token_type_ids, \
        seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        ''' Language BERT '''
        language_inputs = {'mode': 'language',
                           'sentence': sentence,
                           'attention_mask': language_attention_mask,
                           'lang_mask': language_attention_mask,
                           'token_type_ids': token_type_ids}
        if args.vlnbert == 'oscar':
            language_features = self.vln_bert(**language_inputs)
            h_t_l = None
        elif args.vlnbert == 'prevalent':
            h_t, language_features = self.vln_bert(**language_inputs)
            h_t_l = h_t.clone()

        if args.lang_loss_weight > 0:
            lang_feat_a = self.vln_bert.lang_linear(language_features.permute(0, 2, 1)).squeeze(-1)
            lang_feat_a = self.vln_bert.lang_proj_layer(lang_feat_a)
            lang_feat_a = self.vln_bert.lang_pred_layer(lang_feat_a)

            lang_pos = []
            lang_neg = []
            lang_loss = 0
            if args.aug_lang_type is None:
                if args.aug_lang_num < 13:
                    rand_inds = random.sample([ind for ind in range(13)], args.aug_lang_num)
                else:
                    rand_inds = [ind for ind in range(13)]
            else:
                if args.aug_lang_type == 'back_translation':
                    rand_inds = [0]
                    args.aug_lang_num = 1
                elif args.aug_lang_type == 'insert':
                    rand_inds = [1, 2, 3, 4]
                    args.aug_lang_num = 4
                elif args.aug_lang_type == 'substitute':
                    rand_inds = [5, 6, 7, 8]
                    args.aug_lang_num = 4
                elif args.aug_lang_type == 'synonym':
                    rand_inds = [9, 10, 11, 12]
                    args.aug_lang_num = 4
            sent_ps = []
            mask_ps = []
            token_type_ids_ps = []
            sent_ns = []
            mask_ns = []
            token_type_ids_ns = []
            for i in range(args.aug_lang_num):
                sent_p, mask_p, token_type_ids_p, valid_idx = self._glo_aug_sub_instr_sort_batch(perm_obs, ind=rand_inds[i],
                                                                                                 positive=True)
                sent_ps.append(sent_p.clone())
                mask_ps.append(mask_p.clone())
                token_type_ids_ps.append(token_type_ids_p.clone())
                sent_n, mask_n, token_type_ids_n, valid_idx = self._glo_aug_sub_instr_sort_batch(perm_obs, ind=i,
                                                                                                 positive=False)
                sent_ns.append(sent_n.clone())
                mask_ns.append(mask_n.clone())
                token_type_ids_ns.append(token_type_ids_n.clone())

            if len(valid_idx) > 1:
                sent_ps = torch.cat(sent_ps, dim=0)
                mask_ps = torch.cat(mask_ps, dim=0)
                token_type_ids_ps = torch.cat(token_type_ids_ps, dim=0)
                sent_ns = torch.cat(sent_ns, dim=0)
                mask_ns = torch.cat(mask_ns, dim=0)
                token_type_ids_ns = torch.cat(token_type_ids_ns, dim=0)

                language_inputs = {'mode': 'language',
                                   'sentence': sent_ps,
                                   'attention_mask': mask_ps,
                                   'lang_mask': mask_ps,
                                   'token_type_ids': token_type_ids_ps}
                with torch.no_grad():
                    if args.vlnbert == 'oscar':
                        lang_feat_p = self.vln_bert(**language_inputs)
                    elif args.vlnbert == 'prevalent':
                        h_t_l_p, lang_feat_p = self.vln_bert(**language_inputs)
                    lang_feat_p = self.vln_bert.lang_linear(lang_feat_p.permute(0, 2, 1)).squeeze(-1)
                    lang_feat_p = self.vln_bert.lang_proj_layer(lang_feat_p).view(args.aug_lang_num, len(valid_idx), -1).permute(1,0,2)
                    # lang_pos.append(lang_feat_p.clone())

                language_inputs = {'mode': 'language',
                                   'sentence': sent_ns,
                                   'attention_mask': mask_ns,
                                   'lang_mask': mask_ns,
                                   'token_type_ids': token_type_ids_ns}
                with torch.no_grad():
                    if args.vlnbert == 'oscar':
                        lang_feat_n = self.vln_bert(**language_inputs)
                    elif args.vlnbert == 'prevalent':
                        h_t_l_n, lang_feat_n = self.vln_bert(**language_inputs)
                    lang_feat_n = self.vln_bert.lang_linear(lang_feat_n.permute(0, 2, 1)).squeeze(-1)
                    lang_feat_n = self.vln_bert.lang_proj_layer(lang_feat_n)
                    # lang_neg.append(lang_feat_n.clone())

                # lang_pos = torch.cat(lang_pos, dim=0).view(args.aug_lang_num, len(valid_idx), -1).permute(1, 0, 2)
                # lang_neg = torch.cat(lang_neg, dim=0)
                if args.con_path_loss_type == 'nce':
                    for ind in range(lang_feat_p.shape[1]):
                        lang_loss += self.nce_loss(lang_feat_a[valid_idx, :], lang_feat_p[:, ind, :], lang_feat_n.T, type='lang')
                    lang_loss = lang_loss / args.aug_path_num
                elif args.con_path_loss_type == 'circle':
                    lang_loss += self.circle_loss(lang_feat_a[valid_idx, :], lang_feat_p, lang_feat_n, type='lang')
                lang_loss = args.lang_loss_weight * lang_loss
                self.loss += lang_loss
                self.logs['lang_loss'].append(lang_loss.item())

        if args.lang_local_loss_weight > 0:
            lang_loc_loss = 0
            try:
                sent_a, mask_a, token_type_ids_a, valid_idx, sel_ind = self._loc_aug_sub_instr_sort_batch(perm_obs,
                                                                                                          anchor=True)
                language_inputs = {'mode': 'language',
                                   'sentence': sent_a,
                                   'attention_mask': mask_a,
                                   'lang_mask': mask_a,
                                   'token_type_ids': token_type_ids_a}
                if len(valid_idx) > 1:
                    if args.vlnbert == 'oscar':
                        lang_feat_loc_a = self.vln_bert(**language_inputs)
                    elif args.vlnbert == 'prevalent':
                        h_t_loc_a, lang_feat_loc_a = self.vln_bert(**language_inputs)
                    lang_feat_loc_a = self.vln_bert.lang_linear(lang_feat_loc_a.permute(0, 2, 1)).squeeze(-1)
                    lang_feat_loc_a = self.vln_bert.lang_proj_layer(lang_feat_loc_a)
                    lang_feat_loc_a = self.vln_bert.lang_pred_layer(lang_feat_loc_a)

                    sent_n, mask_n, token_type_ids_n, valid_idx, sel_ind = self._loc_aug_sub_instr_sort_batch(perm_obs,
                                                                                                              ind=sel_ind,
                                                                                                              anchor=False,
                                                                                                              valid_idx=valid_idx)
                    language_inputs = {'mode': 'language',
                                       'sentence': sent_n,
                                       'attention_mask': mask_n,
                                       'lang_mask': mask_n,
                                       'token_type_ids': token_type_ids_n}
                    with torch.no_grad():
                        if args.vlnbert == 'oscar':
                            lang_feat_loc_p = self.vln_bert(**language_inputs)
                        elif args.vlnbert == 'prevalent':
                            h_t_loc_p, lang_feat_loc_p = self.vln_bert(**language_inputs)
                        lang_feat_loc_p = self.vln_bert.lang_linear(lang_feat_loc_p.permute(0, 2, 1)).squeeze(-1)
                        lang_feat_loc_p = self.vln_bert.lang_proj_layer(lang_feat_loc_p)
                    if args.con_path_loss_type == 'nce':
                        lang_loc_loss = self.nce_loss(lang_feat_loc_a, lang_feat_loc_p, negative=None,
                                                   type='lang_loc')
                    else:
                        lang_loc_loss += self.circle_loss(lang_feat_loc_a, lang_feat_loc_p.unsqueeze(1), negative=None,
                                                          type='lang_loc')
                    lang_loc_loss = args.lang_local_loss_weight * lang_loc_loss
                    self.loss += lang_loc_loss
                    self.logs['lang_loc_loss'].append(lang_loc_loss.item())
            except Exception as e:
                print(e)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in perm_obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):  # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            if train_rl or train_ml:
                if '+' not in ob['scan']:
                    last_ndtw[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        end_indexes = [0 for ind in range(batch_size)]

        for t in range(self.episode_len):

            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            if (t >= 1) or (args.vlnbert == 'prevalent'):
                language_features = torch.cat((h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1)

            visual_temp_mask = (utils.length2mask(candidate_leng, gpu_id=self.device_id) == 0).long()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode': 'visual',
                             'sentence': language_features,
                             'attention_mask': visual_attention_mask,
                             'lang_mask': language_attention_mask,
                             'vis_mask': visual_temp_mask,
                             'token_type_ids': token_type_ids,
                             'action_feats': input_a_t,
                             # 'pano_feats':         f_t,
                             'cand_feats': candidate_feat}
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng, gpu_id=self.device_id)
            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)  # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i] - 1) or next_id == args.ignoreid or ended[
                    i]:  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]  # Perm the obs for the resu

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    if '+' not in ob['scan']:
                        ndtw_score[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:  # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            end_points = ((cpu_a_t == -1) != ended).nonzero()[0]
            for point in end_points:
                if not ended[point]:
                    end_indexes[point] = t

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if self.feedback == 'teacher':
            h_t_observe = h_t.clone()

        if train_rl:
            # Last action in A2C
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            language_features = torch.cat((h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1)

            visual_temp_mask = (utils.length2mask(candidate_leng, gpu_id=self.device_id) == 0).long()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode': 'visual',
                             'sentence': language_features,
                             'attention_mask': visual_attention_mask,
                             'lang_mask': language_attention_mask,
                             'vis_mask': visual_temp_mask,
                             'token_type_ids': token_type_ids,
                             'action_feats': input_a_t,
                             # 'pano_feats':         f_t,
                             'cand_feats': candidate_feat}
            last_h_, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda(self.device_id)
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda(self.device_id)
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if 'aug_paths' in obs[0]:
            # aug paths
            aug_pos_paths = []
            aug_neg_paths = []
            valid_idx = set()
            for ob_ind, ob_i in enumerate(perm_obs):
                all_aug_paths = ob_i['aug_paths']
                aug_pos_paths_i = []
                aug_neg_paths_i = []

                if 0 < len(all_aug_paths['pos']) < args.aug_path_num:
                    aug_pos_paths_i = all_aug_paths['pos']
                    aug_num = args.aug_path_num - len(aug_pos_paths_i)
                    while aug_num > 0:
                        aug_pos_paths_i.append(random.sample(all_aug_paths['pos'], 1)[0])
                        aug_num -= 1
                elif len(all_aug_paths['pos']) >= args.aug_path_num:
                    aug_pos_paths_i = random.sample(all_aug_paths['pos'], args.aug_path_num)

                if 0 < len(all_aug_paths['neg']) < args.aug_path_num:
                    aug_neg_paths_i = all_aug_paths['neg']
                    aug_num = args.aug_path_num - len(aug_neg_paths_i)
                    while aug_num > 0:
                        aug_neg_paths_i.append(random.sample(all_aug_paths['neg'], 1)[0])
                        aug_num -= 1
                elif len(all_aug_paths['neg']) >= args.aug_path_num:
                    aug_neg_paths_i = random.sample(all_aug_paths['neg'], args.aug_path_num)

                if len(aug_pos_paths_i) > 0 and len(aug_neg_paths_i) > 0:
                    valid_idx.add(ob_ind)

                aug_pos_paths.append(aug_pos_paths_i.copy())
                aug_neg_paths.append(aug_neg_paths_i.copy())

            valid_idx = list(valid_idx)
            if len(valid_idx) > 1:
                if self.feedback == 'teacher':
                    h_t_anchor = h_t_observe[valid_idx, :]
                else:
                    h_t_anchor = self.pos_neg_path_feat(perm_idx, valid_idx, language_features, language_attention_mask,
                                                        token_type_ids, h_t_l, aug_path=None)[0]
                h_t_anchor = self.vln_bert.vision_proj_layer(h_t_anchor)
                h_t_anchor = self.vln_bert.vision_pred_layer(h_t_anchor)

                # aug paths feat
                with torch.no_grad():
                    h_t_pos = self.pos_neg_path_feat(perm_idx, valid_idx, language_features, language_attention_mask,
                                                     token_type_ids, h_t_l, aug_path=aug_pos_paths)
                    h_t_pos = torch.cat(h_t_pos, dim=0)
                    h_t_pos = self.vln_bert.vision_proj_layer(h_t_pos).view(args.aug_path_num, len(valid_idx),
                                                                            -1).permute(1, 0, 2)

                    h_t_neg = self.pos_neg_path_feat(perm_idx, valid_idx, language_features, language_attention_mask,
                                                     token_type_ids, h_t_l, aug_path=aug_neg_paths)
                    h_t_neg = torch.cat(h_t_neg, dim=0)
                    h_t_neg = self.vln_bert.vision_proj_layer(h_t_neg)

                path_loss = 0
                if args.con_path_loss_type == 'nce':
                    for ind in range(args.aug_path_num):
                        path_loss += self.nce_loss(h_t_anchor, h_t_pos[:, ind, :], h_t_neg.T)
                elif args.con_path_loss_type == 'circle':
                    path_loss += self.circle_loss(h_t_anchor, h_t_pos, h_t_neg)
                path_loss = args.con_path_loss_weight * path_loss / args.aug_path_num
                self.loss += path_loss
                self.logs['path_loss'].append(path_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)  # This argument is useless.

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', final_iter=None, cmp=False, **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if not args.polyaxon:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

        if args.cos and final_iter % 1600 == 0:
            self.adjust_learning_rate(self.vln_bert_optimizer, final_iter)
            self.adjust_learning_rate(self.critic_optimizer, final_iter)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])

        def recover_state_other_dataset(name, model):
            state = model.state_dict()
            state_dict = {k: v for k, v in states[name]['state_dict'].items() if (k in state.keys() and 'vln_bert' in k)}
            state.update(state_dict)
            model.load_state_dict(state)

        if args.dataset == 'R2R':
            all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                         ("critic", self.critic, self.critic_optimizer)]
            for param in all_tuple:
                recover_state(*param)
            return states['vln_bert']['epoch'] - 1
        else:
            recover_state_other_dataset('vln_bert', self.vln_bert)
            return 0

    def adjust_learning_rate(self, optimizer, iter):
        """Decay the learning rate based on schedule"""
        lr = args.lr
        if args.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * iter / args.iters))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if iter >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
