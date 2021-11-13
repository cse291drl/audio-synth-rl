import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal, Categorical
from tqdm import tqdm

from data_modules.data_modules import AudioHandler, TargetSoundDataset
from loss import presetParam
from synth.dexed import Dexed

class CNNFeatExtractor(nn.Module):
	"""CNN for extracting feats from sound spectrograms or stacked 
	spectrograms if used in ComparerNetwork. 
	"""
	def __init__(self, n_feat_channels, n_filters=32, dropout=0.2,
			filter_sizes=[5,7,9]):
		super().__init__()
		self.conv1 = nn.Conv1d(in_channels=n_feat_channels, 
			out_channels=n_filters, kernel_size=filter_sizes[0], stride=1, 
			padding=filter_sizes[0]//2
		)
		self.act1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)
		self.conv2 = nn.Conv1d(in_channels=n_filters, 
			out_channels=n_filters, kernel_size=filter_sizes[1], stride=1, 
			padding=filter_sizes[1]//2
		)
		self.act2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)
		self.conv3 = nn.Conv1d(in_channels=n_filters, 
			out_channels=n_filters, kernel_size=filter_sizes[2], stride=1, 
			padding=filter_sizes[2]//2
		)
		self.act3 = nn.ReLU()
		self.dropout3 = nn.Dropout(dropout)

	def forward(self, x):
		x = self.dropout1(self.act1(self.conv1(x)))
		x = self.dropout2(self.act2(self.conv2(x)))
		x = self.dropout3(self.act3(self.conv3(x)))
		return x


class CNNComparer(nn.Module):
	""" For use in ComparerNetwork """
	def __init__(self, n_feat_channels, n_filters=32, dropout=0.2,
			filter_sizes=[9,7,9]):
		super().__init__()
		self.conv1 = nn.Conv1d(in_channels=n_feat_channels, 
			out_channels=n_filters, kernel_size=filter_sizes[0], stride=1, 
			padding=filter_sizes[0]//2
		)
		self.act1 = nn.ReLU()
		self.pool_a = nn.AvgPool1d(kernel_size=5)
		self.dropout1 = nn.Dropout(dropout)
		self.conv2 = nn.Conv1d(in_channels=n_filters, 
			out_channels=n_filters, kernel_size=filter_sizes[1], stride=1, 
			padding=filter_sizes[1]//2
		)
		self.act2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)
		self.conv3 = nn.Conv1d(in_channels=n_filters, 
			out_channels=n_filters, kernel_size=filter_sizes[2], stride=1, 
			padding=filter_sizes[2]//2
		)
		self.act3 = nn.ReLU()
		self.pool_b = nn.AvgPool1d(kernel_size=5, count_include_pad=False)
		self.dropout3 = nn.Dropout(dropout)

	def forward(self, x):
		x = self.act1(self.conv1(x))
		x = self.pool_a(x)
		x = self.dropout1(x)
		x = self.dropout2(self.act2(self.conv2(x)))
		x = self.act3(self.conv3(x))
		x = self.pool_b(x).flatten(start_dim=1)
		x = self.dropout3(x)
		return x

class PresetActivation(nn.Module):
	""" Applies the appropriate activations (e.g. sigmoid, hardtanh, softmax, ...) to different neurons
	or groups of neurons of a given input layer. """
	def __init__(self, 
				 numerical_activation=nn.Hardtanh(min_val=0.0, max_val=1.0),
				 cat_softmax_activation=False):
		"""
		:param idx_helper:
		:param numerical_activation: Should be nn.Hardtanh if numerical params often reach 0.0 and 1.0 GT values,
			or nn.Sigmoid to perform a smooth regression without extreme 0.0 and 1.0 values.
		:param cat_softmax_activation: if True, a softmax activation is applied on categorical sub-vectors.
			Otherwise, applies the same HardTanh for cat and num params (and softmax should be applied in loss function)
		"""
		super().__init__()
		self.numerical_act = numerical_activation
		self.cat_softmax_activation = cat_softmax_activation
		if self.cat_softmax_activation:
			self.categorical_act = nn.Softmax(dim=-1)  # Required for categorical cross-entropy loss
			# Pre-compute indexes lists (to use less CPU)
			self.num_indexes, self.cat_indexes = Dexed.get_learnable_indexes()
		else:
			pass  # Nothing to init....

	def forward(self, x):
		""" Applies per-parameter output activations using the PresetIndexesHelper attribute of this instance. """
		if self.cat_softmax_activation:
			x[:, self.num_indexes] = self.numerical_act(x[:, self.num_indexes])
			for cat_learnable_indexes in self.cat_indexes:  # type: Iterable
				x[:, cat_learnable_indexes] = self.categorical_act(x[:, cat_learnable_indexes])
		else:  # Same activation on num and cat ('one-hot encoded') params
			x = self.numerical_act(x)
		return x

class BasicActorHead(nn.Module):
	def __init__(self, input_size, output_size, hidden_sizes=[256,64],
			dropout=0.2):
		super().__init__()
		self.fc1 = nn.Linear(input_size, hidden_sizes[0])
		self.act1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)
		self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.act2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)
		self.fc3 = nn.Linear(hidden_sizes[1], output_size)
		self.activation = PresetActivation()

	def forward(self, x):
		x = self.dropout1(self.act1(self.fc1(x)))
		x = self.dropout2(self.act2(self.fc2(x)))
		return self.activation(self.fc3(x))


class BasicCriticHead(nn.Module):
	def __init__(self, input_size, output_size, hidden_sizes=[256,64],
			dropout=0.2):
		super().__init__()
		self.fc1 = nn.Linear(input_size, hidden_sizes[0])
		self.act1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)
		self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.act2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)
		self.fc3 = nn.Linear(hidden_sizes[1], 1)

	def forward(self, x):
		x = self.dropout1(self.act1(self.fc1(x)))
		x = self.dropout2(self.act2(self.fc2(x)))
		return self.fc3(x)		


class ComparerNetwork(nn.Module):
	def __init__(self, feature_extractor, comparer_net):
		super().__init__()
		self.feature_extractor = feature_extractor
		self.comparer_net = comparer_net

	def forward(self, state):
		target_sound_emb = self.feature_extractor(state['target_spectrogram'])
		current_sound_emb = self.feature_extractor(state['current_spectrogram'])
		x = torch.cat((target_sound_emb, current_sound_emb), dim=1)
		return self.comparer_net(x)


class PolicyModel(nn.Module):
	"""Models which input state and output an action or value based
	on provided sound_comparer and decision_head.
	"""
	def __init__(self, sound_comparer, decision_head):
		super().__init__()
		self.sound_comparer = sound_comparer
		self.decision_head = decision_head

	def forward(self, state):
		sound_dif_emb = self.sound_comparer(state)
		comb_emb = torch.cat(
			[sound_dif_emb, state['current_params'], state['steps_remaining']], 
			dim=-1
		)
		return self.decision_head(comb_emb)


def n_trainable_params(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PPO:
	def __init__(self, actor, critic, actor_lr=1e-3, critic_lr=1e-3, gamma=0.9,
			cov_matrix_val=.01, reward_metric = "mae"):
		super().__init__()
  
		
		self.actor = actor
		self.critic = critic
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.gamma = gamma
		self.cov_matrix_val = cov_matrix_val
		self.audiohandler = AudioHandler()
		self.continuous_activation = nn.Hardtanh(min_val=0.0, max_val=1.0)
  
		assert reward_metric in ("mae", "sc")
		self.reward_metric = reward_metric

		# Mapping stuff
		self.continuous_params_dict = {
			k:v for element in AudioHandler.get_mapping_dict()['Numerical'] for k,v in element.items()
		}
		self.cont_logit_indices = np.array(list(self.continuous_params_dict.values()))
		self.cont_param_indices = np.array(list(self.continuous_params_dict.keys()))
		self.num_continuous = len(self.cont_logit_indices)

		self.desc_params_dict = {
			k:v for element in AudioHandler.get_mapping_dict()['Categorical'] for k,v in element.items()
		}

		# Initialize optimizers for actor and critic
		self.actor_optim = torch.optim.Adam(
			self.actor.parameters(), lr=self.actor_lr
		)
		self.critic_optim = torch.optim.Adam(
			self.critic.parameters(), lr=self.critic_lr
		)

		# Initialize the covariance matrix used to query the actor for 
		# continuous actions
		self.cov_var = torch.full(
			size=(self.num_continuous,), 
			fill_value=self.cov_matrix_val
		)
		self.cov_mat = torch.diag(self.cov_var)

	@staticmethod
	def unbatch_states(states):
		"""Unbatch a dict of batched tensors into a list of dicts of tensors.
		"""
		ret_states = []

		n_states = len(states['target_spectrogram'])
		for i in range(n_states):
			ret_states.append({
				k: v[i] for k, v in states.items()
			})

		return ret_states

	def step(self, states, actions):
		"""	
  		Performs actions in parallel for batch

		states:  a dict of batched tensors that has the following fields:
					'target_spectrogram': (rollout_batch_size, *spectrogram_shape),
					'current_spectrogram': (rollout_batch_size, *spectrogram_shape),
					'current_params': (rollout_batch_size, n_params),
					'steps_remaining': (rollout_batch_size, 1)
     
		actions: a batched tensor of shape: (rollout_batch_size, n_params)
  
		Returns:
			Tuple of:
				next_state: a dict of batched tensors that has the following fields:
								'target_spectrogram': (rollout_batch_size, *spectrogram_shape),
								'current_spectrogram': (rollout_batch_size, *spectrogram_shape),
								'current_params': (rollout_batch_size, n_params),
								'steps_remaining': (rollout_batch_size, 1)
				rewards:	(rollout_batch_size, 1)
		"""
		batch_size = actions.size(0)
		pred_states = {
			"target_spectrogram" : states['target_spectrogram'], 
			"current_spectrogram" : [],
			"current_params" : actions,
			"steps_remaining" : states['steps_remaining'] - 1
   		}
		
		# batched tensor of shape: (rollout_batch_size, 1)
  		rewards = []
		
		for i in range(len(states)):
			# Convert learnable param to synthesizer param
			param = presetParam(actions[i], learnable=True)

			# Get next state
			predicted_spectrogram = self.audiohandler.generateSpectrogram(param.to_params())
			pred_states['current_spectrogram'].append(predicted_spectrogram)

			# Generate reward
			if self.reward_metric == "mae":
				rew = -self.audiohandler.getMAE(states[i],pred_states[i])
			else:
       			# TODO: how does sc behave? is a higher sc a better/higher reward?
          		rew = self.audiohandler.getSpectralConvergence(states[i],pred_states[i])
			rewards.append(rew)
   
		pred_states['current_spectrogram'] = torch.vstack(pred_states['current_spectrogram'])
		rewards = torch.vstack(rewards).reshape(-1, 1)

		return pred_states, rewards
	
 
	def get_actions(self, states):
		""" 
  		Get actions for a batch of states. 

		states:  a dict of batched tensors that has the following fields:
				'target_spectrogram': (rollout_batch_size, *spectrogram_shape),
				'current_spectrogram': (rollout_batch_size, *spectrogram_shape),
				'current_params': (rollout_batch_size, n_params),
				'steps_remaining': (rollout_batch_size, 1)
    	"""
		action_logits = self.actor(states)

		if action_logits.dim() > 1:
			cov_mat = torch.stack(
				[ppo_model.cov_mat for _ in range(action_logits.shape[0])], 
				dim=0
			)
		else:
			cov_mat = self.cov_mat
		output_actions = torch.zeros_like(action_logits)

		# Sample actions from multivariate normal distribution for 
		# continuous params
		cont_logits = action_logits[:, self.cont_logit_indices]
		dist = MultivariateNormal(cont_logits, cov_mat)
		cont_actions = self.continuous_activation(dist.sample())
		cont_log_probs = dist.log_prob(cont_actions)

		output_actions[:, self.cont_param_indices] = cont_actions

		total_log_probs = cont_log_probs

		# Sample actions from categorical distributions for discrete params
		for dexed_idx, log_idx in self.desc_params_dict.items():
			logits = action_logits[:, log_idx]
			dist = Categorical(logits=logits)

			action_inds = dist.sample()
			total_log_probs += dist.log_prob(action_inds)

			one_hot_actions = F.one_hot(action_inds, num_classes=len(log_idx))
			output_actions[:, log_idx] = one_hot_actions.float()
  
		return output_actions.detach(), total_log_probs.detach()

	def compute_rtgs(self, batch_rewards):
		"""Compute rewards-to-go
  
		batch_rewards shape: (rollout_batch_size, num_steps_in_episode)
		output returns (ie., rewards to go): [batch_size, num_steps_in_episode]
		"""
		batch_returns = torch.zeros_like(batch_rewards)
		
		# The returns at the last time step is just equal to the reward at that last time step
		batch_returns[:,-1] = batch_rewards[:,-1]

		num_steps_in_episode = batch_rewards.shape[1]
		
		# Now, we loop thrrough each time step in reverse, from the second to last time step, to the first timestep
		for step_idx in reversed(range(num_steps_in_episode - 1)):
			batch_returns[:,step_idx] = batch_rewards[:,step_idx] + (self.gamma * batch_returns[:,step_idx+1])

		return batch_returns

	def rollout(self, init_states, n_steps):
		""" 
  		Do rollouts and return whats returned in other imp 
    
		init_states:	a dict of batched tensors that has the following fields:
				'target_spectrogram': (rollout_batch_size, *spectrogram_shape),
				'current_spectrogram': (rollout_batch_size, *spectrogram_shape),
				'current_params': (rollout_batch_size, n_params),
				'steps_remaining': (rollout_batch_size, 1)
    
		n_steps:		int for number of steps to run per episode
    	"""

		# Data on rolled-out batch which will be returned
  
  
  		# batch_obs: dict of batched tensors
		# 'target_spectrogram': (rollout_batch_size, num_steps_per_episode, *spectrogram_shape),
		# 'current_spectrogram': (rollout_batch_size, num_steps_per_episode, *spectrogram_shape),
		# 'current_params': (rollout_batch_size, num_steps_per_episode, n_params),
		# 'steps_remaining': (rollout_batch_size, num_steps_per_episode, 1)
		batch_obs = {
      		"target_spectrogram" : [],
			"current_spectrogram" : [],
			"current_params" : [],
			"steps_remaining" : [],
        }

		batch_acts = [] # batched tensor: (rollout_batch_size, num_steps_per_episode, num_action_dims)
		batch_log_probs = [] # batched tensor: (rollout_batch_size, num_steps_per_episode)
		batch_rews = [] # batched tensor: (rollout_batch_size, num_steps_per_episode)

		obs = init_states

		# Step through episode from each initial state in parallel
		for i in range(n_steps):
			# Add current state info (obs) to batch_obs
			for k,v in obs.items():
				batch_obs[k].append(v)

			# Get action from actor and take step
			# Note on the shapes of the tensors below:
			#	actions: (rollout_batch_size, num_action_dims)
			#	_log_probs: (rollout_batch_size, 1)
			#	obs: dict of batched tensors (representing the next state) 
			# 			with the tensors having the SAME shape as the tensors in init_states
			#	rews: (rollout_batch_size, 1)
			# TODO: Ensure that the variables below match their expected shapes listed above
			actions, _log_probs = self.get_actions(obs)
			obs, rews = self.step(obs, actions)

			# Update actions, action log probs, and rewards
			batch_acts.append(actions)
			batch_log_probs.append(_log_probs)
			batch_rews.append(rews)

		### Correctly reshape all aggregated tensors across all batches & rollouts ###
    
  		# Reshape batch_obs into following shapes:
		# 	'target_spectrogram': (rollout_batch_size * num_steps_per_episode, *spectrogram_shape),
		# 	'current_spectrogram': (rollout_batch_size * num_steps_per_episode, *spectrogram_shape),
		# 	'current_params': (rollout_batch_size * num_steps_per_episode, n_params),
		# 	'steps_remaining': (rollout_batch_size * num_steps_per_episode, 1)

		# Note:
		# vstack will return tensors whose shape starts with: (num_steps_per_episode, rollout_batch_size,...)
		# so, .transpose(0,1) will make the tensors' shapes start with: (rollout_batch_size, num_steps_per_episode)
		batch_obs['target_spectrogram'] = torch.vstack(batch_obs['target_spectrogram']).transpose(0,1).view(-1, *batch_obs['target_spectrogram'].shape[2:])
		batch_obs['current_spectrogram'] = torch.vstack(batch_obs['current_spectrogram']).transpose(0,1).view(-1, *batch_obs['current_spectrogram'].shape[2:])
		batch_obs['current_params'] = torch.vstack(batch_obs['current_params']).transpose(0,1).view(-1, *batch_obs['current_params'].shape[2:])
		batch_obs['steps_remaining'] = torch.vstack(batch_obs['steps_remaining']).transpose(0,1).view(-1, *batch_obs['steps_remaining'].shape[2:])
   
		# Reshape batch_acts into: (rollout_batch_size * num_steps_per_episode, n_params)
		batch_acts = torch.vstack(batch_acts).transpose(0,1).view(-1, *batch_acts.shape[2:])
  
		# Reshape batch_log_probs into (rollout_batch_size * num_steps_per_episode, 1)
		batch_log_probs = torch.vstack(batch_log_probs).transpose(0,1).view(-1,1)
  
		# First, reshape batch_rews into (rollout_batch_size, num_steps_per_episode)
		batch_rews = torch.vstack(batch_rews).transpose(0,1)
  
		# Compute returns to go using batch_rews: shape: (rollout_batch_size, num_steps_per_episode)
		# Also reshape batch_rtgs from (rollout_batch_size, num_steps_per_episode) into (rollout_batch_size*num_steps_per_episode,1)
		batch_rtgs = self.compute_rtgs(batch_rews).transpose(0,1).view(-1,1)
  
		# Finally, reshape batch_rews into (rollout_batch_size*num_steps_per_episode,1)
		batch_rews = batch_rews.view(-1, 1)
  
		return batch_obs, batch_acts, batch_log_probs, batch_rtgs

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.
			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)
			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs) #.squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs


if __name__ == '__main__':
	__spec__ = None

	# hyperparameters
	steps_per_episode = 25
	rollout_batch_size = 32
	num_train_iter = 100
	n_updates_per_iteration = 5	 # Number of times to update actor/critic per iteration

	# Load the target sounds
	dataset = TargetSoundDataset(
		data_dir=os.path.join('data', 'preset_data'),
		split_file='split_dict.json',
		split_name='train',
		return_params=True
	)

	target_sound_loader = DataLoader(
		dataset=dataset,
		batch_size=rollout_batch_size,
		num_workers=4,
		shuffle=True
	)

	# Parameters about the data
	mapping_dict = AudioHandler.get_mapping_dict()
	n_params = len(mapping_dict['Numerical']) + sum(
		[len(*p.values()) for p in mapping_dict['Categorical']])

	spectrogram_shape = (257, 345)

	# define actor and critic models
	actor_comp_net = ComparerNetwork(
		feature_extractor=CNNFeatExtractor(n_feat_channels=spectrogram_shape[0]),
		comparer_net=CNNComparer(n_feat_channels=32*2)
	)
	actor = PolicyModel(
		sound_comparer=actor_comp_net,
		decision_head=BasicActorHead(416 + n_params + 1, n_params)
	)
	critic_comp_net = ComparerNetwork(
		feature_extractor=CNNFeatExtractor(n_feat_channels=spectrogram_shape[0]),
		comparer_net=CNNComparer(n_feat_channels=32*2)
	)
	critic = PolicyModel(
		sound_comparer=critic_comp_net,
		decision_head=BasicActorHead(416 + n_params + 1, 1)
	)

	test_state_batch = {
		'target_spectrogram': torch.rand((rollout_batch_size, *spectrogram_shape)),
		'current_spectrogram': torch.rand((rollout_batch_size, *spectrogram_shape)),
		'current_params': torch.rand((rollout_batch_size, n_params)),
		'steps_remaining': torch.randint(0, steps_per_episode, (rollout_batch_size, 1))
	}

	r_a = actor(test_state_batch)
	r_c = critic(test_state_batch)

	print(n_trainable_params(actor))
	print(n_trainable_params(critic))

	# Initialize the PPO object
	ppo_model = PPO(actor, critic, 120)

	# Training loop
	target_iterator = iter(target_sound_loader)

	for i in tqdm(range(num_train_iter)):
		# Rollout from rollout_batch_size starts
		try:
			data = next(target_iterator)
		except StopIteration:
			target_iterator = iter(target_sound_loader)
			data = next(target_iterator)

		target_spectrograms = data['spectrogram']

		# Get initial guesses to complete starting state
		# TODO: implement this instead of current random
		init_params = torch.rand((rollout_batch_size, n_params))
		init_spectrograms = torch.rand((rollout_batch_size, *spectrogram_shape))

		# Add number of steps remaining to the initial state
		init_steps_remaining = torch.ones((rollout_batch_size, 1)) * steps_per_episode

		rollout_init_states = {
			'target_spectrogram': target_spectrograms,
			'current_spectrogram': init_spectrograms,
			'current_params': init_params,
			'steps_remaining': init_steps_remaining
		}

		# Do rollouts
		"""
		Expected shapes of below variables:

		batch_obs: 				state dict of batched tensors 
									(where the first dimension of each batched tensor is rollout_batch_size * num_steps_per_episode)
		batch_acts:				(rollout_batch_size * num_steps_per_episode, num_action_dims)
		batch_log_probs:		(rollout_batch_size * num_steps_per_episode, 1)
		batch_rtgs:				(rollout_batch_size * num_steps_per_episode, 1)
    	"""
		(batch_obs, batch_acts, batch_log_probs, batch_rtgs) = ppo_model.rollout(rollout_init_states, n_steps=steps_per_episode)
		
		# Compute advantages and normalize
		V, _ = ppo_model.evaluate(batch_obs, batch_acts)
		A_k = batch_rtgs - V.detach()
		A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

		# Update actor and critic
  
		# TODO: Construct a Dataset object here? 
		for j in range(n_updates_per_iteration):
			# Calculate V_phi and pi_theta(a_t | s_t)
			V, curr_log_probs = ppo_model.evaluate(batch_obs, batch_acts)

			# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
			# NOTE: we just subtract the logs, which is the same as
			# dividing the values and then canceling the log with e^log.
			# For why we use log probabilities instead of actual probabilities,
			# here's a great explanation: 
			# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
			# TL;DR makes gradient ascent easier behind the scenes.
			ratios = torch.exp(curr_log_probs - batch_log_probs)
   
			# Calculate surrogate losses.
			surr1 = ratios * A_k
			surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

			# Calculate actor and critic losses.
			# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
			# the performance function, but Adam minimizes the loss. So minimizing the negative
			# performance function maximizes it.
			actor_loss = (-torch.min(surr1, surr2)).mean()
			critic_loss = nn.MSELoss()(V, batch_rtgs)

			# Calculate gradients and perform backward propagation for actor network
			self.actor_optim.zero_grad()
			actor_loss.backward(retain_graph=True)
			self.actor_optim.step()

			# Calculate gradients and perform backward propagation for critic network
			self.critic_optim.zero_grad()
			critic_loss.backward()
			self.critic_optim.step()