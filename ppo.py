import os

import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_modules.data_modules import TargetSoundDataset


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

	def forward(self, x):
		x = self.dropout1(self.act1(self.fc1(x)))
		x = self.dropout2(self.act2(self.fc2(x)))
		return self.fc3(x)


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
	def __init__(self, actor, critic, num_continuous, actor_lr=1e-3, critic_lr=1e-3):
		super().__init__()
		self.actor = actor
		self.critic = critic
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr

		self.num_continuous = num_continuous

		# Initialize optimizers for actor and critic
		self.actor_optim = torch.optim.Adam(
			self.actor.parameters(), lr=self.actor_lr
		)
		self.critic_optim = torch.optim.Adam(
			self.critic.parameters(), lr=self.critic_lr
		)

		# Initialize the covariance matrix used to query the actor for 
		# continuous actions
		self.cov_var = torch.full(size=(self.num_continuous,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

	def rollout(self, init_states):
		""" Do rollouts and return whats returned in other imp """
		pass


if __name__ == '__main__':
	__spec__ = None

	# hyperparameters
	steps_per_episode = 25
	rollout_batch_size = 32
	num_train_iter = 100

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

	# these parameters about the data should become constants once they're set
	spectrogram_shape = (257, 345)
	n_params = 144

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
		sound_comparer=actor_comp_net,
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
		batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout(
			rollout_init_states)
		break