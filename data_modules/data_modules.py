import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class TargetSoundDataset(Dataset):
	""" Dataset containing target sounds.
	
	Sounds are currently saved in two parts, a WAV and a NPZ file. For
	most things, we should only need the NPZ file.

	Attributes:
		data_dir: Directory containing NPZ/WAV files.
		split_file: File containing split information.
		split_name: Name of split to use.
		return_params: Whether to include underlying synth parameters in 
			returned dict. This being false allows us to train on audio 
			where we don't know/there aren't underlying synth params.
		load_wav: Whether to load WAV files. Defaults to False and
			is NOT IMPLEMENTED for True.
	"""
	def __init__(self, data_dir, split_file, split_name, 
					return_params=False, load_wav=False):
		self.data_dir = data_dir
		self.split_file = split_file
		self.split_name = split_name
		self.return_params = return_params
		self.load_wav = load_wav

		if self.load_wav:
			raise NotImplementedError("WAV loading not implemented.")

		# load data for samples in split
		with open(os.path.join(self.data_dir, self.split_file)) as fp:    
			self.split_data = json.load(fp)[self.split_name]

	def __len__(self):
		return len(self.split_data['npz_fnames'])

	def __getitem__(self, idx):
		"""Return sequence in matrix form, sequence as string, chromosome
		location, and label."""
		item_npz_fname = self.split_data['npz_fnames'][idx]
		loaded_npz = np.load(
			os.path.join(self.data_dir, item_npz_fname),
			allow_pickle=True
		)

		if self.return_params:
			return {
				'audio': torch.tensor(loaded_npz['audio']).float(),
				'params': loaded_npz['params'],
			}
		else:
			return {
				'audio': torch.tensor(loaded_npz['audio']).float(),
			}


class TargetSoundDataModule(pl.LightningDataModule):
	def __init__(self, data_dir, split_file, batch_size = 32, num_workers = 1,
					shuffle = True, return_params = False):
		super().__init__()
		self.data_dir = data_dir
		self.split_file = split_file
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.shuffle = shuffle
		self.return_params = return_params

	def setup(self, stage=None):
		self.train_data = TargetSoundDataset(
			self.data_dir, self.split_file, 'train', self.return_params
		)
		self.val_data = TargetSoundDataset(
			self.data_dir, self.split_file, 'val', self.return_params
		)
		self.test_data = TargetSoundDataset(
			self.data_dir, self.split_file, 'test', self.return_params
		)

	def train_dataloader(self):
		return DataLoader(self.train_data, batch_size=self.batch_size,
			num_workers=self.num_workers, shuffle=self.shuffle)

	def val_dataloader(self):
		return DataLoader(self.val_data, batch_size=self.batch_size,
			num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.test_data, batch_size=self.batch_size,
			num_workers=self.num_workers)


if __name__ == '__main__':
	__spec__ = None

	# Test Dataset with preset data
	dataset = TargetSoundDataset(
		data_dir=os.path.join('..', 'data', 'preset_data'),
		split_file='split_dict.json',
		split_name='val'
	)
	d0 = dataset[0]

	# Test DataModule with preset data
	preset_data = TargetSoundDataModule(
	    data_dir=os.path.join('..', 'data', 'preset_data'),
	    split_file='split_dict.json',
	    num_workers=3,
	    shuffle=True
	)
	preset_data.setup()
	v0 = next(iter(preset_data.val_dataloader()))
	print(v0['audio'].shape)
	print(v0)
	print(preset_data.val_data[0])
	print(d0)
	print(torch.all(preset_data.val_data[0]['audio'] == d0['audio']))