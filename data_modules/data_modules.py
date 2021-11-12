import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import sys
sys.path.append("..") #Uncomment if running from main dir
from synth.dexed import Dexed
import audio 

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
				 n_fft = 1024, fft_hop = 256,  #ftt 1024 hop=512: spectrogram is approx. the size of 5.0s@22.05kHz audio
                 n_mel_bins=257, mel_fmin=30.0, mel_fmax=11e3,
                 spectrogram_min_dB=-120.0, return_params=False, load_wav=False):
		self.data_dir = data_dir
		self.split_file = split_file
		self.split_name = split_name
		self.return_params = return_params
		self.load_wav = load_wav
        
        # Spectrogram config vars
		self.n_fft = n_fft
		self.fft_hop = fft_hop
		self.n_mel_bins = n_mel_bins
		self.mel_fmin = mel_fmin
		self.mel_fmax = mel_fmax
		if self.n_mel_bins <= 0:
			self.spectrogram = audio.Spectrogram(self.n_fft, self.fft_hop, spectrogram_min_dB)
		else:  # TODO do not hardcode Fs?
			self.spectrogram = audio.MelSpectrogram(self.n_fft, self.fft_hop, spectrogram_min_dB,
															self.n_mel_bins, 22050)		

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
				'spectrogram':self.spectrogram(loaded_npz['audio']),
			}
		else:
			return {
				'audio': torch.tensor(loaded_npz['audio']).float(),
				'spectrogram':self.spectrogram(loaded_npz['audio'])
			}

class AudioHandler:
	'''
	Wrapper class around audio.py to handle RL based tasks AudioHandler class for handling RL based audio things
	'''
	def __init__(self, plugin_path = 'Dexed.dll', note_val = (60,100),n_fft = 1024, fft_hop = 256,
				n_mel_bins=257, mel_fmin=30.0, mel_fmax=11e3,
				spectrogram_min_dB=-120.0):

		self.n_fft = n_fft
		self.fft_hop = fft_hop
		self.n_mel_bins = n_mel_bins
		self.mel_fmin = mel_fmin
		self.mel_fmax = mel_fmax
		self.synth = Dexed(plugin_path = plugin_path)
		self.note_val = note_val
		if self.n_mel_bins <= 0:
			self.spectrogram = audio.Spectrogram(self.n_fft, self.fft_hop, spectrogram_min_dB)
		else: 
			self.spectrogram = audio.MelSpectrogram(self.n_fft, self.fft_hop, spectrogram_min_dB,
														self.n_mel_bins, 22050)	

	def generateAudio(self, params, return_spec = True ):
		'''
		Generates audio given a particular set of parameters, and returns the associated spectrogram with the audio.
		'''
		self.synth.set_param_array(params)
		audio = self.synth.render_note(self.note_val[0],self.note_val[1])
		if return_spec:
			return audio, self.spectrogram(audio)
		return audio

	def generateSpectrogram(self, params, return_spec = True ):
		'''
		Generates audio given a particular set of parameters, and returns the associated spectrogram with the audio.
		'''
		_, spec = self.generateAudio(params, return_spec = True)
		return spec
	
	def getMAE(self, target_spectrogram, pred_spectrogram):
		'''
		Calculates MAE based on STFT
		'''
		target_stft = self.spectrogram.mel_dB_to_STFT(target_spectrogram)
		pred_stft = self.spectrogram.mel_dB_to_STFT(pred_spectrogram)
		eps = 1e-4  # -80dB  (un-normalized stfts)
		log_target_stft = np.log10(np.maximum(target_stft,eps))
		log_pred_stft = np.log10(np.maximum(pred_stft,eps))
		mae = np.abs(log_target_stft-log_pred_stft).mean()
		return mae

	def getSpectralConvergence(self, target_spectrogram,pred_spectrogram):
		'''
  		Calculates spectral convergence
    	'''
		target_stft = self.spectrogram.mel_dB_to_STFT(target_spectrogram)
		pred_stft = self.spectrogram.mel_dB_to_STFT(pred_spectrogram)
		sc = np.linalg.norm(target_stft - pred_stft, ord='fro') / np.linalg.norm(target_stft, ord='fro')
		return sc

	@staticmethod
	def get_mapping_dict():
		numerical_set = set(Dexed.get_numerical_params_indexes_learnable())
		categorical_set = set(Dexed.get_categorical_params_indexes_learnable())
		preset_length = 155
		mapping_dict = {'Numerical':[],'Categorical':[]}
		learn_idx = 0
		for vst_idx in range(preset_length):
			if vst_idx in categorical_set:
				n_classes = Dexed.get_param_cardinality(vst_idx)
				cat_list = []
				for idx in range(n_classes):
					cat_list.append(learn_idx)
					learn_idx += 1
				mapping_dict['Categorical'].append({vst_idx:cat_list})
			elif vst_idx in numerical_set:
				mapping_dict['Numerical'].append({vst_idx:learn_idx})
				learn_idx += 1
		return mapping_dict  

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
			self.data_dir, split_file = self.split_file, split_name='train', return_params=self.return_params, n_fft = 1024, fft_hop = 256, n_mel_bins=257, spectrogram_min_dB=-120.0
		)
		self.val_data = TargetSoundDataset(
			self.data_dir, split_file = self.split_file, split_name='val', return_params=self.return_params, n_fft = 1024, fft_hop = 256, n_mel_bins=257, spectrogram_min_dB=-120.0
		)
		self.test_data = TargetSoundDataset(
			self.data_dir, split_file = self.split_file, split_name='test', return_params=self.return_params, n_fft = 1024, fft_hop = 256, n_mel_bins=257, spectrogram_min_dB=-120.0
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
		data_dir=os.path.join(os.path.realpath('..'), 'data', 'preset_data'),
		split_file='split_dict.json',
		split_name='val'
	)
	d0 = dataset[0]

	# Test DataModule with preset data
	preset_data = TargetSoundDataModule(
	    data_dir=os.path.join(os.path.realpath('..'), 'data', 'preset_data'),
	    split_file='split_dict.json',
	    num_workers=3,
	    shuffle=True
	)
	preset_data.setup()
	v0 = next(iter(preset_data.val_dataloader()))
	print(dataset.split_data.keys())
	print(v0['audio'].shape)
	print(v0)
	print(preset_data.val_data[0])
	print(d0)
	print(torch.all(preset_data.val_data[0]['audio'] == d0['audio']))