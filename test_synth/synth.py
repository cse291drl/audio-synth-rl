import os

import numpy as np
from scipy.io import wavfile

import dawdreamer as daw


class DexedSynth:
	"""Controls Dexed vst using dawdreamer for generating audio samples.
	
	Attributes:
		dll_path (str): Path to Dexed.dll (Vst2 version, e.g. v0.9.4).
		sample_rate (int): The sample rate of the audio in Hz. Defaults
			NSynth rate 16000.
		buffer_size (int): Audio is rendered one block at a time. This 
			defines the block size. Defaults to 1024.
		midi_note (int): The MIDI note to play. Defaults to 60 (middle C).
		midi_velocity (int): The MIDI velocity to play. Defaults to 127.
		note_hold (int): The number of seconds to hold the note. Defaults
			to NSynth style 3 seconds.
		sample_time (int): The number of seconds to render the audio. 
			Defaults to NSynth style 4 seconds.

		engine (dawdreamer.RenderEngine): The engine used to render audio.
		synth (dawdreamer.PluginProcessor): The Dexed synth.
	"""
	def __init__(self, dll_path, 
					sample_rate=16000, buffer_size=1024, midi_note=60,
					midi_velocity=127, note_hold=3, sample_time=4):
		self.dll_path = dll_path
		self.sample_rate = sample_rate
		self.buffer_size = buffer_size
		self.midi_note = midi_note
		self.midi_velocity = midi_velocity
		self.note_hold = note_hold
		self.sample_time = sample_time

		# Create the engine and synth
		self.engine = daw.RenderEngine(self.sample_rate, self.buffer_size)
		self.synth = self.engine.make_plugin_processor("dexed", self.dll_path)

		# Add note to synth
		self.synth.add_midi_note(
			self.midi_note, 
			self.midi_velocity, 
			0.0, # note start time
			self.note_hold)

		# Get info about initial synth params
		self.n_params = self.synth.get_plugin_parameter_size()
		self.params = np.zeros(self.n_params, dtype=np.float32)
		self.param_names = np.empty(self.n_params, dtype="S20")

		for i in range(self.n_params):
			self.param_names[i] = self.synth.get_parameter_name(i)
			self.params[i] = self.synth.get_parameter(i)

		self.param_names = self.param_names.astype(str)

	def get_current_params(self):
		"""Updates self.params to match vst state and returns the current 
		values of the synth parameters."""
		for i in range(self.n_params):
			self.params[i] = self.synth.get_parameter(i)
		return self.params

	def param_desc_str(self):
		"""Returns a string describing the parameters of the synth."""
		self.get_current_params()
		return '\n'.join(
			['{:<5d}{:<21s}{}'.format(i, self.param_names[i], self.params[i]) 
				for i in range(self.n_params)]
		)

	def set_param(self, param_idx, value):
		"""Sets the value of a synth parameter by index.

		Args:
			param_idx (int): The index of the parameter to set.
			value (float): The value to set the parameter to.
		"""
		self.synth.set_parameter(param_idx, value)
		self.get_current_params()

	def set_param_by_name(self, param_name, value):
		"""Sets the value of a synth parameter by name.

		Args:
			param_name (str): The name of the parameter to set.
			value (float): The value to set the parameter to.
		"""
		param_idx = np.where(self.param_names == param_name)[0]
		self.set_param(param_idx, value)
		
	def set_params(self, params):
		"""Sets synth parameters to be given params array.

		Args:
			params (list-like): A list-like of values to set the 
				synth parameters with corresponding indices to.
		"""
		for i in range(self.n_params):
			self.synth.set_parameter(i, params[i])
		self.get_current_params()

	def get_audio(self):
		"""Renders audio using the current synth parameters.
		
		Returns:
			numpy.ndarray: The rendered audio.
		"""
		self.engine.load_graph([(self.synth, [])])
		self.engine.render(self.sample_time)
		return self.engine.get_audio()

	def get_and_save_audio(self, filename):
		"""Renders audio using the current synth parameters and saves it to
		filename.

		Args:
			filename (str): The path to save the audio to.
		"""
		audio = self.get_audio()
		wavfile.write(filename, self.sample_rate, audio.transpose())
		return audio


if __name__ == '__main__':
	dexed_path = os.path.join('..', '..', 'Dexed.dll')
	synth = DexedSynth(dexed_path)

	print(synth.param_desc_str())
	a0 = synth.get_and_save_audio('test0.wav')
	
	synth.set_param(5, 0.555)
	synth.set_param_by_name('ALGORITHM', 0.444)
	print(synth.param_desc_str())
	a1 = synth.get_and_save_audio('test1.wav')

	new_params = np.random.rand(synth.n_params)
	synth.set_params(new_params)
	print(synth.param_desc_str())
	a2 = synth.get_and_save_audio('test2.wav')

	new_params = np.random.rand(synth.n_params)
	synth.set_params(new_params)
	print(synth.param_desc_str())
	a3 = synth.get_and_save_audio('test3.wav')

	# Fix output volume to 1.0
	new_params = np.random.rand(synth.n_params)
	new_params[2] = 1.0
	synth.set_params(new_params)
	print(synth.param_desc_str())
	a4 = synth.get_and_save_audio('test4.wav')

	new_params = np.random.rand(synth.n_params)
	new_params[2] = 1.0
	synth.set_params(new_params)
	print(synth.param_desc_str())
	a5 = synth.get_and_save_audio('test5.wav')

	# Also set all osc params to 1.0
	new_params = np.random.rand(synth.n_params)
	new_params[2] = 1.0
	new_params[[154, 132, 110, 88, 66, 44]] = 1.0
	synth.set_params(new_params)
	print(synth.param_desc_str())
	a6 = synth.get_and_save_audio('test6.wav')

	new_params = np.random.rand(synth.n_params)
	new_params[2] = 1.0
	new_params[[154, 132, 110, 88, 66, 44]] = 1.0
	synth.set_params(new_params)
	print(synth.param_desc_str())
	a7 = synth.get_and_save_audio('test7.wav')

	new_params = np.random.rand(synth.n_params)
	new_params[2] = 1.0
	new_params[[154, 132, 110, 88, 66, 44]] = 1.0
	synth.set_params(new_params)
	print(synth.param_desc_str())
	a8 = synth.get_and_save_audio('test8.wav')

	new_params = np.random.rand(synth.n_params)
	new_params[2] = 1.0
	new_params[[154, 132, 110, 88, 66, 44]] = 1.0
	synth.set_params(new_params)
	print(synth.param_desc_str())
	a9 = synth.get_and_save_audio('test9.wav')