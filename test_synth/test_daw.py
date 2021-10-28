import os

import numpy as np
from scipy.io import wavfile

import dawdreamer as daw
import librosa



if __name__ == '__main__':
	# SAMPLE_RATE is the number of audio samples per second.
	SAMPLE_RATE = 44100  

	# Audio is rendered one block at a time, and the size of the block is
	# the BUFFER_SIZE. If you're using the set_automation function, you should
	# choose a smaller power of 2 buffer size such as 64 or 128.
	BUFFER_SIZE = 512

	# Note: all paths must be absolute paths
	# Note: only VST2 plugins are supported for now (i.e. VST3 plugins do not work, for now)
	SYNTH_PLUGIN = os.path.join('..', '..', 'Dexed.dll')
	# REVERB_PLUGIN = "C:/path/to/reverb.dll"

	# fxp is a conventional file extension for VST presets.
	SYNTH_PRESET = os.path.join('..', '..', 'DX7_AllTheWeb', 'Aminet', '2.syx')

	DURATION = 5 # How many seconds we want to render.

	# # a path to a stereo audio file.
	# VOCALS_PATH = "C:/path/to/vocals.wav"

	# Make an engine. We'll only need one.
	engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)

	# Make a processor and give it the name "my_synth", which we must remember later.
	synth = engine.make_plugin_processor("my_synth", SYNTH_PLUGIN)
	print(synth.get_parameter_name(4), synth.get_parameter_text(4), synth.get_parameter(4))

	synth.load_preset(SYNTH_PRESET)
	print(synth.get_parameter_name(4), synth.get_parameter_text(4), synth.get_parameter(4))

	synth.add_midi_note(60, 127, 0.0, 3) # (MIDI note, velocity, start sec, duration sec)
	# 60 is middle C. Velocity is an integer from 0-127.

	# Render audio
	engine.load_graph([(synth, [])])
	engine.render(DURATION)
	audio = engine.get_audio()
	wavfile.write('test_sound.wav', SAMPLE_RATE, audio.transpose()) # don't forget to transpose!

	synth.set_parameter(4, .5)
	print(synth.get_parameter_name(4), synth.get_parameter_text(4), synth.get_parameter(4))
	engine.load_graph([(synth, [])])
	engine.render(DURATION)
	audio = engine.get_audio()
	wavfile.write('test_sound2.wav', SAMPLE_RATE, audio.transpose())