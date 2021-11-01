import os

# SAMPLE_RATE is the number of audio samples per second.
SAMPLE_RATE = 44100

# Audio is rendered one block at a time, and the size of the block is
# the BUFFER_SIZE. If you're using the set_automation function, you should
# choose a smaller power of 2 buffer size such as 64 or 128.
BUFFER_SIZE = 512

# Note: all paths must be absolute paths
# Note: only VST2 plugins are supported for now (i.e. VST3 plugins do not work, for now)
SYNTH_PLUGIN = os.path.join('..', '..', 'Dexed.dll')

# If False, this will use a constant midi note + velocity for all synth presets.
# Otherwise, a random midi note + velocity will be used for each synth preset.
USE_RANDOM_MIDI_NOTE = False