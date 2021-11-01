import argparse
import glob
import json
import os
import random
from collections import defaultdict

import dawdreamer as daw
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

import settings


def generate_audio(engine, synth, preset_author, audio_id, dest_dir, duration_secs, use_random_midi_note=False):
    if not use_random_midi_note:
        # 60 is middle C. Velocity is an integer from 0-127.
        midi_note_number = 60
        velocity = 127
    else:
        midi_note_number = random.randint(0, 127)
        velocity = random.randint(0, 127)

    synth.add_midi_note(midi_note_number, velocity, 0.0,
                        duration_secs)  # (MIDI note, velocity, start sec, duration sec)

    # Render audio
    engine.load_graph([(synth, [])])
    engine.render(duration_secs)
    audio = engine.get_audio()
    wavfile.write(os.path.join(dest_dir, f"{audio_id}.wav"), settings.SAMPLE_RATE,
                  audio.transpose())  # don't forget to transpose!

    # Get synth params
    n_params = synth.get_plugin_parameter_size()
    synth_params = np.zeros(n_params, dtype=np.float32)

    for i in range(n_params):
        synth_params[i] = synth.get_parameter(i)

    # Save metadata json file for this audio
    with open(os.path.join(dest_dir, f"{audio_id}.json"), "w") as f:
        json.dump({
            "preset_author": preset_author,
            "synth_params": synth_params.tolist(),
            "note_params": {
                "midi_note_number": midi_note_number,
                "velocity": velocity,
                "duration": duration_secs,
            }
        }, f)


def create_synth(engine, preset_filepath):
    synth = engine.make_plugin_processor("my_synth", settings.SYNTH_PLUGIN)
    synth.load_preset(preset_filepath)
    # Set the synth's middle C parameter to 0.5, to avoid the synth from transposing the input midi note
    synth.set_parameter(13, 0.5)
    return synth


def create_dataset(
        presets_root_dir,
        dataset_dest_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        duration_secs=4,
        use_random_midi_note=False,
):
    assert not os.path.exists(dataset_dest_dir), "The dataset destination directory must be empty!"
    assert train_ratio + val_ratio + test_ratio == 1.0, "Train/val/test ratios must sum to 1.0"

    # Make an engine. We'll only need one.
    engine = daw.RenderEngine(settings.SAMPLE_RATE, settings.BUFFER_SIZE)

    if not presets_root_dir.endswith("/"):
        presets_root_dir += "/"

    # Get a list of all the preset files
    preset_filepaths = set(glob.glob(os.path.join(presets_root_dir, '**/*.syx'), recursive=True) + glob.glob(
        os.path.join(presets_root_dir, '**/*.SYX'), recursive=True))

    # Create a mapping from preset author to all the preset files under that author
    preset_author_x_preset_filepaths = defaultdict(set)

    for preset_filepath in preset_filepaths:
        # Extract preset author
        preset_author = preset_filepath.replace(presets_root_dir, "").split("/")[1]
        # Add this preset filepath to the list of presets associated with this preset author
        preset_author_x_preset_filepaths[preset_author].add(preset_filepath)

    # Construct a sorted list of preset authors in a descending order of their number of associated presets
    preset_authors = sorted(preset_author_x_preset_filepaths.keys(),
                            key=lambda preset_author: len(preset_author_x_preset_filepaths[preset_author]),
                            reverse=True)

    # Split the preset authors list into train/val/test
    num_train = int(train_ratio * len(preset_authors))
    num_val = int(val_ratio * len(preset_authors))

    train_preset_authors = preset_authors[:num_train]
    val_preset_authors = preset_authors[num_train: num_train + num_val]
    test_preset_authors = preset_authors[num_train + num_val:]

    assert len(train_preset_authors) > 0, "Number of training preset authors is empty!"
    assert len(val_preset_authors) > 0, "Number of validation preset authors is empty!"
    assert len(test_preset_authors) > 0, "Number of testing preset authors is empty!"

    train_preset_files = []
    for preset_author in train_preset_authors:
        train_preset_files.extend(
            [
                (preset_author, preset)
                for preset in preset_author_x_preset_filepaths[preset_author]
            ]
        )

    val_preset_files = []
    for preset_author in val_preset_authors:
        val_preset_files.extend(
            [
                (preset_author, preset)
                for preset in preset_author_x_preset_filepaths[preset_author]
            ]
        )

    test_preset_files = []
    for preset_author in test_preset_authors:
        test_preset_files.extend(
            [
                (preset_author, preset)
                for preset in preset_author_x_preset_filepaths[preset_author]
            ]
        )

    print(f"Number of training presets: {len(train_preset_files)}")
    print(f"Number of validation presets: {len(val_preset_files)}")
    print(f"Number of test presets: {len(test_preset_files)}")

    """
    We now prepare the dataset directories. Dataset layout will be as follows:
    
    dataset_root_dir/
        - train/
            - 0.wav
            - 0.json
            - 1.wav
            - 1.json
            - 2.wav
            - 2.json
            ...
        - val/
            ...
        - test/
            ...
            
    The <sample_idx>.json files will have the following information:
    {
        "preset_author" : "<preset_author>",
        "synth_params" : <synth parameter array>,
        "note_params" : {
            "midi_note_number" : 60,
            "velocity" : 127,
            "duration" : 4,
        }
    }
    """
    os.makedirs(dataset_dest_dir)
    os.makedirs(os.path.join(dataset_dest_dir, "train"))
    os.makedirs(os.path.join(dataset_dest_dir, "val"))
    os.makedirs(os.path.join(dataset_dest_dir, "test"))

    current_audio_id = 0

    print("Creating training split...")
    dest_dir = os.path.join(dataset_dest_dir, "train")

    for preset_author, preset_filepath in tqdm(train_preset_files):
        # Create a synth using this preset
        synth = create_synth(engine=engine, preset_filepath=preset_filepath)

        # Generate an audio with this preset
        generate_audio(engine, synth, preset_author, current_audio_id, dest_dir, duration_secs,
                       use_random_midi_note=use_random_midi_note)

        # Increment global audio id
        current_audio_id += 1

    print("Creating validation split...")
    dest_dir = os.path.join(dataset_dest_dir, "val")

    for preset_author, preset_filepath in tqdm(val_preset_files):
        # Create a synth using this preset
        synth = create_synth(engine=engine, preset_filepath=preset_filepath)

        # Generate an audio with this preset
        generate_audio(engine, synth, preset_author, current_audio_id, dest_dir, duration_secs,
                       use_random_midi_note=use_random_midi_note)

        # Increment global audio id
        current_audio_id += 1

    print("Creating testing split...")
    dest_dir = os.path.join(dataset_dest_dir, "test")

    for preset_author, preset_filepath in tqdm(test_preset_files):
        # Create a synth using this preset
        synth = create_synth(engine=engine, preset_filepath=preset_filepath)

        # Generate an audio with this preset
        generate_audio(engine, synth, preset_author, current_audio_id, dest_dir, duration_secs,
                       use_random_midi_note=use_random_midi_note)

        # Increment global audio id
        current_audio_id += 1

    print("Done! :)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("presets_root_dir", help="Path to the root directory containing all the preset files", type=str,
                        required=True)
    parser.add_argument("dataset_dest_dir",
                        help="Path to the destination directory that will be created to store the dataset", type=str,
                        required=True)
    parser.add_argument("train_ratio",
                        help="Training ratio (0.0-1.0) for the relative number of preset authors to be included in the training split",
                        type=float, default=0.8)
    parser.add_argument("val_ratio",
                        help="Validation ratio (0.0-1.0) for the relative number of preset authors to be included in the validation split",
                        type=float, default=0.1)
    parser.add_argument("test_ratio",
                        help="Test ratio (0.0-1.0) for the relative number of preset authors to be included in the test split",
                        type=float, default=0.1)
    parser.add_argument("duration_secs", help="The number of seconds for each generated audio", type=int, default=4)
    args = parser.parse_args()

    create_dataset(
        presets_root_dir=args.presets_root_dir,
        dataset_dest_dir=args.dataset_dest_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        duration_secs=args.duration_secs,
        use_random_midi_note=settings.USE_RANDOM_MIDI_NOTE,
    )