import os
import json

import numpy as np
from tqdm import tqdm

from dexed import PresetDatabase, Dexed


if __name__ == "__main__":
    __spec__ = None

    # Set parameters
    save_dir = os.path.join(os.path.realpath('..'), 'data', 'preset_data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load needed data and synth
    dexed_db = PresetDatabase()
    dexed_synth = Dexed(plugin_path='Dexed.dll')
    split_inds = np.load('preset_indices.pickle', allow_pickle=True)

    # Get UID sets for each split
    split_inds = {k: set(v) for k, v in split_inds.items()}

    # Dictionary that will be used be dataloader to determine splits
    split_dict = {
        'train': {'npz_fnames': [], 'wav_fnames': []}, 
        'val': {'npz_fnames': [], 'wav_fnames': []}, 
        'test': {'npz_fnames': [], 'wav_fnames': []}
    }

    for i in tqdm(range(len(dexed_db.all_presets_df))):
        # Get preset info
        preset_data = dexed_db.all_presets_df.iloc[i]
        preset_UID = preset_data['index_preset']
        param_values = dexed_db.presets_mat[i, :]

        save_path = os.path.join(save_dir, str(preset_UID))

        # Generate audio and save as wav and npz
        dexed_synth.set_param_array(param_values)
        audio = dexed_synth.render_note_to_file(60, 100, '{}.wav'.format(save_path))
        np.savez('{}.npz'.format(save_path), audio=audio, params=param_values)

        if preset_UID in split_inds['train']:
            split_dict['train']['npz_fnames'].append('{}.npz'.format(preset_UID))
            split_dict['train']['wav_fnames'].append('{}.wav'.format(preset_UID))
        elif preset_UID in split_inds['validation']:
            split_dict['val']['npz_fnames'].append('{}.npz'.format(preset_UID))
            split_dict['val']['wav_fnames'].append('{}.wav'.format(preset_UID))
        elif preset_UID in split_inds['test']:
            split_dict['test']['npz_fnames'].append('{}.npz'.format(preset_UID))
            split_dict['test']['wav_fnames'].append('{}.wav'.format(preset_UID))

    # Save split dict
    with open(os.path.join(save_dir, 'split_dict.json'), 'w') as f:
        json.dump(split_dict, f)