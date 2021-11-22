import os
import pickle

import torch
from torch.utils.data import DataLoader

import vae_model.build
from data_modules.data_modules import TargetSoundDataset


if __name__ == '__main__':
    __spec__ = None

    config_dir = 'vae_model'

    # Create model
    with open(os.path.join(config_dir, 'subset_samplers.pickle'), 'rb') as f:
        idx_helper = pickle.load(f)
    with open(os.path.join(config_dir, 'config_train.pickle'), 'rb') as f:
        conf_train = pickle.load(f)
    with open(os.path.join(config_dir, 'config_model.pickle'), 'rb') as f:
        conf_model = pickle.load(f)

    _, _, _, model = vae_model.build.build_extended_ae_model(conf_model, conf_train,
                                                                     idx_helper)

    # Load from checkpoint
    checkpoint_state_dict = torch.load(
        os.path.join(config_dir, '00133.tar'), 
        map_location=torch.device('cpu')
    )
    model.load_state_dict(checkpoint_state_dict['ae_model_state_dict'])

    # Use with our dataset
    dataset = TargetSoundDataset(
		data_dir=os.path.join('data', 'preset_data'),
		split_file='split_dict.json',
		split_name='train',
		return_params=True
	)

    target_sound_loader = DataLoader(
		dataset=dataset,
		batch_size=4,
		num_workers=2,
		shuffle=True,
		persistent_workers=True
	)

    demo_batch = next(iter(target_sound_loader))
    pred = model(demo_batch['spectrogram'].unsqueeze(1))

    # Options we can use as our feature extractor
    fe_emb1 = model.ae_model.encoder.single_ch_cnn(demo_batch['spectrogram'].unsqueeze(1))
    fe_emb2 = model.ae_model.encoder.features_mixer_cnn(fe_emb1)
    fe_emb3 = model.ae_model.encoder(demo_batch['spectrogram'].unsqueeze(1))
    fe_emb4 = model.ae_model.encoder(demo_batch['spectrogram'].unsqueeze(1))

    # TODO use VAE to get synth parameters