import os

import mido


if __name__ == '__main__':
    patch_path = os.path.join('..', '..', 'DX7_AllTheWeb', 'Aminet', '2.syx')
    patch = mido.read_syx_file(patch_path)
    print(patch[0].hex())