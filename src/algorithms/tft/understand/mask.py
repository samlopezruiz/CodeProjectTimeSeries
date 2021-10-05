import numpy as np

from algorithms.tft.libs.tft_model import get_decoder_mask

if __name__ == '__main__':
    #%%
    a = get_decoder_mask(np.zeros((3, 2))).numpy()