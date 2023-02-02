import cv2
import numpy as np

def draw_output(layer, idx=0, img_path=None):
    '''
    draw the network layer output as image output
    '''

    arr = layer[idx, :, :, :].cpu().detach().numpy()
    if arr.ndim > 3:
        arr = arr.squeeze()
    arr = np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)
    if arr.shape[2] == 3:
        arr = arr[:, :, [2, 1, 0]]
    if img_path:
        cv2.imwrite(img_path, (arr * 255).astype(np.uint8))

