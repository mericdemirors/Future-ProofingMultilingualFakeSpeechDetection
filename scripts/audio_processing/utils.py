import cv2
import numpy as np
import librosa
from scipy.linalg import toeplitz

import GLOBAL_VARIABLES

def add_noise(data):
    if GLOBAL_VARIABLES.NOISE_SIGMA == 0:
        return data
    noise = np.random.normal(0, GLOBAL_VARIABLES.NOISE_SIGMA, data.shape)
    noisy_data = data + noise
    noisy_data = np.clip(noisy_data, -1.0, 1.0)
    return noisy_data

def get_cum3(data):
    cum3_dim = 2 * GLOBAL_VARIABLES.maxlag
    cum3 = np.zeros((cum3_dim, cum3_dim))

    ind = np.arange((GLOBAL_VARIABLES.SEGMENT_SIZE - GLOBAL_VARIABLES.maxlag), GLOBAL_VARIABLES.SEGMENT_SIZE)
    ind_t = np.arange(GLOBAL_VARIABLES.maxlag, GLOBAL_VARIABLES.SEGMENT_SIZE)
    zero_maxlag = np.zeros((1, GLOBAL_VARIABLES.maxlag))
    zero_maxlag_t = zero_maxlag.transpose()

    signal = np.reshape(data, (1, GLOBAL_VARIABLES.SEGMENT_SIZE))
    signal = signal - np.mean(data)
    sig = signal.transpose()

    rev_signal = np.array([signal[0][::-1]])
    col = np.concatenate((sig[ind], zero_maxlag_t), axis=0)
    row = np.concatenate((rev_signal[0][ind_t], zero_maxlag[0]), axis=0)

    toep = toeplitz(col, row)
    rev_signal = np.repeat(rev_signal, [2 * GLOBAL_VARIABLES.maxlag], axis=0)

    cum3 = cum3 + np.matmul(np.multiply(toep, rev_signal), toep.transpose())
    cum3 = cum3/GLOBAL_VARIABLES.SEGMENT_SIZE

    return cum3

def get_hamming_window():
    N = 2 * GLOBAL_VARIABLES.maxlag

    n = np.arange(N)
    window_even = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N - 1))

    # 2d even window
    window2d = np.array(
        [
            window_even,
        ]
        * N
    )

    ## One-sided window with zero padding
    window = np.zeros(N)
    window[: GLOBAL_VARIABLES.maxlag] = window_even[GLOBAL_VARIABLES.maxlag :]
    window[GLOBAL_VARIABLES.maxlag :] = 0

    # 2d window function to apply to bispectrum
    row = np.concatenate(([window[0]], np.zeros(2 * GLOBAL_VARIABLES.maxlag-1)))
    toep_matrix = toeplitz(window, row)
    toep_matrix += np.tril(toep_matrix, -1).transpose()
    window = toep_matrix[..., ::-1] * window2d * window2d.transpose()
    return window

window = None
def get_mag_and_phase(cum3):
    global window
    if window is None:
        window = get_hamming_window()
    bispec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cum3 * window)))

    mag = np.abs(bispec)
    phase = np.angle(bispec)

    return mag, phase

def get_features(audio_path, max_K=-1):
    segment_size=GLOBAL_VARIABLES.SEGMENT_SIZE
    overlap=GLOBAL_VARIABLES.SEGMENT_OVERLAP
    
    data, samplerate = librosa.load(audio_path, sr=16000)
    data = add_noise(data)
        
    num_segments = (len(data) - segment_size) // overlap + 1
    if max_K > 0:
        num_segments = min(num_segments, max_K)

    RC_layers = np.zeros((num_segments, segment_size, segment_size), dtype=complex)
    cum3_sum = np.zeros((segment_size, segment_size))

    for idx, segment_start in enumerate(range(0, len(data), overlap)):
        if idx == num_segments:
            break
        segment = data[segment_start:segment_start + segment_size]
        
        cum3 = get_cum3(segment)
        mag, phase = get_mag_and_phase(cum3)

        cum3_sum = cum3_sum + cum3

        R = mag * np.cos(phase)
        C = mag * np.sin(phase)
        RC_layers[idx] = R + C * 1j

    return RC_layers, cum3_sum/num_segments

def create_signature_image(RC_layers):
    RC_layers = RC_layers[..., np.newaxis]
    signature_image = np.zeros(RC_layers.shape[1:], dtype=complex)
    tops = np.sum(RC_layers, axis=0)

    signature_image = np.reshape(np.array([tops[r][c]/(np.sqrt(np.dot(RC_layers[:,r,c,:].T,np.conjugate(RC_layers[:,r,c,:])).real) + 0.0001) 
                                        for r in range(signature_image.shape[0]) 
                                        for c in range(signature_image.shape[1])]), signature_image.shape)

    # list comprehension is for this for loop
    # for r in range(signature_image.shape[0]):
    #     for c in range(signature_image.shape[1]):
    #         L = RC_layers[:,r,c,:]
    #         top = tops[r][c]
    #         bottom = np.sqrt(np.dot(L.T, np.conjugate(L)).real)
    #         signature_image[r,c] = top/(bottom + 0.0001)

    return signature_image

def save_images(signature_image, cum3_avg, absolute_path, angle_path, real_path, imag_path, cum3_path):
    absolute = np.absolute(signature_image)
    absolute_norm = (absolute - absolute.min()) / (absolute.max() - absolute.min())
    cv2.imwrite(absolute_path, (absolute_norm*255).astype(np.uint8))

    angle = np.angle(signature_image)
    angle_norm = (angle - angle.min()) / (angle.max() - angle.min())
    cv2.imwrite(angle_path, (angle_norm*255).astype(np.uint8))

    real = signature_image.real
    real_norm = (real - real.min()) / (real.max() - real.min())
    cv2.imwrite(real_path, (real_norm*255).astype(np.uint8))

    imag = signature_image.imag
    imag_norm = (imag - imag.min()) / (imag.max() - imag.min())
    cv2.imwrite(imag_path, (imag_norm*255).astype(np.uint8))

    cum3_norm = (cum3_avg - cum3_avg.min()) / (cum3_avg.max() - cum3_avg.min())
    cv2.imwrite(cum3_path, (cum3_norm*255).astype(np.uint8))
