import scipy.signal as signal
import numpy as np

def find_phase(pred, delta=1.0, height=0.3, dist=1):
    pred=pred*2
    shape=np.shape(pred)
    all_phase = []
    for itr in range(shape[0]):
        phase =[]
        p=pred[itr, 1, :]
        h=height
        peaks, values = signal.find_peaks(p, height=h, distance=100)
        for itr_p in peaks:
            phase.append([1, itr_p, pred[itr, 1, itr_p]])
        all_phase.append(phase)
    return all_phase
        
        
    