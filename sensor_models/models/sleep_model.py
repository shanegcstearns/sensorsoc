import numpy as np

def extract_motion_features(signal):
    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "activity_level": np.sum(np.abs(signal))
    }
    return features

def extract_ppg_features(signal):
    # Simple heart-rate proxy
    peaks = np.where(signal > np.mean(signal) + np.std(signal))[0]
    heart_rate_est = len(peaks) / (len(signal) / 60 + 1e-8)
    
    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "estimated_hr": heart_rate_est
    }
    return features

def simple_sleep_classifier(motion_feat, ppg_feat):
    if motion_feat["activity_level"] < 50 and ppg_feat["estimated_hr"] < 80:
        return "Sleep"
    else:
        return "Awake"
