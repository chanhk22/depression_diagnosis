import numpy as np, webrtcvad
import soundfile as sf

def speech_flags_from_wav(wav_path, aggressiveness=2, frame_ms=30):
    y, sr = sf.read(wav_path)
    if y.ndim > 1: y = y.mean(axis=1)
    i16 = (y * 32767).astype('int16')
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_ms / 1000)
    n = (len(i16) // frame_len) * frame_len
    flags = []
    for i in range(0, n, frame_len):
        frame = i16[i:i+frame_len].tobytes()
        flags.append(1 if vad.is_speech(frame, sr) else 0)
    return np.array(flags, dtype=np.uint8)
