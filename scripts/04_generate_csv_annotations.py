import re
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

FPS = 75

RAW_DIR = Path("data/raw/piano/fur_elise")
SYN_DIR = Path("data/synthetic/piano/fur_elise_aug")

RE_TS = re.compile(r"_ts([0-9]+\.[0-9]+)")
RE_PS = re.compile(r"_ps(-?[0-9]+\.[0-9]+)")

HEADER = "time_stretch,pitch_shift,loudness_rms,spectral_centroid,source\n"

def parse_params(stem: str):
    ts_m = RE_TS.search(stem)
    ps_m = RE_PS.search(stem)
    ts = float(ts_m.group(1)) if ts_m else 1.0
    ps = float(ps_m.group(1)) if ps_m else 0.0
    return ts, ps

def compute_frame_features(wav_path: Path):
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    hop_length = max(1, int(round(sr / FPS)))

    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=hop_length,
        center=True,
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=hop_length,
        center=True,
    )[0]

    n = int(min(len(rms), len(centroid)))
    return rms[:n].astype(float), centroid[:n].astype(float)

def write_csv(csv_path: Path, ts: float, ps: float, rms: np.ndarray, centroid: np.ndarray, source_id: int):
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(HEADER)
        for i in range(len(rms)):
            f.write(f"{ts},{ps},{rms[i]},{centroid[i]},{source_id}\n")

def process_dir(d: Path, source_id: int):
    if not d.exists():
        return 0

    count = 0
    for wav in sorted(d.glob("*.wav")):
        if not wav.exists():
            continue

        _ = sf.info(str(wav))

        ts, ps = parse_params(wav.stem)
        rms, centroid = compute_frame_features(wav)

        csv_path = wav.with_suffix(".csv")
        write_csv(csv_path, ts, ps, rms, centroid, source_id)
        count += 1

    return count

def main():
    n_raw = process_dir(RAW_DIR, source_id=0)
    n_syn = process_dir(SYN_DIR, source_id=1)
    print("ok: csv created for", n_raw + n_syn, "files")

if __name__ == "__main__":
    main()
