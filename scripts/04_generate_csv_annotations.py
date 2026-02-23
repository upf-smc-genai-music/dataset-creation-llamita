import re
from pathlib import Path
import soundfile as sf

FPS = 75

RAW_DIR = Path("data/raw/piano/fur_elise")
SYN_DIR = Path("data/synthetic/piano/fur_elise_aug")

# Fur_elise_10__aug1_ts0.954_ps-0.13.wav
RE_TS = re.compile(r"_ts([0-9]+\.[0-9]+)")
RE_PS = re.compile(r"_ps(-?[0-9]+\.[0-9]+)")

def parse_params(stem: str):
    ts_m = RE_TS.search(stem)
    ps_m = RE_PS.search(stem)
    ts = float(ts_m.group(1)) if ts_m else 1.0
    ps = float(ps_m.group(1)) if ps_m else 0.0
    return ts, ps

def write_csv(csv_path: Path, n_frames: int, ts: float, ps: float):
    # header exact match with parameters.json names
    header = "time_stretch,pitch_shift\n"
    line = f"{ts},{ps}\n"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for _ in range(n_frames):
            f.write(line)

def process_dir(d: Path):
    if not d.exists():
        return 0
    count = 0
    for wav in sorted(d.glob("*.wav")):
        info = sf.info(str(wav))
        dur = info.frames / info.samplerate if info.samplerate else 0.0
        n_frames = max(1, int(round(dur * FPS)))

        ts, ps = parse_params(wav.stem)
        csv_path = wav.with_suffix(".csv")
        write_csv(csv_path, n_frames, ts, ps)
        count += 1
    return count

def main():
    n1 = process_dir(RAW_DIR)
    n2 = process_dir(SYN_DIR)
    print("ok: csv created for", n1 + n2, "files")

if __name__ == "__main__":
    main()
