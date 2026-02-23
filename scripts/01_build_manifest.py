from pathlib import Path
import pandas as pd
import soundfile as sf

AUDIO_DIR = Path("data/raw/piano/fur_elise")
OUT_CSV = Path("data/processed/manifest.csv")

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(AUDIO_DIR.glob("*.wav")):
        info = sf.info(str(p))
        rows.append({
            "path": str(p),
            "filename": p.name,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_sec": info.frames / info.samplerate if info.samplerate else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"ok: {OUT_CSV} ({len(df)} files)")

if __name__ == "__main__":
    main()
