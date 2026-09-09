import argparse
import json
import os
import soundfile as sf
import tqdm

parser = argparse.ArgumentParser(description="Build a JSON audio manifest.")
parser.add_argument("input_txt", help="Text file containing one WAV path per line")
parser.add_argument("output_json", help="Output JSON manifest path")
args = parser.parse_args()

manifest = []

with open(args.input_txt, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines, desc="Processing WAV files", total=len(lines)):
        wav_path = line.strip()

        if not wav_path or not os.path.exists(wav_path):
            continue

        info = sf.info(wav_path)
        duration = info.frames / info.samplerate

        manifest.append(
            {
                "audio_filepath": wav_path,
                "duration": round(duration, 2),
            }
        )

with open(args.output_json, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=4)

print(f"Saved {len(manifest)} entries to {args.output_json}")
