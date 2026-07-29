"""Demo: load Emilia-Dataset samples and run ASR with pipeline or direct backend."""

import argparse
from pathlib import Path

from whisper_asr.config import load_config
from whisper_asr.load_data import EMILIA_LANGUAGES, load_emilia_samples
from whisper_asr.transcriber import Transcriber

project_root = Path(__file__).resolve().parent.parent


def main():
    config = load_config(project_root / "configs" / "default.yml")

    parser = argparse.ArgumentParser(description="Transcribe Emilia-Dataset samples with Whisper")
    parser.add_argument(
        "--backend",
        choices=["pipeline", "direct"],
        default="direct",
        help="Transcription backend (default: pipeline)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to process (default: 3)",
    )
    parser.add_argument(
        "--language",
        choices=list(EMILIA_LANGUAGES.keys()),
        default="en",
        help="Filter by language (EN, ZH, DE, FR, JA, KO). Omit for full dataset.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Full path subset, e.g. Emilia/EN or Emilia-YODAS/DE. Overrides --language if set.",
    )
    parser.add_argument(
        "--no-save-audio",
        action="store_true",
        help="Do not save mp3 files to sample_audios folder.",
    )
    args = parser.parse_args()

    # Streaming is driven by the config; defaults to False if absent.
    args.stream = config.get("enable_streaming", False)

    # Create folder for saving mp3s
    save_dir = project_root / config["sample_audios_folder"]
    if not args.no_save_audio:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving mp3 files to: {save_dir}")

    # Override backend from CLI arg and build the Transcriber once
    # Streaming requires the direct backend (model.generate with streamer)
    if args.stream:
        config["default_backend"] = "direct"
    else:
        config["default_backend"] = args.backend
    transcriber = Transcriber(config)

    mode = "streaming" if args.stream else args.backend
    print(f"Using mode: {mode}")
    print(f"Processing {args.num_samples} samples...\n")

    samples = load_emilia_samples(args.language, args.num_samples, subset=args.subset)

    for i, s in enumerate(samples):
        if not args.no_save_audio:
            mp3_path = save_dir / f"sample_{i}.mp3"
            mp3_path.write_bytes(s["mp3_bytes"])
            print(f"Saved: {mp3_path}")

        try:
            ground_truth = s["ground_truth"] or "N/A"
            print(f"--- Sample {i} ---")
            print(f"Ground truth: {ground_truth}")

            if args.stream:
                print("Transcribed:  ", end="", flush=True)
                for token in transcriber.transcribe_stream(s["audio"], s["sr"]):
                    print(token, end="", flush=True)
                print()
            else:
                text = transcriber.transcribe(s["audio"], s["sr"])
                print(f"Transcribed:  {text}")

            print()
        except Exception as e:
            print(f"Sample {i}: Error - {e}\n")

    print("Done.")


if __name__ == "__main__":
    main()
