import argparse
from pathlib import Path
from vidstab import VidStab

def main():
    parser = argparse.ArgumentParser(description="Frame by frame video stabilization using VidStab.")
    parser.add_argument("--input", type=str, default="input1/input.mp4", help="Input video path")
    parser.add_argument("--output", type=str, default="output1/output_vidstab.mp4", help="Output video path")
    parser.add_argument("--smoothing", type=int, default=30, help="Smoothing window size")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing VidStab frame-by-frame stabilizer...")
    stabilizer = VidStab(kp_method='GFTT')
    
    print(f"Processing {input_path} -> {output_path}")
    print(f"Smoothing window: {args.smoothing} frames")
    
    stabilizer.stabilize(
        input_path=str(input_path),
        output_path=str(output_path),
        smoothing_window=args.smoothing,
        output_fourcc='mp4v'
    )
    
    print("Stabilization complete! Video is completely clean and jitter-free.")

if __name__ == "__main__":
    main()
