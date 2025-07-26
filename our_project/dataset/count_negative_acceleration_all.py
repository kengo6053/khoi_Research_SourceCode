import os
import json
import argparse

def count_accelerating_segments_non_overlapping(dataset_path, output_folder, threshold=1.0, frame_window=10):
    # Initialize the count and list to store results
    accelerating_segments = []

    # Iterate over subdirectories and JSON files
    for root, _, files in os.walk(dataset_path):
        json_files = sorted([f for f in files if f.endswith('.json')])
        
        # Process each set of files in non-overlapping windows
        accelerations = []
        start_frame = None
        for i, file in enumerate(json_files):
            with open(os.path.join(root, file), 'r') as f:
                data = json.load(f)
                accelerations.append(data.get("acceleration", 0.0))
            
            # Record the start frame for this window
            if start_frame is None:
                start_frame = i

            # When we have enough frames for a window
            if len(accelerations) == frame_window:
                # Calculate the sum of accelerations
                acceleration_sum = sum(accelerations)
                if acceleration_sum <= threshold:
                    # Store the start and end frame numbers
                    accelerating_segments.append((start_frame, start_frame + frame_window - 1))
                    # Clear the window to ensure non-overlapping count
                    accelerations = []
                    start_frame = None
                else:
                    # Slide the window completely if no acceleration threshold met
                    accelerations.pop(0)
                    start_frame += 1

    # Save results to a text file in the parent folder
    output_file = os.path.join(output_folder, "accelerating_segments_negative.txt")
    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, "w") as f:
        # Write the total count
        f.write(f"Number of non-overlapping accelerating segments: {len(accelerating_segments)}\n\n")
        # Write details of each segment
        for segment in accelerating_segments:
            f.write(f"Start Frame: {segment[0]}, End Frame: {segment[1]}\n")
    
    return len(accelerating_segments), output_file


if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Count non-overlapping accelerating segments in dataset for multiple parent folders.")
    parser.add_argument("--root_folder", type=str, help="Path to the root folder containing parent folders")
    parser.add_argument("--output_folder", type=str, help="Path to the folder where the summary results will be saved")
    parser.add_argument("--threshold", type=float, default=-1.0, help="Acceleration sum threshold (default: 1.0 m/s²)")
    parser.add_argument("--frame_window", type=int, default=10, help="Number of frames in the window (default: 10)")

    args = parser.parse_args()

    # Get all parent folders in the root folder
    parent_folders = [
        os.path.join(args.root_folder, folder) for folder in os.listdir(args.root_folder)
        if os.path.isdir(os.path.join(args.root_folder, folder))
    ]

    # Initialize summary results and total count
    summary_results = []
    total_accelerating_segments = 0

    # Process each parent folder
    for parent_folder in parent_folders:
        dataset_path = os.path.join(parent_folder, "measurements")
        if not os.path.exists(dataset_path):
            print(f"Skipping {parent_folder}: 'measurements' folder does not exist.")
            continue

        # Define the output folder for the current parent folder
        parent_output_folder = parent_folder  # Save results directly in the parent folder

        # Count non-overlapping accelerating segments and save results
        accelerating_count, output_file = count_accelerating_segments_non_overlapping(
            dataset_path, parent_output_folder, args.threshold, args.frame_window
        )
        total_accelerating_segments += accelerating_count
        summary_results.append(f"Folder: {os.path.basename(parent_folder)}, "
                               f"Number of non-overlapping accelerating segments: {accelerating_count}, "
                               f"Results saved to: {output_file}")
        print(f"Processed {parent_folder}: Number of non-overlapping accelerating segments: {accelerating_count}")
        print(f"Results saved to: {output_file}")

    # Save summary results to a single file in the root folder
    summary_file = os.path.join(args.output_folder, "summary_results_negative.txt")
    with open(summary_file, "w") as f:
        f.write("Summary of all results:\n\n")
        for line in summary_results:
            f.write(line + "\n")
        f.write(f"\nTotal Number of Non-Overlapping Accelerating Segments: {total_accelerating_segments}\n")

    print(f"Summary of results saved to: {summary_file}")
    print(f"Total Number of Non-Overlapping Accelerating Segments: {total_accelerating_segments}")
