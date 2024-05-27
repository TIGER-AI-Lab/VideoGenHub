import argparse
import os
from moviepy.editor import VideoFileClip

def reprocess_video(input_path, output_path):
    # Load the video file
    clip = VideoFileClip(input_path)
    
    # Write the clip to a new file with the desired encoding.
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

def find_and_replace_videos(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                print(f"Processing {full_path}...")
                
                # Define the output path, could overwrite or create a new file
                output_path = full_path  # This will overwrite the original file
                
                # To prevent overwriting, uncomment the following line and comment out the above line
                # output_path = os.path.splitext(full_path)[0] + "_corrected.mp4"
                reprocess_video(full_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reprocess MP4 files in a folder with correct encoding.")
    parser.add_argument("directory", help="The directory to search for MP4 files")

    args = parser.parse_args()

    find_and_replace_videos(args.directory)