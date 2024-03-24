import os

# Function to count audio files in a directory
def count_audio_files(directory):
    audio_formats = ['.wav', '.mp3', '.aiff', '.flac', '.ogg', '.aac', '.wma', '.alac', '.opus', '.m4a', '.ape', '.au', '.midi', '.ac3', '.amr']
    audio_count = {format: 0 for format in audio_formats}
    total_files = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in audio_formats:
                audio_count[ext.lower()] += 1
                total_files += 1

    return audio_count, total_files

# Function to get the names of all folders in a directory
def get_folder_names(root_directory):
    folder_names = []
    for item in os.listdir(root_directory):
        item_path = os.path.join(root_directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

# Main function
if __name__ == "__main__":
    root_directory = "/content/drive/MyDrive/Project/audio_dataset/train"  # Change this to your parent directory path
    folder_names = get_folder_names(root_directory)
    total_audio_count = {format: 0 for format in ['.wav', '.mp3', '.aiff', '.flac', '.ogg', '.aac', '.wma', '.alac', '.opus', '.m4a', '.ape', '.au', '.midi', '.ac3', '.amr']}
    total_files = 0

    for folder_name in folder_names:
        folder_path = os.path.join(root_directory, folder_name)
        audio_count, folder_total_files = count_audio_files(folder_path)
        total_files += folder_total_files
        for format, count in audio_count.items():
            total_audio_count[format] += count

    print("Total File Counts:")
    for format, count in total_audio_count.items():
        print(f"Number of {format} files: {count}")
    print(f"Total number of audio files: {total_files}")