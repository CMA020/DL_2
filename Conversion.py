import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import gc
# Function to generate spectrogram
def generate_spectrogram(audio_path, output_dir):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Create the spectrogram
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save the spectrogram as an image
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    plt.savefig(os.path.join(output_dir, f'{filename}_spectrogram.png'))
    plt.close()
    gc.collect()



if __name__ == '__main__':
    parent_dir = os.path.expanduser("~/DL_1/audio_dataset/val")
    os.makedirs("~/DL_1/Test2")
    output_dir= os.path.expanduser("~/DL_1/Test2")

    # Iterate over directories inside the parent directory
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        os.makedirs(os.path.join(output_dir, folder_name))
        if os.path.isdir(folder_path):
            print(f"Folder Name: {folder_name}")

            for audio_file in os.listdir(folder_path):
                if audio_file.endswith('.wav') or audio_file.endswith('.mp3') or audio_file.endswith(
                        '.aiff') or audio_file.endswith('.flac') or audio_file.endswith('.ogg') or audio_file.endswith(
                        '.m4a'):
                    audio_path = os.path.join(folder_path, audio_file)
                    generate_spectrogram(audio_path, os.path.join(os.path.expanduser('~/DL_1/Test2'), folder_name))


