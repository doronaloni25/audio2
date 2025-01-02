import wave

import librosa
from scipy.io import wavfile
from scipy.signal import resample, spectrogram, stft, istft
import os
import numpy as np
from librosa.feature import melspectrogram
import matplotlib.pyplot as plt


def main():
    # resample_data()
    create_mel()



def create_mel():
    sample_rate = 16000
    window_size = int(0.025 * sample_rate)
    hop_size = int(0.01 * sample_rate)
    input_folder = "audio_files/segmented_resampled"
    output_folder = "audio_files/mel/"
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        input_f = os.path.join(input_folder, file_name)
        audio, _ = librosa.load(input_f, sr=16000)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))
        mel_spectrogram = melspectrogram(y=audio, sr=sample_rate, n_fft=window_size, hop_length=hop_size, n_mels=80)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()

        # Save the figure to the output folder
        base_name = os.path.splitext(file_name)[0]
        output_path = output_folder + base_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()  # Close the figure to free memory


def resample_data():
    # A. resample data
    input_folder = "audio_files/segmented"
    output_folder = "audio_files/segmented_resampled"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Target sample rate
    target_rate = 16000

    # Iterate over each .wav file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Read the input wav file
            original_rate, data = wavfile.read(input_path)

            if original_rate != target_rate:
                data = downsample_resample_method(data, original_rate, target_rate)

                # Ensure data is in correct format (convert to int16 if necessary)
                if data.dtype == np.int16:
                    data = np.round(data).astype(np.int16)

                # Write the resampled audio to the output folder
                wavfile.write(output_path, target_rate, data)

            else:
                # If the sample rate matches, copy the file without modification
                wavfile.write(output_path, original_rate, data)


def downsample_resample_method(data, old_sample_rate, new_sample_rate):
    duration = len(data) / old_sample_rate
    new_num_samples = int(duration * new_sample_rate)
    return resample(data, new_num_samples)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
