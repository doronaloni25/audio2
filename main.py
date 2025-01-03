import librosa
from scipy.io import wavfile
from scipy.signal import resample, spectrogram, stft, istft
import os
import numpy as np
from librosa.feature import melspectrogram
import matplotlib.pyplot as plt

# CONSTANTS FOR AUDIO FILE PATHS
representative_folder = "audio_files/groups_audio/representative"
training_folder = "audio_files/groups_audio/train"
evaluation_folder = "audio_files/groups_audio/evaluation"

def main():
    # resample_data()
    spectrogram_dict = create_mel()



def create_mel():
    sample_rate = 16000
    window_size = int(0.025 * sample_rate)
    hop_size = int(0.01 * sample_rate)
    input_folder = "audio_files/segmented_resampled"
    output_folder = "audio_files/mel/"
    os.makedirs(output_folder, exist_ok=True)
    spectrogram_dict = {}
    for file_name in os.listdir(input_folder):
        input_f = os.path.join(input_folder, file_name)
        audio, _ = librosa.load(input_f, sr=16000)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))
        mel_spectrogram = melspectrogram(y=audio, sr=sample_rate, n_fft=window_size, hop_length=hop_size, n_mels=80)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        spectrogram_dict[os.path.splitext(file_name)[0]] = mel_spectrogram_db
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
    return spectrogram_dict

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

def dtw(seq_a, seq_b):
    m, n = len(seq_a), len(seq_b)
    DTW = np.full((m, n), np.inf)

    DTW[0][0] = np.linalg.norm(seq_a[0], seq_b[0])
    for i in range(1, m):
        DTW[i][0] = DTW[i-1][0] + np.linalg.norm(seq_a[i] - seq_b[0])
    for j in range(1, n):
        DTW[0][j] = DTW[0][j-1] + np.linalg.norm(seq_a[0] - seq_b[j])


    for i in range(1, m):
        for j in range(1, n):
            DTW[i][j] = np.linalg.norm(seq_a[i], seq_b[j]) + np.minimum(
                DTW[i-1][j],    # Insertion
                DTW[i][j-1],    # Deletion
                DTW[i-1][j-1]   # Match
            )

    path = []
    i, j = m-1, n-1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if DTW[i-1][j] == np.minimum(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1]):
                i -= 1
            elif DTW[i][j-1] == np.minimum(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
    path.append((0, 0))
    path.reverse()

    return DTW[m-1][n-1], path


def compare_audio_recording(spectrogram_dict):
    representative_spectrograms = []
    training_spectrograms = []
    for file_name in os.listdir(representative_folder):
        representative_spectrograms.append(spectrogram_dict[os.path.splitext(file_name)[0]])
    for file_name in os.listdir(training_folder):
        training_spectrograms.append(spectrogram_dict[os.path.splitext(file_name)[0]])
    representative_spectrograms = np.array(representative_spectrograms)
    training_spectrograms = np.array(training_spectrograms).reshape((4, 10))
    dist_matrix = np.zeros((4, 10))
    for i in range(10):
        for j in range(4):
            dist_matrix[j][i], _ = dtw(representative_spectrograms[i], training_spectrograms[j][i])
    return dist_matrix

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
