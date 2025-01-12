import librosa
from scipy.io import wavfile
from scipy.signal import resample, spectrogram, stft, istft
import os
import numpy as np
from librosa.feature import melspectrogram
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# CONSTANTS FOR AUDIO FILE PATHS
representative_folder = "audio_files/groups_audio/representative"
training_folder = "audio_files/groups_audio/train"
evaluation_folder = "audio_files/groups_audio/evaluation"

def main():
    # resample_data()
    # 3h, change to True to improve results
    normalize = False
    spectrogram_dict = create_mel(normalize)
    dtw_mat = compare_audio_recording(spectrogram_dict, normalize)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print(dtw_mat)
    # we want to classify every training audio correctly - therefore we want each sample to have its dtw dist <= threshold
    # a simple yet elegant solution would be to set the threshold as the max of the DTW matrix.
    threshold = np.max(dtw_mat)
    representative_spectrograms, training_spectrograms, eval_spectrograms = spectrogram_dict_to_array(spectrogram_dict)
    # 3f
    accuracy, _ = determine_classifications_and_accuracy(training_spectrograms, representative_spectrograms, threshold, normalize)
    print(f"Training accuracy = {accuracy}") #0.2
    # 3g
    accuracy, classifications = determine_classifications_and_accuracy(eval_spectrograms, representative_spectrograms, threshold, normalize)
    print(f"Eval accuracy = {accuracy}") #0.275
    confusion_matrix = calc_confusion_matrix(classifications)
    # 3h. display confusion matrix
    labels = np.arange(0, 10)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


def calc_confusion_matrix(classifications):
    confusion_matrix = np.zeros((10, 10))
    for i in range(len(classifications)):
        for j in range(10):
            if classifications[i, j] != -1:
                confusion_matrix[j, classifications[i, j]] += 1
    return confusion_matrix



def determine_classifications_and_accuracy(samples, representative_samples, threshold, normalize=False):
    # samples = spectrograms of 0 to 9 of speakers (multiple), representative_samples = spectrograms of 0 to 9 of representative
    true_labels = np.arange(10)
    correct_classifications = 0
    classifications = np.zeros((len(samples), 10))
    for speaker_idx in range(len(samples)):
        classifications[speaker_idx] = determine_classification_single_speaker(samples[speaker_idx], representative_samples, threshold, normalize)
        correct_classifications += np.sum(classifications[speaker_idx] == true_labels)
    return correct_classifications / (len(samples)*10), classifications.astype(int)


def determine_classification_single_speaker(samples, representative_samples, threshold, normalize=False):
    # samples = spectrograms of 0 to 9 of speaker, representative_samples = spectrograms of 0 to 9 of representative
    dtw_scores = np.zeros((len(samples), len(representative_samples)))
    for i in range(len(samples)):
        for j in range(len(representative_samples)):
            dtw_scores[i][j], _ = dtw(samples[i], representative_samples[j], normalize)
    min_scores = np.min(dtw_scores, axis=1)
    classifications = np.argmin(dtw_scores, axis=1)
    classifications[min_scores > threshold] = -1
    return classifications


def calculate_accuracy(predictions, true_labels):
    correct = np.sum(predictions == true_labels)
    return (correct / len(true_labels)) * 100


def create_mel(normalize = False):
    sample_rate = 16000
    window_size = int(0.025 * sample_rate)
    hop_size = int(0.01 * sample_rate)
    input_folder = "audio_files/segmented_resampled"
    output_folder = "audio_files/mel/"
    os.makedirs(output_folder, exist_ok=True)
    spectrogram_dict = {}
    for file_name in os.listdir(input_folder):
        input_f = os.path.join(input_folder, file_name)
        audio, _ = librosa.load(input_f, sr=sample_rate)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))
        if normalize:
            audio = auto_gain_control(audio, sample_rate)
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

# AGC from ex1
def auto_gain_control(audio, sample_rate):
    rms_values = []
    noise_floor_db = -20
    desired_rms_db = -10
    window_size = int(0.02 * sample_rate)  # 20ms
    hop_size = int(0.02 * sample_rate)  # 10ms

    for start in range(0, len(audio) - window_size + 1, hop_size):
        frame = audio[start:start + window_size]
        rms_values.append(np.sqrt(np.mean(frame**2)))

    rms_values = np.array(rms_values)
    rms_values_db = 20 * np.log10(rms_values + 1e-12)
    low_rms_indices = rms_values_db < noise_floor_db
    high_rms_indices = rms_values_db >= noise_floor_db
    smooth_rms_db = np.zeros(len(rms_values_db))
    gains_db = np.zeros(len(rms_values_db))
    for i in range(len(rms_values)):
        start_idx = max(0, i - (sample_rate // 2))
        end_idx = min(len(rms_values), i + (sample_rate // 2))
        smooth_rms_db[i] = np.mean(rms_values[start_idx:end_idx])
    gains_db[low_rms_indices] = 0
    gains_db[high_rms_indices] = desired_rms_db - smooth_rms_db[high_rms_indices]
    gains = 10 ** (gains_db / 20.0)

    gain_interpolated = np.interp(np.arange(len(audio)), np.arange(0, len(audio)-window_size+1, hop_size), gains)
    return audio * gain_interpolated


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


def dtw(spectrogram_a, spectrogram_b, normalize=False):
    m, n = np.shape(spectrogram_a)[1], np.shape(spectrogram_b)[1]
    DTW = np.full((m, n), np.inf)

    DTW[0][0] = np.linalg.norm(spectrogram_a[:,0] - spectrogram_b[:,0])
    for i in range(1, m):
        DTW[i][0] = DTW[i-1][0] + np.linalg.norm(spectrogram_a[:,i] - spectrogram_b[:,0])
    for j in range(1, n):
        DTW[0][j] = DTW[0][j-1] + np.linalg.norm(spectrogram_a[:,0] - spectrogram_b[:,j])


    for i in range(1, m):
        for j in range(1, n):
            DTW[i][j] = np.linalg.norm(spectrogram_a[:,i] - spectrogram_b[:,j]) + min(
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
            if DTW[i-1][j] == min(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1]):
                i -= 1
            elif DTW[i][j-1] == min(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
    path.append((0, 0))
    path.reverse()
    if normalize:
        DTW[m-1][n-1] /= max(m, n)
    return DTW[m-1][n-1], path


def compare_audio_recording(spectrogram_dict, normalize=False):
    representative_spectrograms = []
    training_spectrograms = []
    for file_name in os.listdir(representative_folder):
        representative_spectrograms.append(spectrogram_dict[os.path.splitext(file_name)[0]])
    for file_name in os.listdir(training_folder):
        training_spectrograms.append(spectrogram_dict[os.path.splitext(file_name)[0]])
    dist_matrix = np.zeros((4, 10))
    for i in range(10):
        for j in range(4):
            dist_matrix[j][i], _ = dtw(representative_spectrograms[i], training_spectrograms[j*10+i], normalize)
            print(f"done with i = {i}, j = {j}")
    return dist_matrix


def spectrogram_dict_to_array(spectrogram_dict):
    representative_spectrograms = []
    training_spectrograms = []
    eval_spectrograms = []

    for file_name in os.listdir(representative_folder):
        representative_spectrograms.append(spectrogram_dict[os.path.splitext(file_name)[0]])
    for file_name in os.listdir(training_folder):
        training_spectrograms.append(spectrogram_dict[os.path.splitext(file_name)[0]])
    for file_name in os.listdir(evaluation_folder):
        eval_spectrograms.append(spectrogram_dict[os.path.splitext(file_name)[0]])
    training_spectrograms_f = [training_spectrograms[i : i+10] for i in range(0, 40, 10)]
    eval_spectrograms_f = [eval_spectrograms[i: i + 10] for i in range(0, 40, 10)]
    return representative_spectrograms, training_spectrograms_f, eval_spectrograms_f


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
