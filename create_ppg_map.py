import os
import re
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import welch


# Frames zu einzelnen Signalabschnitten zusammenfügen der Länge window * fps. Koordinaten beibehalten
def connect_frames(window, fps, path):
    num_files = len(os.listdir(path + "/csv"))  # Amount of csv files in directory
    first_iteration = True
    window_size = int(window * fps)

    for file in range(3301, 5400 + 1):  # num_files statt 300
        frame_df = pd.read_csv(path + f"/csv/rgb_pixel_values_frame{file}.csv")

        if first_iteration or (file % window_size) == 1:
            first_iteration = False
            r_signal_df = pd.DataFrame(columns=['x', 'y'] + [f'f{i}' for i in range(1,
                                                                                    window_size + 1)])  # Empty DataFrame with columns for every frame that is filled with signals
            g_signal_df = pd.DataFrame(columns=['x', 'y'] + [f'f{i}' for i in range(1, window_size + 1)])
            b_signal_df = pd.DataFrame(columns=['x', 'y'] + [f'f{i}' for i in range(1, window_size + 1)])
            r_signal_df[['x', 'y']] = frame_df[['x', 'y']]
            g_signal_df[['x', 'y']] = frame_df[['x', 'y']]
            b_signal_df[['x', 'y']] = frame_df[['x', 'y']]
            r_temp_columns = []  # Temporäre Liste für neue Spalten
            g_temp_columns = []
            b_temp_columns = []

        # Füllen der temporären Listen statt direktes Hinzufügen zum DataFrame
        r_temp_columns.append(frame_df.iloc[:, 2])
        g_temp_columns.append(frame_df.iloc[:, 3])
        b_temp_columns.append(frame_df.iloc[:, 4])

        if file % window_size == 0:
            # Neue Spalten in einem Schritt hinzufügen
            r_signal_df.iloc[:, 2:] = pd.concat(r_temp_columns, axis=1).values
            g_signal_df.iloc[:, 2:] = pd.concat(g_temp_columns, axis=1).values
            b_signal_df.iloc[:, 2:] = pd.concat(b_temp_columns, axis=1).values

            # Speichern des DataFrames als CSV-Datei
            r_signal_df.to_csv(path + f'/csv/signal_r/r_values_frame_{file - window_size + 1}_to_{file}.csv',
                               index=False)
            g_signal_df.to_csv(path + f'/csv/signal_g/g_values_frame_{file - window_size + 1}_to_{file}.csv',
                               index=False)
            b_signal_df.to_csv(path + f'/csv/signal_b/b_values_frame_{file - window_size + 1}_to_{file}.csv',
                               index=False)

            print(f'Frames {file - window_size + 1} bis {file} als csv gespeichert.')

            r_temp_columns = []  # Leeren der temporären Listen für die nächste Iteration
            g_temp_columns = []
            b_temp_columns = []


# Signale der drei Kanäle kombinieren und zu einem zusammenfassen
def combine_channels(path):
    return 1


# 5th Order Bandpass Butterworth-Filter mit lowcut und highcut Frequenz
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y


def fft(ppg_signal, fps):
    # FFT berechnen
    fft_result = np.fft.fft(ppg_signal)
    fft_magnitude = np.abs(fft_result)  # Betrag der FFT-Werte
    fft_freq = np.fft.fftfreq(len(fft_result), 1 / fps)  # Frequenzwerte

    # Nur die positiven Frequenzen betrachten
    positive_freqs = fft_freq[:len(fft_freq) // 2]
    positive_magnitude = fft_magnitude[:len(fft_magnitude) // 2]

    return positive_freqs, positive_magnitude


# Korrektes Sortieren der Dateien in Verzeichnis
def extract_number(filename):
    # Suche nach den Zahlen in den Dateinamen
    match = re.search(r'(\d+)_to_(\d+)', filename)
    if match:
        # Rückgabe der Zahlen als Integer-Tuple
        return int(match.group(1))
    return 0


# Einfache SNR Berechnung: Alle Frequenzen zwischen Low und High ist Signal, Rest ist Rauschen
def calc_snr(signal_array, fps, low, high):  # De Haan Methode
    # Leistungsspektraldichte (PSD) des Signals berechnen
    f, pxx = welch(signal_array, fps, nperseg=len(signal_array))

    # Spektrale Leistungsdichte im Herzfrequenzbereich extrahieren
    signal_band = np.logical_and(f >= low, f <= high)

    # Berechnen der Signalenergie (im relevanten Frequenzbereich)
    signal_power = np.sum(pxx[signal_band])

    # Rauschenergie berechnen (außerhalb des relevanten Frequenzbereichs)
    noise_band = np.logical_not(signal_band)
    noise_power = np.sum(pxx[noise_band])

    # Schritt 5: Berechnung des SNR
    if signal_power != 0 and noise_power != 0:
        snr = signal_power / (noise_power)
        snr_db = 10 * np.log10(snr)

        return snr_db


def calc_snr_windowed(signal_array, ref_hand_signal, fps):  # De Haan Methode
    freq_bandwidth = 0.2  # Fensterbreite in Hz um die Herzfrequenz bzw. die Harmonische
    min_freq = 0.9  # Frequenz in Hz, ab der Maximum gesucht wird
    max_freq = 1.6

    # Referenzsignal (kontaktlos) der Referenzhand zur Bestimmung der Herzfrequenz
    ref_hand_signal = bandpass_filter(ref_hand_signal, 0.8, 2, 30)

    # Filterung des Signalabschnitts vor SNR Berechnung
    signal_array = bandpass_filter(signal_array, 0.5, 4, 30)

    # Leistungsspektraldichte (PSD) des Signals berechnen
    f, pxx = welch(signal_array, fps, nperseg=len(signal_array))
    f_ref, pxx_ref = welch(ref_hand_signal, fps, nperseg=len(ref_hand_signal))

    # Filtere Frequenzen unterhalb des Schwellwerts aus
    valid_freqs = (f_ref >= min_freq) & (f_ref <= max_freq)
    f_valid = f_ref[valid_freqs]
    pxx_valid = pxx_ref[valid_freqs]

    # Bestimme die Frequenz mit der höchsten Leistung im gültigen Bereich
    heart_rate_freq = f_valid[np.argmax(pxx_valid)]

    # Definiere Frequenzbereiche um die Herzfrequenz und ihre Harmonischen
    harmonics = [round(heart_rate_freq, 1), round(2 * heart_rate_freq, 1), round(3 * heart_rate_freq, 1)]

    signal_power = 0
    pre_noise_bands = 0  # np.logical_and(f >= (harmonics[0] - freq_bandwidth / 2), f <= (harmonics[0] + freq_bandwidth / 2))

    # Summiere die PSD-Werte innerhalb der Frequenzbereiche der Signale
    for harmonic in harmonics:
        signal_band = np.logical_and(f >= (harmonic - freq_bandwidth / 2), f <= (harmonic + freq_bandwidth / 2))
        pre_noise_bands += signal_band
        signal_power += np.sum(pxx[signal_band])

    # Berechne die Rauschenergie als PSD-Werte außerhalb der Signalbereiche
    noise_band = np.logical_not(pre_noise_bands)
    noise_band[:np.where(noise_band == False)[0][0]] = True  # Alles bis zum ersten HR Peak zählt als Noise
    noise_band[np.where(noise_band == False)[0][-1] + 1:] = True
    # noise_band[np.where(noise_band == False)[0][-1]:np.argmin(np.abs(f - 5))] = True  # Alles zwischen dem letzten HR Peak und 5 Hz zählt als Noise
    # noise_band[np.argmin(np.abs(f - 5)):] = False  # Alles ab 5 Hz zählt nicht als Noise
    noise_power = np.sum(pxx[noise_band])

    # Schritt 5: Berechnung des SNR
    if signal_power != 0 and noise_power != 0:
        snr = signal_power / (noise_power)
        snr_db = 10 * np.log10(snr)

        return snr_db
    return np.nan


# Funktion, die Signale einzelner Pixel mit einem Referenz-PPG korreliert
def calc_correlation(signal_array, ref_hand_signal):
    ref_hand_signal = bandpass_filter(ref_hand_signal, 0.5, 4, 30)
    signal_array = bandpass_filter(signal_array, 0.5, 4, 30)

    corr_coefficient = np.corrcoef(ref_hand_signal, signal_array)[0, 1]

    return corr_coefficient


def calc_feature(signal, ref_hand_signal, fps, feature):
    if feature == 'SNR':
        signal_array = signal.to_numpy()
        snr_db = calc_snr_windowed(signal_array, ref_hand_signal, fps=30)
        # snr_db = calc_snr(signal_array, low=0.8, high=5, fps=30)
        return snr_db
    elif feature == 'Corr':
        signal_array = signal.to_numpy()
        corr = calc_correlation(signal_array, ref_hand_signal)
        return corr
    elif feature == 'HR':
        pass
    return 1


# Feature Matrix erstellen
def feature_matrix(feature, channel, path, window_length, fps):
    if channel != 'all':
        files = sorted(os.listdir(path + "/csv/signal_" + channel), key=extract_number)
        for file in files:
            df_feature = pd.DataFrame(columns=['x', 'y', 'Signal'])  # + [f'S{i}' for i in range(1, len(files) + 1)])
            signal_df = pd.read_csv(path + "/csv/signal_" + channel + "/" + file)
            signal_df_left = signal_df[signal_df.iloc[:, 1] < 75].iloc[:, 2:].mean()
            df_feature[['x', 'y']] = signal_df[['x', 'y']]
            for row in range(signal_df.shape[0]):
                df_feature.iloc[row, 2] = calc_feature(signal_df.iloc[row, 2:], signal_df_left, fps, feature)
            pd.set_option('future.no_silent_downcasting', True)
            df_feature = df_feature.apply(lambda col: col.fillna(col.min()),
                                          axis=0)  # Ersetze NaN durch Minimum der Spalte
            df_feature.to_csv(path + f'/csv/features/feature_{feature}_' + file + '.csv', index=False)
            print('Feature Calculation for Signal Window Completed')

            # calc feature for every line (pixel) and save feature in df_feature cell
    signal_rgb_df = combine_channels(path=path)


# Color coded Map anfertigen basierend auf Werterbereich der Features
def create_heatmap(feature, channel, path, window_length, fps, downscaling_factor):
    feature_files = sorted(
    [file for file in os.listdir(path + "/csv/features") if file.startswith("feature_" + feature + "_" + channel)],
    key=extract_number
)
    signal_files = sorted(os.listdir(path + "/csv/signal_" + channel + "/"), key=extract_number)
    all_features = []
    frame = 1

    for file in feature_files:
        # Lesen der Features
        df_feature = pd.read_csv(path + "/csv/features/" + file)
        all_features.append(df_feature)

    combined_features = pd.concat([df['Signal'] for df in all_features], axis=0)
    all_features.clear()

    for file in signal_files:
        # Lesen der Features und zugehörigen Signale
        df_feature = pd.read_csv(path + "/csv/features/feature_" + feature + "_" + file + ".csv")
        # Zugehöriges Signal zu Features, gefiltert
        df_signal = pd.read_csv(path + "/csv/signal_" + channel + "/" + file)
        df_signal_right = df_signal[df_signal.iloc[:, 1] > 85]
        df_signal_left = df_signal[df_signal.iloc[:, 1] < 75]
        signal_right = bandpass_filter(df_signal_right.iloc[:, 2:].mean(), 0.5, 4, fps)
        signal_left = bandpass_filter(df_signal_left.iloc[:, 2:].mean(), 0.5, 4, fps)

        # Nach Frame Anzahl im Dateiname suchen, um Bild zuzuordnen
        match = re.search(r"frame_(\d+)_to_(\d+)", file)
        start_frame = int(match.group(1))

        # Bild laden
        image_path = path + f'/image{start_frame}.bmp'
        img = Image.open(image_path)
        img = img.resize((img.size[0] // downscaling_factor, img.size[1] // downscaling_factor))
        img = img.convert("RGB")
        pixels = img.load()

        # Normalisierung der Signalwerte (0, 1), Dabei die oberen und unteren 5% auf 1 bzw. 0 normalisieren -> bessere farbliche Auflösung im mittleren Wertebereich
        norm = Normalize(vmin=combined_features.quantile(0.05), vmax=combined_features.quantile(0.97))

        # Erstelle eine Farbskala (coolwarm, jet, seismic)
        cmap = plt.get_cmap('jet')
        sm = ScalarMappable(norm=norm, cmap=cmap)

        # Markiere die Pixel im Bild
        for _, row in df_feature.iterrows():
            x = row['x']
            y = row['y']
            signal_value = row['Signal']

            # Farbe basierend auf dem signal_value
            color = sm.to_rgba(signal_value)[:3]  # Farbe im RGB-Format

            # Setze den Pixel im Bild
            pixels[x, y] = tuple(int(255 * c) for c in color)  # Konvertiere zu RGB-Werten (0-255)

        # Zeige das bearbeitete Bild an
        img = img.rotate(90, expand=True)
        plt.figure(figsize=(5, 7))
        plt.subplot(1, 1, 1)
        plt.imshow(img, cmap=cmap, norm=norm)
        plt.colorbar().set_label(feature, rotation=90)
        plt.title(f'Correlation of rPPG')
        #plt.xticks([])
        #plt.yticks([])
        #plt.subplot(1, 3, 2)
        #plt.plot(np.linspace(0, window_length, int(window_length * fps)), signal_left)
        #plt.title('Filtered Mean Green Channel PPG Signal (Left Hand)')
        #plt.xlabel('Time [s]')
        #plt.ylabel('Amplitude')
        #plt.ylim([min(min(signal_left), min(signal_right)), max(max(signal_left), max(signal_right))])
        #plt.subplot(1, 3, 3)
        #plt.plot(np.linspace(0, window_length, int(window_length * fps)), signal_right)
        #plt.title('Filtered Mean Green Channel PPG Signal (Right Hand)')
        #plt.ylim([min(min(signal_left), min(signal_right)), max(max(signal_left), max(signal_right))])
        #plt.xlabel('Time [s]')
        #plt.ylabel('Amplitude')
        plt.savefig(
            path + f'/Test Plot Images/heatmap_frame_{frame}_to_frame_{frame + (int(window_length * fps) - 1)}_windowed.png',
            dpi=300)
        print('Plot saved')
        plt.close()
        #plt.tight_layout()
        #plt.show()

        frame += int(window_length * fps)


window_length = 10                                  # Länge des Signalfensters in Sekunden
fps = 30                                            # Framerate der Kamera
measurement = 'E'                                   # [D: Basline, E: Full Stenosis, F: Medium Stenosis]
path = f'C:/Users/tauber/Desktop/{measurement}_9'
feature = 'Corr'                                    # Features = ['SNR', 'Corr', 'HR']
channel = 'g'                                       # channel = ['r', 'g', 'b', 'all']

create_signal_snips = 0  # 0: Signalabschnitte werden nicht erzeugt, 1: Signalabschnitte werden
calc_features = 0  # 0: Features werden nicht berechnet, 1: Features werden berechnet
heatmap = 1  # 0: Heatmap wird nicht erstellt, 1: Heatmap wird erstellt

if create_signal_snips == 1:
    connect_frames(window=window_length, fps=fps, path=path)

if calc_features == 1:
    feature_matrix(feature=feature, channel=channel, path=path, window_length=window_length,
                   fps=fps)

if heatmap == 1:
    create_heatmap(feature=feature, channel=channel, path=path, window_length=window_length, fps=fps, downscaling_factor=8)
