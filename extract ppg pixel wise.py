import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

time = 60
fps = 30

def is_skin_pixel(pixel):
    # Konvertierung des Pixels von RGB zu YCrCb
    pixel_ycrcb = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_RGB2YCrCb)[0][0]
    y, cr, cb = pixel_ycrcb
    # Schwellwerte für Hautfarbe im YCrCb-Farbraum
    return 0 <= y <= 255 and 123 <= cr <= 173 and 77 <= cb <= 127

def plot_skin_pixels(frame_rgb, mask):
    # Kopiere das Originalbild
    highlighted_image = frame_rgb.copy()

    # Erstelle einen grünen Hintergrund
    green_background = np.full_like(frame_rgb, [0, 0, 0])  # RGB für Grün [0, 255, 0]

    # Kombiniere das Bild mit der Hautmaske: Hautpixel bleiben wie sie sind, Nicht-Hautpixel werden grün
    highlighted_image[mask] = [255, 255, 255]
    highlighted_image[~mask] = green_background[~mask]

    plt.figure(figsize=(10, 5))

    # Originalbild anzeigen
    plt.subplot(1, 2, 1)
    plt.imshow(frame_rgb)
    plt.title('Original RGB Bild')

    # Bild mit hervorgehobenen Hautpixeln anzeigen
    plt.subplot(1, 2, 2)
    plt.imshow(highlighted_image)
    plt.title('Erkannte Hautpixel hervorgehoben')

    plt.show()

def create_skin_mask(image, lower_threshold=0, upper_threshold=255, tolerance=3):
    # Konvertierung in Graustufen
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Anwenden der Thresholds
    if lower_threshold > 0:
        # Schwellwertbereich anwenden
        _, lower_mask = cv2.threshold(gray_image, lower_threshold, 255, cv2.THRESH_BINARY)
        _, upper_mask = cv2.threshold(gray_image, upper_threshold, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(lower_mask, upper_mask)  # Nur Pixel im Bereich [lower_threshold, upper_threshold]
    else:
        # Otsu-Schwellwertmethode anwenden
        _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, upper_mask = cv2.threshold(gray_image, upper_threshold, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(mask, upper_mask)  # Entfernen von sehr hellen Bereichen

    # Verkleinerung der Maske durch Erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance, tolerance))
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask

def process_video_frames(folder_path, down_scaling_factor, window_length):
    files = [file for file in os.listdir(folder_path) if file.endswith('.bmp')]

    for i in range(1, len(files) + 1):
        frame = cv2.imread(folder_path + "/image{}.bmp".format(i))

        height, width, _ = frame.shape

        frame = cv2.resize(frame, (width//down_scaling_factor, height//down_scaling_factor), interpolation=cv2.INTER_AREA)
        # Konvertiere den Frame in RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, _ = frame_rgb.shape

        # Nur bei erstem Durchlauf eine Maske generieren (Annahme: keine Bewegung der Hände)
        if i == 1 or (i % window_length) == 1:

            mask = create_skin_mask(frame_rgb)
            mask = (mask / 255).astype(bool)
            #plot_skin_pixels(frame_rgb, mask)
            #mask2 = np.zeros(frame_rgb.shape[:2], dtype=bool)
            #
            # # Erstellen der Hautmaske
            # for k in range(height):
            #     for j in range(width):
            #         if is_skin_pixel(frame_rgb[k, j]):
            #             mask[k, j] = True

        skin_pixels = frame_rgb[mask]

        # Plot Skin Mask
        #plot_skin_pixels(frame_rgb, mask)

        # Finde die Indizes der True-Werte
        y_indices, x_indices = np.nonzero(mask)

        # Erstelle ein DataFrame mit den X- und Y-Koordinaten
        mask_coordinates_df = pd.DataFrame({
            'x': x_indices,
            'y': y_indices,
            'R': np.nan * len(x_indices),
            'G': np.nan * len(x_indices),
            'B': np.nan * len(x_indices),
        })

        mask_coordinates_df['R'] = skin_pixels[:, 0]
        mask_coordinates_df['G'] = skin_pixels[:, 1]
        mask_coordinates_df['B'] = skin_pixels[:, 2]

        mask_coordinates_df.to_csv(folder_path + "/csv/rgb_pixel_values_frame{}.csv".format(i), index=False)

        print("Frame {} processed".format(i))

if __name__ == "__main__":

    measurement = 'E'       # [D: Basline, E: Full Stenosis, F: Medium Stenosis]
    folder_path = f'C:/Users/tauber/Desktop/{measurement}_9'
    process_video_frames(folder_path, down_scaling_factor=8, window_length=300)
