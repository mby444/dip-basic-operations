import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ==========================================
# FUNGSI HELPER PENYIMPANAN & HISTOGRAM
# ==========================================
def save_image(img, name, folder='hasil_output'):
    """Menyimpan gambar hasil pemrosesan ke folder hasil_output"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Konversi RGB ke BGR untuk OpenCV imwrite agar warna tidak tertukar
    img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    path = os.path.join(folder, f"{name}.jpg")
    cv2.imwrite(path, img_to_save)
    print(f"   [Saved Image] {path}")

def save_histogram_comparison(img_before, img_after, name, label_before="Sebelum", label_after="Sesudah", folder='hasil_output_bonus'):
    """Membuat dan menyimpan plot perbandingan citra beserta histogramnya ke hasil_output_bonus"""
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Baris 1: Tampilan Citra
    axes[0, 0].imshow(img_before, cmap='gray' if len(img_before.shape) == 2 else None)
    axes[0, 0].set_title(f"Citra {label_before}")
    axes[0, 1].imshow(img_after, cmap='gray' if len(img_after.shape) == 2 else None)
    axes[0, 1].set_title(f"Citra {label_after}")

    # Baris 2: Histogram
    if len(img_before.shape) == 2: # Citra Grayscale
        axes[1, 0].hist(img_before.ravel(), 256, [0, 256], color='black')
        axes[1, 1].hist(img_after.ravel(), 256, [0, 256], color='black')
    else: # Citra Berwarna (RGB)
        colors = ('red', 'green', 'blue')
        for i, col in enumerate(colors):
            axes[1, 0].hist(img_before[:,:,i].ravel(), 256, [0, 256], color=col, alpha=0.5)
            axes[1, 1].hist(img_after[:,:,i].ravel(), 256, [0, 256], color=col, alpha=0.5)

    axes[1, 0].set_title(f"Histogram {label_before}")
    axes[1, 1].set_title(f"Histogram {label_after}")
    
    plt.tight_layout()
    path = os.path.join(folder, f"hist_{name}.png")
    plt.savefig(path)
    plt.close()
    print(f"   [Saved Hist]  {path}")

# ==========================================
# FUNGSI OPERASI (Sesuai Tugas)
# ==========================================
def convert_to_grayscale(img_rgb, method='luminance'):
    M, N = img_rgb.shape[:2]
    gray = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            R, G, B = img_rgb[i, j].astype(float)
            if method == 'average':
                # [cite_start]Metode rata-rata: (R + G + B)/3 [cite: 16]
                gray[i, j] = (R + G + B) / 3 
            else:
                # [cite_start]Metode luminance: 0.299R + 0.587G + 0.114B [cite: 17]
                gray[i, j] = (0.299 * R) + (0.587 * G) + (0.114 * B) 
    return gray

def adjust_brightness(img, b):
    M, N = img.shape[:2]
    res = np.zeros_like(img)
    for i in range(M):
        for j in range(N):
            # [cite_start]f(x,y)' = f(x,y) + b [cite: 19]
            res[i, j] = np.clip(img[i, j].astype(int) + b, 0, 255) 
    return res

def apply_threshold(gray_img, threshold):
    M, N = gray_img.shape
    biner = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            # [cite_start]Thresholding biner [cite: 20]
            biner[i, j] = 255 if gray_img[i, j] > threshold else 0 
    return biner

def mean_filter_3x3(gray_img):
    M, N = gray_img.shape
    res = np.zeros_like(gray_img)
    # [cite_start]Mask 1/9 untuk mean filter [cite: 26]
    padded = np.pad(gray_img, ((1, 1), (1, 1)), mode='constant')
    for i in range(M):
        for j in range(N):
            region = padded[i:i+3, j:j+3]
            res[i, j] = np.sum(region) / 9 
    return res

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("--- Memulai Skrip Pemrosesan Citra + Histogram ---")
    start_total = time.time()
    
    # Definisi direktori tujuan
    proc_dir = 'hasil_output'       # Untuk file gambar hasil
    bonus_dir = 'hasil_output_bonus' # Untuk file histogram bonus

    # 1. Load & Resize
    # Pastikan file image1.jpg tersedia di folder yang sama
    img1 = cv2.cvtColor(cv2.imread('image1.jpg'), cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (400, 300)) # Resize agar loop manual lebih cepat
    save_image(img1, "00_original", proc_dir)

    # [cite_start]2. Grayscale & Histogram (Analisis Luminance vs Average) [cite: 38]
    print("[1/4] Memproses Grayscale & Histogram...")
    g_avg = convert_to_grayscale(img1, 'average')
    g_lum = convert_to_grayscale(img1, 'luminance')
    
    # Simpan gambar hasil pemrosesan
    save_image(g_avg, "01_gray_average", proc_dir)
    save_image(g_lum, "02_gray_luminance", proc_dir)
    # Simpan histogram perbandingan di folder bonus
    save_histogram_comparison(g_avg, g_lum, "grayscale_comp", "Average", "Luminance", folder=bonus_dir)

    # [cite_start]3. Brightness & Histogram [cite: 39]
    print("[2/4] Memproses Brightness & Histogram...")
    img_bright = adjust_brightness(img1, 60)
    
    # Simpan gambar hasil pemrosesan
    save_image(img_bright, "03_brightness_plus_60", proc_dir)
    # Simpan histogram perbandingan di folder bonus
    save_histogram_comparison(img1, img_bright, "brightness", "Original", "Bright (+60)", folder=bonus_dir)

    # [cite_start]4. Thresholding & Histogram [cite: 20]
    print("[3/4] Memproses Thresholding...")
    bin_127 = apply_threshold(g_lum, 127)
    
    # Simpan gambar hasil pemrosesan
    save_image(bin_127, "04_threshold_127", proc_dir)
    # Simpan histogram perbandingan di folder bonus
    save_histogram_comparison(g_lum, bin_127, "thresholding", "Grayscale", "Biner (T=127)", folder=bonus_dir)

    # [cite_start]5. Filtering (Lokal) [cite: 40]
    print("[4/4] Memproses Filtering...")
    filtered = mean_filter_3x3(g_lum)
    
    # Simpan gambar hasil pemrosesan
    save_image(filtered, "05_mean_filtered", proc_dir)
    # Simpan histogram perbandingan di folder bonus
    save_histogram_comparison(g_lum, filtered, "filtering", "Grayscale_Original", "Filtered", folder=bonus_dir)

    print(f"\n--- Selesai! ---")
    print(f"Gambar hasil di: '{proc_dir}'")
    print(f"Histogram bonus di: '{bonus_dir}'")
    print(f"Total waktu: {time.time() - start_total:.2f} detik")

if __name__ == "__main__":
    main()