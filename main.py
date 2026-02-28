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
    
    img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    path = os.path.join(folder, f"{name}.jpg")
    cv2.imwrite(path, img_to_save)
    print(f"   [Saved Image] {path}")

def save_histogram_comparison(img_before, img_after, name, label_before="Sebelum", label_after="Sesudah", folder='hasil_output_bonus'):
    """Membuat plot perbandingan citra dan histogram yang mendukung mix RGB & Grayscale"""
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    def plot_column(img, col_index, label):
        is_gray = len(img.shape) == 2
        axes[0, col_index].imshow(img, cmap='gray' if is_gray else None)
        axes[0, col_index].set_title(f"Citra {label}")

        if is_gray:
            axes[1, col_index].hist(img.ravel(), 256, [0, 256], color='black')
        else:
            colors = ('red', 'green', 'blue')
            for i, col in enumerate(colors):
                axes[1, col_index].hist(img[:,:,i].ravel(), 256, [0, 256], color=col, alpha=0.5)
        
        axes[1, col_index].set_title(f"Histogram {label}")

    plot_column(img_before, 0, label_before)
    plot_column(img_after, 1, label_after)

    plt.tight_layout()
    path = os.path.join(folder, f"hist_{name}.png")
    plt.savefig(path)
    plt.close()
    print(f"   [Saved Hist]  {path}")

# ==========================================
# 1. OPERASI TITIK
# ==========================================
def convert_to_grayscale(img_rgb, method='luminance'):
    M, N = img_rgb.shape[:2]
    gray = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            R, G, B = img_rgb[i, j].astype(float)
            if method == 'average':
                gray[i, j] = (R + G + B) / 3 
            else:
                gray[i, j] = (0.299 * R) + (0.587 * G) + (0.114 * B) 
    return gray

def adjust_negative(img_rgb):
    M, N, C = img_rgb.shape
    res = np.zeros_like(img_rgb)
    for i in range(M):
        for j in range(N):
            for c in range(C):
                res[i, j, c] = 255 - img_rgb[i, j, c] 
    return res

def adjust_brightness(img_rgb, b):
    M, N, C = img_rgb.shape
    res = np.zeros_like(img_rgb)
    for i in range(M):
        for j in range(N):
            for c in range(C):
                res[i, j, c] = np.clip(int(img_rgb[i, j, c]) + b, 0, 255) 
    return res

def apply_threshold(gray_img, threshold):
    M, N = gray_img.shape
    biner = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            biner[i, j] = 255 if gray_img[i, j] > threshold else 0 
    return biner

# ==========================================
# 2. OPERASI ARITMATIKA
# ==========================================
def arithmetic_ops(img1, img2, scalar=1.5):
    M, N, C = img1.shape
    add = np.zeros_like(img1)
    sub = np.zeros_like(img1)
    mul = np.zeros_like(img1)
    for i in range(M):
        for j in range(N):
            for c in range(C):
                add[i, j, c] = np.clip(int(img1[i, j, c]) + int(img2[i, j, c]), 0, 255) 
                sub[i, j, c] = np.clip(int(img1[i, j, c]) - int(img2[i, j, c]), 0, 255) 
                mul[i, j, c] = np.clip(img1[i, j, c] * scalar, 0, 255) 
    return add, sub, mul

# ==========================================
# 3. OPERASI LOKAL (FILTERING)
# ==========================================
def mean_filter_3x3(gray_img):
    M, N = gray_img.shape
    res = np.zeros_like(gray_img)
    padded = np.pad(gray_img, ((1, 1), (1, 1)), mode='constant')
    for i in range(M):
        for j in range(N):
            region = padded[i:i+3, j:j+3]
            res[i, j] = np.sum(region) / 9 
    return res

# ==========================================
# 4. OPERASI BOOLEAN
# ==========================================
def boolean_ops(bin1, bin2):
    M, N = bin1.shape
    op_and = np.zeros((M, N), dtype=np.uint8)
    op_or = np.zeros((M, N), dtype=np.uint8)
    op_not = np.zeros((M, N), dtype=np.uint8)
    
    for i in range(M):
        for j in range(N):
            # AND: Putih jika keduanya putih 
            op_and[i, j] = 255 if bin1[i, j] == 255 and bin2[i, j] == 255 else 0
            # OR: Putih jika salah satu putih 
            op_or[i, j] = 255 if bin1[i, j] == 255 or bin2[i, j] == 255 else 0
            # NOT: Inversi citra 1 
            op_not[i, j] = 255 - bin1[i, j]
            
    return op_and, op_or, op_not

# ==========================================
# 5. IMAGE BLENDING
# ==========================================
def blend_images(img1, img2, alpha):
    # O = alpha*I1 + (1-alpha)*I2 
    return (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("--- Memulai Skrip Pemrosesan Citra Lengkap ---")
    start_total = time.time()
    proc_dir = 'hasil_output'
    bonus_dir = 'hasil_output_bonus'

    # Load & Pre-processing
    img1 = cv2.cvtColor(cv2.imread('image1.jpg'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('image2.jpg'), cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (400, 300))
    img2 = cv2.resize(img2, (400, 300))
    save_image(img1, "00_original_1", proc_dir)
    save_image(img2, "00_original_2", proc_dir)

    # 1. Operasi Titik
    print("[1/5] Memproses Operasi Titik...")
    g_avg1 = convert_to_grayscale(img1, 'average')
    g_lum1 = convert_to_grayscale(img1, 'luminance')
    g_lum2 = convert_to_grayscale(img2, 'luminance')
    
    negatif = adjust_negative(img1)
    bright_pos = adjust_brightness(img1, 50)
    bright_neg = adjust_brightness(img1, -50)
    
    # Thresholding untuk kedua gambar guna keperluan Boolean
    bin1_100 = apply_threshold(g_lum1, 100) # Threshold Gambar 1
    bin2_100 = apply_threshold(g_lum2, 100) # Threshold Gambar 2
    bin1_180 = apply_threshold(g_lum1, 180) # Variasi Threshold Gambar 1
    bin2_180 = apply_threshold(g_lum2, 180) # Variasi Threshold Gambar 2

    save_image(g_avg1, "01a_grayscale_avg_img1", proc_dir)
    save_image(g_lum1, "01a_grayscale_luminance_img1", proc_dir)
    save_image(negatif, "01b_negatif", proc_dir)
    save_image(bright_pos, "01c_brightness_pos", proc_dir)
    save_image(bright_neg, "01c_brightness_neg", proc_dir)
    save_image(bin1_100, "01d_threshold_100_img1", proc_dir)
    save_image(bin2_100, "01d_threshold_100_img2", proc_dir)
    save_image(bin1_180, "01d_threshold_180_img1", proc_dir)
    save_image(bin2_180, "01d_threshold_180_img2", proc_dir)
    
    # Histogram Operasi Titik
    save_histogram_comparison(g_avg1, g_lum1, "01a_grayscale_comp", "Average", "Luminance", bonus_dir)
    save_histogram_comparison(img1, g_avg1, "01a_grayscale_avg", "Original 1", "Gray Average", bonus_dir)
    save_histogram_comparison(img1, g_lum1, "01a_grayscale_lum", "Original 1", "Gray Luminance", bonus_dir)
    save_histogram_comparison(img1, negatif, "01b_negatif1", "Original 1", "Negatif 1", bonus_dir)
    save_histogram_comparison(img2, negatif, "01b_negatif2", "Original 2", "Negatif 2", bonus_dir)
    save_histogram_comparison(img1, bright_pos, "01c_brightness_pos", "Original 1", "Bright", bonus_dir)
    save_histogram_comparison(img1, bright_neg, "01c_brightness_neg", "Original 1", "Dark", bonus_dir)
    save_histogram_comparison(g_lum1, bin1_100, "01d_threshold_100_img1", "Gray 1", "Bin1_100", bonus_dir)
    save_histogram_comparison(g_lum2, bin2_100, "01d_threshold_100_img2", "Gray 2", "Bin2_100", bonus_dir)
    save_histogram_comparison(g_lum1, bin1_180, "01d_threshold_180_img1", "Gray 1", "Bin1_180", bonus_dir)
    save_histogram_comparison(g_lum2, bin2_180, "01d_threshold_180_img2", "Gray 2", "Bin2_180", bonus_dir)

    # 2. Operasi Aritmatika
    print("[2/5] Memproses Operasi Aritmatika...")
    add, sub, mul = arithmetic_ops(img1, img2, scalar=1.5)
    save_image(add, "02a_aritmatika_penjumlahan", proc_dir)
    save_image(sub, "02b_aritmatika_pengurangan", proc_dir)
    save_image(mul, "02c_aritmatika_perkalian_skalar", proc_dir)
    
    save_histogram_comparison(img1, add, "02a_aritmatika_add", "Img1", "Hasil Tambah", bonus_dir)
    save_histogram_comparison(img1, sub, "02a_aritmatika_sub", "Img1", "Hasil Kurang", bonus_dir)
    save_histogram_comparison(img1, mul, "02a_aritmatika_mul", "Img1", "Hasil Kali", bonus_dir)

    # 3. Operasi Lokal
    print("[3/5] Memproses Operasi Lokal (Filtering)...")
    filtered = mean_filter_3x3(g_lum1)
    save_image(filtered, "03_mean_filter_3x3", proc_dir)
    save_histogram_comparison(g_lum1, filtered, "03_filtering", "Sebelum", "Sesudah", bonus_dir)

    # 4. Operasi Boolean (MENGGUNAKAN BIN_100 IMG1 DAN BIN_100 IMG2)
    print("[4/5] Memproses Operasi Boolean (Img1 vs Img2)...")
    b_and, b_or, b_not = boolean_ops(bin1_100, bin2_100)
    
    save_image(b_and, "04a_boolean_AND", proc_dir)
    save_image(b_or, "04b_boolean_OR", proc_dir)
    save_image(b_not, "04c_boolean_NOT", proc_dir)
    
    # Histogram Perbandingan Boolean
    save_histogram_comparison(bin1_100, b_and, "04a_boolean_AND", "Bin1_100", "Hasil AND", bonus_dir)
    save_histogram_comparison(bin1_100, b_or, "04b_boolean_OR", "Bin1_100", "Hasil OR", bonus_dir)
    save_histogram_comparison(bin1_100, b_not, "04c_boolean_NOT", "Bin1_100", "Hasil NOT", bonus_dir)

    # 5. Image Blending
    print("[5/5] Memproses Image Blending...")
    alphas = [0.3, 0.5, 0.7]
    for a in alphas:
        blended = blend_images(img1, img2, a)
        save_image(blended, f"05_blending_alpha_{a}", proc_dir)
        save_histogram_comparison(img1, blended, f"05_blending_alpha_{a}", "Original 1", f"Blend {a}", bonus_dir)

    print(f"\n--- Batch Selesai! Total waktu: {time.time() - start_total:.2f} detik ---")

if __name__ == "__main__":
    main()