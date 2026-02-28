import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ==========================================
# FUNGSI HELPER PENYIMPANAN
# ==========================================
def save_image(img, name, folder='hasil_output'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Konversi kembali ke BGR sebelum disimpan jika citra berwarna (RGB)
    if len(img.shape) == 3:
        img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_to_save = img
        
    path = os.path.join(folder, f"{name}.jpg")
    cv2.imwrite(path, img_to_save)
    print(f"   [Saved] {path}")

# ==========================================
# 1. OPERASI TITIK [cite: 14]
# ==========================================
def convert_to_grayscale(img_rgb, method='luminance'):
    M, N, _ = img_rgb.shape
    gray = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            R, G, B = img_rgb[i, j].astype(float)
            if method == 'average': # [cite: 16]
                gray[i, j] = (R + G + B) / 3
            else: # [cite: 17]
                gray[i, j] = (0.299 * R) + (0.587 * G) + (0.114 * B)
    return gray

def adjust_negative(img): # [cite: 18]
    M, N = img.shape[:2]
    res = np.zeros_like(img)
    for i in range(M):
        for j in range(N):
            res[i, j] = 255 - img[i, j]
    return res

def adjust_brightness(img, b): # [cite: 19]
    M, N = img.shape[:2]
    res = np.zeros_like(img)
    for i in range(M):
        for j in range(N):
            res[i, j] = np.clip(img[i, j].astype(int) + b, 0, 255)
    return res

def apply_threshold(gray_img, threshold): # [cite: 20]
    M, N = gray_img.shape
    biner = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            biner[i, j] = 255 if gray_img[i, j] > threshold else 0
    return biner

# ==========================================
# 2. OPERASI ARITMATIKA [cite: 21]
# ==========================================
def arithmetic_ops(img1, img2, scalar=1.5):
    M, N, C = img1.shape
    add = np.zeros_like(img1)
    sub = np.zeros_like(img1)
    mul = np.zeros_like(img1)
    
    for i in range(M):
        for j in range(N):
            for c in range(C):
                # Penjumlahan, Pengurangan, & Perkalian Skalar [cite: 22, 23, 24]
                add[i, j, c] = np.clip(int(img1[i, j, c]) + int(img2[i, j, c]), 0, 255)
                sub[i, j, c] = np.clip(int(img1[i, j, c]) - int(img2[i, j, c]), 0, 255)
                mul[i, j, c] = np.clip(img1[i, j, c] * scalar, 0, 255)
    return add, sub, mul

# ==========================================
# 3. OPERASI LOKAL (FILTERING) [cite: 25]
# ==========================================
def mean_filter_3x3(gray_img): # [cite: 26]
    M, N = gray_img.shape
    res = np.zeros_like(gray_img)
    padded = np.pad(gray_img, ((1, 1), (1, 1)), mode='constant')
    
    for i in range(M):
        for j in range(N):
            region = padded[i:i+3, j:j+3]
            res[i, j] = np.sum(region) / 9
    return res

# ==========================================
# 4. OPERASI BOOLEAN [cite: 28]
# ==========================================
def boolean_ops(bin1, bin2):
    op_and = cv2.bitwise_and(bin1, bin2) # [cite: 29]
    op_or = cv2.bitwise_or(bin1, bin2)   # [cite: 30]
    op_not = cv2.bitwise_not(bin1)        # [cite: 31]
    return op_and, op_or, op_not

# ==========================================
# 5. IMAGE BLENDING [cite: 33]
# ==========================================
def blend_images(img1, img2, alpha): # 
    return (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("--- Memulai Skrip Pemrosesan Citra ---")
    start_total = time.time()
    out_dir = 'hasil_output'

    # 1. Load & Resize
    print("[1/6] Memuat citra...")
    img1 = cv2.cvtColor(cv2.imread('image1.jpg'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('image2.jpg'), cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (480, 320))
    img2 = cv2.resize(img2, (480, 320))
    save_image(img1, "00_original_resized", out_dir)

    # 2. Operasi Titik
    print("[2/6] Memproses Operasi Titik...")
    g_avg = convert_to_grayscale(img1, 'average')
    g_lum = convert_to_grayscale(img1, 'luminance')
    negatif = adjust_negative(img1)
    cerah = adjust_brightness(img1, 50)
    bin_100 = apply_threshold(g_lum, 100)
    bin_150 = apply_threshold(g_lum, 150)
    
    save_image(g_avg, "01_gray_average", out_dir)
    save_image(g_lum, "02_gray_luminance", out_dir)
    save_image(negatif, "03_negatif", out_dir)
    save_image(cerah, "04_brightness_plus_50", out_dir)
    save_image(bin_100, "05_threshold_100", out_dir)
    save_image(bin_150, "06_threshold_150", out_dir)

    # 3. Operasi Aritmatika
    print("[3/6] Memproses Operasi Aritmatika...")
    tambah, kurang, kali = arithmetic_ops(img1, img2)
    save_image(tambah, "07_aritmatika_tambah", out_dir)
    save_image(kurang, "08_aritmatika_kurang", out_dir)
    save_image(kali, "09_aritmatika_skalar", out_dir)

    # 4. Operasi Lokal
    print("[4/6] Memproses Operasi Lokal (Filtering)...")
    filtered = mean_filter_3x3(g_lum)
    save_image(filtered, "10_mean_filter_3x3", out_dir)

    # 5. Boolean & Blending
    print("[5/6] Memproses Boolean & Blending...")
    b_and, b_or, b_not = boolean_ops(bin_100, bin_150)
    save_image(b_and, "11_boolean_AND", out_dir)
    save_image(b_or, "12_boolean_OR", out_dir)
    save_image(b_not, "13_boolean_NOT", out_dir)
    
    for a in [0.3, 0.5, 0.7]: # 
        blended = blend_images(img1, img2, a)
        save_image(blended, f"14_blending_alpha_{a}", out_dir)

    # 6. Finalisasi
    end_total = time.time()
    print(f"\n--- Selesai! Semua file disimpan di folder '{out_dir}' ---")
    print(f"Total waktu: {end_total - start_total:.2f} detik")

if __name__ == "__main__":
    main()