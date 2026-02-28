import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. OPERASI TITIK 
# ==========================================

def convert_to_grayscale(img_rgb, method='luminance'):
    M, N, _ = img_rgb.shape
    gray = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            R, G, B = img_rgb[i, j].astype(float)
            if method == 'average':
                # Metode rata-rata: (R + G + B)/3 
                gray[i, j] = (R + G + B) / 3
            else:
                # Metode luminance: 0.299R + 0.587G + 0.114B 
                gray[i, j] = (0.299 * R) + (0.587 * G) + (0.114 * B)
    return gray

def adjust_negative(img):
    # f(x,y)' = 255 - f(x,y) 
    M, N = img.shape[:2]
    res = np.zeros_like(img)
    for i in range(M):
        for j in range(N):
            res[i, j] = 255 - img[i, j]
    return res

def adjust_brightness(img, b):
    # f(x,y)' = f(x,y) + b 
    M, N = img.shape[:2]
    res = np.zeros_like(img)
    for i in range(M):
        for j in range(N):
            # Gunakan np.clip agar nilai tetap 0-255 
            res[i, j] = np.clip(img[i, j].astype(int) + b, 0, 255)
    return res

def apply_threshold(gray_img, threshold):
    # Thresholding (citra biner) 
    M, N = gray_img.shape
    biner = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            biner[i, j] = 255 if gray_img[i, j] > threshold else 0
    return biner

# ==========================================
# 2. OPERASI ARITMATIKA 
# ==========================================

def arithmetic_ops(img1, img2, scalar=1.2):
    M, N, C = img1.shape
    add = np.zeros_like(img1)
    sub = np.zeros_like(img1)
    mul = np.zeros_like(img1)
    
    for i in range(M):
        for j in range(N):
            for c in range(C):
                # Penjumlahan, Pengurangan, & Perkalian Skalar 
                add[i, j, c] = np.clip(int(img1[i, j, c]) + int(img2[i, j, c]), 0, 255)
                sub[i, j, c] = np.clip(int(img1[i, j, c]) - int(img2[i, j, c]), 0, 255)
                mul[i, j, c] = np.clip(img1[i, j, c] * scalar, 0, 255)
    return add, sub, mul

# ==========================================
# 3. OPERASI LOKAL (FILTERING) 
# ==========================================

def mean_filter_3x3(gray_img):
    # Mean filter 3x3 (mask 1/9) 
    M, N = gray_img.shape
    res = np.zeros_like(gray_img)
    # Padding manual sederhana dengan nol
    padded = np.pad(gray_img, ((1, 1), (1, 1)), mode='constant')
    
    for i in range(M):
        for j in range(N):
            # Ambil jendela 3x3
            region = padded[i:i+3, j:j+3]
            res[i, j] = np.sum(region) / 9
    return res

# ==========================================
# 4. OPERASI BOOLEAN (Input Biner) 
# ==========================================

def boolean_ops(bin1, bin2):
    # AND, OR, NOT 
    op_and = cv2.bitwise_and(bin1, bin2)
    op_or = cv2.bitwise_or(bin1, bin2)
    op_not = cv2.bitwise_not(bin1)
    return op_and, op_or, op_not

# ==========================================
# 5. IMAGE BLENDING 
# ==========================================

def blend_images(img1, img2, alpha):
    # O = αI1 + (1-α)I2 
    return (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # Load dan Pre-processing
    img1 = cv2.cvtColor(cv2.imread('image1.jpg'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('image2.jpg'), cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (480, 320))
    img2 = cv2.resize(img1, (480, 320))
    # img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Jalankan Fungsi Modular
    g_avg = convert_to_grayscale(img1, 'average')
    g_lum = convert_to_grayscale(img1, 'luminance')
    negatif = adjust_negative(img1)
    cerah = adjust_brightness(img1, 50)
    biner_100 = apply_threshold(g_lum, 100)
    biner_150 = apply_threshold(g_lum, 150)
    
    tambah, kurang, kali = arithmetic_ops(img1, img2)
    filtered = mean_filter_3x3(g_lum)
    b_and, b_or, b_not = boolean_ops(biner_100, biner_150)
    
    # Blending dengan 3 nilai alpha 
    blend_03 = blend_images(img1, img2, 0.3)
    blend_05 = blend_images(img1, img2, 0.5)
    blend_07 = blend_images(img1, img2, 0.7)

    # --- Visualisasi (Gunakan subplot untuk laporan) ---
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1); plt.imshow(img1); plt.title("Original")
    plt.subplot(2, 3, 2); plt.imshow(g_lum, cmap='gray'); plt.title("Luminance")
    plt.subplot(2, 3, 3); plt.imshow(negatif); plt.title("Negatif")
    plt.subplot(2, 3, 4); plt.imshow(filtered, cmap='gray'); plt.title("Mean Filter")
    plt.subplot(2, 3, 5); plt.imshow(blend_05); plt.title("Blending 0.5")
    plt.subplot(2, 3, 6); plt.imshow(b_and, cmap='gray'); plt.title("Boolean AND")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()