"""
Digital Image Processing Assignment - Module: brightness_gui.py
--------------------------------------------------------------
Modul ini menyediakan antarmuka grafis (GUI) interaktif untuk melakukan
penyesuaian kecerahan (brightness adjustment) pada citra secara real-time.
Implementasi ini menggunakan teknik vektorisasi NumPy untuk memastikan
performa yang responsif saat menggunakan slider.

Tujuan: Memenuhi poin bonus tugas implementasi GUI sederhana.
"""

import cv2
import numpy as np

def adjust_brightness(img_rgb, b):
    """
    Melakukan penyesuaian kecerahan pada citra menggunakan teknik vektorisasi.
    Optimasi ini digunakan agar pemrosesan pada slider GUI berjalan lancar.
    
    Rumus: f(x,y)' = f(x,y) + b 
    
    Args:
        img_rgb (numpy.ndarray): Citra input dalam format RGB.
        b (int): Nilai offset kecerahan (-100 s/d 100).
        
    Returns:
        numpy.ndarray: Citra hasil penyesuaian kecerahan dalam format uint8.
    """
    # Mengonversi ke int16 untuk mencegah overflow saat perhitungan matematika
    # sebelum melakukan clipping kembali ke rentang 0-255.
    res = np.clip(img_rgb.astype(np.int16) + b, 0, 255)
    return res.astype(np.uint8)

# ==========================================
# GUI SLIDER BRIGHTNESS
# ==========================================
def brightness_gui(img_rgb):
    """
    Membuat jendela interaktif dengan slider (Trackbar) untuk mengatur 
    kecerahan citra secara langsung.
    
    Mekanisme:
    1. Membuat window OpenCV.
    2. Menambahkan Trackbar dengan rentang 0-200 (Posisi 100 = Normal).
    3. Loop interaktif untuk menangkap perubahan nilai slider.
    
    Args:
        img_rgb (numpy.ndarray): Citra input yang akan ditampilkan di GUI.
    """
    window_name = 'Brightness GUI'
    cv2.namedWindow(window_name)

    # Callback kosong: Diperlukan oleh fungsi cv2.createTrackbar sebagai parameter wajib
    def nothing(x): pass

    # Inisialisasi Slider: Rentang 0 s/d 200 untuk merepresentasikan offset -100 s/d 100
    cv2.createTrackbar('Nilai B', window_name, 100, 200, nothing)

    print("\n--- Menjalankan GUI Interaktif ---")
    print("Gunakan slider untuk mengubah brightness.")
    print("Tekan tombol 'ESC' pada jendela gambar untuk keluar.")

    while True:
        # Membaca posisi slider saat ini dan menghitung nilai offset sebenarnya
        b_val = cv2.getTrackbarPos('Nilai B', window_name) - 100
        
        # Eksekusi fungsi brightness (menggunakan versi cepat/vektorisasi)
        res_rgb = adjust_brightness(img_rgb, b_val)
        
        # Konversi format RGB kembali ke BGR agar warna tampil akurat di window OpenCV
        res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
        
        # Update tampilan gambar pada window
        cv2.imshow(window_name, res_bgr)
        
        # Deteksi input keyboard: Keluar dari loop jika tombol ESC (ASCII 27) ditekan
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    # Membersihkan dan menutup semua jendela GUI setelah selesai
    cv2.destroyAllWindows()

def main():
    """
    Entry point untuk menjalankan aplikasi GUI Brightness.
    Membaca citra referensi dan memulai antarmuka interaktif.
    """
    # Membaca citra dan memastikan urutan kanal warna benar (RGB) sebelum diproses
    img1 = cv2.cvtColor(cv2.imread("image1.jpg"), cv2.COLOR_BGR2RGB)
    
    # Memulai antarmuka slider
    brightness_gui(img1)

if __name__ == "__main__":
    main()