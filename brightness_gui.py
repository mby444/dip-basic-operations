import cv2
import numpy as np

def adjust_brightness(img_rgb, b):
  res = np.clip(img_rgb.astype(np.int16) + b, 0, 255)
  return res.astype(np.uint8)

# ==========================================
# GUI SLIDER BRIGHTNESS
# ==========================================
def brightness_gui(img_rgb):
  window_name = 'Brightness GUI'
  cv2.namedWindow(window_name)

  # Callback kosong (diperlukan OpenCV)
  def nothing(x): pass

  # Slider dari 0 ke 200, posisi awal di 100 (netral)
  cv2.createTrackbar('Nilai B', window_name, 100, 200, nothing)

  print("\n--- Menjalankan GUI Interaktif ---")
  print("Gunakan slider untuk mengubah brightness.")
  print("Tekan tombol 'ESC' pada jendela gambar untuk keluar.")

  while True:
      # Ambil posisi slider dan hitung offset (-100 sampai 100)
      b_val = cv2.getTrackbarPos('Nilai B', window_name) - 100
      
      # Proses brightness
      res_rgb = adjust_brightness(img_rgb, b_val)
      
      # Konversi ke BGR untuk ditampilkan di OpenCV window
      res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
      
      cv2.imshow(window_name, res_bgr)
      
      # Berhenti jika tombol ESC (ASCII 27) ditekan
      if cv2.waitKey(1) & 0xFF == 27:
          break
          
  cv2.destroyAllWindows()

def main():
  img1 = cv2.cvtColor(cv2.imread("image1.jpg"), cv2.COLOR_BGR2RGB)
  brightness_gui(img1)

if __name__ == "__main__":
  main()