from deepface import DeepFace
import cv2
import os
import numpy as np

def main():
    # Inisialisasi webcam
    cap = cv2.VideoCapture(0)

    img_counter = 0
    detector_backend = "opencv"  # Atur backend deteksi wajah

    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()

        # Tampilkan frame dari webcam
        cv2.imshow('Webcam', frame)

        # Tunggu input dari user, jika tekan 's', ambil gambar
        key = cv2.waitKey(1)
        if key % 256 == 27:
            # ESC ditekan untuk keluar
            print("Escape hit, closing...")
            break

        elif key % 256 == 32:
            # SPACE ditekan untuk capture gambar
            img_counter += 1
            print(f"Capturing image {img_counter}...")

            # Konversi frame ke uint8 jika perlu
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            # Pastikan frame dalam format BGR
            if len(frame.shape) == 2:  # Jika grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # Jika RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Ekstrak wajah menggunakan DeepFace
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=detector_backend,
                enforce_detection=False
            )

            # Jika wajah terdeteksi, simpan semua wajah yang ditemukan
            if len(faces) > 0:
                for i, face_dict in enumerate(faces):
                    face = face_dict["face"]
                    face_filename = os.path.join(f"extracted_face_{img_counter}_{i+1}.png")

                    # Konversi face ke uint8 jika perlu
                    if face.dtype != np.uint8:
                        face = (face * 255).astype(np.uint8)

                    # Simpan wajah hasil ekstraksi menggunakan cv2.imwrite
                    cv2.imwrite(face_filename, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                    print(f"Saved extracted face as {face_filename}")
            else:
                print("No faces detected in this frame.")



    # Lepaskan resource webcam dan tutup window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()