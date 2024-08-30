import streamlit as st
from multiapp import MultiApp
import cnn
import knn
import randomforest
import svm

# Membuat instance aplikasi
app = MultiApp()

# Halaman Home yang Ditingkatkan
def home():
    st.title("Selamat Datang di Aplikasi Klasifikasi Beras")

    # Menambahkan gambar header khusus (opsional: Anda bisa menghapus ini jika tidak diperlukan)
    st.image("image/rice.jpg", use_column_width=True)

    # Menambahkan pesan selamat datang
    st.markdown(
        """
        ### Mulai
        Selamat datang di **Aplikasi Klasifikasi Beras**! Aplikasi ini membantu Anda mengklasifikasikan berbagai jenis beras menggunakan algoritma pembelajaran mesin yang canggih.
        
        - **Pilih Algoritma**: Pilih salah satu algoritma dari sidebar untuk memulai.
        - **Unggah Gambar Anda**: Unggah gambar beras, dan aplikasi ini akan mengklasifikasikannya ke dalam salah satu kategori: **Basmati, IR64, Pandanwangi, Rojolele**.
        
        ### Algoritma yang Tersedia:
        - **Convolutional Neural Network (CNN)**
        - **Random Forest**
        - **K-Nearest Neighbors (KNN)**
        - **Support Vector Machine (SVM)**
        
        **Instruksi**:
        1. Pilih algoritma dari sidebar.
        2. Unggah gambar untuk diklasifikasikan.
        3. Dapatkan hasil dan prediksi secara instan!
        
        """,
        unsafe_allow_html=True
    )

    # Penambahan styling dan konten tambahan
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background: #e8effa;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Menambahkan halaman aplikasi Anda
app.add_app("Home", home)
app.add_app("Random Forest", randomforest.app)
app.add_app("CNN", cnn.app)
app.add_app("KNN", knn.app)
app.add_app("SVM", svm.app)

# Menjalankan aplikasi
app.run()
