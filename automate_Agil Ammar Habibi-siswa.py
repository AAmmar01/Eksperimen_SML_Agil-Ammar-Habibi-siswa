import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_raw_data():
    """
    Memuat data mentah (Iris Dataset) sebagai simulasi data input.
    Dalam proyek nyata, ini akan membaca dari file mentah.
    """
    # Mengambil data Iris dari Scikit-learn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Melakukan Data Preprocessing secara otomatis pada DataFrame.
    Tahapan: Pemisahan Fitur/Target dan Standarisasi (sesuai langkah di notebook).

    Args:
        df (pd.DataFrame): DataFrame input (data mentah).
        test_size (float): Proporsi data untuk set pengujian.
        random_state (int): Seed untuk reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) data yang sudah siap dilatih.
    """
    print("--- Memulai Data Preprocessing Otomatis ---")
    
    # 1. Pemisahan Fitur (X) dan Target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 2. Standarisasi Fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)
    print(f"Fitur berhasil distandardisasi. Bentuk data: {X_processed.shape}")
    
    # 3. Memisahkan Data Training dan Testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data dipisahkan. Training set: {X_train.shape}, Testing set: {X_test.shape}")
    print("--- Data Preprocessing Selesai ---")

    return X_train, X_test, y_train, y_test

def save_processed_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, folder_path: str = 'iris_processed'):
    """
    Menyimpan data yang sudah diproses ke folder yang ditentukan.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Gabungkan fitur dan target kembali untuk disimpan (opsional, tergantung kebutuhan Kriteria 2)
    train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    
    train_df.to_csv(os.path.join(folder_path, 'train_processed.csv'), index=False)
    test_df.to_csv(os.path.join(folder_path, 'test_processed.csv'), index=False)
    print(f"Data yang sudah diproses disimpan di folder: {folder_path}")

# --- Eksekusi Utama (Untuk Menguji Fungsi) ---
if __name__ == "__main__":
    # 1. Muat data mentah
    raw_df = load_raw_data()
    
    # 2. Proses data
    X_train, X_test, y_train, y_test = preprocess_data(raw_df)
    
    # 3. Simpan data yang sudah diproses
    # Pastikan nama folder sesuai dengan yang Anda gunakan di repository
    save_processed_data(X_train, X_test, y_train, y_test, folder_path='iris_processed') 
    
    # Contoh akses setelah disimpan
    # processed_data = pd.read_csv('iris_processed/train_processed.csv')
    # print("\nContoh data yang sudah disimpan:")
    # print(processed_data.head())
