import streamlit as st  # Library untuk bikin web app dengan mudah
import pandas as pd  # Library untuk manipulasi data (kayak Excel)
import numpy as np  # Library untuk operasi matematika dan array
import plotly.express as px  # Library untuk bikin grafik yang bagus
import plotly.graph_objects as go  # Library untuk grafik yang lebih custom
from sklearn.ensemble import RandomForestClassifier  # Algoritma ML yang kita pakai
from sklearn.model_selection import train_test_split  # Untuk bagi data jadi training & testing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Untuk evaluasi model
import pickle  # Untuk simpan/load model dan dataset
import os  # Untuk operasi file system
from datetime import datetime  # Untuk dapat tanggal dan waktu
import sqlite3  # Untuk database SQLite
import warnings  # Untuk sembunyiin warning yang nggak penting
warnings.filterwarnings('ignore')  # Matiin semua warning

# ============================================================================
# KONFIGURASI HALAMAN - Setting tampilan web app
# ============================================================================

# Set judul halaman, icon, dan layout
st.set_page_config(
    page_title="🍎  Makanan Sehat - Random Forest",  # Judul yang muncul di tab browser
    page_icon="🍎",  # Icon yang muncul di tab browser
    layout="wide",  # Layout lebar biar lebih lega
    initial_sidebar_state="expanded"  # Sidebar langsung terbuka pas pertama kali
)

# ============================================================================
# CSS STYLING - Bikin tampilan lebih menarik dengan CSS
# ============================================================================

# Ini CSS untuk styling elemen HTML biar lebih bagus
st.markdown("""
<style>
    /* Styling untuk header utama - judul besar di atas */
    .main-header {
        font-size: 3rem;  /* Ukuran font besar banget */
        font-weight: bold;  /* Teks tebal */
        color: #2E7D32;  /* Warna hijau tua */
        text-align: center;  /* Rata tengah */
        margin-bottom: 1rem;  /* Jarak bawah */
    }
    
    /* Styling untuk sub-header - judul kedua */
    .sub-header {
        font-size: 1.5rem;  /* Ukuran font sedang */
        color: #4CAF50;  /* Warna hijau muda */
        text-align: center;  /* Rata tengah */
        margin-bottom: 2rem;  /* Jarak bawah lebih besar */
    }
    
    /* Kotak untuk pesan sukses - warna hijau */
    .success-box {
        padding: 1rem;  /* Jarak dalam kotak */
        border-radius: 0.5rem;  /* Sudut kotak agak bulat */
        background-color: #C8E6C9;  /* Background hijau muda */
        border-left: 5px solid #2E7D32;  /* Border kiri hijau tebal */
        margin: 1rem 0;  /* Jarak atas bawah */
    }
    
    /* Kotak untuk peringatan - warna orange */
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFE0B2;  /* Background orange muda */
        border-left: 5px solid #FF9800;  /* Border kiri orange */
        margin: 1rem 0;
    }
    
    /* Kartu untuk menampilkan metric - angka penting */
    .metric-card {
        background-color: #F5F5F5;  /* Background abu-abu muda */
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True biar bisa pakai HTML

# ============================================================================
# DATABASE MAKANAN - Kumpulan data makanan populer beserta nutrisinya
# ============================================================================

# Dictionary yang isinya data nutrisi berbagai makanan
# Format: "Nama Makanan": {nutrisi: nilai}
FOOD_DATABASE = {
    "Salad Sayuran": {
        "kalori": 150, "protein": 8, "karbohidrat": 20, 
        "lemak": 5, "serat": 6, "gula": 5
    },
    "Burger Fast Food": {
        "kalori": 600, "protein": 25, "karbohidrat": 65, 
        "lemak": 30, "serat": 2, "gula": 10
    },
    "Buah Apel": {
        "kalori": 95, "protein": 0.5, "karbohidrat": 25, 
        "lemak": 0.3, "serat": 4, "gula": 19
    },
    "Keripik Kentang": {
        "kalori": 550, "protein": 7, "karbohidrat": 53, 
        "lemak": 35, "serat": 4, "gula": 0.2
    },
    "Nasi Putih (1 porsi)": {
        "kalori": 200, "protein": 4, "karbohidrat": 45, 
        "lemak": 0.4, "serat": 0.6, "gula": 0.1
    },
    "Ayam Panggang (100g)": {
        "kalori": 165, "protein": 31, "karbohidrat": 0, 
        "lemak": 3.6, "serat": 0, "gula": 0
    },
    "Ikan Salmon (100g)": {
        "kalori": 208, "protein": 20, "karbohidrat": 0, 
        "lemak": 13, "serat": 0, "gula": 0
    },
    "Pizza Slice": {
        "kalori": 285, "protein": 12, "karbohidrat": 36, 
        "lemak": 10, "serat": 2, "gula": 3
    },
    "Smoothie Buah": {
        "kalori": 120, "protein": 2, "karbohidrat": 28, 
        "lemak": 0.5, "serat": 3, "gula": 24
    },
    "Cokelat (100g)": {
        "kalori": 546, "protein": 7, "karbohidrat": 45, 
        "lemak": 31, "serat": 7, "gula": 24
    },
    "Brokoli (100g)": {
        "kalori": 34, "protein": 2.8, "karbohidrat": 7, 
        "lemak": 0.4, "serat": 2.6, "gula": 1.5
    },
    "Donat": {
        "kalori": 452, "protein": 5, "karbohidrat": 51, 
        "lemak": 25, "serat": 1.5, "gula": 19
    }
}

# ============================================================================
# FUNGSI UNTUK MEMBUAT DATA TRAINING - Generate data contoh untuk latih model
# ============================================================================

@st.cache_data  # Decorator ini bikin fungsi di-cache, jadi nggak perlu jalan berkali-kali
def create_sample_data():
    """
    Fungsi untuk load dataset dari file atau generate baru.
    Prioritas: training_data.pkl > dataset.csv > generate baru
    """
    # Cek apakah file training_data.pkl ada
    if os.path.exists("training_data.pkl"):
        try:
            with open("training_data.pkl", 'rb') as f:
                df = pickle.load(f)
            st.sidebar.success("✅ Dataset loaded dari training_data.pkl")
            return df
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error loading PKL: {e}. Menggunakan CSV...")
    
    # Cek apakah file dataset.csv ada
    if os.path.exists("dataset.csv"):
        try:
            df = pd.read_csv("dataset.csv")
            st.sidebar.success("✅ Dataset loaded dari dataset.csv")
            # Simpan juga ke PKL untuk next time
            try:
                with open("training_data.pkl", 'wb') as f:
                    pickle.dump(df, f)
            except:
                pass  # Skip kalau gagal save PKL
            return df
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error loading CSV: {e}. Generating baru...")
    
    # Kalau file tidak ada, generate baru
    st.sidebar.info("ℹ️ File dataset tidak ditemukan. Generating dataset baru...")
    
    # Set random seed biar hasil random selalu sama setiap kali jalan
    # Ini penting biar hasilnya bisa di-reproduce
    np.random.seed(42)
    
    # Data untuk makanan sehat (label = 1)
    # Makanan sehat biasanya punya kalori rendah, protein cukup, lemak rendah, serat tinggi
    healthy_data = {
        'kalori': np.random.randint(80, 300, 150),  # Kalori antara 80-300, sebanyak 150 data
        'protein': np.random.randint(5, 25, 150),  # Protein 5-25 gram
        'karbohidrat': np.random.randint(10, 40, 150),  # Karbohidrat 10-40 gram
        'lemak': np.random.randint(1, 10, 150),  # Lemak rendah, 1-10 gram
        'serat': np.random.randint(3, 15, 150),  # Serat tinggi, 3-15 gram
        'gula': np.random.randint(2, 15, 150),  # Gula rendah, 2-15 gram
        'label': [1] * 150  # Label 1 berarti sehat, ada 150 data
    }
    
    # Data untuk makanan tidak sehat (label = 0)
    # Makanan tidak sehat biasanya punya kalori tinggi, protein rendah, lemak tinggi, serat rendah
    unhealthy_data = {
        'kalori': np.random.randint(400, 800, 150),  # Kalori tinggi, 400-800
        'protein': np.random.randint(1, 8, 150),  # Protein rendah, 1-8 gram
        'karbohidrat': np.random.randint(50, 100, 150),  # Karbohidrat tinggi, 50-100 gram
        'lemak': np.random.randint(15, 40, 150),  # Lemak tinggi, 15-40 gram
        'serat': np.random.randint(0, 3, 150),  # Serat rendah, 0-3 gram
        'gula': np.random.randint(30, 60, 150),  # Gula tinggi, 30-60 gram
        'label': [0] * 150  # Label 0 berarti tidak sehat, ada 150 data
    }
    
    # Ubah dictionary jadi DataFrame (format tabel dari pandas)
    healthy_df = pd.DataFrame(healthy_data)
    unhealthy_df = pd.DataFrame(unhealthy_data)
    
    # Gabungkan data sehat dan tidak sehat jadi satu
    df = pd.concat([healthy_df, unhealthy_df], ignore_index=True)
    
    # Acak urutan data biar nggak semua sehat dulu baru tidak sehat
    # frac=1 berarti ambil semua data, random_state=42 biar hasilnya konsisten
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Simpan dataset yang baru dibuat ke file
    try:
        df.to_csv("dataset.csv", index=False)
        with open("training_data.pkl", 'wb') as f:
            pickle.dump(df, f)
        st.sidebar.success("✅ Dataset baru disimpan ke file")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Gagal menyimpan dataset: {e}")
    
    return df  # Kembalikan data yang sudah digabung dan diacak

# ============================================================================
# FUNGSI UNTUK MELATIH MODEL - Training Random Forest model
# ============================================================================

@st.cache_resource  # Cache resource ini di memory, lebih efisien untuk model ML
def train_model():
    """
    Fungsi untuk melatih model Random Forest.
    Model ini bakal belajar dari data untuk bisa bedain makanan sehat vs tidak sehat.
    """
    # Panggil fungsi buat data training
    df = create_sample_data()
    
    # Pisahkan fitur (X) dan target (y)
    # Fitur = kolom yang dipake buat prediksi (semua kecuali label)
    X = df.drop('label', axis=1)  # axis=1 berarti drop kolom, bukan baris
    
    # Target = kolom yang mau diprediksi (label: 0 atau 1)
    y = df['label']
    
    # Bagi data jadi training set (80%) dan testing set (20%)
    # Training set = data buat latih model
    # Testing set = data buat test seberapa bagus modelnya
    # stratify=y berarti proporsi label di training dan testing sama
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Buat model Random Forest
    # Random Forest = banyak pohon keputusan yang voting bareng
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Jumlah pohon keputusan (100 pohon)
        max_depth=10,  # Kedalaman maksimal pohon (biar nggak overfit)
        random_state=42,  # Random seed biar konsisten
        n_jobs=-1  # Pakai semua core CPU (-1 = semua core)
    )
    
    # Latih model dengan data training
    # Ini yang bikin model belajar pola dari data
    rf_model.fit(X_train, y_train)
    
    # Test model dengan data testing
    # Prediksi label untuk data testing
    y_pred = rf_model.predict(X_test)
    
    # Hitung akurasi model (seberapa benar prediksinya)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Kembalikan model yang sudah dilatih, nama fitur, akurasi, dan data testing
    return rf_model, X.columns, accuracy, X_test, y_test, y_pred

# ============================================================================
# FUNGSI UNTUK PREDIKSI - Prediksi apakah makanan sehat atau tidak
# ============================================================================

def predict_food(model, kalori, protein, karbohidrat, lemak, serat, gula):
    """
    Fungsi untuk prediksi apakah makanan sehat atau tidak.
    
    Parameters:
    - model: Model Random Forest yang sudah dilatih
    - kalori, protein, karbohidrat, lemak, serat, gula: Nilai nutrisi makanan
    
    Returns:
    - prediction: 0 (tidak sehat) atau 1 (sehat)
    - probability: Probabilitas untuk masing-masing kelas (array)
    """
    # Susun data nutrisi jadi array 2D (format yang diminta model)
    # [[kalori, protein, karbohidrat, lemak, serat, gula]]
    data = np.array([[kalori, protein, karbohidrat, lemak, serat, gula]])
    
    # Prediksi dengan model (hasilnya 0 atau 1)
    prediction = model.predict(data)[0]
    
    # Dapat probabilitas (seberapa yakin model dengan prediksinya)
    # Hasilnya array: [prob_tidak_sehat, prob_sehat]
    probability = model.predict_proba(data)[0]
    
    return prediction, probability

# ============================================================================
# FUNGSI UNTUK HITUNG SKOR NUTRISI - Kalkulasi skor kesehatan makanan
# ============================================================================

def calculate_nutrition_score(kalori, protein, karbohidrat, lemak, serat, gula):
    """
    Fungsi untuk hitung skor nutrisi dari 0-100.
    Skor tinggi = lebih sehat, skor rendah = kurang sehat.
    """
    score = 100  # Mulai dari 100 (perfect score)
    
    # Kurangi skor kalau kalori terlalu tinggi
    if kalori > 500:
        score -= 30  # Kalori sangat tinggi, kurangi banyak
    elif kalori > 300:
        score -= 15  # Kalori agak tinggi, kurangi sedikit
    
    # Tambah skor kalau protein tinggi (protein bagus buat tubuh)
    if protein > 20:
        score += 10  # Protein tinggi, bonus banyak
    elif protein > 10:
        score += 5  # Protein cukup, bonus sedikit
    
    # Kurangi skor kalau lemak terlalu tinggi
    if lemak > 25:
        score -= 25  # Lemak sangat tinggi, kurangi banyak
    elif lemak > 15:
        score -= 10  # Lemak agak tinggi, kurangi sedikit
    
    # Tambah skor kalau serat tinggi (serat bagus buat pencernaan)
    if serat > 5:
        score += 15  # Serat tinggi, bonus banyak
    elif serat > 3:
        score += 8  # Serat cukup, bonus sedikit
    
    # Kurangi skor kalau gula terlalu tinggi
    if gula > 30:
        score -= 30  # Gula sangat tinggi, kurangi banyak
    elif gula > 20:
        score -= 15  # Gula agak tinggi, kurangi sedikit
    
    # Pastikan skor tetap antara 0-100
    return max(0, min(100, score))

# ============================================================================
# FUNGSI DATABASE SQLITE - Setup dan operasi database
# ============================================================================

DB_NAME = "food_app.db"  # Nama file database

def init_database():
    """
    Inisialisasi database SQLite dan buat tabel jika belum ada.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Tabel untuk history prediksi
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu TEXT NOT NULL,
            nama TEXT NOT NULL,
            kalori REAL NOT NULL,
            protein REAL NOT NULL,
            karbohidrat REAL NOT NULL,
            lemak REAL NOT NULL,
            serat REAL NOT NULL,
            gula REAL NOT NULL,
            prediksi TEXT NOT NULL,
            skor REAL NOT NULL,
            keyakinan REAL NOT NULL
        )
    ''')
    
    # Tabel untuk daily nutrition tracking (per hari)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_nutrition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal TEXT NOT NULL UNIQUE,
            kalori REAL DEFAULT 0,
            protein REAL DEFAULT 0,
            karbohidrat REAL DEFAULT 0,
            lemak REAL DEFAULT 0,
            serat REAL DEFAULT 0,
            gula REAL DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

def save_prediction_history(waktu, nama, kalori, protein, karbohidrat, lemak, serat, gula, prediksi, skor, keyakinan):
    """
    Simpan history prediksi ke database.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO prediction_history 
        (waktu, nama, kalori, protein, karbohidrat, lemak, serat, gula, prediksi, skor, keyakinan)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (waktu, nama, kalori, protein, karbohidrat, lemak, serat, gula, prediksi, skor, keyakinan))
    
    conn.commit()
    conn.close()

def get_prediction_history(limit=None):
    """
    Ambil history prediksi dari database.
    limit: jumlah data yang diambil (None = semua)
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    if limit:
        cursor.execute('''
            SELECT * FROM prediction_history 
            ORDER BY waktu DESC 
            LIMIT ?
        ''', (limit,))
    else:
        cursor.execute('''
            SELECT * FROM prediction_history 
            ORDER BY waktu DESC
        ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert ke list of dictionaries
    columns = ['id', 'waktu', 'nama', 'kalori', 'protein', 'karbohidrat', 
               'lemak', 'serat', 'gula', 'prediksi', 'skor', 'keyakinan']
    history = []
    for row in rows:
        history.append(dict(zip(columns, row)))
    
    return history

def delete_all_history():
    """
    Hapus semua history prediksi dari database.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM prediction_history')
    conn.commit()
    conn.close()

def get_daily_nutrition(tanggal=None):
    """
    Ambil total nutrisi harian dari database.
    tanggal: format YYYY-MM-DD (None = hari ini)
    """
    if tanggal is None:
        tanggal = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM daily_nutrition WHERE tanggal = ?
    ''', (tanggal,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'tanggal': row[1],
            'kalori': row[2],
            'protein': row[3],
            'karbohidrat': row[4],
            'lemak': row[5],
            'serat': row[6],
            'gula': row[7]
        }
    else:
        return {
            'tanggal': tanggal,
            'kalori': 0,
            'protein': 0,
            'karbohidrat': 0,
            'lemak': 0,
            'serat': 0,
            'gula': 0
        }

def update_daily_nutrition(tanggal, kalori, protein, karbohidrat, lemak, serat, gula):
    """
    Update atau insert total nutrisi harian ke database.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Cek apakah sudah ada data untuk tanggal ini
    cursor.execute('SELECT * FROM daily_nutrition WHERE tanggal = ?', (tanggal,))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing
        cursor.execute('''
            UPDATE daily_nutrition 
            SET kalori = kalori + ?, 
                protein = protein + ?,
                karbohidrat = karbohidrat + ?,
                lemak = lemak + ?,
                serat = serat + ?,
                gula = gula + ?
            WHERE tanggal = ?
        ''', (kalori, protein, karbohidrat, lemak, serat, gula, tanggal))
    else:
        # Insert new
        cursor.execute('''
            INSERT INTO daily_nutrition 
            (tanggal, kalori, protein, karbohidrat, lemak, serat, gula)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (tanggal, kalori, protein, karbohidrat, lemak, serat, gula))
    
    conn.commit()
    conn.close()

def reset_daily_nutrition(tanggal=None):
    """
    Reset total nutrisi harian ke 0.
    """
    if tanggal is None:
        tanggal = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE daily_nutrition 
        SET kalori = 0, protein = 0, karbohidrat = 0, 
            lemak = 0, serat = 0, gula = 0
        WHERE tanggal = ?
    ''', (tanggal,))
    
    # Jika tidak ada data, insert dengan nilai 0
    if cursor.rowcount == 0:
        cursor.execute('''
            INSERT INTO daily_nutrition 
            (tanggal, kalori, protein, karbohidrat, lemak, serat, gula)
            VALUES (?, 0, 0, 0, 0, 0, 0)
        ''', (tanggal,))
    
    conn.commit()
    conn.close()

# ============================================================================
# INISIALISASI DATABASE - Setup database saat aplikasi pertama kali jalan
# ============================================================================

# Inisialisasi database (buat tabel jika belum ada)
init_database()

# ============================================================================
# INISIALISASI SESSION STATE - Setup data yang disimpan selama sesi
# ============================================================================

# Catatan: History dan daily nutrition sekarang disimpan di database SQLite
# Session state tidak lagi digunakan untuk data persisten

# ============================================================================
# HEADER HALAMAN - Judul utama aplikasi
# ============================================================================

# Tampilkan header utama dengan styling CSS yang udah dibuat
st.markdown('<div class="main-header">🍎 AI Makanan Sehat</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning dengan Random Forest Algorithm</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Menu navigasi di samping
# ============================================================================

with st.sidebar:  # Buka sidebar (panel di samping)
    st.header("📋 Menu Navigasi")
    
    # Radio button buat pilih halaman (hanya bisa pilih satu)
    page = st.radio(
        "Pilih Halaman:",
        ["🏠 Beranda", "🔮 Prediksi Makanan", "📊 Analisis Model", "📝 History Prediksi"]
    )

# ============================================================================
# TRAINING MODEL - Latih model sekali di awal (cached)
# ============================================================================

# Panggil fungsi train_model yang sudah di-cache
# Ini cuma jalan sekali, setelah itu pakai hasil yang di-cache
model, feature_names, accuracy, X_test, y_test, y_pred = train_model()

# ============================================================================
# HALAMAN BERANDA - Halaman utama aplikasi
# ============================================================================

if page == "🏠 Beranda":
    # Tampilkan 3 kolom metric (kartu angka penting)
    col1, col2, col3 = st.columns(3)  # Bagi jadi 3 kolom sama lebar
    
    with col1:
        st.metric("🎯 Akurasi Model", f"{accuracy*100:.2f}%")  # Tampilkan akurasi model
    
    with col2:
        st.metric("📊 Total Sampel", "300 makanan")  # Total data training
    
    with col3:
        st.metric("🌳 Jumlah Pohon", "100 pohon")  # Jumlah pohon di Random Forest
    
    st.markdown("---")
    
    st.markdown("""
    **Random Forest** menggunakan 100 pohon keputusan untuk klasifikasi makanan sehat.
    """)
    
    st.markdown("---")
    st.header("📋 Database Makanan")
    
    # Ambil semua nama makanan dari database
    food_options = list(FOOD_DATABASE.keys())
    
    # Dropdown buat pilih makanan
    selected_food = st.selectbox("Pilih makanan untuk melihat nutrisi:", food_options)
    
    # Kalau ada makanan yang dipilih
    if selected_food:
        # Ambil data nutrisi makanan yang dipilih
        food_data = FOOD_DATABASE[selected_food]
        
        # Tampilkan nutrisi dalam 2 kolom
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 Nutrisi {selected_food}")
            st.write(f"**Kalori:** {food_data['kalori']} kcal")
            st.write(f"**Protein:** {food_data['protein']} g")
            st.write(f"**Karbohidrat:** {food_data['karbohidrat']} g")
        
        with col2:
            st.write(f"**Lemak:** {food_data['lemak']} g")
            st.write(f"**Serat:** {food_data['serat']} g")
            st.write(f"**Gula:** {food_data['gula']} g")
        
        # Prediksi cepat apakah makanan ini sehat
        pred, prob = predict_food(
            model, food_data['kalori'], food_data['protein'],
            food_data['karbohidrat'], food_data['lemak'],
            food_data['serat'], food_data['gula']
        )
        
        # Tentukan status dan warna berdasarkan prediksi
        status = "✅ SEHAT" if pred == 1 else "⚠️ TIDAK SEHAT"
        status_color = "#2E7D32" if pred == 1 else "#D32F2F"  # Hijau kalau sehat, merah kalau tidak
        
        # Tampilkan hasil prediksi dengan styling
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: {'#C8E6C9' if pred == 1 else '#FFCDD2'}; 
                    border-left: 5px solid {status_color}; margin: 1rem 0;">
            <h3>Prediksi: {status}</h3>
            <p><strong>Tingkat Keyakinan:</strong> {prob[pred]*100:.2f}%</p>
            <p>Probabilitas Sehat: {prob[1]*100:.2f}% | Tidak Sehat: {prob[0]*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# HALAMAN PREDIKSI MAKANAN - Halaman buat prediksi makanan
# ============================================================================

elif page == "🔮 Prediksi Makanan":
    st.header("🔮 Prediksi Makanan")
    
    tab1, tab2 = st.tabs(["📝 Input Manual", "🍽️ Database"])
    
    # TAB 1: Input Manual
    with tab1:
        st.subheader("Input Nutrisi")
        
        # Bagi form jadi 2 kolom
        col1, col2 = st.columns(2)
        
        with col1:
            # Input field buat nama makanan
            nama_makanan = st.text_input("Nama Makanan", placeholder="Contoh: Nasi Goreng")
            
            # Input field buat kalori (0-2000, default 200)
            kalori = st.number_input("Kalori (kcal)", min_value=0, max_value=2000, value=200)
            
            # Input field buat protein (0-100 gram, bisa desimal)
            protein = st.number_input("Protein (gram)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
            
            # Input field buat karbohidrat
            karbohidrat = st.number_input("Karbohidrat (gram)", min_value=0.0, max_value=200.0, value=30.0, step=0.1)
        
        with col2:
            # Input field buat lemak
            lemak = st.number_input("Lemak (gram)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            
            # Input field buat serat
            serat = st.number_input("Serat (gram)", min_value=0.0, max_value=50.0, value=3.0, step=0.1)
            
            # Input field buat gula
            gula = st.number_input("Gula (gram)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        
        # Tombol buat mulai prediksi
        if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
            # Panggil fungsi prediksi
            pred, prob = predict_food(model, kalori, protein, karbohidrat, lemak, serat, gula)
            
            # Hitung skor nutrisi
            score = calculate_nutrition_score(kalori, protein, karbohidrat, lemak, serat, gula)
            
            # Tentukan status, warna, dan background berdasarkan prediksi
            status = "✅ SEHAT" if pred == 1 else "⚠️ TIDAK SEHAT"
            status_color = "#2E7D32" if pred == 1 else "#D32F2F"  # Hijau atau merah
            status_bg = "#C8E6C9" if pred == 1 else "#FFCDD2"  # Background hijau atau merah muda
            
            # Tampilkan hasil prediksi dengan box yang menarik
            st.markdown(f"""
            <div style="padding: 2rem; border-radius: 1rem; background-color: {status_bg}; 
                        border-left: 8px solid {status_color}; margin: 2rem 0;">
                <h2 style="color: {status_color};">{status}</h2>
                <p style="font-size: 1.2rem;"><strong>Tingkat Keyakinan:</strong> {prob[pred]*100:.2f}%</p>
                <p><strong>Skor Nutrisi:</strong> {score}/100</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tampilkan probabilitas dalam 2 kolom
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probabilitas Sehat", f"{prob[1]*100:.2f}%")
            with col2:
                st.metric("Probabilitas Tidak Sehat", f"{prob[0]*100:.2f}%")
            
            # Bikin grafik bar buat visualisasi probabilitas
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Tidak Sehat', 'Sehat'],  # Label sumbu X
                y=[prob[0]*100, prob[1]*100],  # Nilai sumbu Y (dalam persen)
                marker_color=['#FF5252', '#4CAF50'],  # Warna bar (merah dan hijau)
                text=[f'{prob[0]*100:.1f}%', f'{prob[1]*100:.1f}%'],  # Teks di atas bar
                textposition='auto'  # Posisi teks otomatis
            ))
            fig.update_layout(
                title="Probabilitas Prediksi",
                yaxis_title="Probabilitas (%)",
                height=300  # Tinggi grafik
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Simpan hasil prediksi ke database SQLite
            waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            nama = nama_makanan or "Makanan"
            
            save_prediction_history(
                waktu=waktu,
                nama=nama,
                kalori=kalori,
                protein=protein,
                karbohidrat=karbohidrat,
                lemak=lemak,
                serat=serat,
                gula=gula,
                prediksi=status,
                skor=score,
                keyakinan=prob[pred]*100
            )
            
            # Update total nutrisi harian di database
            tanggal = datetime.now().strftime("%Y-%m-%d")
            update_daily_nutrition(
                tanggal=tanggal,
                kalori=kalori,
                protein=protein,
                karbohidrat=karbohidrat,
                lemak=lemak,
                serat=serat,
                gula=gula
            )
            
            st.success("✅ Prediksi berhasil disimpan ke database!")
    
    # TAB 2: Pilih dari Database
    with tab2:
        st.subheader("Pilih Makanan")
        
        # Dropdown buat pilih makanan dari database
        selected_food = st.selectbox("Pilih Makanan:", list(FOOD_DATABASE.keys()))
        
        # Kalau ada makanan yang dipilih
        if selected_food:
            # Ambil data nutrisi
            food_data = FOOD_DATABASE[selected_food]
            
            # Tampilkan nutrisi dalam 3 kolom (metric cards)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Kalori", f"{food_data['kalori']} kcal")
                st.metric("Protein", f"{food_data['protein']} g")
            with col2:
                st.metric("Karbohidrat", f"{food_data['karbohidrat']} g")
                st.metric("Lemak", f"{food_data['lemak']} g")
            with col3:
                st.metric("Serat", f"{food_data['serat']} g")
                st.metric("Gula", f"{food_data['gula']} g")
            
            # Tombol buat prediksi makanan yang dipilih
            if st.button("🔮 Prediksi Makanan Ini", type="primary", use_container_width=True):
                # Prediksi dengan model
                pred, prob = predict_food(
                    model, food_data['kalori'], food_data['protein'],
                    food_data['karbohidrat'], food_data['lemak'],
                    food_data['serat'], food_data['gula']
                )
                
                # Hitung skor nutrisi
                score = calculate_nutrition_score(
                    food_data['kalori'], food_data['protein'],
                    food_data['karbohidrat'], food_data['lemak'],
                    food_data['serat'], food_data['gula']
                )
                
                # Tentukan status dan styling
                status = "✅ SEHAT" if pred == 1 else "⚠️ TIDAK SEHAT"
                status_color = "#2E7D32" if pred == 1 else "#D32F2F"
                status_bg = "#C8E6C9" if pred == 1 else "#FFCDD2"
                
                # Tampilkan hasil prediksi
                st.markdown(f"""
                <div style="padding: 2rem; border-radius: 1rem; background-color: {status_bg}; 
                            border-left: 8px solid {status_color}; margin: 2rem 0;">
                    <h2 style="color: {status_color};">{status}</h2>
                    <p style="font-size: 1.2rem;"><strong>Tingkat Keyakinan:</strong> {prob[pred]*100:.2f}%</p>
                    <p><strong>Skor Nutrisi:</strong> {score}/100</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# HALAMAN ANALISIS MODEL - Lihat performa dan detail model ML
# ============================================================================

elif page == "📊 Analisis Model":
    st.header("📊 Analisis Model Machine Learning")
    
    # Tampilkan 4 metric penting tentang model
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Akurasi", f"{accuracy*100:.2f}%")  # Seberapa akurat modelnya
    with col2:
        st.metric("Training Samples", "240")  # Jumlah data training
    with col3:
        st.metric("Test Samples", "60")  # Jumlah data testing
    with col4:
        st.metric("Features", "6")  # Jumlah fitur (kalori, protein, dll)
    
    st.markdown("---")
    
    # FLOWCHART RANDOM FOREST - Visualisasi bagaimana 100 pohon bekerja
    st.subheader("🌳 Flowchart: Bagaimana 100 Pohon Bekerja")
    
    # Buat flowchart sederhana menggunakan HTML/CSS
    st.markdown("""
    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <div style="text-align: center;">
            <div style="background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; margin: 10px; display: inline-block; width: 200px;">
                <strong>Data Input</strong><br>
                (Kalori, Protein, dll)
            </div>
            <div style="margin: 20px 0; font-size: 30px;">⬇️</div>
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 10px;">
                <div style="background-color: #2196F3; color: white; padding: 10px; border-radius: 5px; width: 80px; font-size: 12px;">
                    Pohon 1
                </div>
                <div style="background-color: #2196F3; color: white; padding: 10px; border-radius: 5px; width: 80px; font-size: 12px;">
                    Pohon 2
                </div>
                <div style="background-color: #2196F3; color: white; padding: 10px; border-radius: 5px; width: 80px; font-size: 12px;">
                    ...
                </div>
                <div style="background-color: #2196F3; color: white; padding: 10px; border-radius: 5px; width: 80px; font-size: 12px;">
                    Pohon 100
                </div>
            </div>
            <div style="margin: 20px 0; font-size: 30px;">⬇️</div>
            <div style="background-color: #FF9800; color: white; padding: 15px; border-radius: 5px; margin: 10px; display: inline-block; width: 200px;">
                <strong>Voting</strong><br>
                (Mayoritas suara)
            </div>
            <div style="margin: 20px 0; font-size: 30px;">⬇️</div>
            <div style="background-color: #9C27B0; color: white; padding: 15px; border-radius: 5px; margin: 10px; display: inline-block; width: 200px;">
                <strong>Hasil Prediksi</strong><br>
                Sehat / Tidak Sehat
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Cara Kerja:**
    1. Data nutrisi masuk ke 100 pohon keputusan
    2. Setiap pohon memberikan prediksi (Sehat/Tidak Sehat)
    3. Voting: mayoritas suara menentukan hasil akhir
    4. Hasil lebih akurat karena menggunakan banyak pohon
    """)
    
    st.markdown("---")
    
    # CONFUSION MATRIX - Tabel yang nunjukkin seberapa benar prediksi model
    st.subheader("📈 Confusion Matrix")
    
    # Hitung confusion matrix dari data testing
    cm = confusion_matrix(y_test, y_pred)
    
    # Bikin grafik heatmap buat confusion matrix
    fig_cm = px.imshow(
        cm,  # Data confusion matrix
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),  # Label sumbu
        x=['Tidak Sehat', 'Sehat'],  # Label kolom
        y=['Tidak Sehat', 'Sehat'],  # Label baris
        text_auto=True,  # Tampilkan angka di setiap sel
        color_continuous_scale='Blues'  # Skala warna biru
    )
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # FEATURE IMPORTANCE - Fitur mana yang paling penting buat prediksi
    st.subheader("🎯 Feature Importance")
    
    # Buat DataFrame dari feature importance
    feature_importance = pd.DataFrame({
        'Fitur': feature_names,  # Nama fitur
        'Importance': model.feature_importances_  # Nilai importance dari model
    }).sort_values('Importance', ascending=True)  # Urutkan dari terkecil ke terbesar
    
    # Bikin grafik bar horizontal
    fig_importance = px.bar(
        feature_importance,
        x='Importance',  # Sumbu X = nilai importance
        y='Fitur',  # Sumbu Y = nama fitur
        orientation='h',  # Horizontal bar chart
        color='Importance',  # Warna berdasarkan nilai importance
        color_continuous_scale='Greens',  # Skala warna hijau
        title="Pengaruh Fitur terhadap Prediksi"
    )
    fig_importance.update_layout(height=400)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # PARAMETER MODEL
    st.subheader("⚙️ Parameter Model")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**N Estimators:** 100 pohon")
        st.write("**Max Depth:** 10")
    with col2:
        st.write("**Random State:** 42")
        st.write("**Split:** 80% Training, 20% Testing")
    

# ============================================================================
# HALAMAN HISTORY PREDIKSI - Riwayat semua prediksi yang pernah dilakukan
# ============================================================================

elif page == "📝 History Prediksi":
    st.header("📝 History Prediksi")
    
    # Ambil history dari database SQLite
    history = get_prediction_history()
    
    # Cek apakah ada history atau belum
    if history:
        # Tampilkan jumlah total prediksi
        st.subheader(f"Total: {len(history)} Prediksi")
        
        # Buat DataFrame dari history (exclude kolom 'id')
        history_df = pd.DataFrame(history)
        if 'id' in history_df.columns:
            history_df = history_df.drop('id', axis=1)
        
        # Tampilkan tabel history
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Bikin grafik trend skor nutrisi kalau ada lebih dari 1 prediksi
        if len(history) > 1:
            st.subheader("📊 Grafik History")
            
            # Bikin grafik line chart
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                x=list(range(len(history))),  # Sumbu X = urutan prediksi (0, 1, 2, ...)
                y=[h['skor'] for h in history],  # Sumbu Y = skor nutrisi setiap prediksi
                mode='lines+markers',  # Tampilkan garis dan titik
                name='Skor Nutrisi',
                line=dict(color='#4CAF50', width=3)  # Garis hijau tebal
            ))
            fig_history.update_layout(
                title="Trend Skor Nutrisi",
                xaxis_title="Urutan Prediksi",
                yaxis_title="Skor Nutrisi",
                height=400
            )
            st.plotly_chart(fig_history, use_container_width=True)
        
        # Tombol buat hapus semua history
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Hapus Semua History", type="secondary", use_container_width=True):
                delete_all_history()
                st.success("✅ Semua history berhasil dihapus!")
                st.rerun()  # Refresh halaman
        
        with col2:
            # Export ke CSV
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"history_prediksi_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        # Kalau belum ada history, tampilkan pesan
        st.info("Belum ada history prediksi. Silakan lakukan prediksi di halaman 'Prediksi Makanan'")

# ============================================================================
# FOOTER - Bagian bawah halaman
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #757575; padding: 2rem;">
    <p>🍎 AI Makanan Sehat - Machine Learning dengan Random Forest</p>
    <p>Dibuat dengan ❤️ menggunakan Streamlit</p>
</div>
""", unsafe_allow_html=True)
