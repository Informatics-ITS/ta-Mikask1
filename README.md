# 🏁 Tugas Akhir (TA) - Final Project

**Nama Mahasiswa**: Darren Prasetya<br/>
**NRP**: 5025211162<br/>
**Judul TA**: PERBANDINGAN KINERJA SISTEM TANYA JAWAB HUKUM INDONESIA DENGAN LLM OPEN-SOURCE BERBASIS RAG<br/>
**Dosen Pembimbing**: Dini Adni Navastara, S.Kom., M.Sc.<br/>
**Dosen Ko-pembimbing**: Prof. Dr. Ir. Diana Purwitasari, S.Kom., M.Sc.<br/>

---

## 📺 Demo Aplikasi  
Embed video demo di bawah ini (ganti `VIDEO_ID` dengan ID video YouTube Anda):  

[![Demo Aplikasi](https://i.ytimg.com/vi/zIfRMTxRaIs/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)  
*Klik gambar di atas untuk menonton demo*

---

## 🛠 Panduan Instalasi & Menjalankan Software  

### Prasyarat  
- Daftar dependensi:
   - Python 3.10
   - pandas
   - python-dotenv
   - transformers
   - accelerate
   - langchain
   - typing_extensions
   - pydantic
   - langgraph
   - langchain-huggingface
   - langchain_community
   - sentence-transformers
   - tavily-python
   - pinecone
   - pinecone-client
   - requests
   - tqdm
   - pymupdf4llm
   - beautifulsoup4
   - streamlit
   - numpy 1.26.2

### Langkah-langkah menggunakan model
1. **Clone Repository**  
   ```bash
   git clone https://github.com/Informatics-ITS/ta-Mikask1
   ```
2. **Instalasi Dependensi**
   ```bash
   cd [folder-proyek]
   python -m venv venv
   ./venv/Scripts/activate
   pip install -r requirements.txt
   ```
3. **Konfigurasi**
- Salin/rename file .env.example menjadi .env
- Isi variabel lingkungan
- Pada app.py, tentukan model yang digunakan dengan mengubah MODEL_NAME
4. **Jalankan Aplikasi**
   ```bash
   streamlit run app.py
   ```
5. Buka browser dan kunjungi: `http://localhost:8501`

### Langkah-langkah update Knowledge Base
1. **Clone Repository**  
   ```bash
   git clone https://github.com/Informatics-ITS/ta-Mikask1
   ```
2. **Instalasi Dependensi**
   ```bash
   cd [folder-proyek]
   python -m venv venv
   ./venv/Scripts/activate
   pip install -r requirements.txt
   ```
3. **Konfigurasi**
- Salin/rename file .env.example menjadi .env
- Isi variabel lingkungan
4. **Jalankan Update Script**
   ```bash
   cd "Update KB"
   python main.py
   ```
---

## ⁉️ Pertanyaan?

Hubungi:
- Penulis: 5025211162@student.its.ac.id
- Pembimbing Utama: dini_navastara@if.its.ac.id
