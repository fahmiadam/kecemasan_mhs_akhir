KLASIFIKASI TINGKAT KECEMASAN MASUK DUNIA KERJA 
PADA MAHASISWA TINGKAT AKHIR MENGGUNAKAN 
METODE NEIGHBOR WEIGHTED K- NEAREST NEIGHBOR
========================
Kecemasan menghadapi dunia kerja adalah perasaan khawatir yang dialami seseorang ketika menghadapi atau memasuki dunia kerja. Salah satunya Mahasiswa tingkat akhir. Banyaknya kendala yang dihadapi mahasiswa tingkat akhir memunculkan gejala dan gangguan psikologis seperti stres, kesulitan tidur, mudah marah, frustasi, dan kehilangan motivasi. Maka dari itu, dibutuhkan suatu cara untuk deteksi dini dan mengklasifikasikan kecemasan mahasiswa tingkat akhir. Salah satu algoritma machine learning yang umum digunakan dalam pengklasifikasian diagnosis medis seperti kecemasan adalah NWKNN. Metode NWKNN merupakan metode perkembangan dari metode KNN, yang membedakan adalah pada NWKNN terdapat proses pembobotan terhadap setiap jenis yang akan di klasifikasikan. Penelitian ini akan dilakukan klasifikasi kecemasan menghadapi dunia kerja berdasarkan gejala yang muncul menggunakan metode klasifikasi Neighbor Weigted K-Nearest Neighbor (NWKNN). Ada empat proses utama dalam algoritma NWKNN, proses-proses tersebut yaitu proses menghitung kedekatan ketetanggaan menggunakan eucliedean distance, mengurutkan nilai eucliedean distance, pembobotan setiap jenis gejala dan menghitung skor di setiap jenis sesuai nilai K. Proses pertama yaitu menghitung tetangga terdekat eucliedean distance untuk mengetahui data mana yang masuk klasifikasi kelas yang dicari. Kedua yaitu mengurutkan nilai hasil eucliedean distance sesuai dengan terkecil ke terbesar. Ketiga yaitu proses pembobotan tiap kelas kecemasan, proses ini berfungsi untuk memberikan bobot pada tiap kelas kecemasan dengan bobot terbesar diberikan pada kelas dengan jumlah paling sedikit dan bobot kecil diberikan pada kelas paling banyak. Terakhir yaitu proses perhitungan nilai skor hasil klasifikasi, proses ini berfungsi untuk mengetahui hasil klasifikasi. Nilai skor terbesar akan menjadi hasil klasifikasi. Peneltian ini menggunakan metodologi penelitian prototype. Pada penelitian ini akan dilakukan klasifikasi yang terdiri atas 3 kelas kecemasan meliputi kecemasan rendah, kecemasan sedang dan kecemasan tinggi. Penelitian ini menggunakan 1009 data yang didapat melalui kuesioner. Hasil dari penelitian ini menunjukkan bahwa metode NWKNN dapat melakukan klasifikasi kecemasan menghadapi dunia kerja pada mahasiswa tingkat akhir dengan baik ketika rasio 90:10. 90% menjadi data latih atau sebanyak 909 data dan 10% menjadi data uji atau sebanyak 100 data. Dari hasil pengujian didapatkan akurasi terbaik ketika nilai K=20, dan nilai E=4 dengan hasil akurasinya mencapai 94% dengan rata-rata akurasi sebesar 91%

Kata kunci: Mahasiswa, Kecemasan, Dunia Kerja, NWKNN, Pembobotan

Ini merupakan hasil proyek Tugas Akhir yang dibuat guna menyelesaikan pendidikan S-1

Analisis Kebutuhan
------------------
Python 3.1.2

Numpy 1.26.2

pandas 2.1.4

streamlit 1.29.0

scikit-learn 1.3.2

Demo Program
------------
https://kecemasanmhsakhir-8yw629ggsaqfadxfnv2e2g.streamlit.app/
