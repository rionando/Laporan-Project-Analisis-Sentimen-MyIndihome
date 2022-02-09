# Laporan Proyek Analisis Sentimen Aplikasi MyIndohome 
- Muhammad Rionando D
## Perbandingan Analisis Sentimen Review Aplikasi MyIndihome Menggunakan Algoritma Support Vector Machine (SVM) dan Logistic Regression  

### Project Intern Telkom REG 4
![alternate text](https://github.com/rionando/Laporan-Project-Analisis-Sentimen-MyIndihome/blob/main/Telkom%20Semarang.jpg)

## Domain Proyek
Domain yang akan digunakan dalam proyek ini adalah di bidang Telekomunikasi dan layanan internet. PT Telkom Indonesia (Persero) Tbk, biasa disebut Telkom Indonesia atau Telkom saja adalah perusahaan informasi dan komunikasi serta penyedia jasa dan jaringan telekomunikasi secara lengkap di Indonesia. Telkom mengklaim sebagai perusahaan telekomunikasi terbesar di Indonesia. Salah produk layanannya adalah Indihome. yang merupakan paket layanan komunikasi dan data seperti telepon rumah (voice), internet (Internet on Fiber atau High Speed Internet), dan layanan televisi interaktif (UseeTV Cable, IPTV). Untuk mempermudah penggunaanya Telkom meluncurkan aplikasi MyIndihome akan tetapi banyak komplain dari pengguna mengenai aplikasi tersebut. sedangkan sentiment analysis adalah proses menganalisis tulisan online untuk menentukan nada emosional dari penulisnya

## Business Understanding
Pada kasus ini saya ingin melakukan analisis sentimen terhadap review aplikasi MyIndihome yang terdapat pada PlayStore yang telah di unduh lebih dari 5juta kali dan mendapatkan ratings rata rata 3/5 dari 130.706 pengguna dan bayak yang memberikan rating 1 atas aplikasi ini.
![alternate text](https://github.com/rionando/Laporan-Project-Analisis-Sentimen-MyIndihome/blob/main/appmyindihome.jpeg)


### Problem Statements
1. Banyak review negatif dan ratings yang kecil pada aplikasi myIndihome yang terdapat pada playstore
2. Dari 2 model yaitu SVM dan logistics Regression mana yang memiliki akurasi paling tinggi

### Goals
1. Mengetahui hasil analisis sentimen data review pengguna aplikasi myIndiHome yang ada pada playstore
2. Membuat model machine learning terbaik untuk memprediksi kasus ini

### Solution statements
Pada kasus kali ini saya menggunakan 2 model machine learning yaitu:
- **SVM**. Support Vector Machine (SVM) merupakan salah satu metode dalam supervised learning yang biasanya digunakan untuk klasifikasi (seperti Support Vector Classification) dan regresi (Support Vector Regression). Dalam pemodelan klasifikasi, SVM memiliki konsep yang lebih matang dan lebih jelas secara matematis dibandingkan dengan teknik-teknik klasifikasi lainnya.
- **Logistics Regression**. Regresi logistik (kadang disebut model logistik atau model logit), dalam statistika digunakan untuk prediksi probabilitas kejadian suatu peristiwa dengan mencocokkan data pada fungsi logit kurva logistik..

## Data Understanding
Dataset yang saya gunakan pada kasus ini bersumber dari kaggle [Bank Loan Modelling](https://www.kaggle.com/itsmesunil/bank-loan-modelling/code) yang memiliki dimensi 5000 X 14 variabel-variabelnya antaralain:

- id : ID pelanggan
- age : Usia
- experience : Pengalaman Kerja
- income : Pendapatan 
- zip_code : Kode Pos
- family : Status Keluarga
- ccavg : Rata-rata pengeluaran kartu kredit
- education : Pendidikan
- mortgage : Tanggungan KPR.
- personal_loan : Apakah pelanggan menerima tawaran campaing sebelumnya
- securities_account : Apakah memiliki akun pengamab?
- cd_account : Apakah Pelanggan memiliki deposio?
- online : Apakah Menggunakan Internet Banking?
- creditcard : apakah menggunakan credit card?

### Data Visualization
![alternate text](https://github.com/rionando/MLT-1/blob/main/image%201.jpg?raw=true)
1. variabel **'age'** dan **'experience'** berdistribusi normal
2. varibael **'income''CCAvg' dan 'mortgage'** memiliki kemiringan positif
3. variabel **ZIP code** memiliki kemiringan negatif

![alternate text](https://github.com/rionando/MLT-1/blob/main/image%202.jpg?raw=true)
1. Kebanyakan pelanggan tidak memiliki **Securities Account, CD Account dan CreditCard**
2. Kebanyakan pelanggan menggunakan **internet banking**
3. Persebaran **tipe keluarga** di dominasi tipe 1 dan 2

![alternate text](https://github.com/rionando/MLT-1/blob/main/image%203.jpg?raw=true)
1. Variable **'age'** beridistribusi normal dengan rata rata usia antara 30-60 tahun
2. Variabel **'Experience'** berdistribusi normal dan banyak customer yang memiliki pengalaman mulai 8 tahun. 
3. Variabel **Income** memiliki kemiringan positif. mayoritas pelanggan memiliki pemasukan antara 45K-55K. 
4. Variabel **CCAvg** juga memiliki kemiringan positif dan rata rata pengeluaran antara 0K-10K dan mayoritas kurang dari 2.5K.
5. Variabel **Mortgage** 70% pelanggan memiliki mortgage kurang dari 40K. Namun nilai maksimalnya adalah 635K.


![alternate text](https://github.com/rionando/MLT-1/blob/main/image%204.jpg?raw=true)
1. Variabel **Income dan CCAvg** memiliki korelasi yang cukup tinggi
2. Variabel **Age dan Experience** memiliki korelasi yang sangat tinggi

## Data Preparation
### Melakukan Split dataset menjadi dua bagian dengan rasio 80% untuk train set dan 20% untuk test set

Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. Ketahuilah bahwa setiap transformasi yang kita lakukan pada data sebelum pemodelan juga merupakan bagian dari model. Karena data uji (test set) berperan sebagai data baru, kita perlu melakukan semua proses transformasi dalam data latih. Inilah alasan mengapa langkah awal adalah membagi dataset sebelum melakukan transformasi apa pun. Tujuannya adalah agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. 

### Melakukan standarisasi data

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn, StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1. Untuk menghindari kebocoran informasi pada data uji, kita akan menerapkan fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi, kita akan melakukan standarisasi pada data uji. Untuk lebih jelasnya, mari kita terapkan StandardScaler pada data. 

## Modeling
Pada Kasus ini saya menggunakan 3 model machine learning yaitu Decision tree, Random forest dan KNN dan metric yang saya gunakan kali ini hanyalah Accuracy

### Decision Tree
Cara kerja model decision tree dimulai dengan satu node atau simpul. Kemudian, node tersebut bercabang untuk menyatakan pilihan-pilihan yang ada. Selanjutnya, setiap cabang tersebut akan memiliki cabang-cabang baru. Oleh karenanya, metode ini disebut 'tree' karena bentuknya menyerupai pohon yang memiliki banyak cabang.

Untuk Hasil Accuracynya
![alternate text](https://github.com/rionando/MLT-1/blob/main/iamge%205.jpg?raw=true)

### Random Forest
Random forest termasuk ke dalam kelompok model ensemble (group). Apa itu model ensemble? Sederhananya, ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. 

Untuk Hasil Accuracynya
![alternate text](https://github.com/rionando/MLT-1/blob/main/iamge%206.jpg?raw=true)


### KNN
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat. Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi. Pada modul ini, kita akan menggunakannya untuk kasus regresi.

Untuk Hasil Accuracynya
![alternate text](https://github.com/rionando/MLT-1/blob/main/image%207.jpg?raw=true)

## Evaluation
### Confusion Matrics
Confusion matrix terdapat 4 hasil kesimpulan yang bisa kita ambil yaitu
- True Positive (TP) :
Interpretasi: Anda memprediksi positif dan itu benar.
- True Negative (TN):
Interpretasi: Anda memprediksi negatif dan itu benar.
- False Positive (FP): (Kesalahan Tipe 1)
Interpretasi: Anda memprediksi positif dan itu salah.
- False Negative (FN): (Kesalahan Tipe 2, kesalahan tipe 2 ini sangat berbahaya)
Interpretasi: Anda memprediksi negatif dan itu salah.

### Metrik evaluasi
Pada kasus klasifikasi kali ini saya hanya menggunakan 1 metrik yaitu akurasi, alasan saya menggunkan akurasi adalah karena akurasi merupakan metrik klasifikasi klasik. dan cukup mudah untuk dipahami serta cocok untuk masalah klasifikasi multiclass.
Rumus dari akurasi sendiri adalah sebagai berikut:
Accuracy = (TP+TN)/(TP+FP+FN+TN)
Akurasi adalah proporsi hasil yang benar di antara jumlah total kasus yang diperiksa.

Berikut merupakan perbandingan hasil akurasi dari 3 model yang digunakan
![alternate text](https://github.com/rionando/MLT-1/blob/main/image%208.jpg?raw=true)

dari hasil tersebut model Random forest memiliki tingkat akurasi paling tinggi
