# Laporan Proyek Analisis Sentimen Aplikasi MyIndohome - Muhammad Rionando D
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
2. Membuat model machine learning terbaik untuk analisis sentimen pada kasus ini

### Solution statements
Pada kasus kali ini saya menggunakan 2 model machine learning yaitu:
- **SVM**. Support Vector Machine (SVM) merupakan salah satu metode dalam supervised learning yang biasanya digunakan untuk klasifikasi (seperti Support Vector Classification) dan regresi (Support Vector Regression). Dalam pemodelan klasifikasi, SVM memiliki konsep yang lebih matang dan lebih jelas secara matematis dibandingkan dengan teknik-teknik klasifikasi lainnya.
- **Logistics Regression**. Regresi logistik (kadang disebut model logistik atau model logit), dalam statistika digunakan untuk prediksi probabilitas kejadian suatu peristiwa dengan mencocokkan data pada fungsi logit kurva logistik.

## Data Scraping
Dataset yang saya gunakan pada kasus ini bersumber dari playsotre review aplikasi myIndihome yang saya dapatkan dari scraping menggunakan library google-play-scraper yang terdapat pada python dan di run menggunakan google colab berikut merupakan link [notebooknya](https://github.com/rionando/Scraping-Data-Playstore/blob/main/Scrapping_MyIndihome.ipynb) 
![alternate text](https://github.com/rionando/Laporan-Project-Analisis-Sentimen-MyIndihome/blob/main/data%20scraping%201.jpg)
![alternate text](https://github.com/rionando/Laporan-Project-Analisis-Sentimen-MyIndihome/blob/main/data%20scraping%202.jpg)

## Data Understanding
Jumlah data yang berhasil di dapat dari scraping adalah 65066

![alternate text](https://github.com/rionando/Laporan-Project-Analisis-Sentimen-MyIndihome/blob/main/jumlah%20data.jpg)

berikut merupakan variable yang akan kita gunakan dalam proses analisis sentimen
![alternate text](https://github.com/rionando/Laporan-Project-Analisis-Sentimen-MyIndihome/blob/main/data%20understanding.jpg)

- Username : Username Pengguna
- score : ratings
- at : waktu review
- content : review pengguna

## Data Preparation
### Pelabelan
Melakukan pelabelan terhadap komen review oleh pengguna berdasarkan nilai ratings yang mereka berikan disini saya mebaginya menjadi 2 yaitu ratings 1-3 artinya buruk atau negatif dan ratings 4-5 artinya positif atau bagus 
![alternate text](https://github.com/rionando/Laporan-Project-Analisis-Sentimen-MyIndihome/blob/main/pelabelan.jpg)

### Cleansing dan penggunaan stopword
Cleansing merupakan step data yang akan diolah harus dibersihkan terlebih dahulu. Proses cleansing dalam text mining bisa berupa menghapus tanda baca seperti koma, titik, tanda seru dan lain lain. Selain itu juga jika diperlukan stopwords, stemming, lemmatization, dan top frequent dan low frequent.
![alternate text](https://github.com/rionando/Laporan-Project-Analisis-Sentimen-MyIndihome/blob/main/preprosses.jpg)

## Modeling
Pada Kasus ini saya menggunakan 2 model machine learning yaitu Support Vector Machine (SVM) dan Logistics Regression serta metric yang saya gunakan kali ini hanyalah Accuracy

### SVM
Cara kerja dari metode Support Vector Machine khususnya pada masalah non-linear adalah dengan memasukkan konsep kernel ke dalam ruang berdimensi tinggi. Tujuannya adalah untuk mencari hyperplane atau pemisah yang dapat memaksimalkan jarak (margin) antar kelas data.

Untuk Hasil Accuracynya
![alternate text](https://github.com/rionando/Laporan-Project-Analisis-Sentimen-MyIndihome/blob/main/acc%20svm.jpg)

### Logistics Regression
Logistic Regression adalah sebuah algoritma klasifikasi untuk mencari hubungan antara fitur (input) diskrit/kontinu dengan probabilitas hasil output diskrit tertentu.

Untuk Hasil Accuracynya.
![alternate text](https://github.com/rionando/Laporan-Project-Analisis-Sentimen-MyIndihome/blob/main/acc%20logistics%20regression.jpg)

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
Pada kasus kali ini saya hanya menggunakan 1 metrik yaitu akurasi, alasan saya menggunkan akurasi adalah karena akurasi merupakan metrik klasik. dan cukup mudah untuk dipahami serta cocok untuk mencari model terbaik
Rumus dari akurasi sendiri adalah sebagai berikut:
Accuracy = (TP+TN)/(TP+FP+FN+TN)
Akurasi adalah proporsi hasil yang benar di antara jumlah total kasus yang diperiksa.

dari model svm dan logistics reggression sebenarnya memiliki hasil yang tidak jauh beda yaitu

SVM                   : 0.9580766461316905

Logistics Regression  : 0.9583166653332267

dari hasil tersebut model Logistics Regression memiliki tingkat akurasi paling tinggi
