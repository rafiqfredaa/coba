# Laporan Proyek Pertama Modul Machine Learning Terapan

## Domain Proyek
Domain proyek yang dipilih untuk proyek pertama machine learning terapan adalah mengenai ekonomi dengan membahas tentang harga emas yang dilakukan adalah memprediksi harga emas.

- Latar Belakang

Dalam catatan sejarah, emas telah digunakan sebagai mata uang di berbagai belahan dunia. Saat ini, logam mulia seperti emas dipegang oleh bank sentral di semua negara untuk menjamin pembayaran kembali utang luar negeri, dan juga untuk mengendalikan inflasi yang mencerminkan kekuatan keuangan negara. Baru-baru ini, negara berkembang seperti Cina, Rusia, dan India menjadi pembeli emas yang besar, sedangkan Amerika Serikat, Afrika Selatan, dan Australia termasuk di antara penjual emas yang besar.

Memprediksi kenaikan dan penurunan harga emas harian dapat membantu investor memutuskan kapan harus membeli atau menjual komoditas tersebut. tetapi harga emas bergantung pada banyak faktor seperti harga logam mulia lainnya seperti harga minyak mentah, kinerja bursa saham, harga obligasi, nilai tukar mata uang, dan sebagainya. 

Tantangan pada proyek ini adalah untuk secara akurat memprediksi harga penutupan ETF emas yang disesuaikan di masa depan selama periode waktu tertentu di masa mendatang. Masalahnya adalah masalah regresi, karena nilai output yang merupakan harga penutupan yang disesuaikan dalam proyek ini adalah nilai kontinu.

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68459186/138650396-7b5242eb-4287-4b70-af33-ab6465db02e1.png">
</p>

## Business Understanding

### Problem

Dari latar belakang masalah di atas, berikut merupakan rumusan masalah yang didapatkan :
- Dari banyaknya fitur yang ada fitur manakah yang memiliki hubungan atau pengaruh terhadap data harga penjualan emas?
- Bagaimana cara pemrosesan data yang dapat dilakukan pada data harga penjualan emas?
- Bagaimana membuat model machine learning yang mampu memprediksi harga penjualan emas secara akurat?

### Goals
Berikut ini adalah tujuan yang akan dicapai :
- Memilih fitur-fitur yang memiliki hubungan atau pengaruh terhadap data harga penjualan emas.
- Melakukan pemrosesan terhadap data harga penjualan emas.
- Membuat model machine learning terbaik untuk memprediksi harga penjualan emas. 

### Solution Statements
Berikut ini adalah solusi yang mungkin dapat dilakukan :
- Melihat persebaran data pada data penjualan emas dan memilih variabel utama yang berhubungan langsung dengan harga penjualan emas.
- Pemrosesan terhadap data penjualan emas yang dapat dilakukan antara lain, melihat apakah ada data yang hilang/kosong, memvisualisasikan data, melakukan beberapa perhitungan (MACD, RSI, SMA, dan Bollinger Bands), normalisasi, encoding fitur dan membagi data menjadi data latih, data validasi dan data uji. 
- Membuat beberapa algoritma model seperti, 
  - Decision Tree Regression, metode pembelajaran terawasi non-parametrik yang digunakan untuk klasifikasi dan regresi. Tujuannya adalah untuk membuat model yang memprediksi nilai variabel target dengan mempelajari aturan keputusan sederhana yang disimpulkan dari fitur data. Sebuah pohon dapat dilihat sebagai pendekatan konstan sepotong demi sepotong. 
    - Kelebihan, mudah disiapkan, lebih sedikit pembersihan data yang diperlukan dan mudah dibaca dan ditafsirkan.   
    - Kekurangan, overfitting ketika kelas dan kriteria yang digunakan sangat banyak dan kurang efektif dalam memprediksi hasil dari variabel kontinu karena pohon keputusan cenderung kehilangan informasi saat mengkategorikan variabel ke dalam beberapa kategori.    
  - Support Vector Regressor, penerapan SVM yang digunakan untuk kasus regresi yang outputnya berupa bilangan riil atau kontinu. Fungsi regresi dengan batasan deviasi tertentu sehingga dapat menghasilkan prediksi yang mendekati nilai aktual.
    - Kelebihan, cocok untuk outlier, model keputusan dapat dengan mudah diperbarui, memiliki kemampuan generalisasi yang baik dan akurasi prediksi yang tinggi, dan implementasinya mudah.
    - Kekurangan, kurang cocok untuk kumpulan data besar, jika jumlah fitur untuk setiap titik data melebihi jumlah sampel data pelatihan-SVM akan berperforma buruk, dan model keputusan tidak berkinerja sangat baik ketika kumpulan data memiliki lebih banyak noise, yaitu kelas target tumpang tindih.
  - Random Forest, Pendekatan ensemble untuk menemukan pohon keputusan yang paling sesuai dengan data pelatihan dengan membuat banyak pohon keputusan dan kemudian menentukan yang "rata-rata". Bagian "acak" dari istilah tersebut mengacu pada pembangunan masing-masing pohon keputusan dari pilihan fitur secara acak; "hutan" mengacu pada kumpulan pohon keputusan.
    - Kelebihan, serbaguna dapat digunakan untuk tugas regresi dan klasifikasi, mengotomatiskan nilai-nilai hilang yang ada dalam data, normalisasi data tidak diperlukan karena menggunakan pendekatan berbasis aturan dan bekerja baik dengan nilai kategoris dan berkelanjutan.
    - Kekurangan, sejumlah besar pohon dapat membuat algoritma terlalu lambat dan tidak efektif untuk prediksi waktu nyata, cukup lambat untuk membuat prediksi setelah dilatih, dan membutuhkan banyak daya komputasi serta sumber daya karena membangun banyak pohon untuk digabungkan. 
  - LassoCV, (Least Absolute Shrinkage and Selection Operator) melakukan regularisasi L1, yaitu menambahkan faktor penjumlahan nilai mutlak koefisien dalam tujuan optimasi dan meminimalkan fungsi tujuan dengan menambahkan penalti ke jumlah nilai absolut koefisien atau biasa dikenal dengan metode deviasi absolut terkecil. 
    - Kelebihan, dapat menghindari overfitting, dapat diterapkan bahkan ketika jumlah fitur lebih besar dari jumlah data, dapat melakukan seleksi fitur dan cepat dalam hal inferensi dan fitting.
    - Kekurangan, model yang dipilih oleh lasso tidak stabil, hasil pemilihan model tidak intuitif untuk ditafsirkan.
  - RidgeCV, melakukan regularisasi L2 dan meminimalkan fungsi tujuan dengan menambahkan penalti ke jumlah kuadrat koefisien.
    - Kelebihan, dapat menghindari overfitting, bekerja dengan baik bahkan di hadapan fitur yang sangat berkorelasi karena akan mencakup semuanya dalam model tetapi koefisien akan didistribusikan di antara mereka tergantung pada korelasinya.
    - Kekurangan, memasukkan semua prediktor dalam model akhir, tidak dapat melakukan pemilihan fitur dan mengecilkan koefisien menuju nol.
  - Gradient Boosting Regressor, termasuk dalam algoritma ensemble yang menggunakan peningkatan akurasi predictor. Gradient boost membangun tree dengan 8 sampai 32 daun, menggunakan boosting untuk proses pengoptimalan dengan menggunakan loss function untuk meminimalisir kesalahan, dan cara kerja algoritma gradient boost adalah membangun satu tree untuk menyesuaikan data, lalu tree berikutnya dibangun untuk mengurangi residual (error).
    - Kelebihan, fleksibilitas, dapat mengoptimalkan fungsi kerugian yang berbeda dan menyediakan beberapa opsi penyetelan hyperparameter yang membuat fungsi tersebut sangat fleksibel, tidak diperlukan pra-pemrosesan data yang sering berfungsi dengan baik dengan nilai kategori dan numerik apa adanya.
    - Kekurangan, komputasi mahal sering membutuhkan banyak pohon yang dapat menghabiskan waktu dan memori, membutuhkan pencarian grid yang besar selama penyetelan, dan GB akan terus ditingkatkan untuk meminimalkan semua kesalahan dapat terlalu menekankan outlier dan menyebabkan overfitting.
  - Stochastic Gradient Descent, Algoritma penurunan gradien di mana ukuran batch adalah satu. Dengan kata lain, SGD bergantung pada satu contoh yang dipilih secara seragam secara acak dari kumpulan data untuk menghitung perkiraan gradien pada setiap langkah.
    - Kelebihan, secara komputasi cepat, hemat memori karena mempertimbangkan satu pengamatan pada satu waktu dari kumpulan data lengkap, dan untuk kumpulan data yang besar dapat menyatu lebih cepat karena menyebabkan pembaruan parameter lebih sering.
    - Kekurangan, karena pembaruan sering maka langkah-langkah yang diambil menuju minimal sangat bising sehingga menyebabkan penurunan gradien ke arah lain dan diperlukan waktu lebih lama untuk mencapai konvergensi ke fungsi kerugian minimum.


## Data Understanding

![image](https://user-images.githubusercontent.com/68459186/138727998-25fa3504-e84f-473c-af31-68eeb6de0a34.png)

Informasi dataset :

| Hal                     | Keterangan                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset : Gold Price Prediction Dataset](https://www.kaggle.com/sid321axn/gold-price-prediction-dataset) |
| Lisensi                 | CC0: Public Domain                                                                      |
| Kategori                | Finance, Tabular data, Beginner, Economics, Regression                                   |
| Rating Penggunaan       | 9.4                                                                                     |
| Jenis dan Ukuran Berkas | CSV (1.04 MB)                                                                           |

### Deskripsi Variabel
Atribut pada dataset :

- Gold ETF, komoditas yang diperdagangkan seperti saham. Meski terdiri dari aset berupa emas, investor sebenarnya tidak memiliki komoditas fisiknya. Investasi emas ETF dikenakan biaya tambahan baik biaya broker dan penebusan ETF.
  - Open, harga pembukaan
  - High, harga tertinggi
  - Low,  harga terendah
  - Close, harga penutupan
  - Adj Close, harga penutupan yang telah disesuaikan ketika terjadi aksi korporasi perusahaan, dalam hal ini adalah dividen dan stock split.
  - Volume, menunjukkan jumlah perdagangan atau transaksi yang terjadi dalam perdagangan di suatu sesi.
- S&P 500 Index, indeks yang terdiri atas 500 saham dengan market capitalization terbesar di Amerika Serikat. Indeks ini dimiliki oleh Standard & Poor. Indeks saham S&P 500 meliputi 80% dari kapitalisasi pasar di USA.
  - SP_open, harga pembukaan S&P 500 Index
  - SP_high, harga tertinggi S&P 500 Index
  - SP_low, harga terendah S&P 500 Index
  - SP_close, harga penutupan S&P 500 Index
  - SP_Ajclose, harga penutupan S&P 500 Index yang telah disesuaikan ketika terjadi aksi korporasi perusahaan, dalam hal ini adalah dividen dan stock split.
  - SP_volume, menunjukkan jumlah perdagangan atau transaksi S&P 500 Index yang terjadi dalam perdagangan di suatu sesi.
- Dow Jones Index, indeks pasar saham yang didirikan oleh editor The Wall Street Journal dan pendiri Dow Jones & Company Charles Cow. Bursa saham ini terdiri dari 30 perusahaan terbesar di Amerika Serikat.
  - DJ_open, harga pembukaan Dow Jones Index
  - DJ_high, harga tertinggi Dow Jones Index
  - DJ_low, harga terendah Dow Jones Index
  - DJ_close, harga penutupan Dow Jones Index
  - DJ_Ajclose, harga penutupan Dow Jones Index yang telah disesuaikan ketika terjadi aksi korporasi perusahaan, dalam hal ini adalah dividen dan stock split.
  - DJ_volume, menunjukkan jumlah perdagangan atau transaksi Dow Jones Index yang terjadi dalam perdagangan di suatu sesi.
- Eldorado Gold Corporation, perusahaan Kanada yang memiliki dan mengoperasikan tambang emas di Turki, Yunani, dan Kanada.
  - EG_open, harga pembukaan Eldorado Gold Corporation
  - EG_high, harga tertinggi Eldorado Gold Corporation
  - EG_low, harga terendah Eldorado Gold Corporation
  - EG_close, harga penutupan Eldorado Gold Corporation
  - EG_Ajclose, harga penutupan Eldorado Gold Corporation yang telah disesuaikan ketika terjadi aksi korporasi perusahaan, dalam hal ini adalah dividen dan stock split.
  - EG_volume, menunjukkan jumlah perdagangan atau transaksi Eldorado Gold Corporation yang terjadi dalam perdagangan di suatu sesi.
- EURO - USD Exchange Rate, Nilai tukar satuan mata uang Euro terhadap USD. 
  - EU_Price, harga jual
  - EU_open, harga pembukaan EURO - USD Exchange Rate
  - EU_high, harga tertinggi EURO - USD Exchange Rate
  - EU_low, harga terendah EURO - USD Exchange Rate
  - EU_Trend, rekam jejak harga EURO - USD Exchange Rate
- Brent Crude Oil Futures, harga patokan utama untuk pembelian minyak di seluruh dunia. Sementara minyak Brent Crude bersumber dari Laut Utara, produksi minyak yang berasal dari Eropa, Afrika dan Timur Tengah yang mengalir ke Barat cenderung dihargai relatif terhadap minyak ini.
  - OF_Price, harga jual Brent Crude Oil Futures
  - OF_Open, harga pembukaan Brent Crude Oil Futures
  - OF_High, harga tertinggi Brent Crude Oil Futures
  - OF_Low, harga terendah Brent Crude Oil Futures
  - OF_Volume, menunjukkan jumlah perdagangan atau transaksi Brent Crude Oil Futures yang terjadi dalam perdagangan di suatu sesi.
  - OF_Trend, rekam jejak harga Brent Crude Oil Futures
- Crude Oil WTI USD, West Texas Intermediate - Patokan WTI untuk minyak mentah AS adalah komoditas dunia yang paling aktif diperdagangkan.
  - OS_Price, harga jual Crude Oil WTI USD
  - OS_Open, harga pembukaan Crude Oil WTI USD
  - OS_High, harga tertinggi Crude Oil WTI USD
  - OS_Low, harga terendah Crude Oil WTI USD
  - OS_Trend, rekam jejak harga Crude Oil WTI USD
- Silver Futures, perdagangan perak berjangka
  - SF_Price, harga jual Silver Futures
  - SF_Open, harga pembukaan Silver Futures
  - SF_High, harga tertinggi Silver Futures
  - SF_Low, harga terendah Silver Futures
  - SF_Volume, menunjukkan jumlah perdagangan atau transaksi Silver Futures yang terjadi dalam perdagangan di suatu sesi.
  - SF_Trend, rekam jejak harga Silver Futures
- US Bond Rate (10 years), surat berharga berupa pengakuan utang negara Amerika Serikat yang dijamin pembayaran bunga dan pokoknya oleh negara. Di Amerika Serikat, berlaku surat utang negara 10 tahun. 
  - USB_Price, harga jual US Bond Rate
  - USB_Open, harga pembukaan US Bond Rate
  - USB_High, harga tertinggi US Bond Rate
  - USB_Low, harga terendah US Bond Rate
  - USB_Trend, rekam jejak harga US Bond Rate
- Platinum Price, perdagangan harga platinum
  - PLT_Price, harga jual Platinum
  - PLT_Open, harga pembukaan Platinum
  - PLT_High, harga tertinggi Platinum
  - PLT_Low, harga terendah Platinum
  - PLT_Trend, rekam jejak harga Platinum
- Palladium Price, perdagangan harga palladium
  - PLD_Price, harga jual Palladium
  - PLD_Open, harga pembukaan Palladium
  - PLD_High, harga tertinggi Palladium
  - PLD_Low, harga terendah Palladium
  - PLD_Trend, rekam jejak harga Palladium
- Rhodium Price, perdagangan harga rhodium
  - RHO_PRICE, harga jual Rhodium
- US Dollar Index, indeks atau ukuran nilai dolar Amerika Serikat relatif terhadap mata uang asing. Indeks ini dirancang, dipelihara dan diterbitkan oleh ICE (International Exchange, Inc).
  - USDI_Price, harga jual US Dollar Index
  - USDI_Open, harga pembukaan US Dollar Index
  - USDI_High, harga tertinggi US Dollar Index
  - USDI_Low, harga terendah US Dollar Index
  - USDI_Volume, menunjukkan jumlah perdagangan atau transaksi US Dollar Index yang terjadi dalam perdagangan di suatu sesi.
  - USDI_Trend, rekam jejak harga US Dollar Index
- Gold Miners ETF, produk investasi yang diperdagangkan di bursa yang berusaha memberikan hasil investasi yang sesuai dengan pergerakan harga saham perusahaan yang secara aktif terlibat dalam penambangan dan aspek lain dari produksi emas.
  - GDX_Open, harga pembukaan Gold Miners ETF
  - GDX_High, harga tertinggi Gold Miners ETF
  - GDX_Low, harga terendah Gold Miners ETF
  - GDX_Close, harga penutupan Gold Miners ETF
  - GDX_Adj Close, harga penutupan Gold Miners ETF yang telah disesuaikan ketika terjadi aksi korporasi perusahaan, dalam hal ini adalah dividen dan stock split.
  - GDX_Volume, menunjukkan jumlah perdagangan atau transaksi Gold Miners ETF yang terjadi dalam perdagangan di suatu sesi.
- Oil ETF USO, produk yang diperdagangkan di bursa yang berusaha memberikan hasil investasi yang sesuai dengan pergerakan harga harian minyak mentah ringan dan manis WTI. 
  - USO_Open, harga pembukaan Oil ETF USO
  - USO_High, harga tertinggi Oil ETF USO
  - USO_Low, harga terendah Oil ETF USO
  - USO_Close, harga penutupan Oil ETF USO
  - USO_Adj Close, harga penutupan Oil ETF USO yang telah disesuaikan ketika terjadi aksi korporasi perusahaan, dalam hal ini adalah dividen dan stock split. 
  - USO_Volume, menunjukkan jumlah perdagangan atau transaksi Oil ETF USO yang terjadi dalam perdagangan di suatu sesi.

Keterangan:
- ETF, Reksa Dana berbentuk Kontrak Investasi Kolektif yang unit penyertaannya diperdagangkan di Bursa Efek. Meskipun ETF pada dasarnya adalah reksa dana, produk ini diperdagangkan seperti saham-saham yang ada di bursa efek. ETF merupakan penggabungan antara unsur reksa dana dalam hal pengelolaan dana dengan mekanisme saham dalam hal transaksi jual maupun beli.
- Trend, rangkaian rekam jejak harga dalam bentuk grafik dengan kecondongan untuk mengarah ke atas (1) atau ke bawah (0)

Atribut pada dataset yang dipilih untuk digunakan untuk mengembangkan model :
- Fitur
  - Open, High, Low, Close
- Variabel target
  - Adj Close

Kemudian dilakukan perhitungan pada data variabel target antara lain :
- MACD (Moving Average Convergence Divergence), sebuah indikator dalam analisis teknikal yang menggambarkan hubungan antara dua moving average dalam sebuah tren harga aset. Adapum, moving average merupakan rerata harga, baik pembukaan atau penutupan perdagangan setiap harinya yang digambarkan dalam sebuah garis tren. Kegunaan untuk memahami kapan harga aset tersebut akan bersifat bullish atau bearish. Pada dasarnya, MACD menghitung Exponential Moving Average (EMA) selama 12 hari dan 26 hari terakhir. EMA adalah jenis moving average yang menitikberatkan pada bobot dan signifikansi dari data yang paling baru. Rumus MACD sebagai berikut.

  ![MACD](https://user-images.githubusercontent.com/68459186/138883468-f5dd57fb-f173-4bb3-89d5-55c1af4242d7.png)

  Dengan demikian, MACD akan bernilai positif jika EMa 12 hari lebih besar dari EMA 26 hari dan berlaku sebaliknya. 

- RSI (Relative Strength Index), indikator yang digunakan dalam mengukur besarnya volatilitas harga sebuah aset. Indikator ini dilakukan untuk mengevaluasi apakah aset tersebut terbilang dalam posisi jenuh beli (overbought) atau jenuh jual (oversold). RSI ditampilkan sebagai osilator (grafik garis yang bergerak antara dua titik ekstrim) dengan nilai berada di antara 0 hingga 100. Rumus RSI sebagai berikut.

  ![RSI](https://user-images.githubusercontent.com/68459186/138889935-e40b7183-a5bc-411a-a4f9-7b9a0e8a9ab4.png)

  Rata-rata keuntungan atau kerugian yang digunakan dalam perhitungan adalah persentase  keuntungan atau kerugian rata-rata selama periode kilas balik (dua titik yang dipilih untuk dibandingkan, bisa selama 7 hari, bisa selama 14 hari, dst).

- SMA (Simple Moving Average), bentuk simpel dari Moving Average. Moving Average untuk memberi petunjuk mengenai arah tren harga sebuah aset di masa depan. Pada Simple  Moving Average indikator dihitung dengan menggunakan rerata aritmatika dari salah satu set nilai tertentu, biasanya harga penutupan dengan jumlah periode dalam kisaran itu. Dengan kata lain, serangkaian data aset digabungkan dulu bersama-sama untuk kemudian dibagi menjadi harga aset di set tertentu tersebut. Rumus SMA sebagai berikut.
  
  ![SMA](https://user-images.githubusercontent.com/68459186/138896477-358fc5e2-a4af-49e7-a493-95a3da06d5a8.png)

- Bollinger  Band, alat analisis teknis yang dikembangkan oleh John Bollinger untuk menghasilkan sinyal oversold atau overbought. Ada tiga baris yang membentuk Bollinger Bands, SMA (middle band), upper band, dan lower band. Upper dan lower band biasanya 2 standar deviasi +/- dari rata-rata bergerak sederhana selama 20 hari, tetapi dapat dimodifikasi. Bollinger band dimanfaatkan untuk menganalisis pergerakan harga sebuah aset atau komoditas tertentu. Rumus bollinger band sebagai berikut.
  
  ![image](https://user-images.githubusercontent.com/68459186/138906534-cc3772a6-99e3-4ada-87a1-17bb252b8a49.png)

Sehingga fitur yang digunakan bertambah dengan adanya hasil perhitungan yang dilakukan, maka variabel yang digunakan antara lain :
- open, harga pembukaan
- high, harga tertinggi
- low,  harga terendah
- close, harga penutupan
- adj close, harga penutupan yang telah disesuaikan ketika terjadi aksi korporasi perusahaan, dalam hal ini adalah dividen dan stock split.
- adj close_returns, menghitung pengembalian harian
- rsi_adj close, mengukur besarnya volatilitas harga pada adj close
- upper_band_adj close, batas atas bollinger band pada adj close
- lower_band_adj close, batas bawah bollinger band pada adj close
- dif_adj close, selisih nilai antara data periode 26 dengan periode 12 pada adj close
- macd_adj close, 

### Memvisualisasikan Data
Berikut ini merupakan visualisasi dari data fitur yang digunakan :
- open,

  ![image](https://user-images.githubusercontent.com/68459186/138909541-8a92fc56-f127-4c09-a743-98472eae798f.png) 

- high,

  ![image](https://user-images.githubusercontent.com/68459186/138909692-d4b5b164-bb33-4bfe-8913-f91098087d49.png)

- low,
  
  ![image](https://user-images.githubusercontent.com/68459186/138909743-38b3213d-64de-430d-bf03-422ad2203469.png)
 
- close, 
  
  ![image](https://user-images.githubusercontent.com/68459186/138909825-10187dcf-41ae-441a-9788-aa8e770ab756.png)

- adj close,
  
  ![image](https://user-images.githubusercontent.com/68459186/138909863-d5c03339-2f50-455a-a004-2ca83e36bd55.png)
 
- adj close_returns,
  
  ![image](https://user-images.githubusercontent.com/68459186/138909893-7667571c-4cfd-44e4-b498-ace379546a0b.png)
 
- rsi_adj close, 
  
  ![image](https://user-images.githubusercontent.com/68459186/138909956-0e378d72-54f7-41eb-a9e4-2f4190869321.png)

- upper_band_adj close, 
  
  ![image](https://user-images.githubusercontent.com/68459186/138910004-9c9b1c2a-c3f4-4a97-9feb-97960756a659.png)

- lower_band_adj close,
  
  ![image](https://user-images.githubusercontent.com/68459186/138910050-a61d4011-81a6-4876-89bf-89de292268ed.png)
 
- dif_adj close, dan 
 
  ![image](https://user-images.githubusercontent.com/68459186/138910082-0e059120-7d5f-4a15-9a00-3efaa1b7edf8.png)

- macd_adj close.

  ![image](https://user-images.githubusercontent.com/68459186/138908914-a5b878b8-d44d-4fe5-9016-8d634d351597.png)

## Data Preparation

Teknik preparation yang digunakan pada proyek ini antara lain :
- Menangani Missing Value
  Menghilangkan data yang bernilai 0 atau kosong
  
  ![image](https://user-images.githubusercontent.com/68459186/139034942-5b1ea3b5-439d-4e8a-a4a3-889ef4df4488.png)
  
  Bisa dilihat pada gambar diatas menunjukan jumlah nilai yang kosong atau NaN yang terdapat pada data dikarenakan jumlahnya tidak terlalu banyak sehingga diputuskan untuk menghapusnya. Selain itu, karena jumlah yang tidak terlalu banyak sehingga tidak terlalu mempengaruhi fitur atau hilangnya informasi yang dibutuhkan. 

- Train-Test-Split 
  Dilakukan pembagian dataset menjadi s bagian, yaitu data latih dan data uji. Pertama, dilakukan pembagian menjadi data latih dan data uji dengan perbandingan 80:20. Setelah pembagian dataset dilakukan pengurutan karena data hasil pembagian berbentuk acak sehingga perlu diurutkan kembali berdasarkan urutan waktu agar data yang akan digunakan sesuai dengan kondisi aslinya. Data latih digunakan untuk proses pelatihan model dengan data sebanyak 80% dari dataset, sedangkan data uji sebanyak 20% dari dataset digunakan untuk menguji model yang sudah dilatih. Pembagian dataset dilakukan menggunakan fungsi train_test_split dari sklearn.

- Normalisasi
  Normalisasi dilakukan dengan tujuan untuk mengubah nilai kolom numerik dalam data ke skala yang sama, tanpa mengganggu perbedaan dalam rentang nilai. Normalisasi dilakukan pada fitur-fitur yang akan digunakan. Proses normalisasi dilakukan menggunakan fungsi MinMaxScaler dari sklearn. Proses normalisasi dilakukan setelah pembagian data dengan tujuan menghindari kebocoran data pada data uji. 

## Modeling

Pada proses pemodelan menggunakan beberapa algoritma sebagai berikut:
- Decision Tree Regression, cara kerjanya dengan cara membagi data menjadi himpunan bagian berdasarkan variabel inputnya, metode pengambilan keputusan dengan cara melihat nilai probabilitas yang terstruktur dan sistematis untuk sampai pada kesimpulan yang tepat. Parameter yang digunakan adalah 'random_state=0' untuk mengatur generator angka acak yang digunakan. 
- Support Vector Regression, pada SVR dataset dimasukkan ke dalam satu zona dengan tetap meminimasi nilai epsilon, kemudian buat kernel untuk memplot data. Kemudian, dilakukan perhitungan kernel untuk menentukan nilai tengah dan nilai batas atas dan bawah. Jarak antara garis tengan dengan garis batas disebut epsilon. Tujuan akhirnya untuk membuat satu kluster yang masuk kriteria. Parameter yang digunakan adalah 'kernel='linear'' untuk merepresentasikan data berupa vector, fungsi kernel liner seperti berikut ![image](https://user-images.githubusercontent.com/68459186/140456247-ba382bd3-b62a-493c-bb66-0aa0b3403730.png).
- Random Forest, prinsip dasar seperti decision tree. Random forest terdiri dari beberapa decision tree yang akan menghasilkan output yang berbeda-beda, lalu random forest akan melakukan voting untuk menentukan hasil mayoritas dari semua decision tree. Parameter yang digunakan adalah 'n_estimators=50' berarti jumlah pohon di hutan yang digunakan berjumlah 50 dan 'random_state=0' untuk mengatur generator angka acak yang digunakan. 
- LassoCV, menggunakan teknik regularisasi L1 digunakan ketika memiliki lebih banyak fitur karena secara otomatis melakukan pemilihan fitur . Parameter yang digunakan adalah 'n_alphas=1000' untuk jumlah alfa di sepanjang jalur regularisasi sebanyak 1000 alfa, 'max_iter=3000' untuk jumlah maksimum iterasi sebanyak 3000 iterasi dan 'random_state=0' sebagai benih generator nomor acak semu yang memilih fitur acak untuk diperbarui.
- RidgeCV, memecahkan model regresi di mana fungsi kerugian adalah fungsi kuadrat terkecil linier dan regularisasi L2. Parameter yang digunakan adalah 'gcv_mode='auto'' untuk menunjukkan strategi mana yang akan digunakan saat melakukan Leave-One-Out Cross-Validation - mode 'auto' adalah default dan dimaksudkan untuk memilih opsi yang lebih murah dari 'svd' atau 'eigen' tergantung pada bentuk data pelatihan.
- Gradient Boosting Regressor, menggunakan ensamble dari decision tree, dengan membuat simple decision tree yang terpisah dan mengkombinasikan output dari setiap decision tree tersebut untuk mendapatkan hasil akhir. Parameter yang digunakan adalah 'n_estimators=70' menunjukkan jumlah tahapan boosting yang harus dilakukan yaitu 70 tahapan, 'learning_rate=0.1' menunjukkan kecepatan pembelajaran, 'max_depth=4' kedalaman maksimum estimator regresi individu, 'random_state=0' untuk mengontrol banih acak yang diberikan ke setiap estimator tree pada setiap iterasi boosting dan 'loss='ls'' yaitu fungsi kerugian untuk dioptimalkan 'ls' mengacu pada kesalahan kuadrat untuk regresi.
- Stochastic Gradient Descent, bekerja dengan memulai dari sebuah tebakan awal dan secara iteratif tebakan ini diperbaiki berdasarkan suatu aturan yang melibatkan gradien/turunan pertama dari fungsi yang ingin diminimumkan, SGD tidak menggunakan semua data training untuk menghitung gradien pada setiap iterasi, tetapi hanya menggunakan satu atau beberapa bagian saja dari data training yang dipilih secara acak. Parameter yang digunakan adalah 'max_iter=1000' untuk jumlah maksimum lintasan di atas data pelatihan (alias epoch) sebanyak 100 epoch, 'tol=1e-3' untuk menghentikan pelatihan ketika kerugian lebih besar dari nilai 'tol' yaitu 1e-3, 'loss='squared_epsilon_insensitive'' merupakan fungsi kerugian yang mengabaikan kesalahan kuadrat melewati toleransi epsilon, 'penalty='l1'' untuk hukuman atau regularisasi yang akan digunakan 'l1' mungkin membawa kelonggaran pada model yang tidak dapat dicapai dengan 'l2' dan 'alpha=0.1' sebagai konstanta yang mengalikan istilah regularisasi-juga untuk menghitung kecepatan belajar ketika diatur ke learning_rate diatur ke 'optimal'.

## Evaluation

Matrik evaluasi yang digunakan :
- RMSE, Metrik kesalahan akar rata-rata kuadrat adalah ukuran yang sering digunakan untuk perbedaan antara nilai yang diprediksi oleh model atau penaksir dan nilai yang diamati. Metrik ini berkisaran dari nol hingga tak terhingga; nilai yang lebih rendah menunjukkan kualitas model yang lebih tinggi. Akar kuadrat dari Mean Squared Error.

- R2 Score, Kuadrat dari koefisien korelasi Pearson antara label dan nilai prediksi. Metrik ini berkisaran antara nol dan satu; nilai yang lebih tinggi menunjukkan kualitas model yang lebih tinggi. 

Berikut ini merupakan hasil dari proses pemodelan yang telah dilakukan menggunakan beberapa algoritma, model dilatih menggunakan data latih dan dicoba memprediksi pada data validasi yang dapat dilihat pada gambar visualisasi berikut ini.
  
![image](https://user-images.githubusercontent.com/68459186/140457451-c4dc6f16-20e1-45b5-a585-59a31250cc62.png)

Berikut ini grafik yang digunakan untuk membandingkan nilai RMSE dari setiap model yang telah dibuat.

![image](https://user-images.githubusercontent.com/68459186/140457437-97a36147-1e67-490f-b90f-046fc2896b1d.png)

Dari data diatas sehingga dibuat sebuah tabel untuk mengurutkan algoritma dengan performa tinggi ke rendah, sebagai berikut ini. 

![image](https://user-images.githubusercontent.com/68459186/140458229-c2b816d2-4379-41dd-beb3-9bab3651a0a7.png)

Dari hasil yang didapatkan dari pembuatan beberapa model diatas maka didapatkan nilai RMSE dan R2 Score seperti diatas. Sehingga dapat digunakan model dengan nilai RMSE terendah dan R2-Score tertinggi untuk digunakan sebagai model prediksi harga emas. Model yang mungkin cocok untuk digunakan adalah Support Vector Regression, Gradient Boosting dan RidgeCV.  

## Referensi
- Fernando, Jason. (2021). _Moving Average Convergence Divergence (MACD)_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/m/macd.asp
-  Fernando, Jason. (2021). _Relative Strength Index (RSI)_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/r/rsi.asp
-  Hayes, Adam. (2021). _Simple Moving Average (SMA)_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/s/sma.asp
- Hayes, Adam. (2021). _Bollinger Band Definition_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/b/bollingerbands.asp
- Anonim. (2021). _Jenis-Jenis Metode Regresi dalam Algoritma Supervised Learning_. Diakses pada 27 Oktober 2021, dari https://www.dqlab.id/jenis-metode-regresi-algoritma-supervised-learning
- IYKRA. (2018). _Mengenal Decision Tree dan Manfaatnya_. Diakses pada 27 Oktober 2021, dari https://medium.com/iykra/mengenal-decision-tree-dan-manfaatnya-b98cf3cf6a8d
- Anonim. (). _Support Vector Machine-Regression (SVR)_. Diakses pada 27 Oktober 2021, dari https://www.saedsayad.com/support_vector_machine_reg.htm
- Anonim. (). _1.5. Stochastic Gradient Descent_. Diakses pada 27 Oktober 2021, dari https://scikit-learn.org/stable/modules/sgd.html
- Anonim. (2021). _Algoritma Machine Learning yang Harus Kamu Pelajari di Tahun 2021_. Diakses pada 27 Oktober 2021, dari https://www.dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari
- Anonim. (2021). _AutoML Tables Documentation : Guide_. Diakses pada 27 Oktober 2021, dari https://cloud.google.com/automl-tables/docs/evaluate?hl=en
- Majumder, Sohom. (2021). _Gold price prediction-Time Series split & LGBM_. Diakses pada 18 Oktober 2021, dari https://www.kaggle.com/sohommajumder21/gold-price-prediction-time-series-split-lgbm
- Siddhartha, Manu. (2021). _Gold Price Prediction Using Machine Learning_. Diakses pada 17 Oktober 2021, dari https://www.kaggle.com/sid321axn/gold-price-prediction-using-machine-learning
- Dranitsyna, Ekaterina. (2021). _gold_price_model_. Diakses pada 18 Oktober 2021, dari https://www.kaggle.com/ekaterinadranitsyna/gold-price-model
-  Venable, Hayden. (2021). _Gold Price Prediction with PCA and Regression_. Diakses pada 17 Oktober 2021, dari https://www.kaggle.com/haydenvenable/gold-price-prediction-with-pca-and-regression
