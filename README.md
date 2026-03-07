# HandSpeak 🤟

HandSpeak, gerçek zamanlı olarak Amerikan İşaret Dili (ASL) harflerini tanıyan modern ve kullanıcı dostu bir masaüstü uygulamasıdır. Kamera üzerinden el hareketlerini algılayarak, ekranda hangi harfin yapıldığını gösterir.

## Özellikler

- **Gerçek Zamanlı Tanıma:** Bilgisayarınızın kamerasını kullanarak anlık olarak işaret dili harflerini algılar.
- **Modern ve Şık Arayüz:** `customtkinter` kullanılarak geliştirilmiş modern ve akıcı bir kullanıcı deneyimi sunar.
- **Yüksek Hassasiyet:** Google MediaPipe kütüphanesi ile el iskeleti (landmarks) çıkarılır ve dinamik olarak harf tespiti yapılır.
- **Tahmin Stabilizasyonu:** Yanlış algılamaları en aza indirmek için tahminleri kısa bir süre biriktirerek en kararlı, net sonucu ekrana yansıtır.

## Gereksinimler

Uygulamanın çalışabilmesi için bilgisayarınızda **Python 3.8 veya daha üstü** bir sürümün yüklü olması önerilir. 

Kullanılan temel kütüphaneler:
- `customtkinter`: Modern masaüstü görünümü ve arayüzü sağlamak için.
- `Pillow`: Görüntü (resim) işleme ve özel arayüz çizimleri gerçekleştirmek için.
- `opencv-python`: Kameradan görüntü akışını almak ve işlemek için.
- `mediapipe`: El işaretlerini/eklemlerini tespit etmek için.

## Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla uygulayın:

### 1. Gerekli Kütüphanelerin Kurulumu
Proje klasörünün (`HandSpeak`) içerisindeyken, sistemin ihtiyaç duyduğu yan paketleri yüklemek için terminal (komut satırı) üzerinden aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

### 2. Uygulamayı Başlatma
Gerekli paketler başarıyla yüklendikten sonra ana uygulamayı çalıştırmak için aşağıdaki komutu girin:

```bash
python app.py
```

## Nasıl Kullanılır?

1. Uygulama açıldığında sol alttaki **"▶ Başlat"** düğmesine tıklayarak kameranızı etkinleştirin.
2. Elinizi kameranın görebileceği bir açıda ve netlikte tutun.
3. Amerikan İşaret Dili (ASL) alfabesindeki harfleri elinizle yapmaya başlayın.
4. Yaptığınız hareketin tahmini, hesaplanan "Güven Skoru" ile birlikte sağ panelde anlık olarak görüntülenecektir.
5. Algılanan harfler geçmiş panelinde kaydedilir, böylece oluşturduğunuz metni geriye dönük görebilirsiniz (dilerseniz "Temizle" butonuyla sıfırlayabilirsiniz).
6. Uygulamanın izlenmesini ve kamerayı kapatmak için **"⏹ Durdur"** düğmesine basabilirsiniz.

## Dosya Yapısı

- `app.py`: Ana uygulama modülü. Kullanıcı arayüzü (UI) tasarımı, kamera görüntü yönetimi ve MediaPipe entegrasyonu bu dosyada yer alır.
- `asl_classifier.py`: Harf sınıflandırma motoru. MediaPipe ile tespit edilen el iskeletindeki eklem açılarını ve mesafelerini hesaplayarak ilgili parmakların açılıp kapanmasına göre kural tabanlı tahmin yürütür.
- `requirements.txt`: Uygulamanın çalışması için gereken Python kütüphaneleri listesi.

## Geliştirme

Bu proje, kural tabanlı bir mantık (`asl_classifier.py`) kullanarak çalışmaktadır. Eğer bazı harflerin yeterince iyi algılanmadığını düşünüyorsanız veya yeni işaretler dahil etmek isterseniz, bu dosyadaki parmak açı ve mesafe kurallarını (`_mesafe` ve `_aci` fonksiyonlarına bağlı kalarak) dilediğiniz gibi güncelleyebilirsiniz.
