# TemizDil Model

Türkçe metinlerdeki nefret söylemi ve saldırgan içeriği tespit etmek için geliştirilmiş hiyerarşik bir sınıflandırma modeli.

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Mimarisi](#-model-mimarisi)
- [Veri Seti](#-veri-seti)
- [Performans](#-performans)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

## 🚀 Proje Hakkında

TemizDil Model, Türkçe metinlerdeki nefret söylemi ve saldırgan içeriği tespit etmek için geliştirilmiş hiyerarşik bir sınıflandırma modelidir. Model, metinleri çoklu seviyede analiz ederek farklı kategorilerde sınıflandırma yapmaktadır.

## ✨ Özellikler

- Hiyerarşik sınıflandırma yaklaşımı
- Çoklu etiket sınıflandırması
- BERT tabanlı derin öğrenme modeli
- Zorluk seviyesi tahmini
- Türkçe metin analizi
- Detaylı içerik kategorizasyonu
- CUDA ve CPU desteği

## 🛠️ Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/kemalersin/temizdil-model.git
cd temizdil-model
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Eğitilmiş model dosyaları `offensive_model_hierarchical` dizininde bulunmaktadır. Modeli kullanmak için bu dizinin proje kök dizininde olduğundan emin olun.

4. CUDA Desteği (Opsiyonel):
   - Model, CUDA destekli GPU'lar ile çalışabilir
   - CUDA kullanmak için PyTorch'un CUDA destekli sürümünü yüklemeniz gerekir
   - CPU kullanımı için herhangi bir ek kurulum gerekmez

## 📝 Kullanım

### Model Eğitimi

Modeli eğitmek için:
```bash
python train.py
```

Eğitilen model `offensive_model_hierarchical` dizinine kaydedilecektir.

### Analiz ve Tahmin

#### 1. Parametreli Kullanım

Tek bir metin analizi için:
```bash
python analyze.py --text "Analiz edilecek metin"
```

Dosyadan çoklu metin analizi için:
```bash
python analyze.py --file metinler.txt
```

#### 2. Etkileşimli Kullanım (Parametresiz)

Programı parametresiz çalıştırarak etkileşimli modda kullanabilirsiniz:
```bash
python analyze.py
```

Bu modda:
- Program sizden metin girmenizi bekleyecektir
- Her metin girişinden sonra analiz sonuçlarını gösterecektir
- Çıkmak için 'q' yazabilirsiniz

#### Donanım Kullanımı

Model, sisteminizde CUDA destekli bir GPU varsa otomatik olarak GPU'yu kullanacaktır. GPU yoksa veya CUDA kurulu değilse CPU'da çalışacaktır. Model ağırlıkları otomatik olarak uygun cihaza taşınır.

#### Parametreler

- `--text`: Analiz edilecek metin (tek bir metin için)
- `--file`: Analiz edilecek metinlerin bulunduğu dosya (her satır bir örnek)
- `--model_path`: Eğitilmiş model klasörü (varsayılan: "./offensive_model_hierarchical")

#### Çıktılar

Analiz sonuçları aşağıdaki bilgileri içerir:

1. **Temel Bilgiler:**
   - Metnin saldırgan olup olmadığı
   - Tahmin edilen etiketler
   - Her etiket için olasılık değerleri
   - Karar vermesi zor olup olmadığı

2. **Saldırgan İçerik Detayları:**
   - Hedefli olup olmadığı
   - Hedef türü (grup, birey, diğer, çoklu hedef)

#### Örnek Çıktı

```
==================================================
METİN: [analiz edilen metin]
==================================================
SALDIRGAN MI: Evet

ETİKETLER:
  - prof
  - grp

ETİKET OLASILIKLARI:
  - non: 0.1234
  - prof: 0.8765
  - grp: 0.7654
  - ind: 0.2345
  - oth: 0.3456

HEDEFLİ Mİ: Evet
HEDEF TÜRÜ: grup

KARAR VERMESİ ZOR MU: Hayır
==================================================
```

## 🏗️ Model Mimarisi

Model, aşağıdaki hiyerarşik yapıyı kullanmaktadır:

1. **Seviye 1**: Saldırgan içerik tespiti (offensive/non-offensive)
2. **Seviye 2**: Hedefli/hedefsiz içerik tespiti
3. **Seviye 3**: Hedef tipi sınıflandırması
   - Grup hedefli (grp)
   - Birey hedefli (ind)
   - Diğer hedefli (oth)
   - Çoklu hedefli

## 📊 Veri Seti

Model, [Türkçe Saldırgan Dil Veri Seti](https://coltekin.github.io/offensive-turkish/) kullanılarak eğitilmiştir. Bu veri seti, sosyal medyadan toplanan Türkçe tweet'lerden oluşmaktadır ve aşağıdaki etiketleri içermektedir:

- `non`: Saldırgan olmayan içerik
- `prof`: Küfürlü içerik
- `grp`: Gruba yönelik saldırı
- `ind`: Bireye yönelik saldırı
- `oth`: Diğer türde saldırı
- `X`: Anlaşılamayan veya Türkçe olmayan içerik

Veri seti, Çağrı Çöltekin tarafından hazırlanmış ve Creative Commons Attribution License (CC-BY) altında dağıtılmaktadır. Veri setini kullanırken aşağıdaki makaleyi referans göstermeniz önerilir:

```
@inproceedings{coltekin2020lrec,
 author  = {\c{C}\"{o}ltekin, \c{C}a\u{g}r{\i}},
 year  = {2020},
 title  = {A Corpus of Turkish Offensive Language on Social Media},
 booktitle  = {Proceedings of The 12th Language Resources and Evaluation Conference},
 pages  = {6174--6184},
 address  = {Marseille, France},
 url  = {https://www.aclweb.org/anthology/2020.lrec-1.758},
}
```

## 📈 Performans

Model performans metrikleri buraya eklenecek.

## 🤝 Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına bakın.

## 📞 İletişim

Proje Sahibi - [@kemalersin](https://github.com/kemalersin)

Proje Linki: [https://github.com/kemalersin/temizdil-model](https://github.com/kemalersin/temizdil-model) 