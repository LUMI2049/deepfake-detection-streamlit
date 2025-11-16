# Hugging Face Hub Integration

Bu klasör, DeepFake Detection modelini Hugging Face Model Hub'a yüklemek ve yönetmek için gerekli tüm araçları içerir.

## 📁 Dosya Yapısı

```
huggingface/
├── README.md                    # Bu dosya
├── HUGGINGFACE_GUIDE.md        # Detaylı kullanım kılavuzu (Türkçe)
├── upload_model.py             # Model yükleme script'i
├── test_inference.py           # Test ve inference script'i
├── README_MODEL.md             # Hugging Face model card şablonu
├── config.json                 # Model konfigürasyon dosyası
├── requirements.txt            # Python bağımlılıkları
└── .gitattributes             # Git LFS konfigürasyonu
```

## 🚀 Hızlı Başlangıç

### 1. Gereksinimleri Yükle

```bash
cd huggingface
pip install -r requirements.txt
```

### 2. Hugging Face Token Al

1. [Hugging Face Settings](https://huggingface.co/settings/tokens) sayfasına git
2. **Write** yetkili yeni bir token oluştur
3. Token'ı kopyala ve kaydet

### 3. Model'i Yükle

```bash
export HF_TOKEN="your_token_here"

python upload_model.py \
    --model_path /path/to/your/best_model_effatt.h5 \
    --repo_id CemRoot/deepfake-detection-model
```

### 4. Model'i Test Et

```bash
python test_inference.py --image path/to/test_image.jpg
```

## 📚 Detaylı Dokümantasyon

Tüm detaylar için [HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md) dosyasına bakın:

- ✅ Gereksinimler ve kurulum
- ✅ Token alma ve yönetimi
- ✅ Model yükleme adımları
- ✅ Test ve doğrulama
- ✅ Model güncelleme
- ✅ En iyi uygulamalar
- ✅ Sorun giderme

## 🔧 Kullanılabilir Script'ler

### `upload_model.py`

Model ve tüm ilgili dosyaları Hugging Face Hub'a yükler.

**Temel Kullanım:**
```bash
python upload_model.py --model_path model.h5 --token YOUR_TOKEN
```

**Tüm Parametreler:**
```bash
python upload_model.py \
    --model_path /path/to/model.h5 \
    --repo_id username/model-name \
    --token YOUR_TOKEN \
    --commit_message "Update message" \
    --private  # Opsiyonel: private repo
```

### `test_inference.py`

Hugging Face Hub'dan model yükleyip test eder.

**Tek Görüntü:**
```bash
python test_inference.py --image test.jpg
```

**Toplu İşleme:**
```bash
python test_inference.py --batch images_folder/
```

## 📦 Ne Yüklenir?

Script çalıştırıldığında şu dosyalar yüklenir:

1. **best_model_effatt.h5** - Model dosyası (~780 MB)
2. **README.md** - Detaylı model dokümantasyonu
3. **config.json** - Model metadata ve konfigürasyonu
4. **inference_example.py** - Kullanım örneği
5. **.gitattributes** - Git LFS ayarları

## 🎯 Özellikler

- ✅ Otomatik repository oluşturma
- ✅ Kapsamlı model card (README)
- ✅ Model metadata (config.json)
- ✅ Test ve inference araçları
- ✅ Git LFS desteği
- ✅ Batch processing
- ✅ Detaylı error handling

## 📝 Model Card İçeriği

`README_MODEL.md` şablonu şunları içerir:

- Model açıklaması ve özellikleri
- Mimari detayları
- Desteklenen generative modeller
- Kullanım örnekleri
- Performans metrikleri
- Sınırlamalar ve etik değerlendirmeler
- Alıntı bilgileri
- Lisans bilgisi

## 🔐 Güvenlik

- ⚠️ Token'ınızı asla kodda saklama
- ✅ Ortam değişkenleri kullanın
- ✅ `.gitignore` token dosyalarını içermeli
- ✅ **Write** yetkili token kullanın

## 🐛 Sorun Giderme

### Token bulunamıyor
```bash
export HF_TOKEN="your_token"
```

### Git LFS hatası
```bash
git lfs install
```

### Model yüklenemedi
```bash
pip install --upgrade tensorflow>=2.15.0
```

Daha fazla sorun giderme için [HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md#sorun-giderme) bölümüne bakın.

## 🔗 Bağlantılar

- **Hugging Face Model Hub**: https://huggingface.co/CemRoot/deepfake-detection-model
- **GitHub Repository**: https://github.com/CemRoot/deepfake-detection-streamlit
- **Streamlit Demo**: https://your-app-url.streamlit.app
- **HF Documentation**: https://huggingface.co/docs/hub/models

## 📞 Destek

Sorular veya sorunlar için:
- GitHub Issues açın
- [Detaylı kılavuza](HUGGINGFACE_GUIDE.md) bakın
- [cemkoyluoglu.codes](https://cemkoyluoglu.codes/) adresinden iletişime geçin

---

**Yazar**: Emin Cem Koyluoglu
**Lisans**: MIT
**Conference**: AICS 2025
