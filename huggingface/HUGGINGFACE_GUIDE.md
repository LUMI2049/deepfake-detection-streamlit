# Hugging Face Hub Integration Guide

Bu kılavuz, DeepFake Detection modelinizi Hugging Face Model Hub'a nasıl yükleyeceğinizi ve yöneteceğinizi gösterir.

## İçindekiler

1. [Gereksinimler](#gereksinimler)
2. [Hugging Face Token Alma](#hugging-face-token-alma)
3. [Model Yükleme](#model-yükleme)
4. [Model Test Etme](#model-test-etme)
5. [Model Güncelleme](#model-güncelleme)
6. [En İyi Uygulamalar](#en-iyi-uygulamalar)

---

## Gereksinimler

### 1. Python Paketlerini Yükleyin

```bash
cd huggingface
pip install -r requirements.txt
```

### 2. Git LFS (Large File Storage) Kurulumu

Hugging Face, büyük model dosyaları için Git LFS kullanır. Sisteminize Git LFS kurmanız gerekir:

**Ubuntu/Debian:**
```bash
sudo apt-get install git-lfs
git lfs install
```

**macOS:**
```bash
brew install git-lfs
git lfs install
```

**Windows:**
Git LFS'i [resmi web sitesinden](https://git-lfs.github.com/) indirin ve kurun, ardından:
```bash
git lfs install
```

---

## Hugging Face Token Alma

1. [Hugging Face](https://huggingface.co/) hesabınıza giriş yapın
2. **Settings** → **Access Tokens** sayfasına gidin: https://huggingface.co/settings/tokens
3. **New token** butonuna tıklayın
4. Token için bir isim girin (örnek: "deepfake-model-upload")
5. **Role** olarak **Write** seçin (model yüklemek için gerekli)
6. **Generate a token** butonuna tıklayın
7. Token'ı kopyalayın ve **güvenli bir yerde saklayın** (tekrar gösterilmeyecek!)

### Token'ı Ortam Değişkeni Olarak Ayarlama

**Linux/macOS:**
```bash
export HF_TOKEN="your_token_here"
```

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="your_token_here"
```

**Kalıcı Olarak Kaydetmek için:**

Linux/macOS'ta `~/.bashrc` veya `~/.zshrc` dosyasına ekleyin:
```bash
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

---

## Model Yükleme

### Temel Kullanım

En basit şekilde, modelinizi şu şekilde yükleyebilirsiniz:

```bash
python upload_model.py \
    --model_path /path/to/your/best_model_effatt.h5 \
    --token YOUR_HF_TOKEN
```

### Gelişmiş Kullanım

Daha fazla kontrol için tüm parametreleri kullanabilirsiniz:

```bash
python upload_model.py \
    --model_path /path/to/your/best_model_effatt.h5 \
    --repo_id CemRoot/deepfake-detection-model \
    --token YOUR_HF_TOKEN \
    --commit_message "Update model with improved accuracy" \
    --private  # Eğer model private olsun istiyorsanız
```

### Parametreler

| Parametre | Açıklama | Varsayılan | Zorunlu |
|-----------|----------|------------|---------|
| `--model_path` | Model dosyasının (.h5) yolu | - | ✅ Evet |
| `--repo_id` | Hugging Face repo ID'si | `CemRoot/deepfake-detection-model` | ❌ Hayır |
| `--token` | HF authentication token | Ortam değişkeninden alınır | ✅ Evet* |
| `--commit_message` | Commit mesajı | `"Upload deepfake detection model"` | ❌ Hayır |
| `--private` | Repo'yu private yap | `False` (public) | ❌ Hayır |

*Token, `HF_TOKEN` ortam değişkeni olarak da sağlanabilir.

### Yükleme Süreci

Script çalıştığında şu işlemleri yapar:

1. ✅ Repository oluşturur/doğrular
2. ✅ Model dosyasını yükler (.h5)
3. ✅ Model card'ı yükler (README.md)
4. ✅ Konfigürasyon dosyasını yükler (config.json)
5. ✅ Örnek inference script'ini yükler
6. ✅ Git attributes dosyasını yükler (.gitattributes)

### Başarılı Yükleme Örneği

```
🔄 Creating repository: CemRoot/deepfake-detection-model
✅ Repository created/verified: CemRoot/deepfake-detection-model
✅ Model card created at temp_hf_upload/README.md
✅ Configuration file created at temp_hf_upload/config.json
✅ Inference example created at temp_hf_upload/inference_example.py
🔄 Uploading model file: best_model_effatt.h5
✅ Model file uploaded successfully
🔄 Uploading model card (README.md)
✅ Model card uploaded successfully
🔄 Uploading configuration file
✅ Configuration uploaded successfully
🔄 Uploading inference example
✅ Inference example uploaded successfully

============================================================
✅ Upload completed successfully!
🔗 Model available at: https://huggingface.co/CemRoot/deepfake-detection-model
============================================================
```

---

## Model Test Etme

Modelinizi Hugging Face Hub'dan test etmek için:

### Tek Bir Görüntü

```bash
python test_inference.py --image path/to/test_image.jpg
```

### Toplu İşleme (Batch)

```bash
python test_inference.py --batch path/to/images_folder/
```

### Farklı Bir Repo'dan Test Etme

```bash
python test_inference.py \
    --image test_image.jpg \
    --repo_id YourUsername/your-model-name
```

### Örnek Çıktı

```
🔄 Loading model from Hugging Face Hub: CemRoot/deepfake-detection-model
✅ Model loaded successfully!

🔍 Analyzing image: test_image.jpg

======================================================================
DETECTION RESULT
======================================================================
📁 Image: test_image.jpg
🚨 FAKE (AI-Generated)

📊 Confidence Scores:
   🚨 Fake (AI-Generated): 87.45%
   ✅ Real (Genuine):       12.55%

🎯 Overall Confidence:    87.45%
======================================================================
```

---

## Model Güncelleme

Modelinizi güncellemek için:

1. **Yeni model dosyasını hazırlayın**
2. **Upload script'ini yeni commit mesajı ile çalıştırın:**

```bash
python upload_model.py \
    --model_path /path/to/new_model.h5 \
    --commit_message "v1.1.0: Improved accuracy on diffusion models" \
    --token $HF_TOKEN
```

3. **Model card'ı güncelleyin** (gerekirse):
   - `README_MODEL.md` dosyasını düzenleyin
   - Script tekrar çalıştırıldığında otomatik olarak yüklenecek

---

## En İyi Uygulamalar

### 1. Versiyonlama

Model güncellemelerinde anlamlı commit mesajları kullanın:

```bash
# İyi örnekler ✅
"v1.1.0: Improved StyleGAN detection accuracy by 5%"
"v1.2.0: Added support for DALL-E 3 detection"
"Fix: Corrected preprocessing normalization bug"

# Kötü örnekler ❌
"update"
"new model"
"fix"
```

### 2. Model Card Güncel Tutma

`README_MODEL.md` dosyasını her zaman güncel tutun:
- Yeni desteklenen generative modelleri ekleyin
- Performans metriklerini güncelleyin
- Bilinen sınırlamaları belirtin
- Versiyon geçmişini kaydedin

### 3. Konfigürasyon Yönetimi

`config.json` dosyasını güncelleyin:
- Model versiyonunu artırın
- Yeni özellikleri belirtin
- Değişiklikleri dokümante edin

### 4. Test ve Doğrulama

Her yüklemeden sonra:

```bash
# Model'i test edin
python test_inference.py --image test_fake.jpg
python test_inference.py --image test_real.jpg

# Batch test yapın
python test_inference.py --batch test_images/
```

### 5. Güvenlik

- ⚠️ **Token'ınızı asla commit etmeyin!**
- ✅ Ortam değişkenleri kullanın
- ✅ `.gitignore` dosyasına token dosyalarını ekleyin
- ✅ Token'ları güvenli bir şekilde saklayın

### 6. Model Dosya Boyutu

- Model dosyası 780 MB civarında olmalı
- Git LFS otomatik olarak büyük dosyaları yönetir
- `.gitattributes` dosyası doğru yapılandırılmış

---

## Örnek Workflow

### İlk Yükleme

```bash
# 1. Gereksinimleri yükle
pip install -r requirements.txt

# 2. Git LFS kur
git lfs install

# 3. Token'ı ayarla
export HF_TOKEN="your_token_here"

# 4. Model'i yükle
python upload_model.py \
    --model_path ../models/best_model_effatt.h5 \
    --repo_id CemRoot/deepfake-detection-model

# 5. Test et
python test_inference.py --image test_image.jpg
```

### Model Güncelleme

```bash
# 1. Yeni modeli yükle
python upload_model.py \
    --model_path ../models/best_model_effatt_v2.h5 \
    --commit_message "v2.0.0: Major architecture improvements"

# 2. Test et
python test_inference.py --batch validation_set/

# 3. Model card'ı güncelle (gerekirse)
# README_MODEL.md'yi düzenle ve tekrar yükle
```

---

## Sorun Giderme

### Problem: "Token not provided" hatası

**Çözüm:**
```bash
# Token'ı ortam değişkeni olarak ayarlayın
export HF_TOKEN="your_token_here"

# Veya script'e doğrudan verin
python upload_model.py --model_path model.h5 --token your_token_here
```

### Problem: "Repository already exists" hatası

**Çözüm:** Bu normal bir durumdur. Script otomatik olarak mevcut repo'yu kullanacak ve güncelleme yapacaktır.

### Problem: Git LFS hatası

**Çözüm:**
```bash
# Git LFS'i yeniden başlatın
git lfs install

# LFS track'leri kontrol edin
git lfs track
```

### Problem: Yükleme çok yavaş

**Çözüm:**
- İnternet bağlantınızı kontrol edin
- Model dosyasının boyutunu kontrol edin (~780 MB normal)
- Git LFS'in doğru kurulu olduğundan emin olun

### Problem: Model yüklenemedi (inference sırasında)

**Çözüm:**
```bash
# TensorFlow versiyonunu kontrol edin
python -c "import tensorflow; print(tensorflow.__version__)"

# En az 2.15.0 olmalı
pip install --upgrade tensorflow>=2.15.0
```

---

## Ek Kaynaklar

### Hugging Face Dokümantasyonu
- [Model Hub Documentation](https://huggingface.co/docs/hub/models)
- [Uploading Models Guide](https://huggingface.co/docs/hub/models-uploading)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [Git LFS Guide](https://huggingface.co/docs/hub/security-git-lfs)

### Proje Kaynakları
- [GitHub Repository](https://github.com/CemRoot/deepfake-detection-streamlit)
- [Streamlit Demo](https://your-app-url.streamlit.app)
- [Model Hub](https://huggingface.co/CemRoot/deepfake-detection-model)

---

## İletişim

Sorularınız veya sorunlarınız için:
- **GitHub Issues**: [Create an issue](https://github.com/CemRoot/deepfake-detection-streamlit/issues)
- **Email**: GitHub profilinde mevcut
- **Website**: [cemkoyluoglu.codes](https://cemkoyluoglu.codes/)

---

**Son Güncelleme**: 2025-01-15
**Yazar**: Emin Cem Koyluoglu
**Lisans**: MIT
