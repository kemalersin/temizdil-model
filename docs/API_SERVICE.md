# api_service.py Teknik Dokümantasyonu

Bu dokümantasyon, Temiz Dil API'nin ana motor dosyası olan `api_service.py`'nin yapısını ve işlevlerini açıklamaktadır. Dosya, Flask web uygulamasını, makine öğrenimi modelini, veritabanı işlemlerini ve API endpoint'lerini içerir.

## İçindekiler

1. [Genel Mimari](#genel-mimari)
2. [Modeller ve Bileşenler](#modeller-ve-bileşenler)
3. [Veritabanı İşlemleri](#veritabanı-i̇şlemleri)
4. [Kullanıcı Kimlik Doğrulama ve Yetkilendirme](#kullanıcı-kimlik-doğrulama-ve-yetkilendirme)
5. [API Endpoint'leri](#api-endpointleri)
6. [Rate Limiting ve Kullanım Takibi](#rate-limiting-ve-kullanım-takibi)
7. [Admin İşlevleri](#admin-i̇şlevleri)
8. [Hata İşleme](#hata-i̇şleme)
9. [Performans Optimizasyonu](#performans-optimizasyonu)
10. [Güvenlik Önlemleri](#güvenlik-önlemleri)
11. [Loglar ve İzleme](#loglar-ve-i̇zleme)
12. [Refactoring Önerileri](#refactoring-önerileri)

## Genel Mimari

`api_service.py` aşağıdaki ana bileşenlerden oluşur:

- **Flask Uygulaması**: Web API'yi sağlayan temel framework
- **PyTorch Modeli**: Metin sınıflandırması için kullanılan makine öğrenimi modeli
- **MySQL Veritabanı**: Kullanıcı bilgileri, API anahtarları ve kullanım istatistikleri için veri depolama
- **Middleware**: İstek işleme, kimlik doğrulama ve yetkilendirme için ara yazılım katmanı
- **Rate Limiting**: İstek sayısı ve token kullanımı sınırlamalarını yöneten mekanizmalar

Uygulama, Flask routing sistemi üzerine kurulmuştur ve RESTful API prensiplerini takip eder. Tüm API cevapları JSON formatında döndürülür.

## Modeller ve Bileşenler

### HierarchicalOffensiveClassifier

`HierarchicalOffensiveClassifier` sınıfı, metinlerdeki saldırgan içeriği tespit etmek için kullanılan ana neural network modelidir. PyTorch ve Hugging Face Transformers kütüphanelerine dayanır.

```python
class HierarchicalOffensiveClassifier(nn.Module):
    def __init__(self, model_name, num_labels=5):
        # Model yapılandırması
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # İleri geçiş işlemi
```

Model hiyerarşik bir yapıya sahiptir ve şu sınıflandırıcıları içerir:
- Saldırgan içerik sınıflandırıcısı (offensive_classifier)
- Hedeflenen içerik sınıflandırıcısı (targeted_classifier)
- Hedef tipi sınıflandırıcısı (target_type_classifier)
- Çoklu etiket sınıflandırıcısı (multi_label_classifier)
- Zorluk tahmini sınıflandırıcısı (difficulty_classifier)

### Tahmin İşlevleri

Kullanıcı isteklerini işleyen ve modelin tahminlerini yorumlayan işlevler:

```python
def predict_offensive_content(model, tokenizer, text):
    # Model tahmini yapan fonksiyon
    
def interpret_predictions(predictions, labels):
    # Model çıktılarını kullanılabilir sonuçlara dönüştüren fonksiyon
```

## Veritabanı İşlemleri

API verilerini saklamak için MySQL kullanılmaktadır. Bağlantı havuzu yaklaşımı, eşzamanlı istekleri verimli bir şekilde yönetmeye yardımcı olur.

### Tablo Yapısı

Veritabanı üç ana tablodan oluşur:

1. **api_keys**: API anahtarları ve kullanım limitleri için
   ```sql
   CREATE TABLE IF NOT EXISTS api_keys (
       id INT AUTO_INCREMENT PRIMARY KEY,
       api_key VARCHAR(64) NOT NULL UNIQUE,
       description VARCHAR(255),
       is_unlimited BOOLEAN DEFAULT FALSE,
       unlimited_ips TEXT,
       monthly_token_limit INT DEFAULT 1000,
       tokens_used INT DEFAULT 0,
       auto_reset BOOLEAN DEFAULT TRUE,
       last_reset_date DATETIME,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
   )
   ```

2. **ip_rate_limits**: IP bazlı kısıtlamalar için
   ```sql
   CREATE TABLE IF NOT EXISTS ip_rate_limits (
       id INT AUTO_INCREMENT PRIMARY KEY,
       ip_address VARCHAR(45) NOT NULL UNIQUE,
       monthly_token_limit INT DEFAULT 10000,
       tokens_used INT DEFAULT 0,
       request_count INT DEFAULT 0,
       last_request_time TIMESTAMP,
       last_reset_date DATETIME,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
   )
   ```

3. **api_usage_logs**: API kullanım kayıtları için
   ```sql
   CREATE TABLE IF NOT EXISTS api_usage_logs (
       id INT AUTO_INCREMENT PRIMARY KEY,
       api_key_id INT,
       request_ip VARCHAR(45),
       endpoint VARCHAR(255),
       text_length INT,
       tokens_used INT,
       is_successful BOOLEAN,
       error_message TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE SET NULL
   )
   ```

### Veritabanı Yardımcı Fonksiyonları

```python
def init_db_pool():
    # Veritabanı bağlantı havuzunu başlatır
    
def create_schema():
    # Gerekli tabloları oluşturur
    
def get_api_key_info(api_key):
    # API anahtarı bilgilerini alır
    
def update_token_usage(api_key_id, tokens_used):
    # Kullanılan token sayısını günceller
    
def log_api_usage(api_key_id, request_ip, endpoint, text_length, tokens_used, is_successful, error_message=None):
    # API kullanımını loglar
    
def get_or_create_ip_info(ip_address):
    # IP adresi için kullanım bilgilerini alır veya oluşturur
```

## Kullanıcı Kimlik Doğrulama ve Yetkilendirme

API, iki tür kimlik doğrulama mekanizması kullanır:

1. **API Anahtarı Kimlik Doğrulaması**: Standart API istekleri için
2. **Admin Şifresi Kimlik Doğrulaması**: Admin panel erişimi için

### Dekoratörler

```python
def require_api_key(f):
    # API anahtarı gerektiren endpoint'ler için dekoratör
    
def admin_required(f):
    # Admin yetkisi gerektiren endpoint'ler için dekoratör
```

## API Endpoint'leri

### Genel API Endpoint'leri

```python
@app.route('/health', methods=['GET'])
def health_check():
    # Servisin sağlık durumunu kontrol eder
    
@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # Tek bir metin tahmini yapar
    
@app.route('/batch_predict', methods=['POST'])
@require_api_key
def batch_predict():
    # Birden fazla metin tahmini yapar
    
@app.route('/usage_info', methods=['GET'])
@require_api_key
def usage_info():
    # Kullanıcının kullanım bilgilerini döndürür
```

### Admin API Endpoint'leri

```python
@app.route('/admin', methods=['GET'])
def admin_panel():
    # Admin panelini gösterir
    
@app.route('/admin/login', methods=['POST'])
def admin_login():
    # Admin girişi yapar
    
@app.route('/admin/logout')
def admin_logout():
    # Admin çıkışı yapar
    
@app.route('/admin/list_api_keys')
@admin_required
def list_api_keys():
    # API anahtarlarını listeler
    
@app.route('/admin/get_api_key/<int:key_id>', methods=['GET'])
@admin_required
def get_api_key(key_id):
    # Belirli bir API anahtarı bilgisini getirir
    
@app.route('/admin/create_api_key', methods=['POST'])
@admin_required
def create_api_key():
    # Yeni API anahtarı oluşturur
    
@app.route('/admin/update_api_key/<int:key_id>', methods=['PUT'])
@admin_required
def update_api_key(key_id):
    # API anahtarını günceller
    
@app.route('/admin/delete_api_key/<int:key_id>', methods=['DELETE'])
@admin_required
def delete_api_key(key_id):
    # API anahtarını siler
    
@app.route('/admin/list_ip_usage')
@admin_required
def list_ip_usage():
    # IP kullanım bilgilerini listeler
    
@app.route('/admin/reset_ip_limits/<string:ip_address>', methods=['POST'])
@admin_required
def reset_ip_limits(ip_address):
    # IP limitlerini sıfırlar
    
@app.route('/admin/usage_summary')
@admin_required
def usage_summary():
    # Kullanım özetini gösterir
```

## Rate Limiting ve Kullanım Takibi

API, iki tür kullanım sınırlaması uygular:

1. **Token Tabanlı Sınırlama**: Aylık kullanılabilecek token sayısı (karakter sayısına göre hesaplanır)
2. **İstek Bazlı Sınırlama**: IP bazlı, belirli bir sürede yapılabilecek maksimum istek sayısı

```python
def calculate_tokens(text):
    # Metin için gereken token sayısını hesaplar
    
def update_ip_request_count(ip_id):
    # IP için istek sayısını günceller
    
def reset_ip_request_count(ip_id):
    # IP için istek sayısını sıfırlar
    
def can_ip_make_request(ip_info):
    # IP'nin istek yapıp yapamayacağını kontrol eder
    
def log_ip_request(ip_address, endpoint, text_length, tokens_used, is_successful, error_message=None):
    # IP bazlı istekleri loglar
```

## Admin İşlevleri

Admin paneli, API'nin yönetimi için bir dizi işlev sunar:

- API anahtarlarını listeleme, oluşturma, düzenleme ve silme
- IP kullanım bilgilerini listeleme ve limitleri sıfırlama
- Genel kullanım istatistiklerini görüntüleme

## Hata İşleme

API, çeşitli hata durumlarını işlemek için bir dizi mekanizma sağlar:

```python
def handle_api_error(e):
    # API hatalarını işler ve uygun HTTP yanıtını döndürür
    
# Belirli hata türleri için HTTP durum kodları ve mesajlar:
# 400: Bad Request
# 401: Unauthorized
# 403: Forbidden (limit aşımı veya yetkisiz erişim)
# 404: Not Found
# 500: Internal Server Error
```

## Performans Optimizasyonu

API, yüksek performans için çeşitli optimizasyonlar kullanır:

1. **Veritabanı Bağlantı Havuzu**: Eşzamanlı istekleri verimli bir şekilde yönetir
2. **Token Hesaplama**: Basit ve hızlı bir token hesaplama algoritması kullanılır
3. **Model Yükleme**: Model bir kez yüklenir ve tüm istekler için yeniden kullanılır
4. **Cache Mekanizmaları**: Sık kullanılan veriler için önbellek kullanımı

## Güvenlik Önlemleri

API, güvenliği sağlamak için çeşitli önlemler alır:

1. **API Anahtarı Doğrulama**: Her istek için API anahtarı kontrolü
2. **Admin Şifre Koruması**: Admin panel erişimi için şifre koruması
3. **Rate Limiting**: Kötüye kullanımı önlemek için istek sınırlaması
4. **IP Filtreleme**: Özel API anahtarları için IP kısıtlaması desteği
5. **Session Güvenliği**: Güvenli session yönetimi
6. **Hata Mesajlarının Sınırlandırılması**: Hassas bilgilerin açığa çıkmasını önleme

## Loglar ve İzleme

API, çeşitli işlemleri loglamak için Python'un logging modülünü kullanır:

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Çeşitli log seviyeleri kullanılır:
# logger.debug(): Detaylı hata ayıklama bilgileri
# logger.info(): Genel bilgi mesajları
# logger.warning(): Uyarı mesajları
# logger.error(): Hata mesajları
# logger.critical(): Kritik hatalar
```

## Refactoring Önerileri

Kodun geliştirilmesi için bazı öneriler:

1. **Modüler Yapı**: Kodu daha küçük, özel amaçlı modüllere ayırma:
   - `models.py`: Model sınıfları ve tahmin işlevleri
   - `database.py`: Veritabanı işlemleri
   - `auth.py`: Kimlik doğrulama ve yetkilendirme
   - `routes/`: API endpoint'leri için ayrı modüller
   - `utils.py`: Yardımcı fonksiyonlar

2. **Yapılandırma Yönetimi**: Çeşitli ortamlar için yapılandırma ayarlarını dışa aktarma

3. **Daha Fazla Birim Test**: Kodun güvenilirliğini ve bakımını kolaylaştırmak için test kapsamını artırma

4. **Asenkron İşleme**: Yoğun hesaplama gerektiren işlemler için asenkron işleme ekleme

5. **Docker Entegrasyonu**: Dağıtımı ve ölçeklendirmeyi kolaylaştırmak için Docker desteği

6. **API Belgelendirme**: Swagger/OpenAPI entegrasyonu ile otomatik API dokümantasyonu

7. **Hata İzleme**: Sentry gibi bir hata izleme servisi entegrasyonu

8. **Önbellek Mekanizması**: Redis gibi bir önbellek sistemi entegrasyonu

9. **Daha Sağlam Güvenlik**: HTTPS zorlaması, CORS politikaları, API anahtarı rotasyonu

10. **Daha Detaylı Metrikler**: Prometheus entegrasyonu ile daha detaylı izleme ve metrik toplama