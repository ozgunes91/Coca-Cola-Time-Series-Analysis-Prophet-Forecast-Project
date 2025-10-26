
# 🥤 Coca-Cola Stock Forecasting — Prophet Model  
**Prepared by / Hazırlayan:** Özge Güneş  
**Date:** 2025  

---

## 🎯 Project Objective / Proje Amacı  
Bu proje, **The Coca-Cola Company (KO:NYSE)** hissesinin 1962–2025 yılları arasındaki fiyat verileriyle 90 günlük fiyat tahmini yapmayı amaçlamaktadır.  
Zaman serisi analizi, Prophet modeli ile trend, mevsimsellik ve volatilite bileşenlerine ayrılmıştır.  

---

## 📊 Dataset & Source / Veri Seti ve Kaynak  
- **Kaynak:** Kaggle → *Yahoo Finance Historical Data*  
- **Dönem:** 1962 – 2025  
- **Gözlem Sayısı:** 16.000+  
- **Değişkenler (8 columns):**  
  | Column | Description |
  |---------|-------------|
  | Date | İşlem tarihi (datetime64 formatına dönüştürülmüştür) |
  | Open | Günün açılış fiyatı (USD) |
  | High | Günün en yüksek fiyatı |
  | Low | Günün en düşük fiyatı |
  | Close | Günün kapanış fiyatı |
  | Volume | Günlük işlem hacmi |
  | ticker | KO (hisse kodu) |
  | name | The Coca-Cola Company |

📎 **Ek özellikler (engineered features):**  
```python
df['Return'] = df['Close'].pct_change()
df['Volatility'] = df['Return'].rolling(window=30).std()
```

🧹 **Temizlik Adımları:**  
- `Date` sütunu datetime tipine dönüştürülüp sıralandı  
- Eksik değer bulunmadı  
- Aykırı değerler silinmedi (piyasa şoklarını korumak için)  

---

## ⚙️ Modeling Approach / Modelleme Yaklaşımı  
Model: **Prophet(daily_seasonality=True, interval_width=0.95)**  
Formül:  
\[
y(t) = Trend + Seasonality + Noise
\]

### Neden Prophet?
- Finansal zaman serilerinde **trend ve mevsimselliği ayrıştırmak** kolaydır.  
- Eksik günler ve düzensiz tarih aralıklarıyla başa çıkabilir.  
- %95 güven aralığı (`yhat_lower`, `yhat_upper`) ile belirsizliği ölçer.

---

## 📈 Forecast Results / Tahmin Sonuçları  
- 2026 Ocak ortası için ortalama tahmin: **64.5 USD**  
- %95 Güven Aralığı: **62.0 – 66.9 USD**  
- Ortalama Mutlak Hata (MAE): **≈ 1.8 USD**  

🧮 Prophet tahmini:  
```python
model = Prophet(daily_seasonality=True, interval_width=0.95)
forecast = model.fit(df_p).predict(future)
```

Bu tahmin bandı, gerçek fiyatın **%95 olasılıkla** 62–67 USD aralığında kalacağını gösterir.  
Bu oran Prophet’in istatistiksel belirsizlik modellemesine dayalıdır, deterministik bir doğruluk oranı değildir.

---

## 📉 Validation & Metrics / Doğrulama & Metrikler  
- **Cross Validation:**  
```python
df_cv = cross_validation(model, initial="2000 days", period="365 days", horizon="180 days")
df_perf = performance_metrics(df_cv)
```
- **MAE (Mean Absolute Error):** 1.8 USD  
- **RMSE (Root Mean Square Error):** 2.3 USD  
- **MAPE (Mean Absolute Percentage Error):** 2.7%  

---

## 🔍 Key Insights / Öne Çıkan İçgörüler  
- Volatilite (30 günlük) 1987, 2008 ve 2020’de zirve yapmıştır.  
- Getiri dağılımı yaklaşık normaldir, ortalama getiri ≈ 0.  
- Kapanış ve açılış fiyatları arasında korelasyon ≈ 0.999’dur.  
- Prophet kısa vadeli tahminlerde tutarlı sonuçlar vermiştir.  

---

## 🚀 Future Work / Gelecek Çalışmalar  
- SARIMAX veya LSTM modelleriyle Prophet performans karşılaştırması  
- Makro ekonomik göstergelerin dahil edilmesi (S&P500, enflasyon, faiz)  
- Otomatik pipeline (Airflow / Streamlit dashboard)  

---

## 🧠 Tools & Libraries / Araçlar  
`Python · pandas · Prophet · matplotlib · seaborn · numpy · scikit-learn`

---

## ✍️ Author / Yazar   
© Özge Güneş | Food Engineer & Aspiring Data Science Specialist 
> Veri bilimi, geçmişin desenlerini çözerek geleceğe anlam kazandırır.  
