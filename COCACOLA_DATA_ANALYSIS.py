# ============================================================
# Coca-Cola Time Series Analysis
# Author: Özge Güneş
# Description: TR–EN bilingual, auto-saving, optimized Prophet forecasting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, logging, os
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")

# ============================================================
# GLOBAL SETTINGS
# ============================================================
plt.style.use("seaborn-v0_8")
sns.set_palette("deep")
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3
})

# ============================================================
#  TR–EN LABEL SYSTEM
# ============================================================
LABEL_MAP = {
    "Coca-Cola Kapanış Fiyatı Zaman Serisi": "Coca-Cola Closing Price Time Series",
    "Açılış vs Kapanış Fiyatı Korelasyonu": "Opening vs Closing Price Correlation",
    "Kapanış Fiyatı ve İşlem Hacmi İlişkisi": "Closing Price vs Trading Volume",
    "Coca-Cola Günlük Getiri Dağılımı": "Coca-Cola Daily Return Distribution",
    "Coca-Cola 30 Günlük Volatilite Trendleri": "Coca-Cola 30-Day Volatility Trends",
    "Coca-Cola 90 Günlük Fiyat Tahmini": "Coca-Cola 90-Day Price Forecast",
    "Gerçek vs Tahmin Edilen Fiyat": "Actual vs Predicted Price",
    "Tahmin Hata Trendi (MAE)": "Forecast Error Trend (MAE)",
    "Trend": "Trend",
    "Haftalık Mevsimsellik": "Weekly Seasonality",
    "Yıllık Mevsimsellik": "Yearly Seasonality",
    "Günlük Döngü": "Daily Cycle",
    "Standart Sapma": "Standard Deviation",
    "Frekans": "Frequency",
    "Getiri": "Return",
    "Tarih": "Date",
    "USD": "USD",
    "Volatilite": "Volatility",
    "Tahmin Ufku (Gün)": "Forecast Horizon (Days)",
    "Ortalama Mutlak Hata": "Mean Absolute Error",
    "Günlük Fiyat Tahmini": "Daily Price Forecast",
    "Açılış": "Open",
    "Kapanış": "Close",
    "İşlem Hacmi": "Trading Volume",
}

def translate_label(label: str) -> str:
    if not label:
        return ""
    if " / " in label:
        return label
    return f"{label} / {LABEL_MAP.get(label, label)}"

def axis_label(xlabel=None, ylabel=None):
    if xlabel:
        plt.xlabel(translate_label(xlabel))
    if ylabel:
        plt.ylabel(translate_label(ylabel))

def plot_title(label: str, fontsize=14, weight="bold"):
    plt.title(translate_label(label), fontsize=fontsize, weight=weight)

# ============================================================
#  AUTO SAVE
# ============================================================
def save_plot(fig, name, save_plots=True):
    if not save_plots:
        return
    os.makedirs("plots", exist_ok=True)
    path = f"plots/{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f" Kaydedildi: {path}")

# ============================================================
#  LOAD DATA
# ============================================================
def load_data(path):
    df = pd.read_csv(path)
    print(f"\n Veri yüklendi! Satır: {df.shape[0]}, Sütun: {df.shape[1]}")
    return df

# ============================================================
#  CLEANING
# ============================================================
def analyze_data(df):
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["Date"]).sort_values("Date")
        print(" Date sütunu datetime tipine dönüştürüldü ve sıralandı.")
    total_missing = df.isna().sum().sum()
    if total_missing > 0:
        print(f" {total_missing} eksik değer bulundu → dolduruluyor...")
        df = df.fillna(method="ffill")
    else:
        print(" Hiç eksik veri bulunamadı.")
    return df

# ============================================================
#  VISUALS
# ============================================================
def plot_close_price(df, save_plots=True):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df["Date"], df["Close"], color="red", linewidth=2)
    plot_title("Coca-Cola Kapanış Fiyatı Zaman Serisi")
    axis_label("Tarih", "USD")
    plt.tight_layout()
    save_plot(fig, "Closing_Price_Time_Series", save_plots)
    plt.show()

def plot_open_close_correlation(df, save_plots=True):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x="Open", y="Close", data=df, alpha=0.6, ax=ax)
    plot_title("Açılış vs Kapanış Fiyatı Korelasyonu")
    axis_label("Açılış", "Kapanış")
    plt.tight_layout()
    save_plot(fig, "Open_vs_Close_Correlation", save_plots)
    plt.show()

def plot_volume_and_price(df, save_plots=True):
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(df["Date"], df["Close"], color="red", label="Kapanış")
    ax1.set_ylabel(translate_label("USD"), color="red")
    ax2 = ax1.twinx()
    ax2.bar(df["Date"], df["Volume"], alpha=0.3, color="blue", label="Volume")
    ax1.set_xlabel(translate_label("Tarih"))
    ax1.set_title(translate_label("Kapanış Fiyatı ve İşlem Hacmi İlişkisi"))
    plt.tight_layout()
    save_plot(fig, "ClosingPrice_TradingVolume_Relationship", save_plots)
    plt.show()

# ============================================================
#  RETURN & VOLATILITY
# ============================================================
def analyze_returns(df, save_plots=True):
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=30).std()

    # Histogram
    fig, ax = plt.subplots()
    sns.histplot(df["Return"].dropna(), bins=50, kde=True, color="purple", ax=ax)
    plot_title("Coca-Cola Günlük Getiri Dağılımı")
    axis_label("Getiri", "Frekans")
    plt.tight_layout()
    save_plot(fig, "DailyReturn_Distribution", save_plots)
    plt.show()

    # Volatility
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df["Date"], df["Volatility"], color="orange")
    plot_title("Coca-Cola 30 Günlük Volatilite Trendleri")
    axis_label("Tarih", "Standart Sapma")
    plt.tight_layout()
    save_plot(fig, "30Day_Volatility_Trends", save_plots)
    plt.show()

    return df

# ============================================================
#  PROPHET (FAST MODE)
# ============================================================
def forecast_price(df, periods=90, save_plots=True):
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    logging.getLogger("prophet").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    print(f"\n Prophet modeli başlatılıyor... ({periods} günlük tahmin)")

    df_p = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df_p)

    # Forecast
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Tahmin CSV olarak kaydet
    os.makedirs("results", exist_ok=True)
    forecast[["ds","yhat","yhat_lower","yhat_upper"]].to_csv("results/forecast_results.csv", index=False)
    print(" Tahmin sonuçları kaydedildi: results/forecast_results.csv")

    # 1️⃣ Ana Grafik
    fig1 = model.plot(forecast)
    plt.title(translate_label(f"Coca-Cola {periods} Günlük Fiyat Tahmini"))
    axis_label("Tarih", "USD")
    plt.tight_layout()
    save_plot(fig1, "Daily_Price_Forecast", save_plots)
    plt.show()

    # 2️⃣ Bileşenler
    fig2 = model.plot_components(forecast)
    for ax in fig2.axes:
        t = ax.get_title().lower()
        ax.set_title("")
        if "trend" in t:
            ax.set_title(translate_label("Trend"))
            axis_label("Tarih", "USD")
        elif "weekly" in t:
            ax.set_title(translate_label("Haftalık Mevsimsellik"))
        elif "yearly" in t:
            ax.set_title(translate_label("Yıllık Mevsimsellik"))
    plt.tight_layout()
    save_plot(fig2, "Trend_Seasonality_Components", save_plots)
    plt.show()

    # 3️⃣ Cross Validation (Hızlı)
    print("\n Cross validation (FAST MODE) çalışıyor...")
    df_cv = cross_validation(model, initial="2000 days", period="365 days", horizon="180 days")
    df_perf = performance_metrics(df_cv)
    df_perf["horizon_days"] = df_perf["horizon"].dt.days

    fig3, ax = plt.subplots()
    ax.plot(df_perf["horizon_days"], df_perf["mae"], marker="o", color="steelblue")
    plot_title("Tahmin Hata Trendi (MAE)")
    axis_label("Tahmin Ufku (Gün)", "Ortalama Mutlak Hata")
    plt.tight_layout()
    save_plot(fig3, "Forecast_Error_Trend(MAE)", save_plots)
    plt.show()

    print("\n Prophet tahmini tamamlandı.")
    return forecast

# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    df = load_data("Coca_Cola_historical_data.csv")
    df = analyze_data(df)
    plot_close_price(df)
    plot_open_close_correlation(df)
    plot_volume_and_price(df)
    df = analyze_returns(df)
    forecast = forecast_price(df, periods=90)
    print("\n✅ Tüm analiz ve tahmin işlemleri tamamlandı!")

