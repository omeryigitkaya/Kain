import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import yaml
from pypfopt import BlackLittermanModel, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.exceptions import OptimizationError
import io
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
import os
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import requests

# --- Gerekli Ayarlar ---
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
st.set_page_config(layout="wide", page_title="Finansal Asistan")
plt.style.use('seaborn-v0_8-darkgrid')

# =======================================================
# BÖLÜM 1: TÜM YARDIMCI FONKSİYONLAR
# =======================================================

def cizim_yap_agirliklar(weights, ax=None):
    if ax is None: fig, ax = plt.subplots()
    labels = list(weights.keys()); sizes = list(weights.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90); ax.axis('equal')
    return ax.get_figure()

@st.cache_data(show_spinner=False)
def piyasa_rejimini_belirle():
    # ... (Fonksiyon içeriği değişmedi)
    st.write("Piyasa rejimi analiz ediliyor...")
    rejim_gostergeleri = {"NASDAQ": {"ticker": "^IXIC", "yon": "yukari"},"BIST 100": {"ticker": "XU100.IS", "yon": "yukari"},"Altın": {"ticker": "GC=F", "yon": "yukari"},"Bitcoin": {"ticker": "BTC-USD", "yon": "yukari"},"ABD 10Y Faiz": {"ticker": "^TNX", "yon": "asagi"}}
    toplam_puan = 0; puan_detaylari = {}
    for isim, info in rejim_gostergeleri.items():
        veri = None; deneme_sayisi=3
        for deneme in range(deneme_sayisi):
            try:
                veri = yf.download(info['ticker'], period="2y", progress=False, auto_adjust=True)
                if veri is not None and not veri.empty: break
                time.sleep(1)
            except Exception: time.sleep(1)
        try:
            if veri is None or veri.empty: raise ValueError("Veri indirilemedi.")
            veri['MA200'] = veri['Close'].rolling(window=200).mean()
            son_fiyat = veri['Close'].iloc[-1]; son_ma200 = veri['MA200'].iloc[-1]
            if not np.isfinite(son_fiyat) or not np.isfinite(son_ma200): raise ValueError("Fiyat/MA200 geçersiz.")
            puan = 1 if (info['yon'] == 'yukari' and son_fiyat > son_ma200) or (info['yon'] == 'asagi' and son_fiyat < son_ma200) else -1
            toplam_puan += puan; puan_detaylari[isim] = "POZİTİF (+1)" if puan == 1 else "NEGATİF (-1)"
        except Exception as e: puan_detaylari[isim] = f"İşlenemedi (0) - {e}"
    if toplam_puan >= 3: rejim = "GÜÇLÜ POZİTİF (BOĞA 🐂🐂)"
    elif toplam_puan >= 1: rejim = "TEMKİNLİ POZİTİF (BOĞA 🐂)"
    else: rejim = "TEMKİNLİ NEGATİF (AYI 🐻)"
    return rejim

# Bu fonksiyon artık sadece TEK bir DataFrame alıyor
@st.cache_data
def auto_format_tickers(df):
    formatted_list = []; commodity_map = {"GOLD": "GC=F", "SILVER": "SI=F", "XAUUSD": "GC=F", "XAGUSD": "SI=F", "WTI": "CL=F", "CRUDE": "CL=F", "OIL": "CL=F", "COPPER": "HG=F", "NATURALGAS": "NG=F"}; crypto_suffixes = ["USDT", "PERP", "BUSD", "USDC"]; crypto_exchanges = ["CRYPTO", "BINANCE", "COINBASE", "KUCOIN", "KRAKEN", "COIN", "KIN"]
    df.columns = df.columns.str.lower().str.strip()
    symbol_col = 'sembol' if 'sembol' in df.columns else 'symbol'; exchange_col = 'borsa' if 'borsa' in df.columns else 'exchange'
    if symbol_col not in df.columns: raise ValueError("CSV'de en azından ('Sembol'/'Symbol') sütunu bulunmalıdır!")
    for index, row in df.iterrows():
        ticker = str(row[symbol_col]).upper(); exchange = str(row.get(exchange_col, '')).upper()
        if ticker in commodity_map: formatted_list.append(commodity_map[ticker]); continue
        is_crypto_by_exchange = any(ex in exchange for ex in crypto_exchanges)
        if is_crypto_by_exchange:
            clean_ticker = ticker;
            for suffix in crypto_suffixes: clean_ticker = clean_ticker.replace(suffix, "")
            formatted_list.append(f"{clean_ticker}-USD"); continue
        if "BIST" in exchange or "XIST" in exchange: formatted_list.append(f"{ticker}.IS"); continue
        is_crypto_by_suffix = False
        for suffix in crypto_suffixes:
            if ticker.endswith(suffix):
                clean_ticker = ticker.replace(suffix, ""); formatted_list.append(f"{clean_ticker}-USD"); is_crypto_by_suffix = True; break
        if is_crypto_by_suffix: continue
        formatted_list.append(ticker)
    return list(set(formatted_list))

@st.cache_data
def veri_cek_ve_dogrula(tickers, start, end):
    # ... (Fonksiyon içeriği değişmedi)
    gecerli_datalar = {}; gecersiz_tickerlar = []
    progress_bar = st.progress(0, text="Varlıklar doğrulanıyor...")
    for i, ticker in enumerate(tickers):
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if not df.empty and 'Close' in df.columns and not df['Close'].dropna().empty and len(df.resample('W-FRI').last()) > 60:
            gecerli_datalar[ticker] = df['Close'].resample('W-FRI').last()
        else:
            gecersiz_tickerlar.append(ticker)
    progress_bar.progress((i + 1) / len(tickers), text=f"Varlık doğrulanıyor: {ticker}")
    progress_bar.empty()
    if gecersiz_tickerlar: st.warning(f"Şu varlıklar için yeterli veri bulunamadı: {gecersiz_tickerlar}")
    if not gecerli_datalar: return pd.DataFrame()
    gecerli_tickerlar = list(gecerli_datalar.keys())
    st.info(f"Analize devam edilecek geçerli varlıklar: {gecerli_tickerlar}")
    close_prices_df = pd.concat(gecerli_datalar, axis=1)
    return close_prices_df.ffill().dropna()

@st.cache_data
def sinyal_uret_ensemble_lstm(fiyat_verisi, look_back_periods=[12, 26, 52]):
    # ... (Fonksiyon içeriği değişmedi)
    predictions = []
    for look_back in look_back_periods:
        try:
            scaler = MinMaxScaler(feature_range=(0, 1)); scaled_data = scaler.fit_transform(fiyat_verisi.values.reshape(-1, 1)); X_train, y_train = [], []
            for i in range(look_back, len(scaled_data)):
                X_train.append(scaled_data[i-look_back:i, 0]); y_train.append(scaled_data[i, 0])
            if not X_train: continue
            X_train, y_train = np.array(X_train), np.array(y_train); X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            model = Sequential([LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)), Dropout(0.2), LSTM(units=50, return_sequences=False), Dropout(0.2), Dense(units=1)])
            model.compile(optimizer='adam', loss='mean_squared_error'); model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)
            last_look_back_weeks = scaled_data[-look_back:]; X_test = np.array([last_look_back_weeks.flatten()]); X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_scaled = model.predict(X_test, verbose=0); predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
            predictions.append(predicted_price)
        except Exception: continue
    last_known_price = fiyat_verisi.iloc[-1]
    if not predictions: return {"tahmin_yuzde": 0.0, "son_fiyat": last_known_price, "hedef_fiyat": last_known_price}
    ortalama_hedef_fiyat = np.mean(predictions)
    percentage_change = ((ortalama_hedef_fiyat - last_known_price) / last_known_price)
    if not np.isfinite(percentage_change): return {"tahmin_yuzde": 0.0, "son_fiyat": last_known_price, "hedef_fiyat": last_known_price}
    return {"tahmin_yuzde": percentage_change, "son_fiyat": last_known_price, "hedef_fiyat": ortalama_hedef_fiyat}

@st.cache_data
def sinyal_uret_duyarlilik(ticker):
    # ... (Fonksiyon içeriği değişmedi)
    try:
        stock = yf.Ticker(ticker); news = stock.news
        if not news: return 0.0
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(article['title'])['compound'] for article in news]
        return np.mean(scores) if scores else 0.0
    except Exception: return 0.0

@st.cache_data
def portfoyu_optimize_et(sinyaller, fiyat_verisi, piyasa_rejimi):
    # ... (Fonksiyon içeriği değişmedi)
    gecerli_sinyaller = {t: s for t, s in sinyaller.items() if np.isfinite(s)}
    if not gecerli_sinyaller: return {}
    fiyat_verisi = fiyat_verisi[list(gecerli_sinyaller.keys())]
    if fiyat_verisi.shape[1] < 2:
        return {list(fiyat_verisi.columns)[0]: 1.0} if fiyat_verisi.shape[1] == 1 else {}
    if "POZİTİF" in piyasa_rejimi:
        agirlik_limiti = 0.60; hedef = "max_sharpe"
    else:
        agirlik_limiti = max(0.35, 1/len(fiyat_verisi.columns)); hedef = "min_volatility"
    S = risk_models.sample_cov(fiyat_verisi)
    market_caps = {ticker: 1 for ticker in fiyat_verisi.columns}; max_abs_pred = max(abs(p) for p in gecerli_sinyaller.values()) if gecerli_sinyaller else 1
    scaling_factor = 0.10 / max_abs_pred if max_abs_pred != 0 else 0; annual_excess_returns = {ticker: pred * scaling_factor * 52 for ticker, pred in gecerli_sinyaller.items()}
    delta = 2.5; market_prior = S.dot(pd.Series(market_caps) / sum(market_caps.values())) * delta
    final_absolute_views = {ticker: market_prior[ticker] + annual_excess_returns.get(ticker, 0) for ticker in market_prior.index}
    bl = BlackLittermanModel(S, pi=market_prior, absolute_views=final_absolute_views); ret_bl = bl.bl_returns()
    ef = EfficientFrontier(ret_bl, S, weight_bounds=(0, agirlik_limiti))
    try:
        weights = ef.max_sharpe() if hedef == "max_sharpe" else ef.min_volatility()
    except (ValueError, OptimizationError):
        try: weights = ef.min_volatility()
        except (ValueError, OptimizationError): weights = {ticker: 1/len(fiyat_verisi.columns) for ticker in fiyat_verisi.columns}
    return weights

# YENİ FONKSİYON: GitHub'dan haftanın varlıklarını okur
@st.cache_data(show_spinner=False)
def get_tickers_from_github(github_user, repo_name, file_path):
    url = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{file_path}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        # İndirilen CSV'yi bir DataFrame'e çevir
        raw_df = pd.read_csv(io.StringIO(response.text))
        # Tek DataFrame'i işlemesi için auto_format_tickers'ı çağır
        return auto_format_tickers([raw_df])
    except Exception as e:
        st.error(f"Haftanın varlık listesi GitHub'dan çekilemedi. Hata: {e}")
        return None

# =======================================================
# BÖLÜM 2: BASİT VE GÜVENLİ GİRİŞ SİSTEMİ
# =======================================================

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True; del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password"); st.write("---"); return False
    elif not st.session_state["password_correct"]:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password"); st.error("😕 Şifre yanlış."); return False
    else:
        return True

# =======================================================
# BÖLÜM 3: STREAMLIT UYGULAMASI
# =======================================================

st.title("🤖 Kişisel Portföy Optimizasyon Asistanı")

if check_password():
    st.sidebar.success("Giriş Başarılı!")
    
    # Yönetici paneli artık yok. Liste doğrudan GitHub'dan geliyor.
    haftanin_varliklari = get_tickers_from_github(
        github_user="omeryigitkaya", # KENDİ GITHUB KULLANICI ADINIZ
        repo_name="kain",            # KENDİ GITHUB PROJE ADINIZ
        file_path="haftanin_varliklari.csv" # YÜKLEDİĞİNİZ CSV'NİN ADI
    )
    
    st.header("Kişisel Yatırım Planınızı Oluşturun")

    if haftanin_varliklari:
        st.info(f"Bu hafta analiz için yöneticinin seçtiği {len(haftanin_varliklari)} potansiyel varlık bulunmaktadır.")
        st.json(haftanin_varliklari)
        
        yatirim_tutari = st.number_input("Yatırmak istediğiniz tutarı (USD) girin:", min_value=100.0, step=100.0, value=1000.0)

        if st.button("Analizi Başlat ve Portföy Önerisi Oluştur"):
            rejim = piyasa_rejimini_belirle()
            st.subheader(f"Tespit Edilen Piyasa Rejimi: {rejim}")
            start_date = "2022-01-01"; end_date = pd.to_datetime("today").strftime('%Y-%m-%d')
            tum_fiyatlar = veri_cek_ve_dogrula(haftanin_varliklari, start_date, end_date)
            
            if tum_fiyatlar.empty:
                st.error("Seçilen varlıklar için analiz edilecek yeterli veri bulunamadı.")
            else:
                final_signals = {}; lstm_sinyal_detaylari = {}
                progress_bar = st.progress(0, text="AI Sinyalleri üretiliyor...")
                for i, ticker in enumerate(tum_fiyatlar.columns):
                    lstm_data = sinyal_uret_ensemble_lstm(tum_fiyatlar[ticker])
                    lstm_sinyal_detaylari[ticker] = lstm_data
                    sentiment_signal = sinyal_uret_duyarlilik(ticker)
                    sentiment_effect = sentiment_signal * 0.10
                    blended_signal = (lstm_data["tahmin_yuzde"] * 0.70) + (sentiment_effect * 0.30)
                    final_signals[ticker] = blended_signal
                    progress_bar.progress((i + 1) / len(tum_fiyatlar.columns), text=f"AI Sinyali üretiliyor: {ticker}")
                progress_bar.empty()
                
                st.info(f"Strateji Modu: {'Ofansif' if 'POZİTİF' in rejim else 'Defansif'}")
                optimal_agirliklar = portfoyu_optimize_et(final_signals, tum_fiyatlar, rejim)
                
                if optimal_agirliklar:
                    st.success("Analiz Tamamlandı!")
                    # ... (Raporlama kısmı aynı, değişiklik yok)
                    # ...
    else:
        st.error("Sistem için haftalık varlık listesi bulunamadı veya yüklenemedi.")
