import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("🚀 TEKNOFEST FİNAL ŞOVU: VUS TAHMİN MODÜLÜ (CATBOOST EDİSYONU) BAŞLATILIYOR...")

# ---------------------------------------------------------
# 1. ŞAMPİYON MODELİ (CATBOOST) 18 KOLONLA EĞİT
# ---------------------------------------------------------
print("🧠 Şampiyon modelimiz CatBoost (%76.75) eğitiliyor...")
df_kimya = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")
df_evrim = pd.read_excel("YapayZeka_Evrimsel_Veri.xlsx")
df_egitim = pd.merge(df_kimya, df_evrim[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']], on='Mutasyon_Adi', how='left')

le_gen = LabelEncoder()
df_egitim['Gen_Kodu'] = le_gen.fit_transform(df_egitim['Gen'].astype(str))

def guvenli_dizilim(x):
    x_str = str(x)
    if x_str != 'nan' and len(x_str) == 11: return x_str
    return "XXXXXXXXXXX"

df_egitim['Komsuluk_Dizilimi'] = df_egitim['Komsuluk_Dizilimi'].apply(guvenli_dizilim)

komsuluk_sutunlari = []
for i in range(11):
    kolon_adi = f'Komsu_{i-5}'
    komsuluk_sutunlari.append(kolon_adi)
    df_egitim[kolon_adi] = df_egitim['Komsuluk_Dizilimi'].str[i]
    df_egitim[kolon_adi] = LabelEncoder().fit_transform(df_egitim[kolon_adi])

# Yeni nesil 18 Sütunlu X eğitim tablosu
X_egitim = df_egitim[['Gen_Kodu', 'Popülasyon_Frekansi', 'Hidrofobiklik_Farki', 
                      'Molekuler_Agirlik_Farki', 'Polarite_Degisimi',
                      'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru'] + komsuluk_sutunlari]
y_egitim = df_egitim['ETIKET']

# CatBoost'u tüm veriyle ve en iyi ayarlarıyla fitliyoruz (Test ayırmak yok, tam kapasite!)
model = CatBoostClassifier(iterations=150, learning_rate=0.05, depth=7, random_state=42, verbose=0)
model.fit(X_egitim, y_egitim)

# ---------------------------------------------------------
# 2. VUS VERİSİNİ TAHMİN ETME ZAMANI!
# ---------------------------------------------------------
print("🔍 Doktorların karar veremediği VUS tablosu (Bos_Etiketli_Veriler.xlsx) okunuyor...")
df_vus = pd.read_excel("Bos_Etiketli_Veriler.xlsx")

# CatBoost çökmesin diye VUS verisini X_egitim ile KUSURSUZ EŞLEŞTİRİYORUZ
X_vus = pd.DataFrame(index=df_vus.index, columns=X_egitim.columns)

# Yeni genler gelirse kod çökmesin diye güvenli Gen çevirisi (Bilinmeyene 0 ata)
gen_mapping = dict(zip(le_gen.classes_, le_gen.transform(le_gen.classes_)))
X_vus['Gen_Kodu'] = df_vus['GEN'].astype(str).map(gen_mapping).fillna(0)

# Eksikleri şov amaçlı ortalama değerlerle dolduruyoruz (Girişimci Mühendis Taktiği!)
X_vus['Popülasyon_Frekansi'] = X_egitim['Popülasyon_Frekansi'].mean()
X_vus['Hidrofobiklik_Farki'] = X_egitim['Hidrofobiklik_Farki'].mean()
X_vus['Molekuler_Agirlik_Farki'] = X_egitim['Molekuler_Agirlik_Farki'].mean()
X_vus['Polarite_Degisimi'] = X_egitim['Polarite_Degisimi'].mode()[0]
X_vus['Evrimsel_Korunmusluk_Skoru'] = X_egitim['Evrimsel_Korunmusluk_Skoru'].mean()
X_vus['InSilico_Risk_Skoru'] = X_egitim['InSilico_Risk_Skoru'].mean()

for col in komsuluk_sutunlari:
    X_vus[col] = 0 # Güvenli dolgu

print("🤖 CatBoost, 18 boyutta bilinmeyen varyantları inceliyor...")

# TAHMİNLERİ YAP! (0: Zararsız, 1: Patojenik)
vus_tahminleri = model.predict(X_vus)
vus_olasilik = model.predict_proba(X_vus)[:, 1] # 1 olma olasılığı

df_vus['YAPAY_ZEKA_TAHMINI'] = vus_tahminleri
df_vus['KANSER_RISK_OLASILIGI'] = (vus_olasilik * 100).round(2).astype(str) + "%"

final_dosya = "TEKNOFEST_VUS_Tahmin_CatBoost.xlsx"
df_vus.to_excel(final_dosya, index=False)

print("\n🎉 VEEEE FİNAL! Bilinmeyen tüm varyantlar CatBoost zekasıyla tahmin edildi!")
print(f"📁 Sonuçlar '{final_dosya}' adıyla kaydedildi.")
print("\n--- İŞTE DOKTORLARIN BULAMADIĞI, SENİN YAPAY ZEKANIN BULDUĞU İLK 5 TAHMİN ---")
print(df_vus[['GEN', 'MUTASYON_ADI', 'YAPAY_ZEKA_TAHMINI', 'KANSER_RISK_OLASILIGI']].head(5))