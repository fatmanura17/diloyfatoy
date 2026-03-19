import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("🥊 NİHAİ KAPIŞMA V3.0: 18 KOLONLU VERİYLE TÜM MODELLER SAHADA!\n")

# 1. Verileri Yükle ve 18 Kolonlu Yapıya Getir
df_kimya = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")
df_evrim = pd.read_excel("YapayZeka_Evrimsel_Veri.xlsx")
df_final = pd.merge(df_kimya, df_evrim[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']], on='Mutasyon_Adi', how='left')

le_gen = LabelEncoder()
df_final['Gen_Kodu'] = le_gen.fit_transform(df_final['Gen'].astype(str))

def guvenli_dizilim(x):
    x_str = str(x)
    if x_str != 'nan' and len(x_str) == 11: return x_str
    return "XXXXXXXXXXX"

df_final['Komsuluk_Dizilimi'] = df_final['Komsuluk_Dizilimi'].apply(guvenli_dizilim)

komsuluk_sutunlari = []
for i in range(11):
    kolon_adi = f'Komsu_{i-5}'
    komsuluk_sutunlari.append(kolon_adi)
    df_final[kolon_adi] = df_final['Komsuluk_Dizilimi'].str[i]
    df_final[kolon_adi] = LabelEncoder().fit_transform(df_final[kolon_adi])

# 18 ÖZELLİKLİ TABLO
X = df_final[['Gen_Kodu', 'Popülasyon_Frekansi', 'Hidrofobiklik_Farki', 
              'Molekuler_Agirlik_Farki', 'Polarite_Degisimi',
              'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru'] + komsuluk_sutunlari]
y = df_final['ETIKET']

# 2. Çapraz Doğrulama Ayarı
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- MODELLER YARIŞIYOR ---

# A) Logistic Regression (Çökmemesi için NaN temizliğiyle)
lr_skor = np.mean(cross_val_score(LogisticRegression(max_iter=500), X.fillna(0), y, cv=skf))
print(f"📉 Logistic Regression: %{lr_skor*100:.2f}")

# B) Random Forest
rf_skor = np.mean(cross_val_score(RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42), X.fillna(0), y, cv=skf))
print(f"🌲 Random Forest: %{rf_skor*100:.2f}")

# C) XGBoost 
xgb_skor = np.mean(cross_val_score(xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=7, random_state=42, eval_metric='logloss'), X, y, cv=skf))
print(f"⚔️ XGBoost: %{xgb_skor*100:.2f}")

# D) ŞAMPİYON: CatBoost
cat_skor = np.mean(cross_val_score(CatBoostClassifier(iterations=150, learning_rate=0.05, depth=7, random_state=42, verbose=0), X, y, cv=skf))
print(f"👑 CatBoost: %{cat_skor*100:.2f}")

print("\n📊 Raporun 5.2 maddesindeki tabloyu bu değerlerle güncelleyebilirsin!")