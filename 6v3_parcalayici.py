import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

print("🚀 V3.0: KOMŞULUK DİZİLİMİ 11 PARÇAYA BÖLÜNÜYOR (Feature Engineering)...")

# 1. Verileri Yükle
df_kimya = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")
df_evrim = pd.read_excel("YapayZeka_Evrimsel_Veri.xlsx")
df_final = pd.merge(df_kimya, df_evrim[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']], on='Mutasyon_Adi', how='left')
df_final = df_final.drop(columns=['Prior_Skoru', 'Align_GVGD_Skoru'])

# 2. Encoding
le_gen = LabelEncoder()
df_final['Gen_Kodu'] = le_gen.fit_transform(df_final['Gen'].astype(str))

# --- HATA DÜZELTİLDİ: FLOAT/NAN KORUMASI EKLENDİ ---
komsuluk_sutunlari = []

def guvenli_dizilim(x):
    x_str = str(x)
    # Eğer hücre boş değilse ve tam 11 harfse dizilimi ver, yoksa X'lerle doldur
    if x_str != 'nan' and len(x_str) == 11:
        return x_str
    return "XXXXXXXXXXX"

df_final['Komsuluk_Dizilimi'] = df_final['Komsuluk_Dizilimi'].apply(guvenli_dizilim)

print("✂️ Dizilimler kesiliyor, 11 yeni mikroskobik özellik yaratılıyor...")
for i in range(11):
    pozisyon = i - 5 
    kolon_adi = f'Komsu_{pozisyon}'
    komsuluk_sutunlari.append(kolon_adi)
    
    # Dizilimin i. harfini al
    df_final[kolon_adi] = df_final['Komsuluk_Dizilimi'].str[i]
    
    # Harfleri sayılara çevir (Label Encoding)
    df_final[kolon_adi] = LabelEncoder().fit_transform(df_final[kolon_adi])

# 3. YENİ BÜYÜK X TABLOMUZ (Toplam 18 özellik!)
temel_ozellikler = ['Gen_Kodu', 'Popülasyon_Frekansi', 'Hidrofobiklik_Farki', 
                    'Molekuler_Agirlik_Farki', 'Polarite_Degisimi',
                    'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']

X = df_final[temel_ozellikler + komsuluk_sutunlari]
y = df_final['ETIKET']

# 4. K-Fold Çapraz Doğrulama
print("\n🔄 5-Fold Sınavı Genişletilmiş Zekayla Tekrar Başlıyor...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=7, random_state=42, eval_metric='logloss')

fold_no = 1
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train_fold, y_train_fold)
    preds = model.predict(X_test_fold)
    acc = accuracy_score(y_test_fold, preds)
    accuracies.append(acc)
    print(f"   Fold {fold_no} Sınav Sonucu: %{acc*100:.2f}")
    fold_no += 1

print(f"\n🏆 YENİ ORTALAMA GERÇEK BAŞARI: %{np.mean(accuracies)*100:.2f}")

import matplotlib.pyplot as plt
import shap

# ---------------------------------------------------------
# 5. YENİ MODEL İÇİN Feature Importance
# ---------------------------------------------------------
print("\n📊 xgboost Feature Importance Grafiği Çiziliyor...")
# Modeli grafik çizebilmek için tüm veriyle son bir kez eğitiyoruz
model.fit(X, y) 

plt.figure(figsize=(10, 8))
# Artık 18 özelliğimiz var, en önemli 10 tanesini görelim
xgb.plot_importance(model, max_num_features=10, title="Hangi Özellik Ne Kadar Önemli? (V3.0)", xlabel="Önem Skoru (F-score)", ylabel="Özellikler")
plt.tight_layout()
plt.savefig("Feature_Importance_V3.png", dpi=300)
print("✅ Feature_Importance_V3.png klasöre kaydedildi!")

# ---------------------------------------------------------
# 6. YENİ MODEL İÇİN SHAP ANALİZİ
# ---------------------------------------------------------
print("\n🧠 xgboost SHAP Analizi Yapılıyor...")
explainer = shap.Explainer(model)
shap_values = explainer(X)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("SHAP_Analizi_V3.png", dpi=300, bbox_inches='tight')
print("✅ SHAP_Analizi_V3.png klasöre kaydedildi!")

