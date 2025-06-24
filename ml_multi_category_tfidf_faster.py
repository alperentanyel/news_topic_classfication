import pandas as pd
import numpy as np
import os
import time
import pickle
import re
import requests
import zipfile
import io
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
from scipy.sparse import vstack  # Sparse matrix birleştirme için
warnings.filterwarnings("ignore")

# Sklearn için gerekli kütüphaneler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import _stop_words

# Kategori ID'leri ve anlamları
# 1: World (Dünya/Politik Haberler)
# 2: Sports (Spor Haberleri)
# 3: Business (İş/Ekonomi Haberleri)
# 4: Sci/Tech (Bilim/Teknoloji Haberleri)

# Ayarlar - OPTIMIZE EDİLMİŞ (Cross-Category Korundu)
cv = 5  # Çapraz doğrulama fold sayısı  
GLOVE_DIM = 300  # GloVe vektör boyutu
MAX_WORDS = 15000
TFIDF_MAX_FEATURES = 5000  # Her kategori için max feature (dengeli)
FINAL_DIM = 800  # PCA sonrası boyut (dengeli)
SAMPLE_SIZE = None  # Hızlı ama anlamlı test için örnek boyutu

# Veri yolları
train_path = "archive/train.csv"
test_path = "archive/test.csv"
models_dir = "models"
glove_dir = "glove"



def enhanced_clean_text(text):
    """Gelişmiş metin ön işleme"""
    if isinstance(text, str):
        # Küçük harfe çevir
        text = text.lower()
        
        # URL'leri kaldır
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Email adreslerini kaldır
        text = re.sub(r'\S+@\S+', '', text)
        
        # Sayıları kaldır
        text = re.sub(r'\d+', '', text)
        
        # Noktalama işaretlerini kaldır
        text = re.sub(r'[^\w\s]', '', text)
        
        # Çok kısa kelimeleri kaldır (1-2 harf)
        text = re.sub(r'\b\w{1,2}\b', '', text)
        
        # Çok uzun kelimeleri kaldır (15+ harf)
        text = re.sub(r'\b\w{15,}\b', '', text)
          # Tekrarlayan harfleri düzenle
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Gereksiz boşlukları kaldır
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    return ""

def download_glove_model():
    glove_path = f"{glove_dir}/glove.6B.{GLOVE_DIM}d.txt"
    
    if not os.path.exists(glove_path):
        print(f"GloVe vektörleri indiriliyor (boyut: {GLOVE_DIM}d)...")
        print("Not: Bu büyük bir dosya, indirme işlemi birkaç dakika sürebilir.")
        
        try:
            url = "https://nlp.stanford.edu/data/glove.6B.zip"
            print("İndirme başlıyor...")
            r = requests.get(url, stream=True, timeout=120)
            if r.status_code == 200:
                print("İndirme tamamlandı, zipten çıkarılıyor...")
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(glove_dir)
                print("GloVe vektörleri başarıyla indirildi ve çıkarıldı.")
            else:
                print(f"İndirme hatası: {r.status_code}")
                raise Exception("GloVe vektörleri indirilemedi.")
        except Exception as e:
            print(f"Hata: {e}")
            print("GloVe vektörlerini manuel olarak indirmeniz gerekebilir.")
            raise Exception("GloVe vektörleri yüklenemedi.")
    else:
        print(f"GloVe vektörleri zaten mevcut: {glove_path}")
    
    return glove_path

def load_glove_model(glove_path):
    """GloVe kelime vektörlerini yükle"""
    print(f"GloVe vektörleri yükleniyor: {glove_path}")
    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="GloVe Yükleniyor")):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')           
            embeddings_index[word] = coefs
    print(f"Toplam {len(embeddings_index)} kelime vektörü yüklendi.")
    return embeddings_index

def calculate_edit_distance(s1, s2):
    """İki string arasındaki edit distance (Levenshtein) hesapla"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def find_similar_words_semantic(target_word, embeddings_index, context_words=None, target_category=None, 
                               word_score_cache=None, max_candidates=10, use_context=True):
    """
    Anlamsal benzerlik (cosine similarity) kullanarak benzer kelimeleri bul
    Bağlamsal analiz ve kategori farkındalığı ile geliştirilmiş
    """
    if len(target_word) < 3:
        return []
    
    # Hedef kelime için vektör oluşturmaya çalış (edit distance ile fallback)
    target_vector = None
    edit_candidates = []
    
    # 1. Edit distance ile potansiyel adayları bul (sadece fallback için)
    target_len = len(target_word)
    for word in embeddings_index:
        word_len = len(word)
        if abs(word_len - target_len) <= 2:  # Sadece benzer uzunluktaki kelimeler
            distance = calculate_edit_distance(target_word, word)
            if distance <= 2:
                edit_candidates.append((word, distance))
    
    # En iyi edit distance adaylarından target vector oluştur
    if edit_candidates:
        edit_candidates.sort(key=lambda x: x[1])
        top_edit_words = [word for word, _ in edit_candidates[:5]]
        vectors = [embeddings_index[word] for word in top_edit_words]
        weights = [1.0 / (1.0 + calculate_edit_distance(target_word, word)) for word in top_edit_words]
        weights = np.array(weights) / sum(weights)
        target_vector = np.average(vectors, axis=0, weights=weights)
    
    if target_vector is None:
        return []
    
    # 2. Anlamsal benzerlik ile en iyi adayları bul
    semantic_candidates = []
    
    # Bağlamsal kelimeler varsa onların vektörlerini de hesaba kat
    context_vector = None
    if use_context and context_words and len(context_words) > 0:
        context_vectors = []
        for ctx_word in context_words:
            if ctx_word in embeddings_index:
                context_vectors.append(embeddings_index[ctx_word])
        
        if context_vectors:
            context_vector = np.mean(context_vectors, axis=0)
    
    # Tüm kelimeler arasında cosine similarity hesapla
    for word in embeddings_index.keys():
        if len(word) < 3 or word == target_word:
            continue
            
        word_vector = embeddings_index[word]
        
        # Cosine similarity hesapla
        cosine_sim = np.dot(target_vector, word_vector) / (
            np.linalg.norm(target_vector) * np.linalg.norm(word_vector)
        )
        
        # Bağlamsal bonus
        context_bonus = 0.0
        if context_vector is not None:
            context_sim = np.dot(context_vector, word_vector) / (
                np.linalg.norm(context_vector) * np.linalg.norm(word_vector)
            )
            context_bonus = context_sim * 0.3  # Bağlamsal benzerlik bonusu
        
        # Kategori bonusu
        category_bonus = 0.0
        if target_category and word_score_cache and word in word_score_cache:
            category_score = word_score_cache[word].get(target_category, 0.0)
            category_bonus = category_score * 0.2  # Kategori skoru bonusu
        
        # Final skor
        final_score = cosine_sim + context_bonus + category_bonus
        
        if final_score > 0.3:  # Minimum benzerlik threshold
            semantic_candidates.append((word, final_score, cosine_sim))
    
    # Skorlara göre sırala ve en iyileri al
    semantic_candidates.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _, _ in semantic_candidates[:max_candidates]]

def find_similar_words(target_word, embeddings_index, max_edit_distance=2, max_candidates=10):
    """Geriye uyumluluk için eski fonksiyon - yeni semantik versiyonu çağırır"""
    return find_similar_words_semantic(target_word, embeddings_index, max_candidates=max_candidates)

def calculate_category_aware_missing_weight(missing_token, similar_words, target_category, word_score_cache):
    """Kategori-farkında eksik kelime ağırlığı hesapla"""
    if not similar_words:
        return 0.2  # Fallback ağırlık
    
    # En iyi benzer kelimeleri kategoriye göre skorla
    category_scores = []
    cross_weights = []
    
    for similar_word in similar_words:
        if similar_word in word_score_cache:
            # Bu kelimenin hedef kategorideki skorunu al
            category_score = word_score_cache[similar_word].get(target_category, 0.0)
            category_scores.append(category_score)
            
            # Cross-category ağırlığını da hesapla
            all_scores = [word_score_cache[similar_word].get(cat, 0.0) for cat in [1, 2, 3, 4]]
            if any(score > 0 for score in all_scores):
                cross_weight = calculate_cross_category_weight_from_scores(all_scores, target_category)
                cross_weights.append(cross_weight)
    
    if category_scores:
        # Kategori-spesifik ortalama ağırlık
        avg_category_score = np.mean(category_scores)
        # Cross-category bilgisi varsa onu da dahil et
        if cross_weights:
            avg_cross_weight = np.mean(cross_weights)
            final_weight = (avg_category_score * 0.6) + (avg_cross_weight * 0.4)
        else:
            final_weight = avg_category_score
        
        # 0.1-1.5 arası sınırla
        return max(0.1, min(1.5, final_weight))
    else:
        return 0.2  # Fallback

def calculate_cross_category_weight_from_scores(category_scores, target_category):
    """Cache olmadan cross-category ağırlık hesapla"""
    if len(category_scores) == 0 or all(score == 0 for score in category_scores):
        return 0.05
    
    # Hedef kategorinin skoru
    target_score = category_scores[target_category - 1]  # 0-indexed
    
    if target_score <= 0:
        return 0.05
    
    # Diğer kategorilerin skorları
    other_scores = [category_scores[i] for i in range(4) if i != (target_category - 1)]
    
    # İstatistiksel metrikler
    mean_other = np.mean(other_scores) if other_scores else 0
    std_other = np.std(other_scores) if len(other_scores) > 1 else 0.01
    
    # Z-SCORE BAZLI AYIRT EDİCİLİK
    if std_other > 0:
        z_score = (target_score - mean_other) / std_other
        normalized_distinctiveness = max(0, min(2, z_score)) / 2.0
    else:
        normalized_distinctiveness = min(target_score / (mean_other + 0.01), 2.0) / 2.0
    
    # Final ağırlık hesaplama (basitleştirilmiş)
    final_weight = target_score * (1.0 + normalized_distinctiveness)
    
    return min(final_weight, 2.0)

def analyze_context_for_missing_word(missing_token, text_tokens, token_index, target_category, word_score_cache):
    """Eksik kelimenin bağlamını analiz ederek kategori tahmin et"""
    
    # Pencere boyutu (önceki 2, sonraki 2 kelime)
    window_size = 2
    context_words = []
    
    # Önceki kelimeler
    for i in range(max(0, token_index - window_size), token_index):
        if i < len(text_tokens):
            context_words.append(text_tokens[i])
    
    # Sonraki kelimeler
    for i in range(token_index + 1, min(len(text_tokens), token_index + window_size + 1)):
        context_words.append(text_tokens[i])
    
    # Bağlam kelimelerinin kategori skorlarını analiz et
    context_category_scores = []
    for context_word in context_words:
        if context_word in word_score_cache:
            category_score = word_score_cache[context_word].get(target_category, 0.0)
            if category_score > 0:
                context_category_scores.append(category_score)
    
    # Bağlam güvenirlik skoru
    if context_category_scores:
        context_strength = np.mean(context_category_scores)
        return context_strength
    else:
        return 0.1

def smart_missing_word_weight_with_context(missing_token, text_tokens, token_index, target_category, 
                                         embeddings_index, word_score_cache):
    """Bağlamsal analiz ile akıllı eksik kelime ağırlık sistemi"""
    
    # 1. Bağlamsal analiz
    context_strength = analyze_context_for_missing_word(
        missing_token, text_tokens, token_index, target_category, word_score_cache
    )
    
    # 2. KNN-like en iyi replacement
    knn_weight = find_best_replacement_with_knn(
        missing_token, target_category, embeddings_index, word_score_cache
    )
    
    # 3. Benzer kelimelerin kategori analizi
    similar_words = find_similar_words(missing_token, embeddings_index)
    category_weight = calculate_category_aware_missing_weight(
        missing_token, similar_words, target_category, word_score_cache
    )
    
    # 4. Weighted combination
    # Bağlam güçlüyse ona daha fazla ağırlık ver
    context_factor = min(context_strength * 2, 1.0)
    
    final_weight = (
        context_strength * context_factor * 0.4 +
        knn_weight * 0.4 +
        category_weight * 0.2
    )
    
    return max(0.1, min(1.5, final_weight))

def find_best_replacement_with_knn(missing_token, target_category, embeddings_index, word_score_cache, k=5):
    """KNN-like yaklaşımla en iyi kelime replacement'ı bul"""
    candidates = []
    
    # Edit distance ile benzer kelimeler bul
    similar_words = find_similar_words(missing_token, embeddings_index, max_edit_distance=2, max_candidates=15)
    
    for similar_word in similar_words:
        if similar_word in word_score_cache:
            # 1. Edit distance skoru (0-1)
            edit_dist = calculate_edit_distance(missing_token, similar_word)
            similarity_score = 1.0 / (1.0 + edit_dist)
            
            # 2. Kategori skoru
            category_score = word_score_cache[similar_word].get(target_category, 0.0)
            
            # 3. Cross-category discriminativeness
            all_scores = [word_score_cache[similar_word].get(cat, 0.0) for cat in [1,2,3,4]]
            cross_weight = calculate_cross_category_weight_from_scores(all_scores, target_category)
            
            # 4. Kombine skor
            final_score = similarity_score * 0.3 + category_score * 0.4 + cross_weight * 0.3
            
            candidates.append((similar_word, final_score, category_score))
    
    # En iyi K adayı seç
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_k = candidates[:k]
    
    if top_k:
        # En iyi adayların kategori skorlarının ortalamasını ağırlık olarak kullan
        top_category_scores = [candidate[2] for candidate in top_k]
        avg_score = np.mean(top_category_scores)
        return max(0.1, min(1.5, avg_score))
    else:
        return 0.2

def create_missing_word_vector(word, similar_words, embeddings_index, category_vectorizers=None, 
                              category_word_indices=None, target_category=None, text_context=None):
    """Eksik kelime için benzer kelimelerden ağırlıklı ortalama vektör oluştur"""
    if not similar_words:
        # Hiç benzer kelime yoksa random vektör (küçük değerlerle)
        return np.random.normal(0, 0.1, size=embeddings_index[list(embeddings_index.keys())[0]].shape)
    
    vectors = []
    weights = []
    
    for similar_word in similar_words:
        if similar_word in embeddings_index:
            vector = embeddings_index[similar_word]
            
            # Edit distance'a göre ağırlık (daha az distance = daha yüksek ağırlık)
            edit_distance = calculate_edit_distance(word, similar_word)
            distance_weight = 1.0 / (1.0 + edit_distance)
            
            # Kategori-özellikli TF-IDF ağırlığı ekle (eğer mevcutsa)
            tfidf_weight = 1.0
            if (category_vectorizers and category_word_indices and 
                target_category and text_context and 
                target_category in category_vectorizers and 
                similar_word in category_word_indices[target_category]):
                
                try:
                    tfidf_vector = category_vectorizers[target_category].transform([text_context])
                    word_idx = category_word_indices[target_category][similar_word]
                    tfidf_weight = tfidf_vector[0, word_idx] + 0.1  # Minimum ağırlık
                except:
                    tfidf_weight = 0.1
            
            # Toplam ağırlık
            total_weight = distance_weight * tfidf_weight
            
            vectors.append(vector)
            weights.append(total_weight)
    
    if not vectors:
        # Hiç vektör yoksa random vektör
        return np.random.normal(0, 0.1, size=embeddings_index[list(embeddings_index.keys())[0]].shape)
    
    # Ağırlıklı ortalama hesapla
    vectors = np.array(vectors)
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize et
    
    return np.average(vectors, axis=0, weights=weights)

def find_best_replacement_with_context(missing_token, target_category, embeddings_index, word_score_cache, context_words, k=5):
    """Bağlamsal bilgi ile en iyi kelime replacement'ı bul"""
    candidates = []
    
    # Anlamsal benzerlik ile adayları bul
    similar_words = find_similar_words_semantic(
        missing_token, embeddings_index, 
        context_words=context_words, 
        target_category=target_category,
        word_score_cache=word_score_cache, 
        max_candidates=20
    )
    
    for similar_word in similar_words:
        if similar_word in word_score_cache:
            # 1. Anlamsal benzerlik skoru
            if similar_word in embeddings_index:
                # Edit distance fallback
                edit_dist = calculate_edit_distance(missing_token, similar_word)
                similarity_score = 1.0 / (1.0 + edit_dist)
            else:
                similarity_score = 0.5
            
            # 2. Kategori skoru
            category_score = word_score_cache[similar_word].get(target_category, 0.0)
            
            # 3. Bağlamsal skor (bağlam kelimeleri ile benzerlik)
            context_score = 0.0
            if context_words and similar_word in embeddings_index:
                context_similarities = []
                similar_vector = embeddings_index[similar_word]
                
                for ctx_word in context_words:
                    if ctx_word in embeddings_index:
                        ctx_vector = embeddings_index[ctx_word]
                        ctx_sim = np.dot(similar_vector, ctx_vector) / (
                            np.linalg.norm(similar_vector) * np.linalg.norm(ctx_vector)
                        )
                        context_similarities.append(max(0, ctx_sim))
                
                if context_similarities:
                    context_score = np.mean(context_similarities)
            
            # 4. Cross-category discriminativeness
            all_scores = [word_score_cache[similar_word].get(cat, 0.0) for cat in [1,2,3,4]]
            cross_weight = calculate_cross_category_weight_from_scores(all_scores, target_category)
            
            # 5. Kombine skor (bağlamsal ağırlık artırıldı)
            final_score = (
                similarity_score * 0.2 + 
                category_score * 0.3 + 
                context_score * 0.3 + 
                cross_weight * 0.2
            )
            
            candidates.append((similar_word, final_score, category_score))
    
    # En iyi K adayı seç
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_k = candidates[:k]
    
    if top_k:
        # En iyi adayların ağırlıklı ortalaması
        weights = [candidate[1] for candidate in top_k]
        category_scores = [candidate[2] for candidate in top_k]
        
        # Ağırlıklı ortalama hesapla
        weighted_avg = np.average(category_scores, weights=weights)
        return max(0.15, min(1.8, weighted_avg))
    else:
        return 0.2

def create_semantic_missing_word_vector(word, similar_words, embeddings_index, context_words=None, target_category=None, word_score_cache=None):
    """Anlamsal benzerlik ve bağlam ile eksik kelime vektörü oluştur"""
    if not similar_words:
        # Hiç benzer kelime yoksa random vektör (küçük değerlerle)
        sample_key = list(embeddings_index.keys())[0]
        return np.random.normal(0, 0.1, size=embeddings_index[sample_key].shape)
    
    vectors = []
    weights = []
    
    # Bağlamsal vektör hesapla
    context_vector = None
    if context_words:
        context_vectors = []
        for ctx_word in context_words:
            if ctx_word in embeddings_index:
                context_vectors.append(embeddings_index[ctx_word])
        
        if context_vectors:
            context_vector = np.mean(context_vectors, axis=0)
    
    for similar_word in similar_words:
        if similar_word in embeddings_index:
            vector = embeddings_index[similar_word]
            
            # 1. Edit distance ağırlığı (fallback)
            edit_distance = calculate_edit_distance(word, similar_word)
            distance_weight = 1.0 / (1.0 + edit_distance)
            
            # 2. Bağlamsal ağırlık
            context_weight = 1.0
            if context_vector is not None:
                context_sim = np.dot(context_vector, vector) / (
                    np.linalg.norm(context_vector) * np.linalg.norm(vector)
                )
                context_weight = max(0.1, context_sim + 0.5)  # Pozitif ağırlık garantisi
            
            # 3. Kategori ağırlığı
            category_weight = 1.0
            if target_category and word_score_cache and similar_word in word_score_cache:
                category_score = word_score_cache[similar_word].get(target_category, 0.0)
                category_weight = max(0.1, category_score + 0.3)
            
            # Toplam ağırlık
            total_weight = distance_weight * context_weight * category_weight
            
            vectors.append(vector)
            weights.append(total_weight)
    
    if not vectors:
        # Hiç vektör yoksa random vektör
        sample_key = list(embeddings_index.keys())[0]
        return np.random.normal(0, 0.1, size=embeddings_index[sample_key].shape)
    
    # Ağırlıklı ortalama hesapla
    vectors = np.array(vectors)
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize et
    
    return np.average(vectors, axis=0, weights=weights)

def analyze_missing_words(texts, embeddings_index, max_examples=10):
    """
    GloVe'da olmayan kelimeleri analiz et
    """
    print("\n🔍 GloVe'da eksik kelimeler analiz ediliyor...")
    
    all_missing_words = set()
    total_words = 0
    missing_count = 0
    
    # Tüm metinlerdeki kelimeleri kontrol et
    for text in tqdm(texts[:1000], desc="Kelime analizi"):  # İlk 1000 metin
        if isinstance(text, str):
            cleaned_text = enhanced_clean_text(text)
            tokens = cleaned_text.split()
            stop_words = _stop_words.ENGLISH_STOP_WORDS
            filtered_tokens = [t for t in tokens if t not in stop_words and len(t) >= 3]
            
            for token in filtered_tokens:
                total_words += 1
                if token not in embeddings_index:
                    missing_count += 1
                    all_missing_words.add(token)
    
    missing_ratio = (missing_count / total_words) * 100 if total_words > 0 else 0
    
    print(f"\n📊 EKSİK KELİME ANALİZİ:")
    print(f"   Toplam kelime: {total_words:,}")
    print(f"   Eksik kelime (unique): {len(all_missing_words):,}")
    print(f"   Eksik kelime (toplam): {missing_count:,}")
    print(f"   Eksik kelime oranı: {missing_ratio:.2f}%")
    
    # Eksik kelimeleri kategorilere ayır
    categorize_missing_words(all_missing_words, max_examples)
    
    return all_missing_words

def categorize_missing_words(missing_words, max_examples=10):
    """
    Eksik kelimeleri kategorilere ayır ve örnekler göster
    """
    print(f"\n🔍 EKSİK KELİME KATEGORİLERİ:")
    
    # Kategori tanımları
    categories = {
        'Sayısal': [],
        'Çok Kısa (1-2 harf)': [],
        'Çok Uzun (15+ harf)': [],
        'Yazım Hatası (tekrarlı harf)': [],
        'Karma (harf+sayı)': [],
        'Özel İsim (büyük harf)': [],
        'Teknik Terim': [],
        'Diğer': []
    }
    
    for word in missing_words:
        if any(char.isdigit() for char in word):
            if word.isdigit():
                categories['Sayısal'].append(word)
            else:
                categories['Karma (harf+sayı)'].append(word)
        elif len(word) <= 2:
            categories['Çok Kısa (1-2 harf)'].append(word)
        elif len(word) >= 15:
            categories['Çok Uzun (15+ harf)'].append(word)
        elif re.search(r'(.)\1{2,}', word):  # 3+ tekrar eden harf
            categories['Yazım Hatası (tekrarlı harf)'].append(word)
        elif word[0].isupper():
            categories['Özel İsim (büyük harf)'].append(word)
        elif any(tech in word.lower() for tech in ['tech', 'cyber', 'ai', 'ml', 'crypto', 'blockchain', 'covid']):
            categories['Teknik Terim'].append(word)
        else:
            categories['Diğer'].append(word)
    
    # Her kategori için istatistik ve örnekler göster
    for category, words in categories.items():
        if words:
            examples = list(words)[:max_examples]
            print(f"   {category}: {len(words)} kelime")
            print(f"      Örnekler: {', '.join(examples)}")
            if len(words) > max_examples:
                print(f"      (... ve {len(words) - max_examples} kelime daha)")
            print()

class MultiCategoryTFIDFGloVeVectorizer:
    """Multi-Category TF-IDF ağırlıklı GloVe vektörleyici"""
      
    def __init__(self, embeddings_index, dim=100, max_features_per_category=2500, final_dim=100, 
                 handle_missing_words=True):
        self.embeddings_index = embeddings_index
        self.dim = dim
        self.max_features_per_category = max_features_per_category
        self.final_dim = final_dim
        self.category_vectorizers = {}
        self.category_word_indices = {}
        self.pca = PCA(n_components=final_dim)
        self.is_fitted = False
        
        # Eksik kelime yönetimi
        self.handle_missing_words = handle_missing_words
        self.missing_word_cache = {}  # Eksik kelimeler için cache
        
        # CACHE SİSTEMİ - Sadece bir kez hesapla, sürekli kullan!
        self.global_word_score_cache = None
        self.category_tfidf_matrices_cache = None
        
    def fit(self, texts, labels):
        """Multi-category TF-IDF vektörleyiciyi eğit"""
        print("Multi-Category TF-IDF vektörleyici eğitiliyor...")
        
        # Metinleri temizle
        cleaned_texts = [enhanced_clean_text(text) for text in texts]
        
        # Her kategori için ayrı TF-IDF vektörleyici oluştur
        categories = [1, 2, 3, 4]
        
        for category in categories:
            print(f"Kategori {category} için TF-IDF eğitiliyor...")
            
            # Bu kategoriye ait metinleri al
            category_texts = [cleaned_texts[i] for i, label in enumerate(labels) if label == category]
            
            if len(category_texts) == 0:
                print(f"Uyarı: Kategori {category} için metin bulunamadı!")
                continue
            
            # TF-IDF vektörleyici oluştur
            vectorizer = TfidfVectorizer(
                max_features=self.max_features_per_category,
                stop_words='english',
                ngram_range=(1, 1),
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{3,}\b'  # En az 3 harfli kelimeler
            )
            
            # Bu kategorideki tüm metinlerle fit et
            vectorizer.fit(category_texts)
            
            # Kaydet
            self.category_vectorizers[category] = vectorizer
            self.category_word_indices[category] = {
                word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())
            }
            
            print(f"Kategori {category}: {len(self.category_word_indices[category])} özellik")        # Şimdi tüm veriyi transform et ve PCA fit et
        print("PCA için multi-category vektörler oluşturuluyor...")
        multi_vectors = self._create_multi_category_vectors_with_cache(cleaned_texts, build_cache=True)
        
        # PCA fit et
        print(f"PCA eğitiliyor: {multi_vectors.shape[1]}D → {self.final_dim}D")
        self.pca.fit(multi_vectors)
        self.is_fitted = True
        print("Multi-Category TF-IDF vektörleyici eğitimi tamamlandı!")
        
    def _create_multi_category_vectors_with_cache(self, cleaned_texts, build_cache=False):
        """
        MEGA OPTIMIZED: Cache sistemi ile ultra-fast vektör oluşturma
        build_cache=True: Cache'i ilk kez oluştur
        build_cache=False: Mevcut cache'i kullan
        """
        n_texts = len(cleaned_texts)
        total_dim = len(self.category_vectorizers) * self.dim  # 4 * 300 = 1200D
        multi_vectors = np.zeros((n_texts, total_dim))
          # CACHE KONTROLÜ
        if build_cache or self.global_word_score_cache is None:
            
            # Cache yoksa veya yüklenemiyorsa yeniden oluştur
            if self.global_word_score_cache is None:
                print("🚀 İlk kez cache oluşturuluyor...")
                # BATCH İŞLEME: Tüm kategoriler için TF-IDF'leri tek seferde hesapla
                print("🚀 Batch TF-IDF hesaplaması başlıyor...")
                category_tfidf_matrices = {}
                
                # Memory-efficient batch processing
                batch_size = min(1000, n_texts)  # Bellek dostu batch boyutu
                
                for category in [1, 2, 3, 4]:
                    if category in self.category_vectorizers:
                        if n_texts <= batch_size:
                            # Küçük veri seti - tek seferde işle
                            category_tfidf_matrices[category] = self.category_vectorizers[category].transform(cleaned_texts)
                        else:
                            # Büyük veri seti - batch'ler halinde işle
                            print(f"   Kategori {category} için batch işleme...")
                            tfidf_parts = []
                            for i in range(0, n_texts, batch_size):
                                batch_texts = cleaned_texts[i:i+batch_size]
                                batch_tfidf = self.category_vectorizers[category].transform(batch_texts)
                                tfidf_parts.append(batch_tfidf)
                            
                            # Sparse matrix'leri birleştir
                            from scipy.sparse import vstack
                            category_tfidf_matrices[category] = vstack(tfidf_parts)
                
                # GLOBAL WORD SCORE CACHE OLUŞTUR (SADECE BİR KEZ!)
                print("🔥 Word score cache oluşturuluyor...")
                self.global_word_score_cache = self._build_global_word_score_cache(category_tfidf_matrices, cleaned_texts)
                self.category_tfidf_matrices_cache = category_tfidf_matrices
                
                # Cache'i dosyaya kaydet
                if build_cache:
                    print("💾 Cache dosyaya kaydediliyor...")
                    try:
                        cache_data = {
                            'word_cache': self.global_word_score_cache,
                            'tfidf_cache': self.category_tfidf_matrices_cache
                        }
                        with open(self.cache_file_path, 'wb') as f:
                            pickle.dump(cache_data, f)
                        print("✅ Cache başarıyla dosyaya kaydedildi!")
                    except Exception as e:
                        print(f"⚠️ Cache dosyaya kaydedilemedi: {e}")
        else:
            print("🚀 CACHE KULLANILIYOR - SÜPER HIZLI! ⚡")
        
        # VECTORIZED İŞLEME: Tüm metinler için hızlı vektörleme
        print("⚡ Vectorized GloVe işleme başlıyor...")
        for i, text in enumerate(tqdm(cleaned_texts, desc="Ultra-Fast Vektörler", mininterval=0.5)):
            category_vectors = []
            
            # Tokenize once (tekrar etme)
            tokens = text.split()
            stop_words = _stop_words.ENGLISH_STOP_WORDS
            filtered_tokens = [t for t in tokens if t not in stop_words and t in self.embeddings_index]
            
            for category in [1, 2, 3, 4]:
                if category in self.category_vectorizers:
                    # CACHE'DEN HIZLI VEKTÖRLEME
                    weighted_vector = self._text_to_weighted_vector_cached(
                        filtered_tokens, category, i, self.global_word_score_cache
                    )
                    category_vectors.append(weighted_vector)
                else:
                    category_vectors.append(np.zeros(self.dim))
            
            # Tüm kategori vektörlerini birleştir
            multi_vectors[i] = np.concatenate(category_vectors)
        
        return multi_vectors
    
    def _build_global_word_score_cache(self, category_tfidf_matrices, cleaned_texts):
        """
        GLOBAL WORD SCORE CACHE - Her kelime için tüm kategorilerdeki skorları önceden hesapla
        Bu sayede aynı hesaplamalar tekrar tekrar yapılmayacak
        """
        print("   Cache tablosu oluşturuluyor...")
        word_score_cache = {}
        
        # Tüm benzersiz kelimeleri topla
        all_words = set()
        for text in cleaned_texts:
            tokens = text.split()
            stop_words = _stop_words.ENGLISH_STOP_WORDS
            words = [t for t in tokens if t not in stop_words and t in self.embeddings_index]
            all_words.update(words)
        
        print(f"   Toplam {len(all_words)} benzersiz kelime için skorlar hesaplanıyor...")
        
        # Her kelime için tüm kategorilerdeki skorları hesapla
        for word in tqdm(all_words, desc="Cache Building", mininterval=1.0):
            word_score_cache[word] = {}
            
            # Her kategoride bu kelimenin genel skorunu hesapla
            for category in [1, 2, 3, 4]:
                if category in self.category_word_indices and word in self.category_word_indices[category]:
                    # Bu kelimenin bu kategorideki TF-IDF matrisindeki ortalama skorunu al
                    word_idx = self.category_word_indices[category][word]
                    tfidf_matrix = category_tfidf_matrices[category]
                    
                    # Tüm dokümanlardaki ortalama skoru hesapla
                    word_column = tfidf_matrix[:, word_idx].toarray().flatten()
                    avg_score = np.mean(word_column[word_column > 0])  # Sadece pozitif skorların ortalaması
                    word_score_cache[word][category] = avg_score if not np.isnan(avg_score) else 0.01
                else:
                    word_score_cache[word][category] = 0.0
        
        print(f"   Cache oluşturuldu: {len(word_score_cache)} kelime x 4 kategori")
        return word_score_cache
     
    def _text_to_weighted_vector_cached(self, filtered_tokens, category, text_idx, word_score_cache):
        """
        CACHED VERSION: Cross-category ağırlıklandırma ile ultra-fast vektör oluşturma
        Cache sayesinde aynı hesaplamalar tekrar yapılmıyor + Eksik kelime yönetimi
        """
        if not filtered_tokens:
            return np.zeros(self.dim)
        
        # Kelimelerin ağırlıklarını cache'den hızlıca hesapla
        token_weights = {}
        missing_tokens = []  # Eksik kelimeler listesi
        
        for token in filtered_tokens:
            if token in self.embeddings_index:
                # GloVe'da mevcut kelime
                if token in word_score_cache:
                    # Cache'den tüm kategori skorlarını al
                    category_scores = [word_score_cache[token].get(cat, 0.0) for cat in [1, 2, 3, 4]]
                    
                    # Cross-category ağırlığını hızlıca hesapla
                    cross_weight = self._calculate_cross_category_weight_from_cache(
                        category_scores, category
                    )
                    
                    if cross_weight > 0.01:  # Minimum threshold
                        token_weights[token] = cross_weight
            elif self.handle_missing_words:
                # GloVe'da eksik kelime - eksik kelime yönetimi
                missing_tokens.append(token)
          # Eksik kelimeleri işle
        if missing_tokens and self.handle_missing_words:
            for missing_token in missing_tokens:
                # Cache kontrolü - aynı eksik kelime daha önce işlendi mi?
                if missing_token in self.missing_word_cache:
                    vector = self.missing_word_cache[missing_token]
                    
                    # AKILLI AĞIRLIK: Cache'deki vektör için kategori-spesifik ağırlık hesapla
                    smart_weight = self._calculate_smart_missing_weight(
                        missing_token, category, word_score_cache
                    )
                    token_weights[missing_token] = smart_weight
                    
                    # Geçici olarak embeddings_index'e ekle
                    self.embeddings_index[missing_token] = vector                
                
                else:
                    # Bağlamsal kelimeleri al (önceki ve sonraki kelimeler)
                    context_words = []
                    all_tokens = text.split()
                    
                    # Missing token'ın indexini bul
                    try:
                        token_index = all_tokens.index(missing_token)
                        # Önceki ve sonraki kelimeleri al
                        window_size = 2
                        for i in range(max(0, token_index - window_size), min(len(all_tokens), token_index + window_size + 1)):
                            if i != token_index and all_tokens[i] not in _stop_words.ENGLISH_STOP_WORDS:
                                context_words.append(all_tokens[i])
                    except ValueError:
                        # Token bulunamazsa yakındaki kelimeleri al
                        context_words = [t for t in filtered_tokens[:4] if t != missing_token]
                    
                    # Anlamsal benzerlik ile benzer kelimeleri bul
                    similar_words = find_similar_words_semantic(
                        missing_token, self.embeddings_index, 
                        context_words=context_words, 
                        target_category=category,
                        word_score_cache=word_score_cache, 
                        max_candidates=10, 
                        use_context=True
                    )
                    
                    if similar_words:
                        # Bağlamsal KNN ağırlığı hesapla
                        contextual_weight = find_best_replacement_with_context(
                            missing_token, category, self.embeddings_index, 
                            word_score_cache, context_words, k=5
                        )
                        
                        # Anlamsal benzerlik vektörü oluştur
                        vector = create_semantic_missing_word_vector(
                            missing_token, similar_words, self.embeddings_index,
                            context_words, category, word_score_cache
                        )
                        
                        # Cache'e kaydet
                        self.missing_word_cache[missing_token] = vector
                        
                        # Geçici olarak embeddings_index'e ekle
                        self.embeddings_index[missing_token] = vector
                        
                        # Bağlamsal ağırlığı kullan
                        token_weights[missing_token] = contextual_weight
                    else:
                        # Anlamsal benzerlik bulunamazsa fallback
                        token_weights[missing_token] = 0.1
        
        if not token_weights:
            return np.zeros(self.dim)
        
        # VECTORIZED GLOVE İŞLEME - NumPy optimizasyonu
        tokens_list = list(token_weights.keys())
        weights_array = np.array([token_weights[token] for token in tokens_list])
        
        # GloVe vektörlerini batch halinde al
        glove_vectors = np.array([self.embeddings_index[token] for token in tokens_list])
        
        # Ağırlıklı vektörleri vectorized hesapla
        weighted_vectors = glove_vectors * weights_array[:, np.newaxis]  # Broadcasting
        
        # Ağırlıklı ortalama hesapla
        total_weight = weights_array.sum()
        return weighted_vectors.sum(axis=0) / total_weight
    
    def _calculate_cross_category_weight_from_cache(self, category_scores, target_category):
        """
        CACHED VERSION: Cache'den alınan skorlarla ultra-fast cross-category ağırlık hesaplama
        """
        # İSTATİSTİKSEL ANALİZ
        if len(category_scores) == 0 or all(score == 0 for score in category_scores):
            return 0.05
        
        # Hedef kategorinin skoru
        target_score = category_scores[target_category - 1]  # 0-indexed
        
        if target_score <= 0:
            return 0.05
        
        # Diğer kategorilerin skorları
        other_scores = [category_scores[i] for i in range(4) if i != (target_category - 1)]
        
        # İstatistiksel metrikler
        mean_other = np.mean(other_scores) if other_scores else 0
        std_other = np.std(other_scores) if len(other_scores) > 1 else 0.01
        
        # Z-SCORE BAZLI AYIRT EDİCİLİK
        if std_other > 0:
            z_score = (target_score - mean_other) / std_other
            normalized_distinctiveness = max(0, min(2, z_score)) / 2.0
        else:
            normalized_distinctiveness = min(target_score / (mean_other + 0.01), 2.0) / 2.0
        
        # VARYANS BAZLI EK AĞIRLIK
        variance_other = np.var(other_scores) if len(other_scores) > 1 else 0.01
        consistency_bonus = 1.0 / (1.0 + variance_other)
        
        # KATEGORI KAPSAMASI CEZASI
        presence_count = sum(1 for score in category_scores if score > 0)
        presence_ratio = presence_count / 4.0
        exclusivity_bonus = 1.0 - (presence_ratio - 0.25) * 0.5
        exclusivity_bonus = max(0.5, min(1.5, exclusivity_bonus))
        
        # FİNAL AĞIRLIK HESAPLAMA
        final_weight = target_score * (1.0 + normalized_distinctiveness) * consistency_bonus * exclusivity_bonus
        
        return min(final_weight, 2.5)
    
    def _calculate_smart_missing_weight(self, missing_token, target_category, word_score_cache):
        """Cache'deki eksik kelime için akıllı ağırlık hesapla"""
        # Benzer kelimeleri yeniden bul ve analiz et
        similar_words = find_similar_words(missing_token, self.embeddings_index, max_edit_distance=2, max_candidates=10)
        
        if similar_words:
            # KNN-like yaklaşımla ağırlık hesapla
            smart_weight = find_best_replacement_with_knn(
                missing_token, target_category, self.embeddings_index, word_score_cache, k=3
            )
            return smart_weight
        else:
            # Benzer kelime yoksa kategori-farkında fallback
            return 0.15
    
    def _calculate_contextual_missing_weight(self, missing_token, text_tokens, token_index, target_category, word_score_cache):
        """Bağlamsal analiz ile eksik kelime ağırlığı hesapla"""
        # 1. Bağlamsal analiz
        context_strength = analyze_context_for_missing_word(
            missing_token, text_tokens, token_index, target_category, word_score_cache
        )
        
        # 2. KNN-like en iyi replacement
        knn_weight = find_best_replacement_with_knn(
            missing_token, target_category, self.embeddings_index, word_score_cache, k=5
        )
        
        # 3. Benzer kelimelerin kategori analizi
        similar_words = find_similar_words(missing_token, self.embeddings_index)
        category_weight = calculate_category_aware_missing_weight(
            missing_token, similar_words, target_category, word_score_cache
        )
        
        # 4. Weighted combination
        # Bağlam güçlüyse ona daha fazla ağırlık ver
        context_factor = min(context_strength * 2, 1.0)
        
        final_weight = (
            context_strength * context_factor * 0.4 +
            knn_weight * 0.4 +
            category_weight * 0.2
        )
        
        return max(0.1, min(1.5, final_weight))
    
    def _text_to_weighted_vector_cross_category(self, text, category):
        """Cross-category ağırlık hesaplaması için optimize edilmiş vektör oluşturma"""
        tokens = text.split()
        stop_words = _stop_words.ENGLISH_STOP_WORDS
        tokens = [t for t in tokens if t not in stop_words]
        
        weighted_vectors = []
        total_weight = 0
        
        for token in tokens:
            # GloVe vektörü var mı?
            if token in self.embeddings_index:
                # Cross-category TF-IDF ağırlığını hesapla
                cross_weight = self._calculate_cross_category_weight(token, category, text)
                
                if cross_weight > 0:
                    weighted_vector = self.embeddings_index[token] * cross_weight
                    weighted_vectors.append(weighted_vector)
                    total_weight += cross_weight
        
        if not weighted_vectors or total_weight == 0:
            return np.zeros(self.dim)
        
        # Ağırlıklı ortalama hesapla
        weighted_vectors = np.array(weighted_vectors)
        return weighted_vectors.sum(axis=0) / total_weight
      
    def _text_to_weighted_vector(self, text, tfidf_vector, category):
        """Tek bir metni belirli kategori için cross-category TF-IDF ağırlıklı GloVe vektörüne dönüştür + Eksik kelime yönetimi"""
        tokens = text.split()
        stop_words = _stop_words.ENGLISH_STOP_WORDS
        tokens = [t for t in tokens if t not in stop_words]
        
        weighted_vectors = []
        total_weight = 0
        
        for token in tokens:
            # GloVe vektörü var mı?
            if token in self.embeddings_index:
                # Cross-category TF-IDF ağırlığını hesapla
                cross_weight = self._calculate_cross_category_weight(token, category, text)
                
                if cross_weight > 0:
                    weighted_vector = self.embeddings_index[token] * cross_weight
                    weighted_vectors.append(weighted_vector)
                    total_weight += cross_weight
            elif self.handle_missing_words:
                # Eksik kelime yönetimi
                # Cache kontrolü
                if token in self.missing_word_cache:
                    vector = self.missing_word_cache[token]
                    # Geçici olarak embeddings_index'e ekle
                    self.embeddings_index[token] = vector
                    # Cross-category ağırlığını hesapla
                    cross_weight = self._calculate_cross_category_weight(token, category, text)
                    if cross_weight > 0:
                        weighted_vector = vector * cross_weight
                        weighted_vectors.append(weighted_vector)
                        total_weight += cross_weight
                else:
                    # Benzer kelimeleri bul ve vektör oluştur
                    similar_words = find_similar_words(token, self.embeddings_index)
                    if similar_words:
                        vector = create_missing_word_vector(
                            token, similar_words, self.embeddings_index,
                            self.category_vectorizers, self.category_word_indices,
                            category, text
                        )
                        
                        # Cache'e kaydet
                        self.missing_word_cache[token] = vector
                        
                        # Geçici olarak embeddings_index'e ekle
                        self.embeddings_index[token] = vector
                        
                        # Cross-category ağırlığını hesapla
                        cross_weight = self._calculate_cross_category_weight(token, category, text)
                        if cross_weight > 0:
                            weighted_vector = vector * cross_weight
                            weighted_vectors.append(weighted_vector)
                            total_weight += cross_weight
        
        if not weighted_vectors or total_weight == 0:
            return np.zeros(self.dim)
        
        # Ağırlıklı ortalama hesapla
        weighted_vectors = np.array(weighted_vectors)
        return weighted_vectors.sum(axis=0) / total_weight
    
    def _calculate_cross_category_weight(self, token, target_category, text):
        """Bir kelime için tüm kategorileri dikkate alan ağırlık hesapla"""
        category_scores = {}
        
        # Her kategori için TF-IDF skorunu hesapla
        for cat in [1, 2, 3, 4]:
            if cat in self.category_vectorizers and token in self.category_word_indices[cat]:
                # Bu kategorideki TF-IDF skorunu al
                tfidf_vector = self.category_vectorizers[cat].transform([text])
                word_idx = self.category_word_indices[cat][token]
                category_scores[cat] = tfidf_vector[0, word_idx]
            else:
                category_scores[cat] = 0.0
        
        # Eğer hiçbir kategoride yoksa minimum ağırlık
        if all(score == 0 for score in category_scores.values()):
            return 0.05
    
        # Hedef kategorideki skor
        target_score = category_scores[target_category]
        
        # Diğer kategorilerdeki ortalama skor
        other_scores = [score for cat, score in category_scores.items() if cat != target_category]
        avg_other_score = np.mean(other_scores) if other_scores else 0
        
        # Ayırt edicilik hesapla: hedef kategoride yüksek, diğerlerinde düşük olması istenir
        if target_score > 0:
            # Ayırt edicilik katsayısı: hedef yüksek, diğerleri düşükse yüksek değer
            distinctiveness = target_score / (avg_other_score + 0.01)  # Sıfıra bölünmeyi engelle
            
            # Normalize et (1-3 arası değer)
            distinctiveness = min(max(distinctiveness, 0.1), 3.0)
            
            # Final ağırlık: hedef skor * ayırt edicilik katsayısı
            final_weight = target_score * distinctiveness
            
            # Çok yüksek değerleri sınırla
            return min(final_weight, 2.0)
        else:
            # Hedef kategoride hiç yoksa küçük ağırlık ver
            return 0.05
    
    def transform(self, texts):
        """Metinleri transform et - CACHE KULLANARAK SÜPER HIZLI!"""
        if not self.is_fitted:
            raise ValueError("Önce fit() metodunu çağırın!")
        
        # Metinleri temizle
        cleaned_texts = [enhanced_clean_text(text) for text in texts]
        
        # Multi-category vektörler oluştur (Cache kullanarak - build_cache=False)
        multi_vectors = self._create_multi_category_vectors_with_cache(cleaned_texts, build_cache=False)
        
        # PCA ile boyut azalt
        reduced_vectors = self.pca.transform(multi_vectors)
        
        return reduced_vectors
    
    def fit_transform(self, texts, labels):
        """Eğit ve dönüştür"""
        self.fit(texts, labels)
        return self.transform(texts)

def main():
    # Veri yükle
    print("Veri yükleniyor...")
    train_df = pd.read_csv(train_path, header=None)
    train_df.columns = ["label", "title", "description"]
    
    # Header satırını kaldır ve label'ları int'e çevir
    train_df = train_df[train_df["label"] != "Class Index"]
    train_df["label"] = train_df["label"].astype(int)
    train_df["text"] = train_df["title"] + " " + train_df["description"]

    test_df = pd.read_csv(test_path, header=None)
    test_df.columns = ["label", "title", "description"]
    
    # Header satırını kaldır ve label'ları int'e çevir
    test_df = test_df[test_df["label"] != "Class Index"]
    test_df["label"] = test_df["label"].astype(int)
    test_df["text"] = test_df["title"] + " " + test_df["description"]    # Hızlı test için veri örnekleme
    if SAMPLE_SIZE is not None and len(train_df) > SAMPLE_SIZE:
        print(f"Hızlı test için {SAMPLE_SIZE} örnek alınıyor...")
        # Her kategoriden eşit örnekleme
        sampled_dfs = []
        for category in [1, 2, 3, 4]:
            cat_df = train_df[train_df["label"] == category]
            sample_per_cat = min(SAMPLE_SIZE // 4, len(cat_df))
            sampled_dfs.append(cat_df.sample(n=sample_per_cat, random_state=42))
        train_df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Örnekleme sonrası eğitim seti: {len(train_df)}")
    
    X_train = train_df["text"]
    y_train = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]
    
    print(f"Eğitim seti boyutu: {len(X_train)}")
    print(f"Test seti boyutu: {len(X_test)}")
    print(f"Eğitim seti kategori dağılımı: {y_train.value_counts().sort_index()}")
    print(f"Test seti kategori dağılımı: {y_test.value_counts().sort_index()}")
    
    print(f"\n{'='*70}")
    print(f"Multi-Category TF-IDF + GloVe + PCA + SVM Sınıflandırması")
    print(f"{'='*70}")
      # GloVe modeli yükle
    try:
        glove_path = download_glove_model()
        embeddings_index = load_glove_model(glove_path)
    except Exception as e:
        print(f"GloVe modeli yüklenemedi: {e}")
        return
    
    # EKSİK KELİME ANALİZİ - GloVe'da olmayan kelimeleri analiz et
    analyze_missing_words(X_train.tolist(), embeddings_index, max_examples=15)
      # Multi-Category vektörleyici oluştur
    print("\nMulti-Category TF-IDF GloVe vektörleyici oluşturuluyor...")
    vectorizer = MultiCategoryTFIDFGloVeVectorizer(
        embeddings_index=embeddings_index,
        dim=GLOVE_DIM,
        max_features_per_category=TFIDF_MAX_FEATURES,
        final_dim=FINAL_DIM,
        handle_missing_words=True  # Eksik kelime yönetimini aktif et
    )
      # Eğitim seti vektörlere dönüştür
    print("\nEğitim seti vektörlere dönüştürülüyor...")
    X_train_vectors_pca = vectorizer.fit_transform(X_train, y_train)  # PCA'lı vektörler
      # PCA'sız vektörleri de al (CACHE KULLANARAK - SÜPER HIZLI!)
    print("PCA'sız vektörler alınıyor...")
    cleaned_train_texts = [enhanced_clean_text(text) for text in X_train]
    cleaned_test_texts = [enhanced_clean_text(text) for text in X_test]
    X_train_vectors_no_pca = vectorizer._create_multi_category_vectors_with_cache(cleaned_train_texts, build_cache=False)
    X_test_vectors_no_pca = vectorizer._create_multi_category_vectors_with_cache(cleaned_test_texts, build_cache=False)
    
    # Test seti vektörlere dönüştür (PCA'lı)
    print("Test seti PCA'lı vektörlere dönüştürülüyor...")
    X_test_vectors_pca = vectorizer.transform(X_test)
    
    print(f"PCA'lı vektör boyutu: {X_train_vectors_pca.shape[1]}D")
    print(f"PCA'sız vektör boyutu: {X_train_vectors_no_pca.shape[1]}D")
    
    # Models dizini kontrolü
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Sonuçları saklamak için
    results = {}
    
    print(f"\n{'='*80}")
    print(f"MODEL KARŞILAŞTIRMA TESTLERİ")
    print(f"{'='*80}")
      # 1. LOGISTIC REGRESSION - PCA'LI
    print(f"\n{'='*60}")
    print(f"1. LOGISTIC REGRESSION (PCA'lı - {FINAL_DIM}D)")
    print(f"{'='*60}")
    
    start_time = time.time()
    lr_pca_model = LogisticRegression(max_iter=2500, random_state=42)
    lr_pca_model.fit(X_train_vectors_pca, y_train)
    
    # Test seti tahmini
    y_pred_lr_pca = lr_pca_model.predict(X_test_vectors_pca)
    lr_pca_test_accuracy = accuracy_score(y_test, y_pred_lr_pca)
    
    # Çapraz doğrulama
    print("Çapraz doğrulama yapılıyor...")
    lr_pca_cv_scores = cross_val_score(lr_pca_model, X_train_vectors_pca, y_train, cv=cv, scoring='accuracy')
    lr_pca_cv_score = lr_pca_cv_scores.mean()
    lr_pca_training_time = time.time() - start_time
    
    results['LR_PCA'] = {
        'cv_score': lr_pca_cv_score,
        'cv_std': lr_pca_cv_scores.std(),
        'test_accuracy': lr_pca_test_accuracy,
        'training_time': lr_pca_training_time,
        'vector_dim': FINAL_DIM
    }
    
    print(f"Çapraz Doğrulama Skoru: {lr_pca_cv_score:.4f} ± {lr_pca_cv_scores.std():.4f}")
    print(f"Test Seti Doğruluğu: {lr_pca_test_accuracy:.4f}")
    print(f"Eğitim Süresi: {lr_pca_training_time:.2f} saniye")
      # 2. LOGISTIC REGRESSION - PCA'SIZ (Orijinal boyut)
    print(f"\n{'='*60}")
    print(f"2. LOGISTIC REGRESSION (PCA'sız - Orijinal Boyut)")
    print(f"{'='*60}")
    
    start_time = time.time()
    lr_no_pca_model = LogisticRegression(max_iter=3000, random_state=42)
    lr_no_pca_model.fit(X_train_vectors_no_pca, y_train)
    
    # Test seti tahmini
    y_pred_lr_no_pca = lr_no_pca_model.predict(X_test_vectors_no_pca)
    lr_no_pca_test_accuracy = accuracy_score(y_test, y_pred_lr_no_pca)
    
    # Çapraz doğrulama
    print("Çapraz doğrulama yapılıyor...")
    lr_no_pca_cv_scores = cross_val_score(lr_no_pca_model, X_train_vectors_no_pca, y_train, cv=cv, scoring='accuracy')
    lr_no_pca_cv_score = lr_no_pca_cv_scores.mean()
    lr_no_pca_training_time = time.time() - start_time
    
    results['LR_NO_PCA'] = {
        'cv_score': lr_no_pca_cv_score,
        'cv_std': lr_no_pca_cv_scores.std(),
        'test_accuracy': lr_no_pca_test_accuracy,
        'training_time': lr_no_pca_training_time,
        'vector_dim': X_train_vectors_no_pca.shape[1]
    }
    
    print(f"Çapraz Doğrulama Skoru: {lr_no_pca_cv_score:.4f} ± {lr_no_pca_cv_scores.std():.4f}")
    print(f"Test Seti Doğruluğu: {lr_no_pca_test_accuracy:.4f}")
    print(f"Eğitim Süresi: {lr_no_pca_training_time:.2f} saniye")
      # 3. SVM - SADECE PCA'LI
    print(f"\n{'='*60}")
    print(f"3. SVM (Sadece PCA'lı - {FINAL_DIM}D)")
    print(f"{'='*60}")
    
    start_time = time.time()
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    #svm_model.fit(X_train_vectors_pca, y_train)
    print("SVM modeli skipped")
    # Test seti tahmini
    y_pred_svm = svm_model.predict(X_test_vectors_pca)
    svm_test_accuracy = accuracy_score(y_test, y_pred_svm)
    
    # Çapraz doğrulama
    print("Çapraz doğrulama yapılıyor...")
    svm_cv_scores = cross_val_score(svm_model, X_train_vectors_pca, y_train, cv=cv, scoring='accuracy')
    svm_cv_score = svm_cv_scores.mean()
    svm_training_time = time.time() - start_time
    
    results['SVM_PCA'] = {
        'cv_score': svm_cv_score,
        'cv_std': svm_cv_scores.std(),
        'test_accuracy': svm_test_accuracy,
        'training_time': svm_training_time,
        'vector_dim': FINAL_DIM
    }
    
    print(f"Çapraz Doğrulama Skoru: {svm_cv_score:.4f} ± {svm_cv_scores.std():.4f}")
    print(f"Test Seti Doğruluğu: {svm_test_accuracy:.4f}")
    print(f"Eğitim Süresi: {svm_training_time:.2f} saniye")
      
    # KARŞILAŞTIRMALI SONUÇLAR
    print(f"\n{'='*80}")
    print(f"KARŞILAŞTIRMALI SONUÇLAR")
    print(f"{'='*80}")
    
    print(f"{'Model':<20} {'CV Score':<12} {'Test Acc':<12} {'Süre(s)':<10} {'Boyut':<8}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
    
    for model_name, result in results.items():
        model_display = model_name.replace('_', ' ')
        print(f"{model_display:<20} {result['cv_score']:.4f}±{result['cv_std']:.3f} {result['test_accuracy']:.4f}{'':8} {result['training_time']:.1f}{'':6} {result['vector_dim']}D")
    
    # En iyi modeli belirle
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 EN İYİ MODEL: {best_model_name.replace('_', ' ')}")
    print(f"   Test Doğruluğu: {best_result['test_accuracy']:.4f}")
    print(f"   CV Skoru: {best_result['cv_score']:.4f} ± {best_result['cv_std']:.4f}")
    print(f"   Eğitim Süresi: {best_result['training_time']:.2f} saniye")
    
    # Detaylı Classification Report'ları
    print(f"\n{'='*80}")
    print(f"DETAYLI CLASSIFICATION REPORTS")
    print(f"{'='*80}")
    
    print(f"\n📊 LOGISTIC REGRESSION (PCA'lı):")
    print(classification_report(y_test, y_pred_lr_pca))
    
    print(f"\n📊 LOGISTIC REGRESSION (PCA'sız):")
    print(classification_report(y_test, y_pred_lr_no_pca))
    
    print(f"\n📊 SVM (PCA'lı):")
    print(classification_report(y_test, y_pred_svm))
    
    # Confusion Matrix'leri
    print(f"\n{'='*80}")
    print(f"CONFUSION MATRICES")
    print(f"{'='*80}")
    
    print(f"\n🔍 LOGISTIC REGRESSION (PCA'lı) - Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr_pca))
    
    print(f"\n🔍 LOGISTIC REGRESSION (PCA'sız) - Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr_no_pca))
    
    print(f"\n🔍 SVM (PCA'lı) - Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))
    
    # Model kaydetme - En iyi modeli kaydet
    print(f"\n{'='*60}")
    print(f"MODEL KAYDETME")
    print(f"{'='*60}")
    
    if best_model_name == 'LR_PCA':
        best_model = lr_pca_model
        model_filename = "best_logistic_regression_pca.pkl"
        print("En iyi model: Logistic Regression (PCA'lı)")
    elif best_model_name == 'LR_NO_PCA':
        best_model = lr_no_pca_model
        model_filename = "best_logistic_regression_no_pca.pkl"
        print("En iyi model: Logistic Regression (PCA'sız)")
    else:
        best_model = svm_model
        model_filename = "best_svm_pca.pkl"
        print("En iyi model: SVM (PCA'lı)")
    
    # Tüm modelleri kaydet
    models_to_save = [
        (lr_pca_model, "logistic_regression_pca.pkl"),
        (lr_no_pca_model, "logistic_regression_no_pca.pkl"),
        (svm_model, "svm_pca.pkl"),
        (best_model, model_filename)
    ]
    
    for model_obj, filename in models_to_save:
        model_path = os.path.join(models_dir, filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model_obj, f)
        print(f"Model kaydedildi: {model_path}")
    
    vectorizer_path = os.path.join(models_dir, "multi_category_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vektörleyici kaydedildi: {vectorizer_path}")
      # PCA bilgileri
    print(f"\n📈 PCA AÇIKLANAN VARYANS: {vectorizer.pca.explained_variance_ratio_.sum():.4f}")
    print(f"📏 ORIJINAL BOYUT: {X_train_vectors_no_pca.shape[1]}D → PCA BOYUTU: {FINAL_DIM}D")
    print(f"🎯 KATEGORI BAŞINA MAX FEATURES: {TFIDF_MAX_FEATURES}")
    
    print(f"\n{'='*80}")
    print(f"TEST TAMAMLANDI! 🎉")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
