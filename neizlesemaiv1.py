import os
import httpx
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NizlesemAPI")

load_dotenv()

app = FastAPI(title="N'izlesem AI - Backend v1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TMDB_BEARER_TOKEN = os.getenv("TMDB_READ_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

MODELLER = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemma-4-31b-it",
    "gemma-3-27b-it",
    "gemini-2.5-pro",
    "gemini-pro-latest",
    "gemini-1.5-flash-latest",  
    "gemini-1.5-flash"
]

LAST_WORKING_MODEL = None

GENRE_IDS = {
    "aksiyon": 28, "macera": 12, "animasyon": 16, "komedi": 35, 
    "suç": 80, "belgesel": 99, "dram": 18, "aile": 10751, 
    "fantastik": 14, "tarih": 36, "korku": 27, "müzik": 10402, 
    "gizem": 9648, "romantik": 10749, "bilim kurgu": 878, 
    "gerilim": 53, "savaş": 10752, "batı": 37
}

def search_by_name(title: str):
    """Spesifik bir film veya dizi ismini arar."""
    url = "https://api.themoviedb.org/3/search/multi"
    params = {"query": title, "language": "tr-TR", "include_adult": "false"}
    headers = {"Authorization": f"Bearer {TMDB_BEARER_TOKEN}", "accept": "application/json"}
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers, params=params)
            data = response.json().get("results", [])
            
            if not data:
                return "Sonuç bulunamadı."
                
            results = []
            for item in data[:5]:
                results.append({
                    "ad": item.get("title") or item.get("name"),
                    "ozet": (item.get("overview")[:200] + "...") if item.get("overview") else "Özet bulunmuyor.",
                    "puan": item.get("vote_average"),
                    "tarih": item.get("release_date") or item.get("first_air_date"),
                    "poster": f"https://image.tmdb.org/t/p/w500{item.get('poster_path')}" if item.get('poster_path') else None
                })
            return results
    except Exception as e:
        logger.error(f"TMDB İsim Arama Hatası: {e}")
        return {"hata": "Veriler şu an alınamıyor."}

def discover_by_filters(genre_name: str = None, year: int = None, min_rating: float = None):
    """Tür, yıl ve puana göre keşif yapar."""
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "language": "tr-TR",
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "page": 1
    }
    
    if genre_name and genre_name.lower() in GENRE_IDS:
        params["with_genres"] = GENRE_IDS[genre_name.lower()]
    if year:
        params["primary_release_year"] = year
    if min_rating:
        params["vote_average.gte"] = min_rating

    headers = {"Authorization": f"Bearer {TMDB_BEARER_TOKEN}", "accept": "application/json"}
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers, params=params)
            data = response.json().get("results", [])
            
            if not data:
                return "Bu kriterlere uygun sonuç bulunamadı."
                
            results = []
            for item in data[:5]:
                results.append({
                    "ad": item.get("title"),
                    "ozet": (item.get("overview")[:200] + "...") if item.get("overview") else "Özet bulunmuyor.",
                    "puan": item.get("vote_average"),
                    "tarih": item.get("release_date"),
                    "poster": f"https://image.tmdb.org/t/p/w500{item.get('poster_path')}" if item.get('poster_path') else None
                })
            return results
    except Exception as e:
        logger.error(f"TMDB Keşif Hatası: {e}")
        return {"hata": "Veriler şu an alınamıyor."}

SYSTEM_PROMPT = f"""
Senin adın N'izlesem AI. Kullanıcıların film ve dizi kararsızlığına son veren, samimi, eğlenceli ve dürüst bir sinema rehberisin.

GÖREV VE ARAÇ KULLANIMI:
Elinde iki adet güçlü arama aracı var. Kullanıcının talebine göre en uygun olanı KESİNLİKLE kullanmalısın:
- search_by_name: Kullanıcı belirli bir film, dizi veya seri ismi verirse bunu kullan (Örn: Dune, Interstellar, Dark).
- discover_by_filters: Kullanıcı genel bir tavsiye, tür, yıl veya puan belirtirse bunu kullan. 
  Tür (genre_name) parametresi için SADECE şu listeyi kullan: {list(GENRE_IDS.keys())}.

KRİTİK KURAL (İNİSİYATİF AL):
- Eğer kullanıcı sadece "Bana aksiyon filmi öner", "Komedi izlemek istiyorum" gibi KISA ve EKSİK bilgiler verirse, ONA ASLA "Hangi yıl olsun?", "Puanı kaç olsun?" GİBİ SORULAR SORMA! 
- Hemen inisiyatif al, eksik parametreleri boş bırak (sadece türü gönder) ve arama aracını çalıştırarak doğrudan film önerisini sun.

YASAKLI İŞARETLER (ÇOK ÖNEMLİ):
- Metinlerinde KESİNLİKLE * (yıldız) veya # (kare) işareti KULLANMA! 
- Kalın (bold) veya italik yazım formatlarını kullanma. 
- Vurgulamak istediğin yerleri BÜYÜK HARFLE yaz.
- Madde imi veya liste yaparken sadece - (tire) işareti kullan.

ÖNERİ ŞABLONU VE YORUM:
Bir yapımı önerirken sadece verileri sunup geçme, bir sinema eleştirmeni gibi aşağıdaki başlıkları (büyük harflerle) kullanarak detaylı yorum yap:

YAPIMIN ADI (Yıl) - Puan: X.X
    
KONUSU:
- Buraya TMDB'den gelen kısa ve ilgi çekici özeti yaz.

KİMLER İÇİN UYGUN:
- Bu filmi ne tarz izleyiciler, hangi ruh halindekiler izlemeli? Film ne hissettiriyor? Kısa ve samimi bir analiz yap.

BENZER YAPIMLAR:
- Bu filmin atmosferine veya senaryosuna benzeyen, bunu sevenlerin kesinlikle hoşuna gideceği 2 veya 3 popüler film/dizi örneği ver.

DİĞER KURALLAR:
- ASLA uydurma film önerme; DAİMA TMDB'den güncel veri çek.
- Poster linklerini kesinlikle gizli tut, metin içinde URL olarak veya görsel olarak gösterme.
- Araçlar "Sonuç bulunamadı" derse, AI olduğunu belli etmeden farklı filtreler iste.
"""

@app.get("/chat")
async def chat(prompt: str = Query(..., min_length=2)):
    global LAST_WORKING_MODEL
    
    models_to_try = []
    if LAST_WORKING_MODEL:
        models_to_try.append(LAST_WORKING_MODEL)
    
    for m in MODELLER:
        if m != LAST_WORKING_MODEL:
            models_to_try.append(m)

    for m_name in models_to_try:
        try:
            model = genai.GenerativeModel(
                model_name=m_name,
                tools=[search_by_name, discover_by_filters],
                system_instruction=SYSTEM_PROMPT
            )
            
            chat_session = model.start_chat(enable_automatic_function_calling=True)
            response = chat_session.send_message(prompt)
            
            LAST_WORKING_MODEL = m_name
            
            return {
                "status": "success",
                "reply": response.text,
                "model_used": m_name
            }
        except Exception as e:
            logger.error(f"Model {m_name} basarisiz oldu: {e}")
            continue

    return {
        "status": "error",
        "message": "Sistem şu an çok yoğun, lütfen birazdan tekrar dene."
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "N'izlesem"}