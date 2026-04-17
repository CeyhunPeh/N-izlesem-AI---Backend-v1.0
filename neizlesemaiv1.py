import os
import httpx
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv

# Loglama ayarları (Render loglarında hataları görmek için)
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

# API Anahtarları
TMDB_BEARER_TOKEN = os.getenv("TMDB_READ_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Senin genişletilmiş model listen
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

# --- GELİŞTİRİLMİŞ TMDB ARACI ---
async def search_tmdb(query: str, category: str = "movie"):
    """TMDB üzerinden film veya dizi bilgilerini çeker."""
    endpoint = "search/movie" if category == "movie" else "search/tv"
    url = f"https://api.themoviedb.org/3/{endpoint}"
    params = {
        "query": query,
        "include_adult": "false",
        "language": "tr-TR",
        "page": "1"
    }
    headers = {
        "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
        "accept": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json().get("results", [])
            
            # Sadece gerekli alanları temizleyip AI'ya veriyoruz (Token tasarrufu)
            results = []
            for item in data[:5]:
                results.append({
                    "ad": item.get("title") or item.get("name"),
                    "ozet": item.get("overview")[:200] + "...", # Özeti kısa tut
                    "puan": item.get("vote_average"),
                    "tarih": item.get("release_date") or item.get("first_air_date"),
                    "poster": f"https://image.tmdb.org/t/p/w500{item.get('poster_path')}" if item.get('poster_path') else None
                })
            return results
    except Exception as e:
        logger.error(f"TMDB Hatası: {e}")
        return {"hata": "Film verileri şu an alınamıyor."}

# --- OPTİMİZE EDİLMİŞ PROMPT ---
SYSTEM_PROMPT = """
Senin adın 'N'izlesem AI'. Kullanıcıların film ve dizi kararsızlığına son veren bir uzmansın.
Kullanıcı seninle konuştuğunda:
1. Mutlaka search_tmdb fonksiyonunu kullanarak gerçek verileri kontrol et.
2. Sadece isim vermekle kalma, neden önerdiğini 'N'izlesem' tarzında (samimi, kısa, ikna edici) açıkla.
3. Eğer bir yapımın poster linki varsa, bunu cevabın içinde gizli tut (Flutter tarafı JSON objesini parse edecek).
4. Tür tercihleri (aksiyon, korku vb.) için TMDB'de o türde popüler aramalar yap.
5. Cevaplarını Markdown formatında (bold, listeler vb.) ver ki Flutter'da güzel görünsün.
"""

def get_best_model():
    for m_name in MODELLER:
        try:
            # Modeli yüklüyoruz
            curr_model = genai.GenerativeModel(
                model_name=m_name,
                tools=[search_tmdb],
                system_instruction=SYSTEM_PROMPT
            )
            # Modeli sadece döndürüyoruz, eğer çağrı anında hata verirse 
            # chat fonksiyonu içindeki try-except bunu yakalayıp 
            # bir sonraki döngüde farklı model denemeli.
            return curr_model
        except Exception as e:
            logger.error(f"{m_name} başlatılamadı, sonrakine geçiliyor: {e}")
            continue
    return None

@app.get("/chat")
async def chat(prompt: str = Query(..., min_length=2)):
    try:
        model = get_best_model()
        # Otomatik fonksiyon çağırma (Function Calling) aktif
        chat_session = model.start_chat(enable_automatic_function_calling=True)
        
        response = chat_session.send_message(prompt)
        
        return {
            "status": "success",
            "reply": response.text,
            "model_used": model.model_name
        }
    except Exception as e:
        logger.error(f"Genel Hata: {e}")
        return {
            "status": "error",
            "message": "Sistem şu an çok yoğun, lütfen birazdan tekrar dene."
        }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "N'izlesem"}