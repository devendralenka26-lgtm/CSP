import os
import sys
import uuid
import datetime
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import jwt
import requests
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from deep_translator import GoogleTranslator
import google.generativeai as genai

# Try to import psycopg2 for Supabase/PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import DictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
IS_VERCEL = 'VERCEL' in os.environ

SECRET_KEY = os.environ.get('SECRET_KEY', 'agriai_super_secret_token_102938')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'agriai_jwt_secret_token_987654')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if IS_VERCEL:
    UPLOAD_FOLDER = '/tmp/uploads'
else:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')

# ==========================================
# DATABASE ADAPTER LAYER
# ==========================================
def get_db_connection():
    db_url = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL')
    
    if not db_url:
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), 'agriai.db')
        try:
            class SQLiteCursor(sqlite3.Cursor):
                def execute(self, query, params=()):
                    query_mod = query.replace('%s', '?')
                    if "CREATE TABLE" in query_mod.upper():
                        query_mod = query_mod.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
                    return super().execute(query_mod, params)

                def executemany(self, query, seq_of_params):
                    query_mod = query.replace('%s', '?')
                    if "CREATE TABLE" in query_mod.upper():
                        query_mod = query_mod.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
                    return super().executemany(query_mod, seq_of_params)

            class SQLiteConnection(sqlite3.Connection):
                def cursor(self, *args, **kwargs):
                    return super().cursor(factory=SQLiteCursor, *args, **kwargs)

            conn = sqlite3.connect(db_path, factory=SQLiteConnection)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            raise RuntimeError(f"Failed to connect to SQLite fallback database: {e}")
        
    if not HAS_PSYCOPG2:
        raise ImportError("psycopg2-binary is not installed or failed to import, but it is required for PostgreSQL/Supabase database connections.")
        
    try:
        conn = psycopg2.connect(db_url, cursor_factory=DictCursor)
        return conn
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Supabase/PostgreSQL: {e}")

def db_execute(conn, query, params=()):
    cursor = conn.cursor()
    cursor.execute(query, params)
    return cursor

def db_write(conn, query, params=()):
    cursor = db_execute(conn, query, params)
    cursor.close()

def db_fetchone(conn, query, params=()):
    cursor = db_execute(conn, query, params)
    row = cursor.fetchone()
    cursor.close()
    return row

def db_fetchall(conn, query, params=()):
    cursor = db_execute(conn, query, params)
    rows = cursor.fetchall()
    cursor.close()
    return rows

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('farmer', 'expert', 'admin')),
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Crops table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crops (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            crop_name TEXT NOT NULL,
            soil_type TEXT,
            moisture_level REAL,
            growth_stage TEXT,
            area_acres REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Disease Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS disease_predictions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            crop_name TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            affected_area_ratio REAL,
            original_img_path TEXT NOT NULL,
            processed_img_path TEXT NOT NULL,
            treatment_organic TEXT,
            treatment_chemical TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Weather Advisory Cache table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            latitude REAL,
            longitude REAL,
            temperature REAL,
            humidity REAL,
            rain_probability REAL,
            description TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Fertilizer Recommendations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fertilizer_recommendations (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            crop_name TEXT NOT NULL,
            recommended_fertilizer TEXT NOT NULL,
            organic_alternative TEXT,
            quantity TEXT,
            application_method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Chat History table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            sender TEXT NOT NULL CHECK(sender IN ('user', 'assistant')),
            message TEXT NOT NULL,
            language TEXT DEFAULT 'en',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Notifications table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('weather', 'disease', 'irrigation', 'system')),
            read_status INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Irrigation Plans table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS irrigation_plans (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            crop_name TEXT NOT NULL,
            recommended_time TEXT NOT NULL,
            water_quantity TEXT,
            drip_guideline TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Market Prices table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_prices (
            id SERIAL PRIMARY KEY,
            crop_name VARCHAR(255) NOT NULL UNIQUE,
            price_per_kg REAL NOT NULL,
            location TEXT NOT NULL,
            trend TEXT NOT NULL CHECK(trend IN ('up', 'down', 'stable')),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Seed initial market prices if empty
    cursor.execute("SELECT COUNT(*) FROM market_prices")
    count = cursor.fetchone()[0]
    if count == 0:
        initial_prices = [
            ('Tomato', 24.50, 'Madanapalle, AP', 'down'),
            ('Paddy', 21.00, 'Kurnool, AP', 'up'),
            ('Potato', 18.00, 'Agra, UP', 'stable'),
            ('Cotton', 68.00, 'Yavatmal, MH', 'up'),
            ('Wheat', 25.50, 'Khanna, PB', 'stable'),
            ('Maize', 22.00, 'Gulbarga, KA', 'up'),
            ('Banana', 15.00, 'Jalgaon, MH', 'stable'),
            ('Chili', 180.00, 'Guntur, AP', 'up'),
            ('Mango', 45.00, 'Ratnagiri, MH', 'down'),
            ('Groundnut', 72.00, 'Rajkot, GJ', 'stable')
        ]
        insert_query = "INSERT INTO market_prices (crop_name, price_per_kg, location, trend) VALUES (%s, %s, %s, %s)"
        cursor.executemany(insert_query, initial_prices)
        
    # Crop Yield Trends Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crop_yield_trends (
            id SERIAL PRIMARY KEY,
            year VARCHAR(50) NOT NULL,
            crop_name VARCHAR(255) NOT NULL,
            yield_value REAL NOT NULL,
            UNIQUE(year, crop_name)
        )
    ''')
    
    # Water Telemetry Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS water_telemetry (
            id SERIAL PRIMARY KEY,
            water_used_liters REAL NOT NULL,
            water_saved_liters REAL NOT NULL,
            drip_efficiency VARCHAR(50) NOT NULL,
            saving_ratio VARCHAR(50) NOT NULL
        )
    ''')
    
    # Seed Yield Trends
    cursor.execute("SELECT COUNT(*) FROM crop_yield_trends")
    if cursor.fetchone()[0] == 0:
        yield_data = [
            ('2022', 'Tomato', 10.2),
            ('2023', 'Tomato', 11.0),
            ('2024', 'Tomato', 11.5),
            ('2025', 'Tomato', 12.8),
            ('2022', 'Paddy', 2.1),
            ('2023', 'Paddy', 2.3),
            ('2024', 'Paddy', 2.2),
            ('2025', 'Paddy', 2.4)
        ]
        cursor.executemany("INSERT INTO crop_yield_trends (year, crop_name, yield_value) VALUES (%s, %s, %s)", yield_data)
        
    # Seed Water Telemetry
    cursor.execute("SELECT COUNT(*) FROM water_telemetry")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO water_telemetry (water_used_liters, water_saved_liters, drip_efficiency, saving_ratio) VALUES (%s, %s, %s, %s)",
            (145000.0, 35000.0, '92%', '24.1%')
        )
        
    conn.commit()
    cursor.close()
    conn.close()

# ==========================================
# CROP DISEASE PROFILES METADATA
# ==========================================
DISEASE_PROFILES = {
    'Tomato': {
        0: {
            'name': 'Healthy Tomato',
            'severity': 'Healthy',
            'organic': 'Maintain proper crop spacing and soil health.',
            'chemical': 'No chemical treatment needed.',
            'prevention': 'Rotate crops annually and use disease-free seed stock.'
        },
        1: {
            'name': 'Tomato Early Blight',
            'severity': 'Moderate',
            'organic': 'Apply copper-based fungicides or compost tea. Prune lower leaves to prevent soil splash.',
            'chemical': 'Apply Chlorothalonil or Mancozeb fungicides according to instructions.',
            'prevention': 'Use drip irrigation instead of overhead watering to keep foliage dry.'
        },
        2: {
            'name': 'Tomato Late Blight',
            'severity': 'High',
            'organic': 'Immediately destroy infected plants. Apply copper spray before rain events.',
            'chemical': 'Apply Metalaxyl or chlorothalonil immediately upon detection.',
            'prevention': 'Do not plant tomatoes near potatoes. Ensure air circulation.'
        },
        3: {
            'name': 'Tomato Leaf Mold',
            'severity': 'Moderate',
            'organic': 'Remove infected leaves. Maintain low humidity in greenhouses using ventilation.',
            'chemical': 'Use Difolatan or Bravo fungicides if disease pressure is high.',
            'prevention': 'Prune lower leaves to improve airflow and water at base.'
        }
    },
    'Paddy': {
        0: {
            'name': 'Healthy Paddy',
            'severity': 'Healthy',
            'organic': 'Optimize water levels and weed growth.',
            'chemical': 'No chemical treatment needed.',
            'prevention': 'Use resistant varieties and balanced nitrogen fertilizers.'
        },
        1: {
            'name': 'Paddy Blast',
            'severity': 'High',
            'organic': 'Use neem seed kernel extract or spray Pseudomonas fluorescens.',
            'chemical': 'Apply Tricyclazole or Edifenphos at early leaf stages.',
            'prevention': 'Avoid excessive nitrogen fertilizer and maintain proper field flooding.'
        },
        2: {
            'name': 'Paddy Bacterial Leaf Blight',
            'severity': 'High',
            'organic': 'Spray fresh cow dung filtrate or bleach soil. Reduce water levels.',
            'chemical': 'Apply Copper Hydroxide combined with Streptomycin sulphate.',
            'prevention': 'Avoid clipping seedlings during transplanting. Keep bunds clean.'
        },
        3: {
            'name': 'Paddy Brown Spot',
            'severity': 'Moderate',
            'organic': 'Apply biofertilizers. Correct potassium deficiencies in soil.',
            'chemical': 'Spray Mancozeb or Carbendazim at tillering and panicle initiation.',
            'prevention': 'Ensure proper soil drainage and test soil nutrients regularly.'
        }
    },
    'Potato': {
        0: {
            'name': 'Healthy Potato',
            'severity': 'Healthy',
            'organic': 'Practice clean seed selection.',
            'chemical': 'No chemical treatment needed.',
            'prevention': 'Practice 3-year crop rotation.'
        },
        1: {
            'name': 'Potato Early Blight',
            'severity': 'Moderate',
            'organic': 'Apply copper fungicides. Remove volunteer potato plants early.',
            'chemical': 'Apply Mancozeb or Chlorothalonil sprays weekly.',
            'prevention': 'Avoid overhead irrigation. Ensure adequate nitrogen and potassium.'
        },
        2: {
            'name': 'Potato Late Blight',
            'severity': 'High',
            'organic': 'Prune affected stems. Apply compost tea. Destroy infected tubers.',
            'chemical': 'Spray Cymoxanil or Metalaxyl-M immediately.',
            'prevention': 'Use certified disease-free seed tubers. Keep potatoes hilled up.'
        },
        3: {
            'name': 'Potato Common Scab',
            'severity': 'Low',
            'organic': 'Keep soil pH below 5.2. Apply sulfur to acidify soil.',
            'chemical': 'Treat seed tubers with PCNB or Mancozeb dust.',
            'prevention': 'Keep soil moist during tuber initiation (first 4-6 weeks).'
        }
    },
    'Cotton': {
        0: {
            'name': 'Healthy Cotton',
            'severity': 'Healthy',
            'organic': 'Maintain soil aeration.',
            'chemical': 'None',
            'prevention': 'Deep summer plowing.'
        },
        1: {
            'name': 'Cotton Bacterial Blight',
            'severity': 'High',
            'organic': 'Spray Streptomycin sulphate mixture or copper oxychloride.',
            'chemical': 'Apply copper oxychloride at 2.5g/L.',
            'prevention': 'Use acid-delinted seeds and clean crop debris.'
        },
        2: {
            'name': 'Cotton Leaf Curl Virus',
            'severity': 'High',
            'organic': 'Uproot infected plants immediately. Spray neem oil to manage whiteflies.',
            'chemical': 'Apply systemic insecticides like Imidacloprid to control the whitefly vector.',
            'prevention': 'Eradicate weeds around the field which act as alternative hosts.'
        }
    },
    'Wheat': {
        0: {
            'name': 'Healthy Wheat',
            'severity': 'Healthy',
            'organic': 'Ensure proper soil drainage.',
            'chemical': 'None',
            'prevention': 'Avoid early sowing.'
        },
        1: {
            'name': 'Wheat Rust (Stripe/Leaf)',
            'severity': 'High',
            'organic': 'Spray botanical oil formulations. Plant rust-resistant cultivars.',
            'chemical': 'Spray Propiconazole or Tebuconazole fungicide.',
            'prevention': 'Avoid over-irrigation. Monitor crop weekly starting from winter.'
        },
        2: {
            'name': 'Wheat Powdery Mildew',
            'severity': 'Moderate',
            'organic': 'Apply sulfur-based powders. Reduce crop density to increase air flow.',
            'chemical': 'Apply Triadimefon or Carbendazim sprays.',
            'prevention': 'Balanced nitrogen application and timely sowing.'
        }
    },
    'Maize': {
        0: {
            'name': 'Healthy Maize',
            'severity': 'Healthy',
            'organic': 'Apply organic compost.',
            'chemical': 'None',
            'prevention': 'Intercropping with legumes.'
        },
        1: {
            'name': 'Maize Common Rust',
            'severity': 'Moderate',
            'organic': 'Apply copper fungicide. Destroy infected leaves after harvest.',
            'chemical': 'Spray Mancozeb or Pyraclostrobin.',
            'prevention': 'Plant resistant hybrids. Maintain soil potassium levels.'
        },
        2: {
            'name': 'Maize Northern Leaf Blight',
            'severity': 'High',
            'organic': 'Rotate with non-grass crops. Plow under residue after harvesting.',
            'chemical': 'Apply Propiconazole or Azoxystrobin at early silking.',
            'prevention': 'Ensure crop rotations of at least 2 years away from maize.'
        }
    },
    'Banana': {
        0: {
            'name': 'Healthy Banana',
            'severity': 'Healthy',
            'organic': 'Use bio-fertilizers and organic mulch.',
            'chemical': 'None',
            'prevention': 'Keep plantation weed-free.'
        },
        1: {
            'name': 'Banana Black Sigatoka',
            'severity': 'High',
            'organic': 'De-leaf old and spotted leaves. Improve soil drainage and spacing.',
            'chemical': 'Apply Mancozeb or systemic triazoles alternately to prevent resistance.',
            'prevention': 'Prune suckers to reduce canopy humidity.'
        },
        2: {
            'name': 'Banana Panama Wilt',
            'severity': 'High',
            'organic': 'Inject Trichoderma viride. Apply bio-agents to planting pits.',
            'chemical': 'No chemical cure available. Disinfect farming tools.',
            'prevention': 'Use tissue-cultured planting materials. Quarantine infected areas.'
        }
    },
    'Chili': {
        0: {
            'name': 'Healthy Chili',
            'severity': 'Healthy',
            'organic': 'Intercrop with marigold.',
            'chemical': 'None',
            'prevention': 'Use healthy nurseries.'
        },
        1: {
            'name': 'Chili Anthracnose (Fruit Rot)',
            'severity': 'High',
            'organic': 'Spray neem formulations. Destroy affected fruits and seeds.',
            'chemical': 'Apply Azoxystrobin or Copper Oxychloride.',
            'prevention': 'Seed treatment with Thiram. Proper field sanitation.'
        },
        2: {
            'name': 'Chili Leaf Curl Virus',
            'severity': 'High',
            'organic': 'Erect yellow sticky traps. Spray neem oil to control thrips/whiteflies.',
            'chemical': 'Spray Dimethoate or Acephate to control vectors.',
            'prevention': 'Grow barrier crops like maize or sorghum around chili plots.'
        }
    },
    'Mango': {
        0: {
            'name': 'Healthy Mango',
            'severity': 'Healthy',
            'organic': 'Prune crowded inner branches.',
            'chemical': 'None',
            'prevention': 'Spray water force to dislodge pests.'
        },
        1: {
            'name': 'Mango Anthracnose',
            'severity': 'High',
            'organic': 'Prune diseased twigs. Spray copper oxychloride before flowering.',
            'chemical': 'Apply Carbendazim or Bordeaux mixture (1%) during flowering.',
            'prevention': 'Avoid harvesting during rain. Clean orchard floor.'
        },
        2: {
            'name': 'Mango Powdery Mildew',
            'severity': 'Moderate',
            'organic': 'Spray wettable sulfur or biological control agents like Bacillus.',
            'chemical': 'Apply Dinocap or Hexaconazole at panicle emergence.',
            'prevention': 'Keep tree canopy open to sun and wind.'
        }
    },
    'Groundnut': {
        0: {
            'name': 'Healthy Groundnut',
            'severity': 'Healthy',
            'organic': 'Maintain soil calcium.',
            'chemical': 'None',
            'prevention': 'Seed inoculation with Rhizobium.'
        },
        1: {
            'name': 'Groundnut Tikka Leaf Spot',
            'severity': 'High',
            'organic': 'Spray neem seed kernel extract or garlic extract.',
            'chemical': 'Apply Carbendazim or Mancozeb on spots.',
            'prevention': 'Burn crop residues. Plant early maturing crop varieties.'
        },
        2: {
            'name': 'Groundnut Rust',
            'severity': 'High',
            'organic': 'Apply neem cake. Remove volunteer groundnut plants.',
            'chemical': 'Spray Chlorothalonil or Tebuconazole.',
            'prevention': 'Crop rotation and avoidance of overlapping groundnut seasons.'
        }
    }
}

# ==========================================
# SERVICES
# ==========================================
class CVService:
    @staticmethod
    def process_leaf_image(image_path, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image file path or unable to read image.")

        target_size = (500, 500)
        img_resized = cv2.resize(img, target_size)
        denoised = cv2.bilateralFilter(img_resized, 9, 75, 75)
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

        # Disease / Lesion detection masks (yellow/brown/grey regions)
        lower_yellow = np.array([10, 40, 40])
        upper_yellow = np.array([25, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        lower_brown = np.array([0, 30, 20])
        upper_brown = np.array([20, 255, 180])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # Background separation (Green color range)
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Combine leaf mask
        leaf_mask = cv2.bitwise_or(green_mask, yellow_mask)
        leaf_mask = cv2.bitwise_or(leaf_mask, brown_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

        disease_mask_raw = cv2.bitwise_or(yellow_mask, brown_mask)
        disease_mask = cv2.bitwise_and(disease_mask_raw, leaf_mask)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Contour Detection
        leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        disease_contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        leaf_area = sum(cv2.contourArea(c) for c in leaf_contours)
        total_pixel_area = target_size[0] * target_size[1]
        leaf_ratio = leaf_area / total_pixel_area if leaf_area > 0 else (np.sum(leaf_mask == 255) / total_pixel_area)
        
        if leaf_ratio < 0.05:
            gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            leaf_mask = cv2.bitwise_not(thresholded)
            leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            disease_mask = cv2.bitwise_and(disease_mask_raw, leaf_mask)
            disease_contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        leaf_pixel_count = np.sum(leaf_mask == 255)
        disease_pixel_count = np.sum(disease_mask == 255)
        
        infection_ratio = (disease_pixel_count / leaf_pixel_count * 100.0) if leaf_pixel_count > 0 else 0.0
        infection_ratio = min(round(infection_ratio, 2), 100.0)

        # Generate Annotated Image
        annotated_img = img_resized.copy()
        cv2.drawContours(annotated_img, leaf_contours, -1, (0, 255, 0), 2)
        cv2.drawContours(annotated_img, disease_contours, -1, (0, 0, 255), 2)

        # Heatmap
        heatmap_blur = cv2.GaussianBlur(disease_mask, (25, 25), 0)
        heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        _, thresh = cv2.threshold(heatmap_blur, 10, 255, cv2.THRESH_BINARY)
        heatmap_overlay = img_resized.copy()
        idx = (thresh > 0)
        heatmap_overlay[idx] = cv2.addWeighted(img_resized, 0.5, heatmap_color, 0.5, 0)[idx]

        # Unique Filenames
        run_id = str(uuid.uuid4())[:8]
        orig_filename = f"orig_{run_id}.jpg"
        annotated_filename = f"annotated_{run_id}.jpg"
        heatmap_filename = f"heatmap_{run_id}.jpg"

        cv2.imwrite(os.path.join(output_dir, orig_filename), img_resized)
        cv2.imwrite(os.path.join(output_dir, annotated_filename), annotated_img)
        cv2.imwrite(os.path.join(output_dir, heatmap_filename), heatmap_overlay)

        return {
            "original_name": orig_filename,
            "annotated_name": annotated_filename,
            "heatmap_name": heatmap_filename,
            "infection_ratio": infection_ratio,
            "lesion_count": len(disease_contours)
        }

class WeatherService:
    @staticmethod
    def get_weather_forecast(lat=16.30, lon=80.45):
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})
                daily = data.get("daily", {})
                
                temp = current.get("temperature_2m", 28.5)
                humidity = current.get("relative_humidity_2m", 65.0)
                wind_speed = current.get("wind_speed_10m", 12.0)
                rain_prob = daily.get("precipitation_probability_max", [20])[0]
                weather_code = current.get("weather_code", 0)
                
                description = WeatherService._map_weather_code(weather_code)
                advisory = WeatherService._generate_advisory(temp, humidity, wind_speed, rain_prob)
                
                forecast = []
                today = datetime.date.today()
                for i in range(5):
                    date_str = (today + datetime.timedelta(days=i)).strftime("%a")
                    min_t = daily.get("temperature_2m_min", [22.0] * 5)[i]
                    max_t = daily.get("temperature_2m_max", [35.0] * 5)[i]
                    prob = daily.get("precipitation_probability_max", [10] * 5)[i]
                    code = daily.get("weather_code", [0] * 5)[i]
                    
                    forecast.append({
                        "day": date_str,
                        "min_temp": min_t,
                        "max_temp": max_t,
                        "rain_probability": prob,
                        "description": WeatherService._map_weather_code(code)
                    })
                
                return {
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "rain_probability": rain_prob,
                    "description": description,
                    "advisory": advisory,
                    "forecast": forecast,
                    "source": "live_api"
                }
        except Exception as e:
            print(f"Weather API request failed: {e}. Using offline simulation data.")
            
        return WeatherService.get_offline_weather()

    @staticmethod
    def get_offline_weather():
        temp = 29.5
        humidity = 62.0
        wind_speed = 10.5
        rain_prob = 15
        description = "Sunny and Warm"
        
        forecast = [
            {"day": "Mon", "min_temp": 23.0, "max_temp": 34.0, "rain_probability": 10, "description": "Sunny"},
            {"day": "Tue", "min_temp": 24.0, "max_temp": 35.0, "rain_probability": 15, "description": "Partly Cloudy"},
            {"day": "Wed", "min_temp": 22.0, "max_temp": 33.0, "rain_probability": 40, "description": "Light Showers"},
            {"day": "Thu", "min_temp": 23.0, "max_temp": 34.0, "rain_probability": 20, "description": "Partly Cloudy"},
            {"day": "Fri", "min_temp": 24.0, "max_temp": 36.0, "rain_probability": 5, "description": "Clear and Dry"}
        ]
        
        advisory = WeatherService._generate_advisory(temp, humidity, wind_speed, rain_prob)
        
        return {
            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "rain_probability": rain_prob,
            "description": description,
            "advisory": advisory,
            "forecast": forecast,
            "source": "offline_cache"
        }

    @staticmethod
    def _map_weather_code(code):
        mapping = {
            0: "Clear Sky", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing Rime Fog", 51: "Light Drizzle", 53: "Moderate Drizzle",
            55: "Dense Drizzle", 61: "Slight Rain", 63: "Moderate Rain", 65: "Heavy Rain",
            71: "Slight Snow Fall", 73: "Moderate Snow Fall", 75: "Heavy Snow Fall",
            80: "Slight Rain Showers", 81: "Moderate Rain Showers", 82: "Violent Rain Showers",
            95: "Thunderstorm"
        }
        return mapping.get(code, "Clear Sky")

    @staticmethod
    def _generate_advisory(temp, humidity, wind_speed, rain_prob):
        advisories = []
        if rain_prob > 50:
            advisories.append("🌧️ Heavy rainfall is expected. Avoid spraying pesticides or chemical fertilizers today as they will wash off.")
            advisories.append("🚫 Suspend irrigation routines to save water and prevent waterlogging.")
        else:
            if temp > 36:
                advisories.append("🔥 High temperatures detected. Irrigate crops in the early morning or evening to avoid root evaporation shock.")
            elif temp < 15:
                advisories.append("🥶 Low temperatures. Frost risk present. Apply light irrigation at night to insulate crops.")
            
            if humidity > 80:
                advisories.append("🍄 High humidity creates favorable conditions for fungal growth. Monitor crop leaves for blight spots.")
            
            if wind_speed > 22:
                advisories.append("💨 Windy conditions. Postpone overhead spraying of pesticides to avoid spray drift.")
            
            if len(advisories) == 0:
                advisories.append("✅ Optimal farming conditions. Good day for crop dusting, weeding, and normal fertilizer application.")
                
        return advisories

class AIService:
    def __init__(self):
        self.classifiers = {}
        self.train_classifiers()

    def train_classifiers(self):
        for crop, diseases in DISEASE_PROFILES.items():
            X = []
            y = []
            for disease_id, data in diseases.items():
                if disease_id == 0:
                    for _ in range(15):
                        X.append([np.random.uniform(0.0, 0.9), np.random.randint(0, 2)])
                        y.append(disease_id)
                elif disease_id == 1:
                    for _ in range(15):
                        X.append([np.random.uniform(1.0, 12.0), np.random.randint(2, 10)])
                        y.append(disease_id)
                elif disease_id == 2:
                    for _ in range(15):
                        X.append([np.random.uniform(10.0, 75.0), np.random.randint(8, 45)])
                        y.append(disease_id)
                else:
                    for _ in range(15):
                        X.append([np.random.uniform(5.0, 30.0), np.random.randint(5, 25)])
                        y.append(disease_id)
            
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X, y)
            self.classifiers[crop] = clf

    def predict_disease(self, crop, infection_ratio, lesion_count):
        if crop not in DISEASE_PROFILES:
            return None
        
        if infection_ratio < 0.8 and lesion_count <= 1:
            disease_id = 0
            confidence = round(100.0 - (infection_ratio * 10), 2)
        else:
            clf = self.classifiers.get(crop)
            if clf:
                features = np.array([[infection_ratio, lesion_count]])
                disease_id = int(clf.predict(features)[0])
                probs = clf.predict_proba(features)[0]
                confidence = round(float(probs[disease_id] * 100.0), 2)
                confidence = max(55.0, min(98.5, confidence + np.random.uniform(-5.0, 5.0)))
                confidence = round(confidence, 1)
            else:
                disease_id = 0
                confidence = 80.0

        crop_diseases = DISEASE_PROFILES[crop]
        if disease_id not in crop_diseases:
            disease_id = 0
            
        disease_info = crop_diseases[disease_id]
        
        if disease_id == 0:
            severity = "Healthy"
        elif infection_ratio < 6.0:
            severity = "Low"
        elif infection_ratio < 18.0:
            severity = "Medium"
        else:
            severity = "High"

        return {
            "crop": crop,
            "disease_name": disease_info['name'],
            "confidence": confidence,
            "severity": severity,
            "affected_area_ratio": infection_ratio,
            "treatment_organic": disease_info['organic'],
            "treatment_chemical": disease_info['chemical'],
            "prevention": disease_info['prevention']
        }

    def recommend_fertilizer(self, crop, soil_type, ph, moisture, growth_stage, temperature):
        base_npk = {"N": 60, "P": 40, "K": 40}
        fertilizer = "NPK 19-19-19"
        organic = "Well-decomposed farmyard manure or vermicompost"
        qty = "50 kg/acre"
        timing = "Basal application at planting stage"
        method = "Broadcasting or side-dressing near roots"
        warnings = "Ensure soil is sufficiently moist during application to prevent nitrogen volatilization."
        health_score = 75
        overuse_detection = "No risk detected."

        ph = float(ph)
        moisture = float(moisture)
        temperature = float(temperature)

        if ph < 5.5:
            warnings += " Acidic soil detected. Apply agricultural lime (calcium carbonate) to raise pH before fertilizing."
            health_score -= 15
        elif ph > 7.8:
            warnings += " Alkaline soil detected. Add gypsum or elemental sulfur to lower pH and release locked phosphorus."
            health_score -= 10

        if moisture < 20:
            overuse_detection = "HIGH RISK: Soil moisture is low. Fertilizer salt accumulation can cause root burn."
            health_score -= 20
        elif moisture > 80:
            warnings += " High soil moisture. Nitrogen leaching may occur. Avoid heavy urea application."

        if crop == 'Tomato':
            base_npk = {"N": 80, "P": 60, "K": 80}
            fertilizer = "NPK 12-32-16 & Calcium Nitrate"
            organic = "Bone meal (phosphorus source) & Wood ash (potassium source)"
            if growth_stage == 'Flowering':
                timing = "Apply at flower initiation stage"
                qty = "30 kg/acre NPK + 15 kg Calcium Nitrate"
                warnings += " Boost Calcium levels to prevent Blossom End Rot."
        elif crop == 'Paddy':
            base_npk = {"N": 120, "P": 60, "K": 60}
            fertilizer = "Urea, Single Super Phosphate (SSP), Muriate of Potash (MOP)"
            organic = "Green manuring with Dhaincha or Sunnhemp prior to transplanting"
            qty = "Split application: 50% Urea at transplanting, 25% at tillering, 25% at panicle initiation"
            if growth_stage == 'Tillering':
                timing = "Top dress Urea at active tillering stage"
                qty = "30 kg/acre Urea"
        elif crop == 'Cotton':
            base_npk = {"N": 100, "P": 50, "K": 50}
            fertilizer = "DAP (Di-Ammonium Phosphate) & Muriate of Potash"
            organic = "Cottonseed meal or composted leaf manure"
            qty = "40 kg/acre DAP and 20 kg/acre MOP"
            
        npk_desc = f"Recommended N-P-K Ratio: {base_npk['N']}-{base_npk['P']}-{base_npk['K']} kg/hectare."
        health_score = max(30, min(95, health_score))

        return {
            "crop": crop,
            "recommended_fertilizer": fertilizer,
            "organic_alternative": organic,
            "quantity": qty,
            "usage_timing": timing,
            "application_method": method,
            "risk_warnings": warnings,
            "npk_ratio": npk_desc,
            "soil_health_score": health_score,
            "overuse_alert": overuse_detection
        }

    def plan_irrigation(self, crop, soil_moisture, temperature, rain_prob, water_avail):
        soil_moisture = float(soil_moisture)
        temperature = float(temperature)
        rain_prob = float(rain_prob)

        status = "Irrigation Required"
        water_qty = "Medium (15,000 liters / acre)"
        timing = "Early morning (5:00 AM - 8:00 AM)"
        method = "Drip Irrigation"
        guideline = "Run drip system for 45 minutes on alternative days."
        water_saving = "Add straw mulch around plant beds to reduce evaporation loss by 35%."

        if rain_prob > 75:
            status = "Suspend Irrigation"
            water_qty = "None"
            timing = "N/A"
            guideline = "Postpone watering. Natural precipitation will cover requirements."
            water_saving = "Ensure drainage channels are clear to prevent waterlogging."
        elif soil_moisture > 70:
            status = "Irrigation Postponed"
            water_qty = "None"
            timing = "Next check in 48 hours"
            guideline = "Soil moisture is optimal."
            water_saving = "No action needed."
        else:
            if crop == 'Paddy':
                method = "Flood Irrigation (Alternate Wetting and Drying)"
                water_qty = "High (30,000 liters / acre)"
                guideline = "Maintain 2-5 cm water level in the field. Let it dry naturally before re-flooding."
                water_saving = "Install plastic lining along field channels to prevent seepage loss."
            elif crop == 'Tomato' or crop == 'Chili':
                method = "Drip Irrigation with emitters near root zones"
                water_qty = "Low to Medium (12,000 liters / acre)"
                guideline = "Daily watering for 30 minutes at a rate of 2 liters/hour per emitter."
                water_saving = "Use root zone sensors to auto-adjust drip intervals."

        return {
            "crop": crop,
            "status": status,
            "water_quantity": water_qty,
            "irrigation_timing": timing,
            "best_method": method,
            "drip_guideline": guideline,
            "water_saving_suggestion": water_saving
        }

    def predict_yield(self, crop, soil_type, moisture, temp, disease_history, fertilizer_usage):
        base_yields = {
            'Tomato': 12.0, 'Paddy': 2.4, 'Potato': 10.5, 'Cotton': 0.8,
            'Wheat': 1.8, 'Maize': 2.2, 'Banana': 15.0, 'Chili': 1.2,
            'Mango': 4.0, 'Groundnut': 1.0
        }
        base = base_yields.get(crop, 2.0)
        moisture = float(moisture)
        temp = float(temp)

        soil_mult = 1.0
        if soil_type == 'Clayey':
            soil_mult = 0.9 if crop != 'Paddy' else 1.15
        elif soil_type == 'Sandy':
            soil_mult = 0.8
        elif soil_type == 'Loamy' or soil_type == 'Black Soil':
            soil_mult = 1.1

        moisture_mult = 1.0
        if moisture < 30:
            moisture_mult = 0.75
        elif moisture > 85:
            moisture_mult = 0.85

        disease_penalty = 1.0
        if disease_history == 'high':
            disease_penalty = 0.65
        elif disease_history == 'medium':
            disease_penalty = 0.85

        expected = base * soil_mult * moisture_mult * disease_penalty
        expected = round(expected, 2)
        improvement = round(expected * 1.25, 2)

        suggestions = [
            f"Introduce organic mulching to increase humus and boost yield output by up to 15%.",
            f"Optimize NPK ratios using precision fertilizer schedules to prevent crop burning.",
            f"Adopt companion planting: intercrop with marigold (for tomato/chili) or legumes (for maize) to manage nematode pests."
        ]

        risks = "Low risk of loss. Soil moisture and local weather metrics are within crop toleration limits."
        if moisture < 35:
            risks = "HIGH RISK: Water stress detected. Foliage loss or flower drop may occur."
        elif disease_history == 'high':
            risks = "CRITICAL RISK: History of leaf blight in field. High susceptibility to spores. Apply preventive fungicides."

        return {
            "crop": crop,
            "expected_yield": f"{expected} Tons/Acre",
            "potential_yield": f"{improvement} Tons/Acre",
            "yield_suggestions": suggestions,
            "risk_analysis": risks
        }

    def chat_response(self, message, language='en', api_key=None):
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    f"You are AgriAI, a helpful, professional agricultural AI assistant. "
                    f"Answer this farmer query. Keep the response to 3 sentences maximum, "
                    f"highly actionable, and tailored to rural farming. Query: {message}"
                )
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API Error, falling back to local NLP: {e}")

        english_msg = message
        if language != 'en':
            try:
                english_msg = GoogleTranslator(source=language, target='en').translate(message)
            except Exception as e:
                print(f"Translation Error (Input): {e}")

        english_msg = english_msg.lower()
        response_en = ""

        if "hello" in english_msg or "hi" in english_msg or "greetings" in english_msg:
            response_en = "Hello! I am AgriAI, your smart farming assistant. How can I help you with your crops, weather, or irrigation today?"
        elif "yellow" in english_msg or "spot" in english_msg or "leaf" in english_msg or "disease" in english_msg:
            response_en = "Leaf yellowing or dark spots often indicate a nitrogen deficiency or fungal infection. I suggest uploading a crop photo to my Disease Scanner module for a precise diagnostic scan."
        elif "water" in english_msg or "irrigation" in english_msg or "dry" in english_msg:
            response_en = "Watering depends on soil moisture. Use the Irrigation Planner module. For sandy soils, water in small amounts frequently; for clayey soils, water deeply and less often."
        elif "fertilizer" in english_msg or "npk" in english_msg or "manure" in english_msg:
            response_en = "Balanced NPK is vital for growth. I suggest checking the Fertilizer Recommendation tool with your soil pH and crop stage to receive a specific dosage schedule."
        elif "price" in english_msg or "market" in english_msg or "cost" in english_msg:
            response_en = "Crop market prices fluctuate daily. You can view regional market prices and trends directly under the Market Prices tab on your dashboard."
        elif "scheme" in english_msg or "subsidy" in english_msg or "government" in english_msg:
            response_en = "Governments offer various agriculture subsidies for solar pumps, organic composting, and micro-irrigation. Please check the Government Schemes section on your dashboard."
        else:
            response_en = "I understand you are asking about agricultural operations. Ensure your soil is tested, keep irrigation balanced, and crop-rotate to maximize yield. Tell me more details about your crop type."

        translated_res = response_en
        if language != 'en':
            try:
                translated_res = GoogleTranslator(source='en', target=language).translate(response_en)
            except Exception as e:
                print(f"Translation Error (Output): {e}")

        return translated_res

    def translate_to(self, text, language='en'):
        if not text or language == 'en':
            return text
        try:
            return GoogleTranslator(source='en', target=language).translate(text)
        except Exception as e:
            print(f"Translation Error from en to {language}: {e}")
            return text

    def translate_data(self, val, language='en', exclude_keys=None):
        if not val or language == 'en':
            return val
        if exclude_keys is None:
            exclude_keys = {'original_url', 'heatmap_url', 'annotated_url', 'original_img_path', 
                            'processed_img_path', 'url', 'image', 'icon', 'trend', 'source', 
                            'sensor_id', 'crop', 'crop_name', 'day', 'day_name', 'created_at', 
                            'id', 'user_id', 'token', 'role', 'full_name', 'username'}
        
        if isinstance(val, str):
            if val.strip() == "" or val.replace('.', '', 1).isdigit():
                return val
            return self.translate_to(val, language)
        elif isinstance(val, list):
            return [self.translate_data(item, language, exclude_keys) for item in val]
        elif isinstance(val, dict):
            new_dict = {}
            for k, v in val.items():
                if k in exclude_keys:
                    new_dict[k] = v
                else:
                    new_dict[k] = self.translate_data(v, language, exclude_keys)
            return new_dict
        return val

# Instantiate Global AI Service
ai_service = AIService()

# ==========================================
# FLASK BACKEND SETUP
# ==========================================
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
app = Flask(__name__, static_url_path='', static_folder=frontend_dir)

app.config['SECRET_KEY'] = SECRET_KEY
app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app, resources={r"/api/*": {"origins": "*"}})

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure database is set up on initialization
with app.app_context():
    init_db()

# ==========================================
# HELPERS & DECORATORS
# ==========================================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(" ")[1]
        
        if not token:
            return jsonify({'message': 'Access token is missing!'}), 401
        
        try:
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
            conn = get_db_connection()
            user = db_fetchone(conn, 'SELECT * FROM users WHERE id = %s', (data['user_id'],))
            conn.close()
            if not user:
                return jsonify({'message': 'User not found!'}), 401
            current_user = {
                'id': user['id'],
                'username': user['username'],
                'role': user['role'],
                'full_name': user['full_name']
            }
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(current_user, *args, **kwargs)
    return decorated

def get_optional_user_id(req):
    token = None
    if 'Authorization' in req.headers:
        auth_header = req.headers['Authorization']
        if auth_header.startswith('Bearer '):
            token = auth_header.split(" ")[1]
            
    if token:
        try:
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
            return data['user_id']
        except Exception:
            pass
    return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
# AUTH ENDPOINTS
# ==========================================
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Missing request data'}), 400
        
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'farmer')
    full_name = data.get('full_name')

    if not username or not password or not full_name:
        return jsonify({'message': 'Fields username, password, and full_name are required'}), 400

    if role not in ['farmer', 'expert', 'admin']:
        return jsonify({'message': 'Invalid user role selected'}), 400

    hashed_pw = generate_password_hash(password, method='scrypt')

    conn = get_db_connection()
    try:
        db_write(conn,
            'INSERT INTO users (username, password_hash, role, full_name) VALUES (%s, %s, %s, %s)',
            (username, hashed_pw, role, full_name)
        )
        conn.commit()
        return jsonify({'message': 'User registered successfully!'}), 201
    except Exception as e:
        err_msg = str(e).lower()
        if "unique" in err_msg or "already exists" in err_msg or "integrityerror" in err_msg:
            return jsonify({'message': 'Username already exists'}), 409
        return jsonify({'message': f'Registration failed: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Missing request data'}), 400
        
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required'}), 400

    conn = get_db_connection()
    user = db_fetchone(conn, 'SELECT * FROM users WHERE username = %s', (username,))
    conn.close()

    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'message': 'Invalid credentials'}), 401

    token = jwt.encode({
        'user_id': user['id'],
        'role': user['role'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, app.config['JWT_SECRET_KEY'], algorithm="HS256")

    return jsonify({
        'token': token,
        'user': {
            'id': user['id'],
            'username': user['username'],
            'role': user['role'],
            'full_name': user['full_name']
        }
    }), 200

@app.route('/api/auth/supabase-login', methods=['POST'])
def supabase_login():
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Missing request data'}), 400
        
    access_token = data.get('access_token')
    if not access_token:
        return jsonify({'message': 'Access token is required'}), 400
        
    supabase_url = os.environ.get('SUPABASE_URL', 'https://fegjbugggonjsscmdawe.supabase.co')
    supabase_key = os.environ.get('SUPABASE_KEY', 'sb_publishable_xjNCy3WZMbO0k32WLTCjng_70V5FOAT')
    
    headers = {
        'apikey': supabase_key,
        'Authorization': f'Bearer {access_token}'
    }
    
    try:
        response = requests.get(f"{supabase_url}/auth/v1/user", headers=headers)
        if response.status_code != 200:
            return jsonify({'message': 'Invalid Supabase access token'}), 401
            
        supabase_user = response.json()
        email = supabase_user.get('email')
        
        # Extract full name from user metadata if available
        user_metadata = supabase_user.get('user_metadata', {})
        full_name = user_metadata.get('full_name') or user_metadata.get('name') or email.split('@')[0]
        
        if not email:
            return jsonify({'message': 'Email not provided by Supabase'}), 400
            
        conn = get_db_connection()
        # Find or create user
        user = db_fetchone(conn, 'SELECT * FROM users WHERE username = %s', (email,))
        if not user:
            # Create a new user with 'farmer' role
            db_write(conn, 
                'INSERT INTO users (username, password_hash, role, full_name) VALUES (%s, %s, %s, %s)',
                (email, 'supabase_oauth_user', 'farmer', full_name)
            )
            conn.commit()
            user = db_fetchone(conn, 'SELECT * FROM users WHERE username = %s', (email,))
            
        conn.close()
        
        token = jwt.encode({
            'user_id': user['id'],
            'role': user['role'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['JWT_SECRET_KEY'], algorithm="HS256")
        
        return jsonify({
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'role': user['role'],
                'full_name': user['full_name']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Supabase login integration error: {str(e)}'}), 500

# ==========================================
# DISEASE SCAN ENDPOINTS
# ==========================================
@app.route('/api/disease/scan', methods=['POST'])
def scan_crop():
    if 'image' not in request.files:
        return jsonify({'message': 'No image file uploaded'}), 400
        
    file = request.files['image']
    crop = request.form.get('crop')

    if not crop:
        return jsonify({'message': 'Crop parameter is required'}), 400

    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        try:
            cv_results = CVService.process_leaf_image(temp_path, app.config['UPLOAD_FOLDER'])
            prediction = ai_service.predict_disease(
                crop=crop,
                infection_ratio=cv_results['infection_ratio'],
                lesion_count=cv_results['lesion_count']
            )

            if not prediction:
                return jsonify({'message': f'Crop disease model not trained or crop {crop} not supported'}), 400

            prediction['original_url'] = f"/uploads/{cv_results['original_name']}"
            prediction['annotated_url'] = f"/uploads/{cv_results['annotated_name']}"
            prediction['heatmap_url'] = f"/uploads/{cv_results['heatmap_name']}"
            prediction['lesions_found'] = cv_results['lesion_count']

            if os.path.exists(temp_path) and filename != cv_results['original_name']:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

            user_id = get_optional_user_id(request)
            if user_id:
                conn = get_db_connection()
                db_write(conn,
                    '''INSERT INTO disease_predictions 
                    (user_id, crop_name, disease_name, confidence, severity, affected_area_ratio, 
                     original_img_path, processed_img_path, treatment_organic, treatment_chemical)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                    (user_id, crop, prediction['disease_name'], prediction['confidence'], 
                     prediction['severity'], prediction['affected_area_ratio'], 
                     prediction['original_url'], prediction['heatmap_url'], 
                     prediction['treatment_organic'], prediction['treatment_chemical'])
                )
                
                if prediction['severity'] in ['Medium', 'High']:
                    title = f"⚠️ Disease Alert: {prediction['disease_name']}"
                    msg = f"Your {crop} crop scan detected {prediction['disease_name']} with {prediction['severity']} severity ({prediction['confidence']}% confidence). Treatment recommended."
                    db_write(conn,
                        'INSERT INTO notifications (user_id, title, message, type) VALUES (%s, %s, %s, %s)',
                        (user_id, title, msg, 'disease')
                    )
                
                conn.commit()
                conn.close()

            lang = request.headers.get('X-Language', 'en')
            translated_prediction = ai_service.translate_data(prediction, lang)
            return jsonify(translated_prediction), 200

        except Exception as e:
            return jsonify({'message': f'Image processing failed: {str(e)}'}), 500
    else:
        return jsonify({'message': 'Allowed image types are png, jpg, jpeg'}), 400

@app.route('/api/disease/history', methods=['GET'])
def get_history():
    user_id = get_optional_user_id(request)
    if not user_id:
        return jsonify({'message': 'Authentication token is required for history'}), 401
        
    conn = get_db_connection()
    rows = db_fetchall(conn, 'SELECT * FROM disease_predictions WHERE user_id = %s ORDER BY created_at DESC', (user_id,))
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            'id': r['id'],
            'crop_name': r['crop_name'],
            'disease_name': r['disease_name'],
            'confidence': r['confidence'],
            'severity': r['severity'],
            'affected_area_ratio': r['affected_area_ratio'],
            'original_url': r['original_img_path'],
            'heatmap_url': r['processed_img_path'],
            'treatment_organic': r['treatment_organic'],
            'treatment_chemical': r['treatment_chemical'],
            'created_at': r['created_at']
        })
    return jsonify(history), 200

# ==========================================
# FARMING ADVISORY ENDPOINTS
# ==========================================
@app.route('/api/farming/weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat', 16.30, type=float)
    lon = request.args.get('lon', 80.45, type=float)
    
    weather_info = WeatherService.get_weather_forecast(lat, lon)
    
    conn = get_db_connection()
    db_write(conn,
        'INSERT INTO weather_data (latitude, longitude, temperature, humidity, rain_probability, description) VALUES (%s, %s, %s, %s, %s, %s)',
        (lat, lon, weather_info['temperature'], weather_info['humidity'], weather_info['rain_probability'], weather_info['description'])
    )
    
    user_id = get_optional_user_id(request)
    if user_id:
        if weather_info['rain_probability'] > 75:
            db_write(conn,
                'INSERT INTO notifications (user_id, title, message, type) VALUES (%s, %s, %s, %s)',
                (user_id, "🌧️ Heavy Rain Incoming", f"High precipitation probability ({weather_info['rain_probability']}%). Avoid pesticide spraying and adjust irrigation.", 'weather')
            )
        elif weather_info['temperature'] > 38:
            db_write(conn,
                'INSERT INTO notifications (user_id, title, message, type) VALUES (%s, %s, %s, %s)',
                (user_id, "🔥 Heatwave Alert", f"Temperature exceeds {weather_info['temperature']}°C. Increase morning irrigation to protect crops.", 'weather')
            )
            
    conn.commit()
    conn.close()
    
    lang = request.headers.get('X-Language', 'en')
    translated_weather = ai_service.translate_data(weather_info, lang)
    return jsonify(translated_weather), 200

@app.route('/api/farming/fertilizer', methods=['POST'])
def recommend_fertilizer():
    data = request.get_json() or {}
    crop = data.get('crop')
    soil_type = data.get('soil_type', 'Loamy')
    ph = data.get('ph', 6.5)
    moisture = data.get('moisture', 45)
    growth_stage = data.get('growth_stage', 'Vegetative')
    temperature = data.get('temperature', 28)
    
    if not crop:
        return jsonify({'message': 'Crop field is required'}), 400
        
    rec = ai_service.recommend_fertilizer(crop, soil_type, ph, moisture, growth_stage, temperature)
    
    user_id = get_optional_user_id(request)
    if user_id:
        conn = get_db_connection()
        db_write(conn,
            '''INSERT INTO fertilizer_recommendations 
            (user_id, crop_name, recommended_fertilizer, organic_alternative, quantity, application_method)
            VALUES (%s, %s, %s, %s, %s, %s)''',
            (user_id, crop, rec['recommended_fertilizer'], rec['organic_alternative'], rec['quantity'], rec['application_method'])
        )
        
        db_write(conn,
            'INSERT INTO notifications (user_id, title, message, type) VALUES (%s, %s, %s, %s)',
            (user_id, f"🌱 Fertilizer Scheduled: {crop}", f"Recommended application of {rec['recommended_fertilizer']} ({rec['quantity']}) for {growth_stage} stage.", 'irrigation')
        )
        conn.commit()
        conn.close()
        
    lang = request.headers.get('X-Language', 'en')
    translated_rec = ai_service.translate_data(rec, lang)
    return jsonify(translated_rec), 200

def update_water_telemetry(conn, water_qty, status):
    import re
    liters = 0.0
    match = re.search(r'([\d,]+)\s+liters', water_qty)
    if match:
        liters = float(match.group(1).replace(',', ''))
        
    saved = 0.0
    if status == "Irrigation Required":
        saved = max(0.0, 30000.0 - liters)
    elif status in ["Suspend Irrigation", "Irrigation Postponed"]:
        saved = 15000.0
        
    db_write(conn, """
        UPDATE water_telemetry 
        SET water_used_liters = water_used_liters + %s,
            water_saved_liters = water_saved_liters + %s
    """, (liters, saved))
    
    row = db_fetchone(conn, "SELECT water_used_liters, water_saved_liters FROM water_telemetry LIMIT 1")
    if row:
        used = row['water_used_liters']
        saved_val = row['water_saved_liters']
        total = used + saved_val
        ratio_str = f"{round((saved_val / total * 100.0), 1)}%" if total > 0 else "0%"
        
        eff_val = min(98.5, 85.0 + (saved_val / used * 10.0 if used > 0 else 0.0))
        eff_str = f"{round(eff_val, 1)}%"
        
        db_write(conn, """
            UPDATE water_telemetry 
            SET saving_ratio = %s,
                drip_efficiency = %s
        """, (ratio_str, eff_str))

@app.route('/api/farming/irrigation', methods=['POST'])
def plan_irrigation():
    data = request.get_json() or {}
    crop = data.get('crop')
    soil_moisture = data.get('soil_moisture', 40)
    temperature = data.get('temperature', 30)
    rain_probability = data.get('rain_probability', 20)
    water_availability = data.get('water_availability', 'Adequate')
    
    if not crop:
        return jsonify({'message': 'Crop field is required'}), 400
        
    plan = ai_service.plan_irrigation(crop, soil_moisture, temperature, rain_probability, water_availability)
    
    user_id = get_optional_user_id(request)
    if user_id:
        conn = get_db_connection()
        db_write(conn,
            '''INSERT INTO irrigation_plans 
            (user_id, crop_name, recommended_time, water_quantity, drip_guideline)
            VALUES (%s, %s, %s, %s, %s)''',
            (user_id, crop, plan['irrigation_timing'], plan['water_quantity'], plan['drip_guideline'])
        )
        
        if plan['status'] == "Irrigation Required":
            db_write(conn,
                'INSERT INTO notifications (user_id, title, message, type) VALUES (%s, %s, %s, %s)',
                (user_id, f"💧 Irrigation Task: {crop}", f"Apply {plan['water_quantity']} using {plan['best_method']} at {plan['irrigation_timing']}.", 'irrigation')
            )
            
        try:
            update_water_telemetry(conn, plan['water_quantity'], plan['status'])
        except Exception as e:
            print(f"Error updating water telemetry: {e}")
            
        conn.commit()
        conn.close()
        
    lang = request.headers.get('X-Language', 'en')
    translated_plan = ai_service.translate_data(plan, lang)
    return jsonify(translated_plan), 200

@app.route('/api/farming/yield', methods=['POST'])
def predict_yield():
    data = request.get_json() or {}
    crop = data.get('crop')
    soil_type = data.get('soil_type', 'Loamy')
    moisture = data.get('moisture', 50)
    temp = data.get('temperature', 28)
    disease_history = data.get('disease_history', 'none')
    fertilizer_usage = data.get('fertilizer_usage', 'balanced')
    
    if not crop:
        return jsonify({'message': 'Crop field is required'}), 400
        
    prediction = ai_service.predict_yield(crop, soil_type, moisture, temp, disease_history, fertilizer_usage)
    lang = request.headers.get('X-Language', 'en')
    translated_prediction = ai_service.translate_data(prediction, lang)
    return jsonify(translated_prediction), 200

@app.route('/api/farming/market', methods=['GET'])
def get_market_data():
    conn = get_db_connection()
    prices = db_fetchall(conn, 'SELECT * FROM market_prices ORDER BY crop_name ASC')
    conn.close()
    
    market_list = []
    for p in prices:
        market_list.append({
            'crop_name': p['crop_name'],
            'price_per_kg': p['price_per_kg'],
            'location': p['location'],
            'trend': p['trend']
        })
        
    schemes = [
        {
            "name": "PM-KISAN Samman Nidhi",
            "benefit": "₹6,000 yearly income support direct to farmer bank accounts in three equal installments.",
            "subsidy": "100% Central Government Funded",
            "eligibility": "Small and marginal landholder farmer families."
        },
        {
            "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
            "benefit": "Comprehensive crop insurance cover against yield losses from natural calamities, pests & diseases.",
            "subsidy": "Farmers pay low premiums: 2% for Kharif, 1.5% for Rabi, 5% for commercial crops.",
            "eligibility": "All farmers growing notified crops in notified areas."
        },
        {
            "name": "PM Krishi Sinchayee Yojana (PMKSY) - Per Drop More Crop",
            "benefit": "Financial assistance for installing micro-irrigation systems (drip and sprinkler units).",
            "subsidy": "Up to 55% subsidy for small/marginal farmers, 45% for other farmers.",
            "eligibility": "Farmers owning agricultural land with a reliable water source."
        },
        {
            "name": "Paramparagat Krishi Vikas Yojana (PKVY)",
            "benefit": "Promotes organic farming practices, cluster-based cultivation, and organic certification support.",
            "subsidy": "₹50,000 per hectare financial aid over 3 years.",
            "eligibility": "Groups of 20 or more farmers forming organic farming clusters."
        }
    ]
    
    recommendations = [
        {"crop": "Chili", "reason": "High market price spikes at Guntur (₹180.00/kg) with increasing regional export trends.", "demand": "High"},
        {"crop": "Cotton", "reason": "Consistent upward price trends in Maharashtra market hubs. Prefers black soil.", "demand": "High"},
        {"crop": "Paddy", "reason": "Monsoon starting soon. Ideal season for crop transplantation in clay-heavy lowlands.", "demand": "Stable"}
    ]
    
    lang = request.headers.get('X-Language', 'en')
    translated_market_data = ai_service.translate_data({
        'prices': market_list,
        'schemes': schemes,
        'crop_recommendations': recommendations
    }, lang)
    return jsonify(translated_market_data), 200

@app.route('/api/farming/notifications', methods=['GET'])
def get_notifications():
    user_id = get_optional_user_id(request)
    if not user_id:
        return jsonify({'message': 'Authentication token required'}), 401
        
    conn = get_db_connection()
    rows = db_fetchall(conn, 'SELECT * FROM notifications WHERE user_id = %s ORDER BY created_at DESC LIMIT 20', (user_id,))
    conn.close()
    
    notifs = []
    for r in rows:
        notifs.append({
            'id': r['id'],
            'title': r['title'],
            'message': r['message'],
            'type': r['type'],
            'read_status': r['read_status'],
            'created_at': r['created_at']
        })
    return jsonify(notifs), 200

@app.route('/api/farming/notifications/read', methods=['POST'])
def mark_read():
    user_id = get_optional_user_id(request)
    if not user_id:
        return jsonify({'message': 'Authentication token required'}), 401
        
    data = request.get_json() or {}
    notif_id = data.get('id')
    
    conn = get_db_connection()
    if notif_id:
        db_write(conn, 'UPDATE notifications SET read_status = 1 WHERE id = %s AND user_id = %s', (notif_id, user_id))
    else:
        db_write(conn, 'UPDATE notifications SET read_status = 1 WHERE user_id = %s', (user_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Notifications marked as read.'}), 200

# ==========================================
# CHAT ENDPOINTS
# ==========================================
@app.route('/api/chat/message', methods=['POST'])
def send_chat_message():
    data = request.get_json() or {}
    message = data.get('message')
    language = data.get('language', 'en')
    api_key = request.headers.get('X-Gemini-Key') or data.get('api_key')

    if not message:
        return jsonify({'message': 'Message parameter is empty'}), 400

    user_id = get_optional_user_id(request)
    reply = ai_service.chat_response(message, language, api_key)
    
    if user_id:
        conn = get_db_connection()
        db_write(conn,
            'INSERT INTO chat_history (user_id, sender, message, language) VALUES (%s, %s, %s, %s)',
            (user_id, 'user', message, language)
        )
        db_write(conn,
            'INSERT INTO chat_history (user_id, sender, message, language) VALUES (%s, %s, %s, %s)',
            (user_id, 'assistant', reply, language)
        )
        conn.commit()
        conn.close()

    return jsonify({
        'reply': reply,
        'sender': 'assistant',
        'language': language
    }), 200

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    user_id = get_optional_user_id(request)
    if not user_id:
        return jsonify({'message': 'Authentication token required'}), 401
        
    conn = get_db_connection()
    rows = db_fetchall(conn, 'SELECT * FROM chat_history WHERE user_id = %s ORDER BY created_at ASC LIMIT 100', (user_id,))
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            'sender': r['sender'],
            'message': r['message'],
            'language': r['language'],
            'created_at': r['created_at']
        })
    return jsonify(history), 200

# ==========================================
# ANALYTICS ENDPOINTS
# ==========================================
@app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary():
    user_id = get_optional_user_id(request)
    conn = get_db_connection()
    
    total_users = db_fetchone(conn, "SELECT COUNT(*) FROM users")[0]
    total_farmers = db_fetchone(conn, "SELECT COUNT(*) FROM users WHERE role = 'farmer'")[0]
    total_experts = db_fetchone(conn, "SELECT COUNT(*) FROM users WHERE role = 'expert'")[0]
    total_scans = db_fetchone(conn, "SELECT COUNT(*) FROM disease_predictions")[0]
    total_chats = db_fetchone(conn, "SELECT COUNT(*) FROM chat_history WHERE sender = 'user'")[0]
    
    disease_rows = db_fetchall(conn, "SELECT disease_name, COUNT(*) as count FROM disease_predictions GROUP BY disease_name")
    disease_dist = {}
    for r in disease_rows:
        disease_dist[r['disease_name']] = r['count']
        
    crop_rows = db_fetchall(conn, "SELECT crop_name, COUNT(*) as count FROM disease_predictions GROUP BY crop_name")
    crop_trends = {}
    for r in crop_rows:
        crop_trends[r['crop_name']] = r['count']

    user_scans = []
    if user_id:
        user_scans_rows = db_fetchall(conn, "SELECT crop_name, disease_name, confidence, severity, created_at FROM disease_predictions WHERE user_id = %s ORDER BY created_at DESC LIMIT 5", (user_id,))
        for r in user_scans_rows:
            user_scans.append({
                'crop_name': r['crop_name'],
                'disease_name': r['disease_name'],
                'confidence': r['confidence'],
                'severity': r['severity'],
                'created_at': r['created_at']
            })
    # Fetch yield trends from the database
    yield_rows = db_fetchall(conn, "SELECT year, crop_name, yield_value FROM crop_yield_trends ORDER BY year ASC")
    labels = sorted(list(set(r['year'] for r in yield_rows)))
    
    # Map yield data to crop arrays
    tomato_yields = [0.0] * len(labels)
    paddy_yields = [0.0] * len(labels)
    for r in yield_rows:
        if r['year'] in labels:
            idx = labels.index(r['year'])
            if r['crop_name'] == 'Tomato':
                tomato_yields[idx] = r['yield_value']
            elif r['crop_name'] == 'Paddy':
                paddy_yields[idx] = r['yield_value']
                
    crop_yield_trends = {
        "labels": labels if labels else ["2022", "2023", "2024", "2025"],
        "tomato_yields": tomato_yields if tomato_yields else [0.0, 0.0, 0.0, 0.0],
        "paddy_yields": paddy_yields if paddy_yields else [0.0, 0.0, 0.0, 0.0]
    }
    
    # Fetch water savings from telemetry table
    water_row = db_fetchone(conn, "SELECT water_used_liters, water_saved_liters, drip_efficiency, saving_ratio FROM water_telemetry LIMIT 1")
    if water_row:
        water_savings = {
            "water_used_liters": water_row['water_used_liters'],
            "water_saved_liters": water_row['water_saved_liters'],
            "drip_efficiency": water_row['drip_efficiency'],
            "saving_ratio": water_row['saving_ratio']
        }
    else:
        water_savings = {
            "water_used_liters": 0.0,
            "water_saved_liters": 0.0,
            "drip_efficiency": "0%",
            "saving_ratio": "0%"
        }
        
    conn.close()

    iot_status = {
        "drone_connection": "Online",
        "drone_battery": "85%",
        "active_drones": 2,
        "soil_moisture_sensors": [
            {"sensor_id": "SMS-01", "location": "North Field", "moisture": "42%", "status": "Active"},
            {"sensor_id": "SMS-02", "location": "South Field", "moisture": "38%", "status": "Active"},
            {"sensor_id": "SMS-03", "location": "West Field Orchard", "moisture": "55%", "status": "Active"}
        ],
        "satellite_sync": "Synchronized (2 hours ago)",
        "hardware_irrigation_valves": "Ready (Automatic Control)"
    }

    return jsonify({
        'metrics': {
            'total_users': total_users,
            'total_farmers': total_farmers,
            'total_experts': total_experts,
            'total_scans': total_scans,
            'total_chats': total_chats,
        },
        'disease_distribution': disease_dist,
        'crop_trends': crop_trends,
        'user_recent_scans': user_scans,
        'iot_status': iot_status,
        'crop_yield_trends': crop_yield_trends,
        'water_savings': water_savings,
        'api_status': {
            'open_meteo': 'Online (Latency 110ms)',
            'speech_engine': 'Native Browser Connected',
            'translation_engine': 'Online',
            'server_cpu': '4.8%',
            'server_memory': '124MB'
        }
    }), 200

# ==========================================
# STATIC ASSET & SPA SERVERS
# ==========================================
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # Fallback to serve static files from frontend folder if they exist
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

# ==========================================
# APPLICATION RUNNER
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
