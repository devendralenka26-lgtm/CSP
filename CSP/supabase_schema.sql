-- AgriAI Database Schema for Supabase (PostgreSQL)
-- Run this script inside the SQL Editor of your Supabase project.

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('farmer', 'expert', 'admin')),
    full_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Crops Table
CREATE TABLE IF NOT EXISTS crops (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    crop_name TEXT NOT NULL,
    soil_type TEXT,
    moisture_level REAL,
    growth_stage TEXT,
    area_acres REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Disease Predictions Table
CREATE TABLE IF NOT EXISTS disease_predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    crop_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    severity TEXT NOT NULL,
    affected_area_ratio REAL,
    original_img_path TEXT NOT NULL,
    processed_img_path TEXT NOT NULL,
    treatment_organic TEXT,
    treatment_chemical TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Weather Advisory Cache Table
CREATE TABLE IF NOT EXISTS weather_data (
    id SERIAL PRIMARY KEY,
    latitude REAL,
    longitude REAL,
    temperature REAL,
    humidity REAL,
    rain_probability REAL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Fertilizer Recommendations Table
CREATE TABLE IF NOT EXISTS fertilizer_recommendations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    crop_name TEXT NOT NULL,
    recommended_fertilizer TEXT NOT NULL,
    organic_alternative TEXT,
    quantity TEXT,
    application_method TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Chat History Table
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    sender TEXT NOT NULL CHECK (sender IN ('user', 'assistant')),
    message TEXT NOT NULL,
    language TEXT DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Notifications Table
CREATE TABLE IF NOT EXISTS notifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('weather', 'disease', 'irrigation', 'system')),
    read_status INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. Irrigation Plans Table
CREATE TABLE IF NOT EXISTS irrigation_plans (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    crop_name TEXT NOT NULL,
    recommended_time TEXT NOT NULL,
    water_quantity TEXT,
    drip_guideline TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 9. Market Prices Table
CREATE TABLE IF NOT EXISTS market_prices (
    id SERIAL PRIMARY KEY,
    crop_name VARCHAR(255) NOT NULL UNIQUE,
    price_per_kg REAL NOT NULL,
    location TEXT NOT NULL,
    trend TEXT NOT NULL CHECK (trend IN ('up', 'down', 'stable')),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed initial market prices if they do not exist
INSERT INTO market_prices (crop_name, price_per_kg, location, trend) 
VALUES
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
ON CONFLICT (crop_name) DO NOTHING;
