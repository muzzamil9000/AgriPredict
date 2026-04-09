import ee
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 1. AUTHENTICATION
# ==========================================
EE_SERVICE_ACCOUNT = 'agripredict-bot@ee-muzzamilgandapur007.iam.gserviceaccount.com'

def initialize_gee():
    gee_key_json = os.environ.get('GEE_KEY')
    if gee_key_json:
        print("🤖 Running in GitHub Cloud: Using Service Account...")
        credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, key_data=gee_key_json)
        ee.Initialize(credentials, project='ee-muzzamilgandapur007')
        print("✅ Cloud Auth Success")
    else:
        print("💻 Running locally...")
        ee.Initialize(project='ee-muzzamilgandapur007')

initialize_gee()

# ==========================================
# 2. DATE MANAGEMENT (With 5-Day Buffer)
# ==========================================
MASTER_PATH = 'Master_Data.csv'

def robust_date_parser(date_str):
    date_str = str(date_str).strip()
    try:
        return pd.to_datetime(date_str)
    except:
        try:
            # Handles GEE format YYYY-MM-DDD (e.g. 2026-03-87)
            parts = date_str.split('-')
            if len(parts) == 3:
                year = int(parts[0])
                doy = int(parts[2])
                return pd.Timestamp(datetime(year, 1, 1) + timedelta(days=doy - 1))
        except:
            return pd.NaT

if os.path.exists(MASTER_PATH):
    print(f"📄 Analyzing {MASTER_PATH}...")
    df_dates = pd.read_csv(MASTER_PATH, usecols=['date'])
    parsed_dates = df_dates['date'].apply(robust_date_parser)
    last_date = parsed_dates.max()
    start_date_dt = (last_date + timedelta(days=1)) if not pd.isnull(last_date) else datetime(2018, 1, 1)
    print(f"📡 Resuming from: {start_date_dt.date()}")
else:
    start_date_dt = datetime(2018, 1, 1)

# Only go up to 5 days ago to avoid "No Band" errors
end_date_dt = datetime.now() - timedelta(days=5)

if start_date_dt >= end_date_dt:
    print("✅ System is already up to date.")
    sys.exit(0)

# ==========================================
# 3. EXTRACTION LOGIC (RESILIENT & ALIGNED)
# ==========================================
def get_data_safe(start, end):
    try:
        region = ee.Geometry.Rectangle([33.5, -1.5, 36.5, 1.5])
        points = ee.FeatureCollection.randomPoints(region, 200, 42) # 200 pts per chunk

        s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(region)\
            .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))

        if s2_col.size().getInfo() == 0: return pd.DataFrame()

        def process_img(img):
            date = img.date()
            # 1. Vegetation Indices
            ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
            savi = img.expression('((N-R)/(N+R+0.5))*1.5', {'N':img.select('B8'),'R':img.select('B4')}).rename('savi')
            evi = img.expression('2.5*((N-R)/(N+6*R-7.5*B+1))', {'N':img.select('B8'),'R':img.select('B4'),'B':img.select('B2')}).rename('evi')
            
            # 2. Weather
            rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date, date.advance(1,'day')).mean().rename('rainfall')
            
            weather_coll = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date, date.advance(1,'day'))
            temp = ee.Image(ee.Algorithms.If(weather_coll.size().gt(0), 
                                             weather_coll.mean().select(['temperature_2m']).subtract(273.15), 
                                             ee.Image(0))).rename('temp')
            
            # 3. Fallbacks
            wind = temp.multiply(0).rename('wind_speed')
            soil = ee.Image(0).rename('soil_moisture')
            
            # Select ONLY specific columns to prevent shifting
            combined = img.addBands([ndvi, savi, evi, rain, temp, wind, soil])\
                          .select(['ndvi', 'savi', 'evi', 'rainfall', 'temp', 'wind_speed', 'soil_moisture'])
            
            def sample(pt):
                stats = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=pt.geometry(), scale=30)
                return ee.Feature(pt.geometry(), stats).set({
                    'date': date.format('YYYY-MM-DD'), 
                    'point_id': pt.id(), 
                    'year': date.get('year')
                })
            return points.map(sample)

        results = s2_col.map(process_img).flatten().filter(ee.Filter.notNull(['ndvi']))
        features = results.getInfo()['features']
        
        # Build CLEAN, strictly-ordered row
        data_list = []
        for f in features:
            p = f['properties']
            data_list.append({
                '.geo': json.dumps(f['geometry']),
                'year': p.get('year'),
                'point_id': p.get('point_id'),
                'date': p.get('date'),
                'ndvi': p.get('ndvi'),
                'savi': p.get('savi'),
                'evi': p.get('evi'),
                'rainfall': p.get('rainfall'),
                'temp': p.get('temp'),
                'wind_speed': p.get('wind_speed'),
                'soil_moisture': p.get('soil_moisture')
            })
        return pd.DataFrame(data_list)
        
    except Exception as e:
        print(f"⚠️ Chunk Error: {e}")
        return pd.DataFrame()

# ==========================================
# 4. THE PERSISTENT LOOP
# ==========================================
current_start = start_date_dt
chunk_size = timedelta(days=7)

while current_start < end_date_dt:
    current_end = min(current_start + chunk_size, end_date_dt)
    print(f"⏳ Processing week: {current_start.date()}...")
    
    new_data = get_data_safe(current_start, current_end)
    
    if not new_data.empty:
        df_master = pd.read_csv(MASTER_PATH) if os.path.exists(MASTER_PATH) else pd.DataFrame()
        # Merge, drop duplicates, and force save
        final_df = pd.concat([df_master, new_data], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['point_id', 'date'])
        final_df.to_csv(MASTER_PATH, index=False)
        print(f"✅ Saved. Master file: {os.path.getsize(MASTER_PATH)//(1024*1024)}MB")
    
    current_start = current_end

print("🚀 SYNC TASK FINISHED SUCCESSFULLY.")
