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
        credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, key_data=gee_key_json)
        ee.Initialize(credentials, project='ee-muzzamilgandapur007')
        print("✅ Cloud Auth Success")
    else:
        ee.Initialize(project='ee-muzzamilgandapur007')

initialize_gee()

# ==========================================
# 2. DATE MANAGEMENT (With 5-Day Buffer)
# ==========================================
MASTER_PATH = 'Master_Data.csv'

def robust_date_parser(date_str):
    try: return pd.to_datetime(str(date_str).strip())
    except: return pd.NaT

if os.path.exists(MASTER_PATH):
    df_dates = pd.read_csv(MASTER_PATH, usecols=['date'])
    last_date = df_dates['date'].apply(robust_date_parser).max()
    start_date_dt = (last_date + timedelta(days=1)) if not pd.isnull(last_date) else datetime(2018, 1, 1)
else:
    start_date_dt = datetime(2018, 1, 1)

# CRITICAL FIX: Only go up to 5 days ago to avoid "No Band" errors
end_date_dt = datetime.now() - timedelta(days=5)

if start_date_dt >= end_date_dt:
    print("✅ System is already up to date (within 5-day satellite lag).")
    sys.exit(0)

print(f"📡 Syncing: {start_date_dt.date()} to {end_date_dt.date()}")

# ==========================================
# 3. EXTRACTION LOGIC (RESILIENT)
# ==========================================
def get_data_safe(start, end):
    try:
        region = ee.Geometry.Rectangle([33.5, -1.5, 36.5, 1.5])
        points = ee.FeatureCollection.randomPoints(region, 200, 42)

        s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(region)\
            .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)) # 70% is the safe limit for data density

        if s2_col.size().getInfo() == 0: return pd.DataFrame()

        def process_img(img):
            date = img.date()
            ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
            savi = img.expression('((N-R)/(N+R+0.5))*1.5', {'N':img.select('B8'),'R':img.select('B4')}).rename('savi')
            evi = img.expression('2.5*((N-R)/(N+6*R-7.5*B+1))', {'N':img.select('B8'),'R':img.select('B4'),'B':img.select('B2')}).rename('evi')
            
            rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date, date.advance(1,'day')).mean().rename('rainfall')
            
            weather = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date, date.advance(1,'day'))
            temp = ee.Image(ee.Algorithms.If(weather.size().gt(0), 
                                             weather.mean().select(['temperature_2m']).subtract(273.15), 
                                             ee.Image(0))).rename('temp')
            
            # Simple fallback for soil/wind to prevent timeouts
            soil = ee.Image(0).rename('soil_moisture')
            
            combined = img.addBands([ndvi, savi, evi, rain, temp, soil])
            return points.map(lambda pt: ee.Feature(pt.geometry(), combined.reduceRegion(ee.Reducer.mean(), pt.geometry(), 30)).set({
                'date': date.format('YYYY-MM-DD'), 'point_id': pt.id(), 'year': date.get('year')
            }))

        results = s2_col.map(process_img).flatten().filter(ee.Filter.notNull(['ndvi']))
        features = results.getInfo()['features']
        return pd.DataFrame([dict(f['properties'], **{'.geo': json.dumps(f['geometry'])}) for f in features])
    except Exception as e:
        print(f"⚠️ Week skipped due to GEE timeout or missing bands: {e}")
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
        # Save IMMEDIATELY so we don't lose data if the script fails later
        df_master = pd.read_csv(MASTER_PATH) if os.path.exists(MASTER_PATH) else pd.DataFrame()
        new_data.columns = [c.lower() for c in new_data.columns]
        final_df = pd.concat([df_master, new_data], ignore_index=True).drop_duplicates(subset=['point_id', 'date'])
        final_df.to_csv(MASTER_PATH, index=False)
        print(f"✅ Chunk Saved. Total size: {len(final_df)} rows.")
    
    current_start = current_end

print("🚀 SYNC TASK FINISHED SUCCESSFULLY.")
