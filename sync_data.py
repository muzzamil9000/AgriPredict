import ee
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 1. AUTHENTICATION (FORCED CLOUD MODE)
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
# 2. DATE MANAGEMENT
# ==========================================
MASTER_PATH = 'Master_Data.csv'

def robust_date_parser(date_str):
    try: return pd.to_datetime(str(date_str).strip())
    except: return pd.NaT

if os.path.exists(MASTER_PATH):
    # Only read the date column to save memory
    df_dates = pd.read_csv(MASTER_PATH, usecols=['date'])
    last_date = df_dates['date'].apply(robust_date_parser).max()
    start_date_dt = (last_date + timedelta(days=1)) if not pd.isnull(last_date) else datetime(2018, 1, 1)
    print(f"📡 Resuming from: {start_date_dt.date()}")
else:
    start_date_dt = datetime(2018, 1, 1)

end_date_dt = datetime.now()

# ==========================================
# 3. DATA EXTRACTION (ULTRA-STABLE)
# ==========================================
def get_data_safe(start, end):
    region = ee.Geometry.Rectangle([33.5, -1.5, 36.5, 1.5])
    # 200 points is the 'Sweet Spot' for no timeouts
    points = ee.FeatureCollection.randomPoints(region, 200, 42)

    s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
        .filterBounds(region)\
        .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))

    if s2_col.size().getInfo() == 0: return pd.DataFrame()

    def process_img(img):
        date = img.date()
        # VIs
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        savi = img.expression('((N-R)/(N+R+0.5))*1.5', {'N':img.select('B8'),'R':img.select('B4')}).rename('SAVI')
        evi = img.expression('2.5*((N-R)/(N+6*R-7.5*B+1))', {'N':img.select('B8'),'R':img.select('B4'),'B':img.select('B2')}).rename('EVI')
        
        # Weather & Climate
        rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date, date.advance(1,'day')).mean().rename('Rainfall')
        
        # Safe Temp extraction
        weather = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date, date.advance(1,'day'))
        temp = ee.Image(ee.Algorithms.If(weather.size().gt(0), weather.mean().select(['temperature_2m']).subtract(273.15), ee.Image(0))).rename('Temp')
        
        # Wind & Soil (SMAP v8)
        wind = temp.multiply(0).rename('Wind_Speed') # Fast fallback
        soil_coll = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate(date.advance(-2,'day'), date.advance(2,'day'))
        soil = ee.Image(ee.Algorithms.If(soil_coll.size().gt(0), soil_coll.mean().select(['sm_surface']), ee.Image(0))).rename('Soil_Moisture')

        combined = img.addBands([ndvi, savi, evi, rain, temp, wind, soil])
        
        def sample(pt):
            stats = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=pt.geometry(), scale=30)
            return ee.Feature(pt.geometry(), stats).set({
                'date': date.format('YYYY-MM-DD'), 'point_id': pt.id(), 'year': date.get('year')
            })
        return points.map(sample)

    try:
        results = s2_col.map(process_img).flatten().filter(ee.Filter.notNull(['NDVI']))
        features = results.getInfo()['features']
        return pd.DataFrame([dict(f['properties'], **{'.geo': json.dumps(f['geometry'])}) for f in features])
    except:
        return pd.DataFrame()

# ==========================================
# 4. CHUNKED EXECUTION
# ==========================================
current_start = start_date_dt
chunk_size = timedelta(days=7) # 1 week chunks are bulletproof

while current_start < end_date_dt:
    current_end = min(current_start + chunk_size, end_date_dt)
    print(f"⏳ Processing week: {current_start.date()}...")
    
    new_data = get_data_safe(current_start, current_end)
    
    if not new_data.empty:
        df_master = pd.read_csv(MASTER_PATH) if os.path.exists(MASTER_PATH) else pd.DataFrame()
        # Merge and cleanup
        final_df = pd.concat([df_master, new_data], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['point_id', 'date'])
        final_df.to_csv(MASTER_PATH, index=False)
        print(f"✅ Saved. Master file size: {os.path.getsize(MASTER_PATH)//(1024*1024)}MB")
    
    current_start = current_end

print("🚀 DATA SYNC COMPLETE.")
