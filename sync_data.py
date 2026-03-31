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
    else:
        ee.Initialize(project='ee-muzzamilgandapur007')

initialize_gee()

# ==========================================
# 2. DATA GAP CHECKING
# ==========================================
MASTER_PATH = 'Master_Data.csv'

def robust_date_parser(date_str):
    date_str = str(date_str).strip()
    try: return pd.to_datetime(date_str)
    except: return pd.NaT

if os.path.exists(MASTER_PATH):
    df_dates = pd.read_csv(MASTER_PATH, usecols=['date'])
    last_date = df_dates['date'].apply(robust_date_parser).max()
    start_date_dt = (last_date + timedelta(days=1)) if not pd.isnull(last_date) else datetime(2018, 1, 1)
    del df_dates
else:
    start_date_dt = datetime(2018, 1, 1)

end_date_dt = datetime.now()

if start_date_dt >= end_date_dt:
    print("✅ System is already up to date.")
    sys.exit(0)

# ==========================================
# 3. CHUNKED EXTRACTION LOGIC
# ==========================================
def get_data_for_range(start, end):
    region = ee.Geometry.Rectangle([33.5, -1.5, 36.5, 1.5])
    landcover = ee.Image("ESA/WorldCover/v100/2020")
    points = landcover.eq(40).selfMask().stratifiedSample(numPoints=1000, region=region, scale=30, geometries=True, seed=42).limit(1000)

    s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
        .filterBounds(region)\
        .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))

    if s2_col.size().getInfo() == 0: return pd.DataFrame()

    def process_image(img):
        date = img.date()
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        savi = img.expression('((N-R)/(N+R+0.5))*1.5', {'N':img.select('B8'),'R':img.select('B4')}).rename('savi')
        evi = img.expression('2.5*((N-R)/(N+6*R-7.5*B+1))', {'N':img.select('B8'),'R':img.select('B4'),'B':img.select('B2')}).rename('evi')
        rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date, date.advance(1,'day')).mean().rename('rainfall')
        weather = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date, date.advance(1,'day')).mean()
        temp = weather.select(['temperature_2m']).rename(['temp']).subtract(273.15)
        wind = weather.select('u_component_of_wind_10m').hypot(weather.select('v_component_of_wind_10m')).rename('wind_speed')
        smap_col = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate(date.advance(-2,'day'), date.advance(2,'day'))
        soil = ee.Image(ee.Algorithms.If(smap_col.size().gt(0), smap_col.mean().select(['sm_surface']), ee.Image.constant(0).rename('sm_surface'))).rename('soil_moisture')
        
        combined = img.addBands([ndvi, savi, evi, rain, temp, wind, soil])
        return points.map(lambda pt: ee.Feature(pt.geometry(), combined.reduceRegion(ee.Reducer.mean(), pt.geometry(), 30)).set({
            'date': date.format('YYYY-MM-DD'), 'point_id': pt.id(), 'year': date.get('year')
        }))

    results = s2_col.map(process_image).flatten().filter(ee.Filter.notNull(['ndvi', 'rainfall', 'temp']))
    
    # Use getInfo with caution - this is where small chunks help
    try:
        features = results.getInfo()['features']
        return pd.DataFrame([dict(f['properties'], **{'.geo': json.dumps(f['geometry'])}) for f in features])
    except:
        return pd.DataFrame()

# ==========================================
# 4. LOOPED EXECUTION (The 30-Day Chunking)
# ==========================================
current_start = start_date_dt
chunk_size = timedelta(days=30)

print(f"📡 Starting catch-up sync from {start_date_dt.date()}...")

while current_start < end_date_dt:
    current_end = min(current_start + chunk_size, end_date_dt)
    print(f"⏳ Processing chunk: {current_start.date()} to {current_end.date()}...")
    
    try:
        new_rows = get_data_for_range(current_start, current_end)
        
        if not new_rows.empty:
            new_rows.columns = [c.lower() for c in new_rows.columns]
            # Append immediately to file to save progress
            df_master = pd.read_csv(MASTER_PATH) if os.path.exists(MASTER_PATH) else pd.DataFrame()
            final_df = pd.concat([df_master, new_rows], ignore_index=True).drop_duplicates(subset=['point_id', 'date'])
            final_df.to_csv(MASTER_PATH, index=False)
            print(f"✅ Chunk saved. Total records now: {len(final_df)}")
            del df_master, final_df
        else:
            print("ℹ️ No clear images in this chunk.")
            
    except Exception as e:
        print(f"⚠️ Chunk failed, skipping or retrying next time: {e}")

    current_start = current_end

print("🚀 CATCH-UP SYNC COMPLETE.")
