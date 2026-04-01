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
    # If the CSV has a gap, let's start from the LAST date we actually have
    start_date_dt = (last_date + timedelta(days=1)) if not pd.isnull(last_date) else datetime(2018, 1, 1)
    del df_dates
else:
    start_date_dt = datetime(2018, 1, 1)

end_date_dt = datetime.now()

print(f"🔍 System Date Check: Last date in CSV is {start_date_dt.date() - timedelta(days=1)}")

# ==========================================
# 3. ROBUST EXTRACTION LOGIC
# ==========================================
def get_data_for_range(start, end):
    # Core Region - Double check these coordinates match your Maize Belt
    region = ee.Geometry.Rectangle([33.5, -1.5, 36.5, 1.5])
    
    # Simple Point Generation (More reliable than landcover masks for catch-up)
    points = ee.FeatureCollection.randomPoints(region, 1000, 42)

    s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
        .filterBounds(region)\
        .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90)) # Relaxed to 90% to find ANY data

    count = s2_col.size().getInfo()
    if count == 0:
        return pd.DataFrame()
    
    print(f"📸 Found {count} images in this chunk. Extracting...")

    def process_image(img):
        date = img.date()
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        # Use simple mean weather - no complex filters here to prevent crashes
        rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date, date.advance(1,'day')).mean().rename('rainfall')
        weather = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date, date.advance(1,'day')).mean()
        temp = weather.select(['temperature_2m']).rename(['temp']).subtract(273.15)
        
        combined = img.addBands([ndvi, rain, temp])
        
        def sample_point(pt):
            stats = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=pt.geometry(), scale=30)
            return ee.Feature(pt.geometry(), stats).set({
                'date': date.format('YYYY-MM-DD'), 'point_id': pt.id(), 'year': date.get('year')
            })
        return points.map(sample_point)

    results = s2_col.map(process_image).flatten().filter(ee.Filter.notNull(['ndvi', 'rainfall', 'temp']))
    
    try:
        features = results.limit(5000).getInfo()['features'] # Limit to 5000 per chunk to avoid timeout
        if not features: return pd.DataFrame()
        return pd.DataFrame([dict(f['properties'], **{'.geo': json.dumps(f['geometry'])}) for f in features])
    except Exception as e:
        print(f"⚠️ Chunk internal error: {e}")
        return pd.DataFrame()

# ==========================================
# 4. LOOPED EXECUTION (15-Day Chunks for speed)
# ==========================================
current_start = start_date_dt
chunk_size = timedelta(days=15) # Smaller chunks are safer

print(f"📡 Starting catch-up sync from {start_date_dt.date()}...")

while current_start < end_date_dt:
    current_end = min(current_start + chunk_size, end_date_dt)
    print(f"⏳ Syncing: {current_start.date()} to {current_end.date()}...")
    
    new_rows = get_data_for_range(current_start, current_end)
    
    if not new_rows.empty:
        new_rows.columns = [c.lower() for c in new_rows.columns]
        df_master = pd.read_csv(MASTER_PATH) if os.path.exists(MASTER_PATH) else pd.DataFrame()
        # Merge and cleanup
        final_df = pd.concat([df_master, new_rows], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['point_id', 'date'])
        final_df.to_csv(MASTER_PATH, index=False)
        print(f"✅ SUCCESS: Saved {len(new_rows)} new records.")
        del df_master, final_df
    else:
        print("ℹ️ Still no images. Checking next date...")

    current_start = current_end

print("🚀 CATCH-UP SYNC COMPLETE.")
