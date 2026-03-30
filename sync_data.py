import ee
import os
import json
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 1. AUTHENTICATION LOGIC (WORKING!)
# ==========================================
EE_SERVICE_ACCOUNT = 'agripredict-bot@ee-muzzamilgandapur007.iam.gserviceaccount.com'

def initialize_gee():
    gee_key_json = os.environ.get('GEE_KEY')
    if gee_key_json:
        print("🤖 Running in GitHub Cloud: Using Service Account...")
        credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, key_data=gee_key_json)
        ee.Initialize(credentials, project='ee-muzzamilgandapur007')
        print("✅ GEE Login Successful!")
    else:
        print("💻 Running locally...")
        ee.Initialize(project='ee-muzzamilgandapur007')

initialize_gee()

# ==========================================
# 2. DATA GAP CHECKING (FIXED FOR KEYERROR)
# ==========================================
master_path = 'Master_Data.csv'

if os.path.exists(master_path):
    print(f"📄 Loading {master_path}...")
    df_master = pd.read_csv(master_path)
    
    # Standardize columns to lowercase to avoid 'Date' vs 'date' issues
    df_master.columns = [c.lower() for c in df_master.columns]
    
    if not df_master.empty and 'date' in df_master.columns:
        df_master['date'] = pd.to_datetime(df_master['date'])
        start_date = (df_master['date'].max() + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"✅ Last record found: {df_master['date'].max().date()}")
    else:
        print("⚠️ File is empty or 'date' column is missing. Starting from 2018.")
        start_date = '2018-01-01'
else:
    print("📁 Master_Data.csv not found. Creating a new one from 2018.")
    df_master = pd.DataFrame()
    start_date = '2018-01-01'

end_date = datetime.now().strftime('%Y-%m-%d')

if start_date >= end_date:
    print("✅ No new data needed. System is up to date.")
    exit()

print(f"📡 Syncing data from {start_date} to {end_date}...")

# ==========================================
# 3. GEE DATA EXTRACTION (THE "PULL")
# ==========================================
def get_new_data(start, end):
    region = ee.Geometry.Rectangle([33.5, -1.5, 36.5, 1.5])
    landcover = ee.Image("ESA/WorldCover/v100/2020")
    cropland = landcover.eq(40)
    
    # Fixed seed ensures we track the SAME farms every time
    points = cropland.selfMask().stratifiedSample(
        numPoints=1200, region=region, scale=30, geometries=True, seed=42
    )

    s2Col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
        .filterBounds(region)\
        .filterDate(start, end)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))

    if s2Col.size().getInfo() == 0:
        return pd.DataFrame()

    def extract_stats(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        date = img.date()
        rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date, date.advance(1,'day')).mean().rename('rainfall')
        weather = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date, date.advance(1,'day')).mean()
        temp = weather.select('temperature_2m').subtract(273.15).rename('temp')
        
        combined = img.addBands([ndvi, rain, temp])
        
        def reduce_point(pt):
            stats = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=pt.geometry(), scale=30)
            return ee.Feature(pt.geometry(), stats).set({
                'date': date.format('YYYY-MM-DD'),
                'point_id': pt.id()
            })
        return points.map(reduce_point)

    results = s2Col.map(extract_stats).flatten()
    clean_results = results.filter(ee.Filter.notNull(['ndvi', 'rainfall', 'temp']))
    
    # Check if there are any results to avoid errors
    if clean_results.size().getInfo() == 0:
        return pd.DataFrame()

    features = clean_results.getInfo()['features']
    data_list = []
    for f in features:
        props = f['properties']
        props['.geo'] = json.dumps(f['geometry'])
        data_list.append(props)
    
    return pd.DataFrame(data_list)

# ==========================================
# 4. EXECUTION
# ==========================================
try:
    new_df = get_new_data(start_date, end_date)
    
    if not new_df.empty:
        # Standardize new data columns to match master
        new_df.columns = [c.lower() for c in new_df.columns]
        
        updated_master = pd.concat([df_master, new_df], ignore_index=True)
        updated_master['year'] = pd.to_datetime(updated_master['date']).dt.year
        
        # Remove any duplicates that might have sneaked in
        updated_master = updated_master.drop_duplicates(subset=['point_id', 'date'])
        
        updated_master.to_csv(master_path, index=False)
        print(f"🚀 SUCCESS: {len(new_df)} new records added to Master_Data.csv")
    else:
        print("ℹ️ No new clear images found for this period.")

except Exception as e:
    print(f"❌ ERROR: {e}")
