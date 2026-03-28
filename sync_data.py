
import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# ==========================================
# 1. GEE AUTHENTICATION (The "Bot" Login)
# ==========================================
# Change this to your actual Service Account Email!
SERVICE_ACCOUNT = 'agripredict-bot@ee-muzzamilgandapur007.iam.gserviceaccount.com'

def initialize_gee():
    gee_key_json = os.environ.get('GEE_KEY')
    if gee_key_json:
        # Running in GitHub Actions
        key_dict = json.loads(gee_key_json)
        credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_data=gee_key_json)
        ee.Initialize(credentials, project='ee-muzzamilgandapur007')
        print("✅ GEE Initialized via GitHub Actions (Service Account)")
    else:
        # Running locally on your computer
        ee.Initialize(project='ee-muzzamilgandapur007')
        print("✅ GEE Initialized locally")

initialize_gee()

# ==========================================
# 2. DATA GAP CHECKING
# ==========================================
master_path = 'Master_Data.csv'
if os.path.exists(master_path):
    df_master = pd.read_csv(master_path)
    df_master['date'] = pd.to_datetime(df_master['date'])
    # Find the day after the last record
    start_date = (df_master['date'].max() + timedelta(days=1)).strftime('%Y-%m-%d')
else:
    df_master = pd.DataFrame()
    start_date = '2018-01-01'

end_date = datetime.now().strftime('%Y-%m-%d')

if start_date >= end_date:
    print("✅ No new data found. Master_Data is already current.")
    exit()

print(f"📡 Syncing missing data from {start_date} to {end_date}...")

# ==========================================
# 3. GEE PIXEL EXTRACTION LOGIC
# ==========================================
def get_new_data(start, end):
    region = ee.Geometry.Rectangle([33.5, -1.5, 36.5, 1.5])
    landcover = ee.Image("ESA/WorldCover/v100/2020")
    cropland = landcover.eq(40)
    
    # Use fixed seed=42 to always get the exact SAME 1200 points
    points = cropland.selfMask().stratifiedSample(
        numPoints=1200, region=region, scale=30, geometries=True, seed=42
    )

    s2Col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
        .filterBounds(region)\
        .filterDate(start, end)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))

    if s2Col.size().getInfo() == 0:
        print("☁️ Sky too cloudy or no new images yet.")
        return pd.DataFrame()

    # This function processes each image and returns features
    def extract_stats(img):
        # Indices
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        savi = img.expression('((N-R)/(N+R+0.5))*1.5', {'N':img.select('B8'),'R':img.select('B4')}).rename('SAVI')
        
        # Weather
        date = img.date()
        rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date, date.advance(1,'day')).mean().rename('Rainfall')
        weather = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date, date.advance(1,'day')).mean()
        temp = weather.select('temperature_2m').subtract(273.15).rename('Temp')
        
        combined = img.addBands([ndvi, savi, rain, temp])
        
        def reduce_point(pt):
            stats = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=pt.geometry(), scale=30)
            return ee.Feature(pt.geometry(), stats).set({
                'date': date.format('YYYY-MM-DD'),
                'point_id': pt.id()
            })
        
        return points.map(reduce_point)

    # Flatten the collection
    results = s2Col.map(extract_stats).flatten()
    
    # Filter out empty results
    clean_results = results.filter(ee.Filter.notNull(['NDVI', 'Rainfall', 'Temp']))
    
    # Convert GEE FeatureCollection to list of dicts for Pandas
    features = clean_results.getInfo()['features']
    data_list = []
    for f in features:
        props = f['properties']
        # Add .geo if needed for mapping
        props['.geo'] = json.dumps(f['geometry'])
        data_list.append(props)
    
    return pd.DataFrame(data_list)

# ==========================================
# 4. EXECUTION & AUTO-SAVE
# ==========================================
try:
    new_df = get_new_data(start_date, end_date)
    
    if not new_df.empty:
        # Merge with existing data
        updated_master = pd.concat([df_master, new_df], ignore_index=True)
        # Ensure year column is set correctly
        updated_master['year'] = pd.to_datetime(updated_master['date']).dt.year
        # Save
        updated_master.to_csv(master_path, index=False)
        print(f"✅ Master_Data updated with {len(new_df)} new records!")
    else:
        print("ℹ️ No processable images found for this period.")

except Exception as e:
    print(f"❌ Error during sync: {e}")