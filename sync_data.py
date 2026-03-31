import ee
import os
import json
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 1. AUTHENTICATION (The "Bot" Login)
# ==========================================
EE_SERVICE_ACCOUNT = 'agripredict-bot@ee-muzzamilgandapur007.iam.gserviceaccount.com'

def initialize_gee():
    gee_key_json = os.environ.get('GEE_KEY')
    if gee_key_json:
        print("🤖 CLOUD MODE: Service Account Key detected.")
        credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, key_data=gee_key_json)
        ee.Initialize(credentials, project='ee-muzzamilgandapur007')
    else:
        print("💻 LOCAL MODE: Using personal login...")
        ee.Initialize(project='ee-muzzamilgandapur007')

initialize_gee()

# ==========================================
# 2. SMART DATE & GAP CHECKING
# ==========================================
master_path = 'Master_Data.csv'

if os.path.exists(master_path):
    df_master = pd.read_csv(master_path)
    # Ensure column names are clean and case-insensitive
    df_master.columns = df_master.columns.str.strip()
    
    if not df_master.empty and 'date' in df_master.columns:
        df_master['date'] = pd.to_datetime(df_master['date'])
        # Start exactly 1 day after the last recorded date
        start_date = (df_master['date'].max() + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_date = '2018-01-01'
else:
    df_master = pd.DataFrame()
    start_date = '2018-01-01'

# Today's date for the end of the search
end_date = datetime.now().strftime('%Y-%m-%d')

if start_date >= end_date:
    print(f"✅ Data is current up to {start_date}. No update needed.")
    exit()

print(f"📡 Syncing all features from {start_date} to {end_date}...")

# ==========================================
# 3. FULL-FEATURE DATA EXTRACTION
# ==========================================
def get_latest_agri_data(start, end):
    region = ee.Geometry.Rectangle([33.5, -1.5, 36.5, 1.5])
    landcover = ee.Image("ESA/WorldCover/v100/2020")
    cropland = landcover.eq(40)
    
    # Use Seed 42 to keep 1200 points consistent forever
    points = cropland.selfMask().stratifiedSample(
        numPoints=1200, region=region, scale=30, geometries=True, seed=42
    )

    s2Col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
        .filterBounds(region)\
        .filterDate(start, end)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35)) # Increased to 35% for more 2026 data

    if s2Col.size().getInfo() == 0:
        return pd.DataFrame()

    def process_image(img):
        date = img.date()
        # 1. Vegetation Indices
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        savi = img.expression('((N-R)/(N+R+0.5))*1.5', {'N':img.select('B8'),'R':img.select('B4')}).rename('SAVI')
        evi = img.expression('2.5*((N-R)/(N+6*R-7.5*B+1))', {'N':img.select('B8'),'R':img.select('B4'),'B':img.select('B2')}).rename('EVI')
        
        # 2. Weather & Climate (ERA5 & CHIRPS)
        rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date, date.advance(1,'day')).mean().rename('Rainfall')
        weather = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date, date.advance(1,'day')).mean()
        temp = weather.select('temperature_2m').subtract(273.15).rename('Temp')
        
        # 3. Wind & Soil
        u = weather.select('u_component_of_wind_10m')
        v = weather.select('v_component_of_wind_10m')
        wind = u.hypot(v).rename('Wind_Speed')
        
        soil = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007")\
            .filterDate(date.advance(-1,'day'), date.advance(1,'day')).mean()\
            .select('sm_surface').rename('Soil_Moisture')
        
        combined = img.addBands([ndvi, savi, evi, rain, temp, wind, soil])
        
        def reduce_point(pt):
            stats = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=pt.geometry(), scale=30)
            return ee.Feature(pt.geometry(), stats).set({
                'date': date.format('YYYY-MM-DD'),
                'point_id': pt.id(),
                'year': date.get('year')
            })
        return points.map(reduce_point)

    # Execute and flatten
    results = s2Col.map(process_image).flatten()
    # Filter only rows that have the critical data
    final_features = results.filter(ee.Filter.notNull(['NDVI', 'Rainfall', 'Temp']))
    
    if final_features.size().getInfo() == 0:
        return pd.DataFrame()

    # Convert to Pandas
    data = final_features.getInfo()['features']
    rows = []
    for f in data:
        p = f['properties']
        p['.geo'] = json.dumps(f['geometry'])
        rows.append(p)
    return pd.DataFrame(rows)

# ==========================================
# 4. MERGE & SAVE
# ==========================================
try:
    new_rows = get_latest_agri_data(start_date, end_date)
    
    if not new_rows.empty:
        # Match columns exactly
        final_df = pd.concat([df_master, new_rows], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['point_id', 'date'])
        
        # Save with clean commas
        final_df.to_csv(master_path, index=False)
        print(f"🚀 SUCCESS: Added data for {new_rows['date'].nunique()} new dates in 2026!")
    else:
        print("☁️ No clear satellite images found for the missing period.")

except Exception as e:
    print(f"❌ Error: {e}")
