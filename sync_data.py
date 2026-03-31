import ee
import os
import json
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 1. AUTHENTICATION LOGIC (WORKING)
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
        ee.Initialize(project='ee-muzzamilgandapur007')

initialize_gee()

# ==========================================
# 2. SMART DATE PARSER (FIXES THE VALUEERROR)
# ==========================================
def robust_date_parser(date_str):
    """Handles standard YYYY-MM-DD and GEE's YYYY-MM-DDD formats"""
    date_str = str(date_str).strip()
    try:
        # Try standard format first
        return pd.to_datetime(date_str)
    except:
        try:
            # Handle YYYY-MM-DDD (e.g., 2018-12-352)
            parts = date_str.split('-')
            if len(parts) == 3:
                year = int(parts[0])
                day_of_year = int(parts[2])
                return pd.Timestamp(datetime(year, 1, 1) + timedelta(days=day_of_year - 1))
        except:
            return pd.NaT

# ==========================================
# 3. DATA GAP CHECKING
# ==========================================
master_path = 'Master_Data.csv'

if os.path.exists(master_path):
    print(f"📄 Analyzing {master_path} (71MB)...")
    # Read only the date column first to save memory
    df_dates = pd.read_csv(master_path, usecols=['date'])
    
    # Apply the robust parser
    parsed_dates = df_dates['date'].apply(robust_date_parser)
    last_date = parsed_dates.max()
    
    if pd.isnull(last_date):
        start_date = '2018-01-01'
    else:
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"✅ Last valid record: {last_date.date()}")
else:
    start_date = '2018-01-01'

end_date = datetime.now().strftime('%Y-%m-%d')

if start_date >= end_date:
    print("✅ System is already up to date.")
    exit()

print(f"📡 Syncing new data from {start_date} to {end_date}...")

# ==========================================
# 4. GEE EXTRACTION (FULL FEATURES)
# ==========================================
def get_latest_data(start, end):
    region = ee.Geometry.Rectangle([33.5, -1.5, 36.5, 1.5])
    landcover = ee.Image("ESA/WorldCover/v100/2020")
    cropland = landcover.eq(40)
    points = cropland.selfMask().stratifiedSample(numPoints=1200, region=region, scale=30, geometries=True, seed=42)

    s2Col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
        .filterBounds(region)\
        .filterDate(start, end)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))

    if s2Col.size().getInfo() == 0: return pd.DataFrame()

    def process_img(img):
        date = img.date()
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        savi = img.expression('((N-R)/(N+R+0.5))*1.5', {'N':img.select('B8'),'R':img.select('B4')}).rename('SAVI')
        evi = img.expression('2.5*((N-R)/(N+6*R-7.5*B+1))', {'N':img.select('B8'),'R':img.select('B4'),'B':img.select('B2')}).rename('EVI')
        
        rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date, date.advance(1,'day')).mean().rename('Rainfall')
        weather = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date, date.advance(1,'day')).mean()
        temp = weather.select('temperature_2m').subtract(273.15).rename('Temp')
        wind = weather.select('u_component_of_wind_10m').hypot(weather.select('v_component_of_wind_10m')).rename('Wind_Speed')
        soil = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007").filterDate(date.advance(-1,'day'), date.advance(1,'day')).mean().select('sm_surface').rename('Soil_Moisture')
        
        combined = img.addBands([ndvi, savi, evi, rain, temp, wind, soil])
        return points.map(lambda pt: ee.Feature(pt.geometry(), combined.reduceRegion(ee.Reducer.mean(), pt.geometry(), 30)).set({
            'date': date.format('YYYY-MM-DD'), 'point_id': pt.id(), 'year': date.get('year')
        }))

    results = s2Col.map(process_img).flatten().filter(ee.Filter.notNull(['NDVI', 'Rainfall', 'Temp']))
    
    if results.size().getInfo() == 0: return pd.DataFrame()
    
    features = results.getInfo()['features']
    return pd.DataFrame([dict(f['properties'], **{'.geo': json.dumps(f['geometry'])}) for f in features])

# ==========================================
# 5. EXECUTION
# ==========================================
try:
    new_rows = get_latest_data(start_date, end_date)
    if not new_rows.empty:
        # Load full master to append (Standardizing columns)
        df_master = pd.read_csv(master_path)
        final_df = pd.concat([df_master, new_rows], ignore_index=True)
        final_df.drop_duplicates(subset=['point_id', 'date']).to_csv(master_path, index=False)
        print(f"🚀 SUCCESS: Added {len(new_rows)} records up to {end_date}")
    else:
        print("☁️ No new clear imagery found.")
except Exception as e:
    print(f"❌ Error: {e}")
