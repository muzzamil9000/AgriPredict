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
# 2. IDENTIFY THE GAPS (Which year needs work?)
# ==========================================
target_years = list(range(2018, 2027))
NEW_PARAMS = ['ndre', 'lswi', 'ndmi', 'chl', 'humidity', 'vpd'] # Key indicators of an updated file

current_target_year = None
start_dt = None

for year in target_years:
    filename = f"Maizedata_1200poi_{year}_Clean.csv"
    
    if not os.path.exists(filename):
        print(f"📂 Year {year} is missing. Starting fresh.")
        current_target_year = year
        start_dt = datetime(year, 1, 1)
        break
    else:
        # Check if existing file has new parameters
        df_check = pd.read_csv(filename, nrows=5)
        df_check.columns = [c.lower() for c in df_check.columns]
        if not all(p in df_check.columns for p in NEW_PARAMS):
            print(f"🔄 Year {year} exists but needs new parameters. Re-syncing.")
            current_target_year = year
            start_dt = datetime(year, 1, 1)
            break
        # If it's the current year 2026, always check for new days
        elif year == 2026:
            df_full = pd.read_csv(filename, usecols=['date'])
            last_date = pd.to_datetime(df_full['date']).max()
            if last_date < datetime.now() - timedelta(days=5):
                print(f"📈 Year 2026 needs a daily update. Resuming from {last_date.date()}")
                current_target_year = 2026
                start_dt = last_date + timedelta(days=1)
                break

if current_target_year is None:
    print("✅ All data from 2018 to 2026 is complete with all parameters!")
    sys.exit(0)

# ==========================================
# 3. EXTRACTION LOGIC (The "Everything" Pull)
# ==========================================
def get_full_spectrum_data(start, end):
    try:
        region = ee.Geometry.Rectangle([33.5, -1.5, 36.5, 1.5])
        points = ee.FeatureCollection.randomPoints(region, 500, 42) # 500 per chunk to stay fast

        s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(region)\
            .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))

        if s2_col.size().getInfo() == 0: return pd.DataFrame()

        def process_img(img):
            date = img.date()
            # TIER 1: Indices
            ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
            ndre = img.normalizedDifference(['B8', 'B5']).rename('ndre')
            lswi = img.normalizedDifference(['B8', 'B11']).rename('lswi')
            ndmi = img.normalizedDifference(['B8A', 'B11']).rename('ndmi')
            savi = img.expression('((N-R)/(N+R+0.5))*1.5', {'N':img.select('B8'),'R':img.select('B4')}).rename('savi')
            evi = img.expression('2.5*((N-R)/(N+6*R-7.5*B+1))', {'N':img.select('B8'),'R':img.select('B4'),'B':img.select('B2')}).rename('evi')
            chl = img.select('B7').divide(img.select('B5')).subtract(1).rename('chl')
            bsi = img.expression('((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))', {'B11': img.select('B11'), 'B4': img.select('B4'), 'B8': img.select('B8'), 'B2': img.select('B2')}).rename('bsi')
            psri = img.expression('(B4-B2)/B6', {'B4': img.select('B4'), 'B2': img.select('B2'), 'B6': img.select('B6')}).rename('psri')

            # TIER 2: Weather (ERA5 & CHIRPS)
            rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date, date.advance(1,'day')).mean().rename('rainfall')
            w_coll = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date, date.advance(1,'day'))
            weather = ee.Image(ee.Algorithms.If(w_coll.size().gt(0), w_coll.mean(), ee.Image(0)))
            
            temp_c = weather.select('temperature_2m').subtract(273.15).rename('temp')
            dew_c  = weather.select('dewpoint_temperature_2m').subtract(273.15)
            
            rh = ee.Image(100).multiply(ee.Image(17.625).multiply(dew_c).divide(ee.Image(243.04).add(dew_c)).exp())\
                .divide(ee.Image(17.625).multiply(temp_c).divide(ee.Image(243.04).add(temp_c)).exp()).rename('humidity')
            
            vpd = ee.Image(0.611).multiply(ee.Image(17.27).multiply(temp_c).divide(temp_c.add(237.3)).exp())\
                .multiply(ee.Image(1).subtract(rh.divide(100))).rename('vpd')

            combined = img.addBands([ndvi, ndre, lswi, ndmi, savi, evi, chl, bsi, psri, rain, temp_c, rh, vpd])
            bands = ['ndvi', 'ndre', 'lswi', 'ndmi', 'savi', 'evi', 'chl', 'bsi', 'psri', 'rainfall', 'temp', 'humidity', 'vpd']
            
            return points.map(lambda pt: ee.Feature(pt.geometry(), combined.select(bands).reduceRegion(ee.Reducer.mean(), pt.geometry(), 30)).set({
                'date': date.format('YYYY-MM-DD'), 'point_id': pt.id(), 'year': date.get('year')
            }))

        results = s2_col.map(process_img).flatten().filter(ee.Filter.notNull(['ndvi']))
        return pd.DataFrame([dict(f['properties'], **{'.geo': json.dumps(f['geometry'])}) for f in results.getInfo()['features']])
    except Exception as e:
        print(f"⚠️ Chunk Error: {e}")
        return pd.DataFrame()

# ==========================================
# 4. CHUNKED REBUILDING
# ==========================================
filename = f"Maizedata_1200poi_{current_target_year}_Clean.csv"
year_end = datetime(current_target_year, 12, 31)
actual_end = min(year_end, datetime.now() - timedelta(days=5))

current_ptr = start_dt
chunk_size = timedelta(days=10)

while current_ptr < actual_end:
    chunk_end = min(current_ptr + chunk_size, actual_end)
    print(f"⏳ {current_target_year}: Building {current_ptr.date()} to {chunk_end.date()}...")
    
    new_data = get_full_spectrum_data(current_ptr, chunk_end)
    
    if not new_data.empty:
        # Load existing file to append or start new
        df_old = pd.read_csv(filename) if os.path.exists(filename) else pd.DataFrame()
        # Ensure lowercase columns
        new_data.columns = [c.lower() for c in new_data.columns]
        if not df_old.empty: df_old.columns = [c.lower() for c in df_old.columns]
        
        # Merge and save
        final_df = pd.concat([df_old, new_data], ignore_index=True).drop_duplicates(subset=['point_id', 'date'])
        final_df.to_csv(filename, index=False)
        print(f"✅ Saved records to {filename}")
        del df_old, final_df
    
    current_ptr = chunk_end

print(f"🚀 SUCCESS: Completed Year {current_target_year}")
