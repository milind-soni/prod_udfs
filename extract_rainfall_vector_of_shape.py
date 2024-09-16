import ee
import json 
import pandas as pd
import urllib.parse
import h3
from shapely.geometry import shape
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EARTH_ENGINE_DATASET = 'ECMWF/ERA5_LAND/DAILY_AGGR'
PRECIPITATION_BAND = 'total_precipitation_sum'
START_DATE = '2022-01-01'
END_DATE = '2022-12-31'
SCALE = 1000
MAX_PIXELS = 1e9

@fused.udf
def udf(geojson: str = '{"type":"Polygon","coordinates":[[[77.1,28.7],[77.1,28.5],[77.3,28.5],[77.3,28.7],[77.1,28.7]]]}', h3_size: int = 5):

    @fused.cache
    def generate_service_account_info():
        # Replace this with a secure method to retrieve your service account info
        # For example, you might load it from environment variables or a secure storage service
        return {}

    @fused.cache(ttl=3600)
    def get_ee_image_collection(geometry):
        return ee.ImageCollection(EARTH_ENGINE_DATASET) \
            .filterDate(START_DATE, END_DATE) \
            .filterBounds(geometry) \
            .select(PRECIPITATION_BAND)
    
    def fetch_data(geojson):
        try:
            service_account_info = generate_service_account_info()
            if not service_account_info:
                raise ValueError("Failed to retrieve service account information")
    
            credentials = ee.ServiceAccountCredentials(service_account_info['client_email'], key_data=json.dumps(service_account_info))
            ee.Initialize(credentials=credentials)
    
            geometry = ee.Geometry(geojson)
            ic = get_ee_image_collection(geometry)
    
            def aggregate_monthly(month):
                filtered = ic.filter(ee.Filter.calendarRange(month, month, 'month'))
                mean = filtered.mean()
                return mean.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=SCALE,
                    maxPixels=MAX_PIXELS
                ).get(PRECIPITATION_BAND)
    
            monthly_means = ee.List.sequence(1, 12).map(aggregate_monthly)
            
            result = ee.Dictionary({
                'monthly_precipitation': monthly_means
            }).getInfo()
    
            return pd.DataFrame({
                'month': range(1, 13),
                'precipitation': result['monthly_precipitation']
            })
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None
    
    def process_data(df, geojson, h3_size):
        if df is None or df.empty:
            return pd.DataFrame({'error': ['Failed to fetch data for the given GeoJSON']})
    
        # Convert precipitation from meters to millimeters
        df['precipitation'] *= 1000
    
        geom = shape(geojson)
        centroid = geom.centroid
        h3_index = h3.latlng_to_cell(centroid.y, centroid.x, h3_size)
    
        return pd.DataFrame({
            'h3_index': [h3_index],
            'rainfall': [df['precipitation'].tolist()]
        })

    try:
        decoded_geojson = json.loads(urllib.parse.unquote(geojson))
        df = fetch_data(decoded_geojson)
        if df is not None:
            result = process_data(df, decoded_geojson, h3_size)
            print(result)
            return result
        else:
            return pd.DataFrame({'error': ['Failed to fetch data']})
    except Exception as e:
        logger.error(f"Error in UDF: {str(e)}")
        return pd.DataFrame({'error': [str(e)]})
