import ee
import json 
import geopandas as gpd
import numpy as np
import xarray as xr
import h3
import pandas as pd
from shapely.geometry import Polygon, box
from utils import duckdb_with_h3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EARTH_ENGINE_DATASET = 'ECMWF/ERA5_LAND/DAILY_AGGR'
PRECIPITATION_BAND = 'total_precipitation_sum'
START_DATE = '2022-01-01'
END_DATE = '2022-12-31'
BUFFER_DISTANCE = 0.5
SCALE = 0.1

@fused.udf
def udf(bbox: fused.types.Bbox = None, layer: str = PRECIPITATION_BAND, time: int = 1, h3_size: int = 5, input_array: list = [
    2.7715942327085665, 0.7863158647851064, 0.12182573777622914, 0.06965268364875025,
    0.7598821146230067, 2.4584453831794897, 6.331687329766313, 2.96728587006872,
    4.837692532803455, 5.647026369003688, 0.14140295145352866, 0.003789064855178973
]):
    con = duckdb_with_h3()
    con.sql("INSTALL vss")
    
    run_query = fused.load("https://github.com/fusedio/udfs/tree/43656f6/public/common/").utils.run_query

    @fused.cache
    def generate_service_account_info():
        # Replace this with a secure method to retrieve your service account info
        return {}

    @fused.cache
    def get_ee_image_collection(aoi):
        return ee.ImageCollection(EARTH_ENGINE_DATASET) \
            .filterDate(START_DATE, END_DATE) \
            .filterBounds(aoi) \
            .select(layer)

    def fetch_data():
        try:
            service_account_info = generate_service_account_info()
            credentials = ee.ServiceAccountCredentials(service_account_info['client_email'], key_data=json.dumps(service_account_info))
            ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", credentials=credentials)

            minx, miny, maxx, maxy = bbox.total_bounds
            buffered_bounds = (minx - BUFFER_DISTANCE, miny - BUFFER_DISTANCE, maxx + BUFFER_DISTANCE, maxy + BUFFER_DISTANCE)
            aoi = ee.Geometry.Rectangle(buffered_bounds)

            ic = get_ee_image_collection(aoi)
            
            ds = xr.open_dataset(ic, engine='ee', crs='EPSG:4326', scale=SCALE)
            ds = ds.sel(lat=slice(buffered_bounds[3], buffered_bounds[1]), lon=slice(buffered_bounds[0], buffered_bounds[2]))

            logger.info("Successfully fetched data")
            return ds, buffered_bounds
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None, None

    @fused.cache
    def ds_to_hex(ds, layer, h3_size, minx, miny, maxx, maxy):
        df = ds[layer].to_dataframe().reset_index()[['time', 'lat', 'lon', layer]].dropna()
        mask = (df['lon'].between(minx, maxx)) & (df['lat'].between(miny, maxy))
        df = df[mask]
        df[layer] *= 1000  # Convert to mm

        qr = f"""
        WITH monthly_data AS (
            SELECT 
                h3_h3_to_string(h3_latlng_to_cell(lat, lon, {h3_size})) AS hex,
                EXTRACT(MONTH FROM time) AS month,
                AVG({layer}) AS avg_data
            FROM df
            GROUP BY 1, 2
        )
        SELECT 
            hex,
            ARRAY_AGG(avg_data ORDER BY month) AS monthly_avg_data
        FROM monthly_data
        GROUP BY 1
        HAVING COUNT(DISTINCT month) = 12
        ORDER BY 1
        """
        return run_query(qr)

    def calculate_cosine_similarity(hex_data, input_array):
        input_array = np.array(input_array)
        hex_array = np.stack(hex_data['monthly_avg_data'].to_numpy())
        similarity = np.dot(hex_array, input_array) / (np.linalg.norm(hex_array, axis=1) * np.linalg.norm(input_array))
        hex_data['similarity'] = similarity
        return hex_data

    @fused.cache
    def hex_to_polygon(hex_id):
        boundaries = h3.cell_to_boundary(hex_id)
        return Polygon([(lon, lat) for lat, lon in boundaries])

    ds, buffered_bounds = fetch_data()
    if ds is not None and buffered_bounds is not None:
        df_hex_combined = ds_to_hex(ds, layer, h3_size, *buffered_bounds)
        df_hex_combined[['lat', 'lng']] = df_hex_combined['hex'].apply(lambda x: pd.Series(h3.cell_to_latlng(x)))

        if input_array is not None:
            df_hex_with_similarity = calculate_cosine_similarity(df_hex_combined, input_array)
            df = df_hex_with_similarity
        else:
            df = df_hex_combined

        # Convert to GeoDataFrame
        df['geometry'] = df['hex'].apply(hex_to_polygon)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

        # Calculate centroid for each polygon
        gdf['centroid'] = gdf['geometry'].centroid
        gdf['lon'] = gdf['centroid'].x
        gdf['lat'] = gdf['centroid'].y

        # Clip to original bounding box
        original_bbox = box(*bbox.total_bounds)
        gdf = gdf[gdf.intersects(original_bbox)]
        gdf['geometry'] = gdf.intersection(original_bbox)

        logger.info("Processing completed successfully")
        return gdf
    else:
        logger.error("Failed to fetch or process data")
        return None
