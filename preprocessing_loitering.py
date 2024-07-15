# %%
import pandas as pd
import numpy as np
import pickle
import geopandas as gpd
import movingpandas as mpd
import shapely as shp
import hvplot.pandas
import warnings
from geopandas import GeoDataFrame, read_file
from datetime import datetime, timedelta
from holoviews import opts
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import LineString
import time
from geopy.distance import geodesic
from folium.plugins import MeasureControl

vessel_type_mapping = {
    0: 'Not available (default)',
    1: 'Reserved for future use',
    2: 'WIG, Reserved for future use',
    30: 'Fishing',
    31: 'Towing',
    32: 'Towing and length of the tow exceeds 200m or breadth exceeds 25m',
    33: 'Dredging or underwater operations',
    34: 'Diving operations',
    35: 'Military operations',
    36: 'Sailing',
    37: 'Pleasure craft',
    50: 'General cargo vessel',
    51: 'Unit carrier',
    52: 'Bulk carrier',
    53: 'Tanker',
    54: 'Liquefied gas tanker',
    55: 'Other special tanker',
    56: 'Spare - Local vessel',
    57: 'Cargo and passenger vessel',
    58: 'Medical transport',
    59: 'Passenger ship',
    60: 'Assistance vessel',
    61: 'Passenger, Hazardous category A',
    62: 'Passenger, Hazardous category B',
    63: 'Passenger, Hazardous category C',
    64: 'Passenger, Hazardous category D',
    70: 'Other sea-going vessel',
    71: 'Cargo, Hazardous category A',
    72: 'Work ship',
    73: 'Push boat',
    74: 'Dredger',
    80: 'Pleasure boat',
    81: 'Speedboat',
    82: 'Sailing boat with auxiliary motor',
    83: 'Sailing yacht',
    84: 'Boat for sport fishing',
    90: 'Fast ship',
    91: 'Hydrofoil',
    92: 'Catamaran, fast',
    93: 'Other types, Hazardous category C',
    94: 'Other types, Hazardous category D',
    501: 'Grain vessel',
    502: 'Timber/log carrier',
    503: 'Wood chips vessel',
    504: 'Steel products vessel',
    505: 'Carrier, general cargo/container',
    506: 'Temperature controlled cargo vessels',
    507: 'Chemical carrier',
    508: 'Irradiated fuel carrier',
    509: 'Heavy cargo vessel',
    510: 'RoRo/Container vessel',
    511: 'Full container ship/cellular vessel',
    512: 'RoRo vessel',
    513: 'Car carrier',
    514: 'Livestock carrier',
    515: 'Barge carrier – Lash ship',
    516: 'Chemical carrier',
    517: 'Irradiated fuel carrier',
    518: 'Heavy cargo vessel',
    519: 'RoRo/Container vessel',
    521: 'Dry bulk carrier',
    522: 'Ore carrier',
    523: 'Cement carrier',
    524: 'Gravel carrier',
    525: 'Coal carrier',
    531: 'Crude oil tanker',
    532: 'Chemical tanker, coaster',
    533: 'Chemical tanker, deep sea',
    534: 'Oil and other derivatives tanker',
    541: 'LPG tanker',
    542: 'LNG tanker',
    543: 'LNG/LPG tanker',
    551: 'Asphalt/bitumen tanker',
    552: 'Molasses tanker',
    553: 'Vegetable oil tanker',
    591: 'Cruise ship',
    592: 'Ferry',
    593: 'Other passenger ship',
    594: 'Passenger ship, sailing',
    601: 'Tug, without tow',
    602: 'Tug, with tow',
    603: 'Salvage vessel',
    604: 'Rescue vessel',
    605: 'Oil combat vessel',
    606: 'Oil rig',
    607: 'Hospital vessel',
    711: 'Pilot boat',
    712: 'Patrol/measure ship',
    721: 'Supply vessel',
    723: 'Offshore support vessel',
    724: 'Pontoon',
    725: 'Stone dumping vessel',
    726: 'Cable layer',
    727: 'Buoyage vessel',
    728: 'Icebreaker',
    729: 'Pipelaying vessel',
    75: 'Fishing boat',
    751: 'Trawler',
    752: 'Cutter',
    753: 'Factory ship',
    76: 'Research and education ship',
    761: 'Fishery research vessel',
    762: 'Climate registration vessel',
    763: 'Ship for environmental measurement',
    764: 'Scientific vessel',
    765: 'Sailing school ship',
    766: 'Training vessel',
    77: 'Navy vessel',
    78: 'Structure, floating',
    781: 'Crane, floating',
    782: 'Dock, floating',
    85: 'Craft, pleasure, longer than 20 metres',
    89: 'Craft, other, recreational',
    1007: 'Drill Ship',
    1010: 'Offshore',
    1001: 'Fishing',
    1004: 'Heavy Load Carrier'
}

vessel_type_mapping_target = {
    75: 'Fishing boat',
    751: 'Trawler',
    752: 'Cutter',
    30: 'Fishing',
    1010: 'Offshore',
    52: 'Bulk carrier',
    82: 'Sailing boat with auxiliary motor'

}

def msg_telegram(message):
    ip_address = "127.0.0.1"
    import socket
    try:
        # Criar um socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Conectar ao servidor
        s.connect((ip_address, 12345))
        
        # Enviar a mensagem
        s.sendall(message.encode())
        
        # Fechar o socket
        s.close()
        print(f"Message sent to {ip_address}:12345")
    except Exception as e:
        print(f"An error occurred: {e}")

class Encounter:
    def __init__(self):
        self.data_set_folder = "/home/vindn/SynologyDrive/4_UFRJ/projeto_tese/codigos/projeto_datasets/loitering/raw_datasets/link_dataset/"
        self.data_set_output_folder = "datasets_output/"

        # self.data_set_folder = self.data_set_folder
        # self.url = self.data_set_folder + "AIS_2022_01_01.csv"
        self.filename = "AIS_2015_01_01.csv"
        self.url = self.data_set_folder + self.filename
        self.chunksize = 10 ** 6
        self.chunks = []
        # large data
        self.gdf = None
        self.vessel_trajectories = None
        # self.raster = None
        self.raster_data = None
        self.transform = None
        self.init_raster_distances( )

    def setAISfile( self, filename ):
        self.filename = filename
        self.url = self.data_set_folder + self.filename
            
    def init_raster_distances( self ):
        import rasterio
        from rasterio.transform import from_origin
        # Distances in kilometers!!!
        raster_file = self.data_set_folder + 'distance-from-shore.tif'
        # 1. Carregar o raster na memória
        with rasterio.open(raster_file) as src:
            self.raster_data = src.read(1)  # Ler o primeiro band do raster
            self.transform = src.transform

    # 2. Função para converter coordenadas geográficas para coordenadas de linha e coluna
    def geo_to_rowcol(self, x, y ):
        col, row = ~self.transform * (x, y)
        return int(row), int(col)

    def get_distance_from_coast_km(self, p):
        # Distances in kilometers!!!
        row, col = self.geo_to_rowcol(p.x, p.y)

        value = self.raster_data[row, col]
        return value
    
    def apply_distance_from_coast_km( self, gdf ):
        gdf['distance_to_coast'] = gdf['geometry'].apply(self.get_distance_from_coast_km)
        return gdf

    def get_gdf( self ):
        return self.gdf
    
    def get_coastline( self ):
        return self.coast
    

    def read_ais_csv(self):
        for chunk in pd.read_csv(self.url, chunksize=self.chunksize):
            self.chunks.append(chunk)

        data = pd.concat(self.chunks, axis=0)

        print(data.head())
        data.dropna()

        return data

    def write_pickle_obj(self, data, file_name):
        with open(self.data_set_output_folder + file_name, 'wb') as data_file:
            pickle.dump(data, data_file)

    def read_pickle_obj(self, file_name):
        try:
            with open(self.data_set_folder + file_name, 'rb') as data_file:
                data = pickle.load(data_file)
                return data
        except Exception as e:
            print(e, "File not Found!")

    def load_gdf(self, df):
        import warnings
        warnings.filterwarnings('ignore')

        opts.defaults(opts.Overlay(
            active_tools=['wheel_zoom'], frame_width=500, frame_height=400))

        gdf = gpd.GeoDataFrame(
            df.set_index('BaseDateTime'), geometry=gpd.points_from_xy(df.LON, df.LAT))

        gdf.set_crs('epsg:4326')
        return gdf
    
    # Trajectories
    def create_trajectory(self, gdf, verbose=True, gap_minutes=120):
        import movingpandas as mpd
        import shapely as shp
        import hvplot.pandas
        import time

        # reset index
        gdf = gdf.reset_index()
        gdf['BaseDateTime'] = pd.to_datetime(gdf['BaseDateTime'], utc=True)
        gdf['traj_id'] = gdf['MMSI']
        # limit to avoid slow
    #     gdf = gdf[:10000]

        # create trajectories

        start_time = time.time()

        # Specify minimum length for a trajectory (in meters)
        minimum_length = 0
        # collection = mpd.TrajectoryCollection(gdf, "imo",  t='timestamp', min_length=0.001)
        collection = mpd.TrajectoryCollection(
            gdf, "traj_id",  t='BaseDateTime', min_length=0.001, crs='epsg:4326')
        # collection.add_direction(gdf.COG)
        # collection.add_speed(gdf.SOG, units='nm')
        # collection.add_traj_id(overwrite=True)

        # set time gap between trajectories for split
        # collection = mpd.ObservationGapSplitter(
        #     collection).split(gap=timedelta(minutes=90))
        collection = mpd.ObservationGapSplitter(
            collection).split(gap=timedelta(minutes=gap_minutes))
        
        # collection.add_speed(overwrite=True, name="speed_nm", units=('nm', 'h'))
        # collection.add_direction(overwrite=True, name='direction')
        # collection.add_timedelta(overwrite=True, name='time_delta')
        # collection.add_angular_difference(overwrite=True, name='ang_diff')
        # collection.add_distance(overwrite=True, name="dist_diff", units="km")     

        for traj in collection.trajectories:
            traj.gdf['dist_diff']  = self.calc_distance_diff_nm( traj.gdf, 'LAT', 'LON')
            traj.gdf['time_diff_h'] = self.calc_time_diff_h( traj.gdf.reset_index(), 'BaseDateTime' )
            traj.gdf['time_diff'] = gdf['time_diff_h'] * 3600
            traj.gdf['speed_nm'] = traj.gdf['dist_diff'] / traj.gdf['time_diff_h']
            traj.gdf['ang_diff_cog'] = self.angular_diff( traj.gdf['COG'], traj.gdf['COG'].shift(1))
            traj.gdf['cog_calculated'] = self.calculate_cog( traj.gdf )
            traj.gdf['ang_diff_cog_calculated'] = self.angular_diff( traj.gdf['cog_calculated'], traj.gdf['cog_calculated'].shift(1))

        end_time = time.time()
        if verbose:
            print("Time creation trajectories: ", (end_time-start_time)/60,  " min")

        return collection
    
    # The trajectories of encounters was trunc in 21 points
    def create_trajectory_21( self, list_gdf, min_rows=10 ):
        trajs = []
        id = 0
        for gdf in list_gdf:
            if len(gdf) >= min_rows:
                gdf['dist_diff']  = self.calc_distance_diff_nm( gdf, 'LAT', 'LON')
                gdf['time_diff_h'] = self.calc_time_diff_h( gdf.reset_index(), 'BaseDateTime' )
                gdf['time_diff'] = gdf['time_diff_h'] * 3600
                gdf['speed_nm'] = gdf['dist_diff'] / gdf['time_diff_h']
                gdf['ang_diff_cog'] = self.angular_diff( gdf['COG'], gdf['COG'].shift(1))
                gdf['cog_calculated'] = self.calculate_cog( gdf )
                gdf['ang_diff_cog_calculated'] = self.angular_diff( gdf['cog_calculated'], gdf['cog_calculated'].shift(1))

            traj = mpd.Trajectory(gdf, id )
            id += 1
            trajs.append( traj )

        return mpd.TrajectoryCollection( trajs )
    
    def list_encounters_to_list_gdf( self, list_encounters ):

        list_gdfs = []
        for i in range(len(list_encounters)):
            for j in range(len(list_encounters[i])):
                list_gdfs.append( list_encounters[i][j][1] )
                list_gdfs.append( list_encounters[i][j][2] )

        return list_gdfs
    
    # list1_gdfs[0] encounter list2_gdfs[0] and so on...
    def list_encounters_to_2gdf( self, list_encounters, min_rows=10 ):
        list1_gdfs = []
        list2_gdfs = []
        for i in range(len(list_encounters)):
            for j in range(len(list_encounters[i])):
                # test if trajectories has at least min_rows
                if len(list_encounters[i][j][1]) > min_rows and len(list_encounters[i][j][2]) > min_rows:
                    list1_gdfs.append( list_encounters[i][j][1] )
                    list2_gdfs.append( list_encounters[i][j][2] )

        return list1_gdfs, list2_gdfs
    
    def remove_impossible_trajectories( self, trajs ):
        filtered_trajs = []
        for traj in trajs:
            # Test if has impossible speed and positions on trajectories
            testSpoofing = (traj.df['speed_nm'] > 50).any()
            # Drop if has impossible speeds
            if not testSpoofing:
                filtered_trajs.append( traj )

        return mpd.TrajectoryCollection( filtered_trajs )

    
    def built_trajectories_from_scratch( self, aisfilename, resolution=9 ):
        self.setAISfile( aisfilename )
        df_ais = self.read_ais_csv( )
        self.gdf = self.load_gdf( df_ais )        
        # self.gdf = self.filter_gdf( self.gdf )
        # self.vessel_trajectories = self.create_trajectory( self.gdf )
        # self.write_pickle_obj( self.vessel_trajectories, "vessel_trajectories.obj" )
        print("Creating H3 cells...")
        self.gdf = self.apply_h3_cells_in_gdf( self.gdf, resolution )
        print("Creating distances points from coast...")
        self.gdf = self.apply_distance_from_coast_km( self.gdf )
        self.write_gdf_to_file( self.gdf )
        # msg_telegram("Finished built trajectories from scratch!")

        return self.gdf, self.vessel_trajectories
    
    def built_trajectories_from_files( self ):
        self.gdf = self.read_gdf_from_file()
        self.gdf['VesselType'] = self.gdf['VesselType'].fillna(0)
        self.vessel_trajectories = self.read_trajectories_from_file()
        self.insert_columns_trajectories( self.vessel_trajectories )
        # self.gdf = self.filter_gdf( self.gdf )

    def build_gdf_from_batch_files_from_scratch( self ):
        import gc
        prefix_filename = "AIS_2015_01_"
        n_files = 2
        for i in range(1, n_files+1):
            file_path = prefix_filename + "{:02}".format(i) + ".csv"
            self.setAISfile( file_path )
            self.read_ais_csv( )

        df_ais = pd.concat(self.chunks, axis=0)
        self.chunks = None
        gc.collect()
        print("df lines: ", len(df_ais) )
        self.gdf = self.load_gdf( df_ais )
        df_ais = None
        gc.collect()
        self.gdf['VesselType'] = self.gdf['VesselType'].fillna(0)

        self.gdf = self.apply_h3_cells_in_gdf( self.gdf, 9 )
        print("Creating distances points from coast...")
        self.gdf = self.apply_distance_from_coast_km( self.gdf )
        self.write_gdf_to_file( self.gdf )
        msg_telegram("Finished built trajectories from scratch!")

        return self.gdf
        

    def read_trajectories_from_file(  self ):
        data = self.read_pickle_obj( "vessel_trajectories.obj" )
        return data

    def write_trajectories_to_file( self, vessel_trajectories ):
        self.write_pickle_obj( vessel_trajectories, "vessel_trajectories.obj" )

    def write_gdf_to_file( self, gdf ):
        self.write_pickle_obj( gdf, "gdf.obj" )

    def read_gdf_from_file( self ):
        data = self.read_pickle_obj( "gdf.obj" )
        return data    
    
    # Insert data columns in trajectories
    def insert_columns_trajectories( self, collection ):
        
        for traj in collection.trajectories:
            traj.df['ais_interval'] = traj.df.index.to_series().diff()
            traj.df['ais_interval'] = traj.df['ais_interval'].dt.total_seconds() / 60
            traj.df['ais_interval'] = traj.df['ais_interval'].abs()

            traj.df['n_points'] = len( traj.df )

    # def filter_gdf( self, gdf ):
    #     data = gdf[ gdf['SOG'] > 1 ]
    #     return data
    
    # def clustering_trajectories( self, trajs ):
    #     from sklearn.cluster import DBSCAN
    #     from geopy.distance import great_circle
    #     from shapely.geometry import MultiPoint

    #     kms_per_radian = 6371.0088
    #     epsilon = 0.1 / kms_per_radian

    #     n_trajs = len( trajs )
    #     distances = np.zeros(( n_trajs, n_trajs ))
    #     for i in range(n_trajs):
    #         for j in range(n_trajs):
    #             d = trajs[i].to_linestring().distance(
    #                 trajs[j].to_linestring() )
    #             distances[i, j] = d

    #     # print(distances)    

    #     db = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed').fit(distances)
    #     cluster_labels = db.labels_
    #     num_clusters = len(set(cluster_labels))
    #     clusters = pd.Series([distances[cluster_labels == n] for n in range(num_clusters)])
    #     print(f'Number of clusters: {num_clusters}')

    #     for traj, i in zip( trajs, cluster_labels ):
    #         traj.df["cluster"] = i


    # def clustering_points( self, gdf ):
    #     from sklearn.cluster import HDBSCAN

    #     matrix = gdf[ ['LAT','LON'] ].values
    #     kms_per_radian = 6371.0088
    #     epsilon = 0.1 / kms_per_radian

    #     db = HDBSCAN(cluster_selection_epsilon=epsilon, min_samples=1, algorithm='balltree', metric='haversine', n_jobs=-1).fit(np.radians(matrix))
    #     cluster_labels = db.labels_
    #     num_clusters = len(set(cluster_labels))
    #     clusters = pd.Series([matrix[cluster_labels == n] for n in range(num_clusters)])
    #     print(f'Number of clusters: {num_clusters}')

    #     gdf["cluster"] = cluster_labels

    #     return gdf    
    
    def distance_points_meters( self, p1, p2 ):
        point1_series = gpd.GeoSeries([p1], crs="EPSG:4326")
        point1_meters = point1_series.to_crs(epsg=32619)

        point2_series = gpd.GeoSeries([p2], crs="EPSG:4326")
        point2_meters = point2_series.to_crs(epsg=32619)

        return point1_meters.distance(point2_meters).min()

    
    def distance_between_ships_inside_cluster_meters( self, cluster ):
        count = cluster.groupby(['MMSI']).size()
        count = count.sort_values(ascending=False)

        ship1 = cluster[ cluster["MMSI"] == count.index[0] ]
        ship2 = cluster[ cluster["MMSI"] == count.index[1] ]

        return self.distance_points_meters( ship1.iloc[0].geometry, ship2.iloc[0].geometry )

   
    # Resolução	    Raio (km)
    # 0	    1279.0
    # 1	    483.4
    # 2	    183.0
    # 3	    69.09
    # 4	    26.10
    # 5	    9.87
    # 6	    3.73
    # 7	    1.41
    # 8	    0.53
    # 9 	0.20
    # 10	0.076
    # 11	0.0287
    # 12	0.0109
    # 13	0.00411
    # 14	0.00155
    # 15	0.000587
    def apply_h3_cells_in_gdf( self, gdf, resolution=9 ):
        import h3

        # Converta os pontos do gdf para índices H3
        gdf['h3_index'] = gdf.apply(lambda row: h3.geo_to_h3(row['geometry'].y, row['geometry'].x, resolution), axis=1)
        return gdf
    
    # def get_h3_cells_with_multiple_points( self, gdf, min_distance_from_coast=10 ):
    #     import h3

    #     # get clusters indexes with only two different MMSIs in cluster
    #     couting = gdf.groupby('h3_index')['MMSI'].nunique()
    #     indexes = couting.where(couting == 2).dropna().index

    #     encounters = []
    #     for i in indexes:
    #         # get cluster by index
    #         g_cluster = gdf[ gdf["h3_index"] == i ]

    #         count = g_cluster.groupby(['MMSI']).size()
    #         count = count.sort_values(ascending=False)

    #         # ship trajectories
    #         ship1 = self.gdf[ self.gdf["MMSI"] == count.index[0] ]
    #         ship2 = self.gdf[ self.gdf["MMSI"] == count.index[1] ]

    #         encounters.append( [g_cluster, ship1, ship2 ] )                

    #     return encounters

    # Get the 20 AIS points before the encounter!
    def get_trajectory_for_train( self, gdf, ts, mmsi, number_positions=20 ):

        # Filtre por MMSI
        filtered_gdf = gdf[gdf['MMSI'] == mmsi]

        # index_at_nearest_timestamp = filtered_gdf.index.get_loc(ts)
        # Encontre o índice do registro no timestamp específico
        positions = filtered_gdf.index.get_loc(ts)

        # Caso get_loc retorne o slice (no caso de retornar mais de uma data)
        if isinstance(positions, slice):
            positions = list(range(positions.start, positions.stop))
        
        # Selecionar o último índice, se houver duplicatas
        if isinstance(positions, (list, np.ndarray)):
            index_at_timestamp = positions[-1]
        else:
            index_at_timestamp = positions

        # obtenha as 20 ultimas posicoes antes do encontro
        if number_positions > index_at_timestamp:
            result = filtered_gdf.iloc[ 0 : index_at_timestamp + 1]               
        else:
            result = result = filtered_gdf.iloc[ index_at_timestamp - number_positions : index_at_timestamp + 1]
       
        return result 

    def get_h3_cells_with_multiple_points( self, gdf ):
        import h3

        # get clusters indexes with only two different MMSIs in cluster
        couting = gdf.groupby('h3_index')['MMSI'].nunique()
        indexes = couting.where(couting == 2).dropna().index

        encounters = []
        # For each h3 index
        for i in indexes:
            # get cluster by index
            g_cluster = gdf[ gdf["h3_index"] == i ]
            # group by mmsi
            count = g_cluster.groupby(['MMSI']).size()
            # sort by number of rows
            count = count.sort_values(ascending=False)
            # get all rows by both mmsis
            gdf_filtered = self.get_gdf()[ 
                (self.get_gdf()["MMSI"] == count.index[0]) |
                (self.get_gdf()["MMSI"] == count.index[1])
                ]
            gdf_filtered = gdf_filtered.sort_index()
            gdf_filtered.index = pd.to_datetime( gdf_filtered.index )
            # get last time inside cluster
            # ship1OldestTime = g_cluster[ g_cluster["MMSI"] == count.index[0] ].index.max()
            # ship2OldestTime = g_cluster[ g_cluster["MMSI"] == count.index[1] ].index.max()
            ship1OldestTime = g_cluster[ g_cluster["MMSI"] == count.index[0] ].index.min()
            ship2OldestTime = g_cluster[ g_cluster["MMSI"] == count.index[1] ].index.min()
            # get 10 positions before and after select ts
            ship1 = self.get_trajectory_for_train( gdf_filtered, ship1OldestTime, count.index[0] )
            ship2 = self.get_trajectory_for_train( gdf_filtered, ship2OldestTime, count.index[1] )

            encounters.append( [g_cluster, ship1, ship2 ] )                

        return encounters


    # remove df that is not the target in clusters    
    def filter_targets( self, encounters ):
        enc_target = []
        for index, value in enumerate(encounters):
            filtered_gdf = value[0][  value[0]['VesselType'].isin(vessel_type_mapping_target.keys())  ]
            if not filtered_gdf.empty:
                enc_target.append( filtered_gdf )
        return enc_target

    def detect_encounters_h3( self, gdf, time_interval_m=240, min_distance_from_coast=10 ):
        import datetime
        import traceback
        # Defina o intervalo de tempo
        time_h = 4
        time_m = 60

        gdf.index = pd.to_datetime(gdf.index)
        min_time = gdf.index.min()
        max_time = gdf.index.max()
        start_time = min_time
        end_time = min_time + datetime.timedelta(minutes=time_interval_m)
        if end_time > max_time:
            end_time = max_time
        encounters = []

        while start_time < max_time:       
            # gdf_filtered = gdf[(gdf.index >= start_time) & (gdf.index <= end_time)]
            # AFAZER: rever se utilizarei SOG ou speed_nm
            gdf_filtered = gdf[(gdf.index >= start_time) & (gdf.index <= end_time) & (gdf.SOG <= 2.0 ) ]
            print('gdf size: ', str(len( gdf_filtered )) )
            try:
                print( "start_time=", start_time, " end_time=", end_time )
                # do filter points longer than 10km from coast
                gdf_filtered = gdf_filtered[ gdf_filtered[ "distance_to_coast" ] > min_distance_from_coast ]
                # cluster points
                print("Clustering points...")
                tmp_encounters = self.get_h3_cells_with_multiple_points( gdf_filtered )
                encounters += tmp_encounters
            except Exception as e: 
                print("Error! Size cluster: ", len( gdf_filtered ) )
                traceback.print_exc()

            start_time += datetime.timedelta(minutes=time_interval_m)
            end_time += datetime.timedelta(minutes=time_interval_m)
            if end_time > max_time:
                end_time = max_time

        return encounters

    def mmsi_to_color(self, mmsi):
        import folium
        import hashlib
        # Converta MMSI para string e obtenha seu hash
        mmsi_hash = hashlib.md5(str(mmsi).encode()).hexdigest()
        # Use os primeiros 6 caracteres do hash como código de cor hexadecimal
        color = '#' + mmsi_hash[:6]
        return color

    # plot gdf points
    def plot_gdf( self, gdf, vessel_description, m=None, color='blue' ):
        import folium

        latitude_initial = gdf.iloc[0]['LAT']
        longitude_initial = gdf.iloc[0]['LON']

        if not m:
            m = folium.Map(location=[latitude_initial, longitude_initial], zoom_start=10)
            m.add_child(MeasureControl())

        for _, row in gdf.reset_index().iterrows():

            # vessel_description = vessel_type_mapping.get( int( row['VesselType'] ), "Unknown")
            # vessel_description = vessel_type_mapping.get( int( row['VesselType'] ), "Unknown")

            # Concatenar colunas para o popup
            popup_content = f"<b>Traj_id:</b> {row.traj_id}<br><b>Timestamp:</b> {row.BaseDateTime}<br><b>VesselName:</b> {row['VesselName']}<br><b>MMSI</b>: {row['MMSI']}<br><b>LAT:</b> {row['LAT']}<br><b>LON:</b> {row['LON']}<br><b>SOG:</b> {row['SOG']}<br><b>Type:</b> {vessel_description}<br><b>COG:</b> {row['COG']}<br><b>Heading:</b> {row['Heading']}"
            # color = mmsi_to_color( row['MMSI'] )
            
            folium.CircleMarker(
                location=[row['geometry'].y, row['geometry'].x],
                popup=popup_content,
                radius=1,  # Define o tamanho do ponto
                color=color,  # Define a cor do ponto
                fill=True,
                fill_opacity=1,
            ).add_to(m)            

        return m
    

    def calculate_initial_compass_bearing(self, coord1, coord2):
        import geopandas as gpd
        import math

        """
        Calcula o rumo entre dois pontos.
        A fórmula é baseada em: http://mathforum.org/library/drmath/view/55417.html
        Retorna o rumo como um valor entre 0 e 360
        """
        lat1 = math.radians(coord1[0])
        lat2 = math.radians(coord2[0])

        diffLong = math.radians(coord2[1] - coord1[1])

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

        initial_bearing = math.atan2(x, y)

        # Convertendo de radianos para graus
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    def calc_angle_between_points( self, f_gdf ):

        angles = []
        for idx, (i, row) in enumerate(f_gdf.iterrows()):
            if idx < len(f_gdf) - 1:  # Se não for o último ponto
                current_point = row['geometry']
                next_point = f_gdf.iloc[idx+1]['geometry']
                angle = self.calculate_initial_compass_bearing((current_point.y, current_point.x), (next_point.y, next_point.x))
                angles.append(angle)
            else:
                angles.append( angles[-1] )  # para o último ponto, não há "próximo ponto"

        f_gdf['angle_to_next'] = angles
        return f_gdf

    # TODO improve triangle angle, its wrong
    def plot_arrow_gdf( self, gdf ):
        import folium

        latitude_initial = gdf.iloc[0]['LAT']
        longitude_initial = gdf.iloc[0]['LON']
        
        gdf = self.calc_angle_between_points( gdf )

        m = folium.Map(location=[latitude_initial, longitude_initial], zoom_start=10)

        for _, row in gdf.iterrows():

            vessel_description = vessel_type_mapping.get( int( row['VesselType'] ), "Unknown")

            # Concatenar colunas para o popup
            popup_content = f"<b>Timestamp:</b> {row.name}<br><b>VesselName:</b> {row['VesselName']}<br><b>MMSI</b>: {row['MMSI']}<br><b>LAT:</b> {row['LAT']}<br><b>LON:</b> {row['LON']}<br><b>SOG:</b> {row['SOG']}<br><b>Type:</b> {vessel_description}<br><b>COG:</b> {row['COG']}<br><b>Heading:</b> {row['Heading']}"
            color = self.mmsi_to_color( row['MMSI'] )
            
            # folium.Marker(
            #     location=[row['geometry'].y, row['geometry'].x],
            #     popup=popup_content
            # ).add_to(m)

            folium.RegularPolygonMarker(
                location=[row['geometry'].y, row['geometry'].x],
                popup=popup_content,
                radius=3,  # Define o tamanho do ponto
                color=color,  # Define a cor do ponto
                fill=False,
                fill_color=color,
                fill_opacity=1,
                number_of_sides=3, 
                rotation=row['angle_to_next']
            ).add_to(m)            

        return m    

    def create_linestring(self, group):        
        # Ordenar por timestamp
        group = group.sort_values(by='BaseDateTime')      
        # Se há mais de um ponto no grupo, crie uma LineString, caso contrário, retorne None
        return LineString(group.geometry.tolist()) if len(group) > 1 else None

    # plot trajectories from points
    # plot trajectories from points
    def plot_trajectory( self, gdf, vessel_description, m=None, color='blue', meta_id=0 ):
        import folium

        lines = gdf.groupby('MMSI').apply(self.create_linestring)

        # Remove possíveis None (se algum grupo tiver apenas um ponto)
        lines = lines.dropna()

        # Crie um novo GeoDataFrame com as LineStrings
        lines_gdf = gpd.GeoDataFrame(lines, columns=['geometry'], geometry='geometry')

        lines_gdf.reset_index(inplace=True)

        # start_point = Point(lines_gdf.iloc[0].geometry.coords[0])
        # m = folium.Map(location=[start_point.y, start_point.x], zoom_start=10)

        if not m:
            m = self.plot_gdf( gdf, vessel_description, color=color )
        else:
            self.plot_gdf( gdf, vessel_description, m, color=color )

        for _, row in lines_gdf.iterrows():            
            if row['geometry'].geom_type == 'LineString':
                popup_content = f"{row['MMSI']}"
                coords = list(row['geometry'].coords)
                    
                folium.PolyLine(locations=[(lat, lon) for lon, lat in coords], 
                            popup=popup_content,
                            weight=0.5,
                            color=color
                ).add_to(m)

        return m

    def plot_encounter( self, gdf1, gdf2, m=None ):
        if m is None:
            m = self.plot_trajectory( gdf1, "vessel 1", m=None, color='blue' )
            m = self.plot_trajectory( gdf2, "vessel 2", m, color='red' )
        else:
            m = self.plot_trajectory( gdf1, "vessel 1", m=m, color='blue' )
            m = self.plot_trajectory( gdf2, "vessel 2", m, color='red' )

        return m



    # TODO improve triangle angle, its wrong
    def plot_trajectory_arrow( self, gdf ):
        import folium

        lines = gdf.groupby('MMSI').apply(self.create_linestring)

        # Remove possíveis None (se algum grupo tiver apenas um ponto)
        lines = lines.dropna()

        # Crie um novo GeoDataFrame com as LineStrings
        lines_gdf = gpd.GeoDataFrame(lines, columns=['geometry'], geometry='geometry')

        lines_gdf.reset_index(inplace=True)

        # start_point = Point(lines_gdf.iloc[0].geometry.coords[0])
        # m = folium.Map(location=[start_point.y, start_point.x], zoom_start=10)

        m = self.plot_arrow_gdf( gdf )

        for _, row in lines_gdf.iterrows():            
            if row['geometry'].geom_type == 'LineString':
                popup_content = f"{row['MMSI']}"
                coords = list(row['geometry'].coords)
                color = self.mmsi_to_color( row['MMSI'] )
                    
                folium.PolyLine(locations=[(lat, lon) for lon, lat in coords], 
                            popup=popup_content,
                            color=color,  # Define a cor do ponto
                            weight=0.5
                ).add_to(m)

        return m
    
    def report_data( self, combined_gdf ):
        combined_gdf['vessel_description'] = combined_gdf['VesselType'].apply( vessel_type_mapping.get )
        combined_gdf['vessel_description'] = combined_gdf['vessel_description'].fillna('Unknow')

        h3_groups_per_description = combined_gdf.groupby('vessel_description')['h3_index'].nunique().reset_index(name='number_of_h3_groups')

        # Sort the dataframe by the number of groups for better visualization
        h3_groups_per_description = h3_groups_per_description.sort_values(by='number_of_h3_groups', ascending=False)

        return h3_groups_per_description
    
    def heat_map_gdf( self, gdf ):
        import folium
        from folium.plugins import HeatMap

        # Converta as colunas de latitude e longitude em uma lista de coordenadas
        data = gdf[['LAT', 'LON']].values.tolist()

        # Crie uma instância básica do mapa usando a média das latitudes e longitudes para centralizar
        m = folium.Map(location=[gdf['LAT'].mean(), gdf['LON'].mean()], zoom_start=10)

        # Adicione os dados ao mapa usando o plugin HeatMap
        HeatMap(data).add_to(m)
        m.save("heatmap.html")
        return m
    
    def heat_map( self, list_encounters ):
        return self.heat_map_gdf( self.get_concat_gdf( list_encounters ) )

    def get_concat_gdf( self, list_encounters, idx=0 ):
        flat_encounter = []
        for i in range(len(list_encounters)):
            for j in range(len(list_encounters[i])):
                flat_encounter.append( list_encounters[i][j][idx] )

        return pd.concat( flat_encounter, ignore_index=False)    
    
    
    def get_not_encounters( self, gdf, list_encounters, distance_from_coast, n_points=21 ):

        concat_gdf = self.get_concat_gdf(list_encounters)
        gdf_not_encounter = gdf[ gdf[ "distance_to_coast" ] > distance_from_coast ]
        gdf_not_encounter = gdf_not_encounter[ ~gdf_not_encounter.index.isin( concat_gdf.index ) ]

        trajs_no_encounter = self.create_trajectory( gdf_not_encounter )
        # get n_points points by trajectory
        t = []
        for traj in trajs_no_encounter:
            if traj.df['MMSI'].count() >= n_points: 
                t.append( traj.df[-n_points:])

        # return trajectories that's haven't encounter in this gdf
        return t
    
    # GFW criteria for detecting Loitering in Vessels
    # Single-Vessel Loitering
    # Loitering events were identified as locations where a transshipment vessel traveled 
    # at speeds of < 2 knots for at least 8 h, while at least 20 nautical miles from shore 
    # (Figure S4). To avoid false-positives that arise from vessels waiting in crowded 
    # nearshore or port locations (McCoy, 2012), loitering events were restricted to
    #  offshore regions using a distance from shore limit, as this was considered more
    #  restrictive than a distance from anchorage filter. Loitering events by their 
    # nature are more speculative (vessels may loiter for a number of reasons, 
    # especially nearshore) and consequently we only considered events where the 
    # vessels were at sea.
    # 20 nm ~ 37 km    
    def gfw_loitering_criteria( self, gdf, list_encounters, distance_from_coast, n_points=21 ):
        concat_gdf = self.get_concat_gdf(list_encounters)
        gdf_not_encounter = gdf[ gdf[ "distance_to_coast" ] > distance_from_coast ]
        gdf_not_encounter = gdf_not_encounter[ ~gdf_not_encounter.index.isin( concat_gdf.index ) ]

        trajs_no_encounter = encounter.create_trajectory( gdf_not_encounter )
        # get n_points points by trajectory
        t = []
        for traj in trajs_no_encounter:
            if traj.df['MMSI'].count() >= n_points: 
                # 20 nm ~ 37 km
                if traj.df["distance_to_coast"].mean() > 37 and traj.df["speed_nm"].mean( ) < 2 :
                    dt_begin = traj.df.index.min( )
                    dt_end = traj.df.index.max( )
                    if dt_end - dt_begin > timedelta(hours=8):
                        t.append( traj.df[-n_points:])

        return t

    
    def calc_distance_diff_nm( self, df, LAT_colLON, lon_coluna):
        """
        Calcula as diferenças de distância entre pares de pontos de LATitudLON longitude em um DataFrame.
        df: DataFrame contendo as colunas de LATitudLON longitude
        LAT_colLON: Nome da coluna de LATitudLON   lon_coluna: Nome da coluna de longitude
        Retorna uma lista com as diferenças de distância entre as linhas.
        """
        diferencas = []
        for i in range(len(df) - 1):
            ponto1 = (df[LAT_colLON].iloc[i], df[lon_coluna].iloc[i])
            ponto2 = (df[LAT_colLON].iloc[i + 1], df[lon_coluna].iloc[i + 1])
            distancia = geodesic(ponto1, ponto2).nautical
            diferencas.append(distancia)

        diferencas.insert(0, diferencas[0])
        return diferencas

    def calc_time_diff_h( self, df, coluna_tempo):
        """
        Calcula as diferenças de tempo em horas entre linhas consecutivas de um DataFrame.
        df: DataFrame contendo a coluna de tempo
        coluna_tempo: Nome da coluna de tempo
        Retorna uma lista com as diferenças de tempo em horas entre as linhas.
        """
        diferencas = []
        for i in range(len(df) - 1):
            tempo1 = pd.to_datetime(df[coluna_tempo].iloc[i])
            tempo2 = pd.to_datetime(df[coluna_tempo].iloc[i + 1])
            diferenca = (tempo2 - tempo1).total_seconds() / 3600  # Diferença em horas
            diferencas.append(diferenca)
        
        diferencas.insert(0, diferencas[0])
        return diferencas

    def angular_diff(self, direction1, direction2):
        """
        Calcula a menor diferença angular entre duas séries de direções.

        Parâmetros:
        direcao1 (pandas.Series ou array-like): Primeira série de direções (em graus).
        direcao2 (pandas.Series ou array-like): Segunda série de direções (em graus).

        Retorna:
        array-like: A menor diferença angular entre as direções.
        """
        # Converter de graus para radianos
        direction1_rad = np.radians(direction1)
        direction2_rad = np.radians(direction2)

        # Calcular a diferença angular em radianos
        difference = np.arctan2(np.sin(direction1_rad - direction2_rad), 
                                np.cos(direction1_rad - direction2_rad))

        # Converter de radianos para graus
        degrees_diff = np.degrees(difference)

        # Ajustar para que o resultado esteja entre -180 e 180 graus
        degrees_diff = (degrees_diff + 180) % 360 - 180

        # Ajustar o primeiro ponto pra ficar igual ao segundo ponto
        degrees_diff[0] = degrees_diff[1]

        return degrees_diff

    def calculate_compass_bearing(self, pointA, pointB):
        """
        Calcular o azimute entre dois pontos.
        :param pointA: Tuple com latitude e longitude do primeiro ponto (latA, lonA)
        :param pointB: Tuple com latitude e longitude do segundo ponto (latB, lonB)
        :return: Azimute em graus
        """
        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError("Only tuples are supported as arguments")

        lat1 = np.radians(pointA[0])
        lat2 = np.radians(pointB[0])

        diffLong = np.radians(pointB[1] - pointA[1])

        x = np.sin(diffLong) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diffLong))

        initial_bearing = np.arctan2(x, y)

        # Converte de radianos para graus e ajusta para 0-360°
        initial_bearing = np.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    def calculate_cog(self, df):
        """
        Calcular o COG para cada ponto de uma trajetória.
        :param df: DataFrame com colunas 'LAT' e 'LON'
        :return: DataFrame com coluna adicional 'COG'
        """
        # Verifica se as colunas 'LAT' e 'LON' existem no DataFrame
        if 'LAT' not in df.columns or 'LON' not in df.columns:
            raise KeyError("DataFrame must contain 'LAT' and 'LON' columns")

        # Certifica-se de que os índices do DataFrame são contínuos
        df = df.reset_index(drop=True)

        cogs = [np.nan]  # O primeiro ponto não tem COG

        for i in range(1, len(df)):
            pointA = (df.iloc[i-1]['LAT'], df.iloc[i-1]['LON'])
            pointB = (df.iloc[i]['LAT'], df.iloc[i]['LON'])        
            cog = self.calculate_compass_bearing(pointA, pointB)
            cogs.append(cog)

        # df['COG'] = cogs
        cogs[0] = cogs[1]
        return cogs


# %%
#######
## MAIN
########

# %%
## Execution for many AIS files
import gc
import traceback

number_ais_messages_processed = 0
time_interval_m=120 #minutes
min_distance_from_coast=18 # km
# resolution = 10 # 0.076 km
# resolution = 11 # 0.0287 km
# resolution = 12 # 0.0109 km
resolution = 13 # 0.00411 km

prefix_filename = "AIS_2015_01_"
n_files = 31
# n_files = 1

list_encounters = []
list_not_encouters = []
erro=False
for i in range(1, n_files+1):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter()
    encounter.built_trajectories_from_scratch( filename, resolution )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=time_interval_m, min_distance_from_coast=min_distance_from_coast) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, min_distance_from_coast) )

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução Janeiro finalizada com erro!" )
        erro=True
        break  

if not erro:
    msg_telegram( "Execução dos arquivos AIS Janeiro finalizada com sucesso!" )

# %%

prefix_filename = "AIS_2015_02_"
n_files = 28
# n_files = 1

erro=False
for i in range(1, n_files+1):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter()
    encounter.built_trajectories_from_scratch( filename, resolution )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=time_interval_m, min_distance_from_coast=min_distance_from_coast) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, min_distance_from_coast) )

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução Fevereiro finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Fevereiro finalizada com sucesso!" )

# %%

prefix_filename = "AIS_2015_03_"
n_files = 31
# n_files = 1

erro=False
for i in range(1, n_files+1):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter()
    encounter.built_trajectories_from_scratch( filename, resolution )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=time_interval_m, min_distance_from_coast=min_distance_from_coast) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, min_distance_from_coast) )

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução Março finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Março finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_abril/AIS_2015_04_"
n_files = 30
# n_files = 1

erro=False
for i in range(1, n_files+1):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter()
    encounter.built_trajectories_from_scratch( filename, resolution )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=time_interval_m, min_distance_from_coast=min_distance_from_coast) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, min_distance_from_coast) )

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução Abril finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Abril finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_maio/AIS_2015_05_"
n_files = 31
# n_files = 1

erro=False
for i in range(1, n_files+1):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter()
    encounter.built_trajectories_from_scratch( filename, resolution )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=time_interval_m, min_distance_from_coast=min_distance_from_coast) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, min_distance_from_coast) )

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução Maio finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Maio finalizada com sucesso!" )


# %%

prefix_filename = "dados_ais_junho/AIS_2015_06_"
n_files = 30
# n_files = 1

erro=False
for i in range(1, n_files+1):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter()
    encounter.built_trajectories_from_scratch( filename, resolution )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=time_interval_m, min_distance_from_coast=min_distance_from_coast) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, min_distance_from_coast) )

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução Junho finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Junho finalizada com sucesso!" )

# %%
# trajs1[0] encounters trajs2[0] and so on...

list1_gdfs, list2_gdfs = encounter.list_encounters_to_2gdf( list_encounters, min_rows=10 )
trajs1 = encounter.create_trajectory_21( list1_gdfs )
trajs1 = encounter.remove_impossible_trajectories( trajs1 )
trajs2 = encounter.create_trajectory_21( list2_gdfs )
trajs2 = encounter.remove_impossible_trajectories( trajs2 )

# # Test if has impossible speed and positions on trajectories
# testSpoofing1 = (list_encounters[i][j][1]['speed_nm'] > 50).any()
# testSpoofing2 = (list_encounters[i][j][2]['speed_nm'] > 50).any()
# # Drop if has impossible speeds
# if not testSpoofing1 and not testSpoofing2:


# %%
# Plot encounters for visual conference...

# id = 0
# m = encounter.plot_trajectory( trajs1.trajectories[0].df, "vessel" + str(id), m=None, color='blue' )
# m = encounter.plot_trajectory( trajs2.trajectories[0].df, "vessel" + str(id), m, color='red' )
# for i in range(1, len(trajs1.trajectories)):
#     id += 1
#     m = encounter.plot_trajectory( trajs1.trajectories[i].df, "vessel" + str(id), m, color='blue' )
#     m = encounter.plot_trajectory( trajs2.trajectories[i].df, "vessel" + str(id), m, color='red' )

# m

# %%
concat1_gdf = pd.concat( list1_gdfs )
concat2_gdf = pd.concat( list2_gdfs )

# write gdf encounter points in pickle file
encounter.write_pickle_obj( pd.concat( [concat1_gdf, concat2_gdf], ignore_index=False), 'gdf_encounters_6meses.pickle'  )

# %%

# write trajectories em pickle file
encounter.write_pickle_obj( trajs1, 'collection_trajectories1_encounters_6meses.pickle'  )
encounter.write_pickle_obj( trajs2, 'collection_trajectories2_encounters_6meses.pickle'  )

# %%
# # plot  NO ENCOUNTER trajectories
# id = 0
# m = encounter.plot_trajectory( list_not_encouters[0][0], "vessel" + str(id), m=None, color='blue' )
# for i in range(1, len(list_not_encouters[0])):
#     id += 1
#     m = encounter.plot_trajectory( list_not_encouters[0][i], "vessel" + str(id), m, color='blue' )

# m

# %% 
# transform a list of gdf not encounters points in a only one
flat_no_encounter = []
for i in range(len(list_not_encouters)):
    for j in range(len(list_not_encouters[i])):
            flat_no_encounter.append( list_not_encouters[i][j] )

concat_gdf_no_encounters = pd.concat( flat_no_encounter, ignore_index=False) 

# %%

# write gdf not encounters in pickle file
encounter.write_pickle_obj( concat_gdf_no_encounters, 'gdf_not_encounters_6meses.pickle'  )

# %%

# create moving pandas trajectories to not encounter trajectories
trajs_no_encounters = encounter.create_trajectory_21( flat_no_encounter )
trajs_no_encounters = encounter.remove_impossible_trajectories( trajs_no_encounters )

# write trajectories in pickle file
encounter.write_pickle_obj( trajs_no_encounters, 'collection_trajectories_no_encounters_6meses.pickle'  )

# %%
print("encounters trajs = " + str(len(trajs1)+len(trajs2)))
print("normal trajs = " + str(len(trajs_no_encounters)))
print("Mensagens AIS processadas: ", number_ais_messages_processed )
msg_telegram( "PROCESSAMENTO DOS ARQUIVOS AIS FINALIZADOS! MSG AIS processadas: " + str(number_ais_messages_processed) )

# %%



# %%
# #####################################
#
# Tests codes
# ######################################


# m = encounter.heat_map( list_encounters  )
# m.save("encounter_heatmap_1.html")

# # %%

# # draw encouter trajectories
# i_file = 1
# i_cluster = 95
# encounter.plot_trajectory( pd.concat([list_encounters[i_file][i_cluster][1], list_encounters[i_file][i_cluster][2]], ignore_index=False) )

# # %%
# # print vessel type by groups
# encounter.report_data(list_encounters[1] )

# encounter.plot_gdf( list_encounters[500][0] )

# encounter.plot_trajectory_arrow( 
#     encounter.get_gdf( )[ (encounter.get_gdf( )["MMSI"] == 367587920) |  
#                           (encounter.get_gdf( )["MMSI"] == 371765000) ] 
#     )

# # %%

# encounter.plot_gdf( encounter.get_gdf( )[:10000] )

# def report_data( combined_gdf ):
#     combined_gdf['vessel_description'] = combined_gdf['VesselType'].apply( vessel_type_mapping.get )
#     combined_gdf['vessel_description'] = combined_gdf['vessel_description'].fillna('Unknow')

#     h3_groups_per_description = combined_gdf.groupby('vessel_description')['h3_index'].nunique().reset_index(name='number_of_h3_groups')

#     # Sort the dataframe by the number of groups for better visualization
#     h3_groups_per_description = h3_groups_per_description.sort_values(by='number_of_h3_groups', ascending=False)

#     return h3_groups_per_description

# encounter.plot_trajectory( 
#     encounter.get_gdf( )[ (encounter.get_gdf( )["MMSI"] == 338433000) ] 
#     )

# # %%

# def heat_map( gdf ):
#     import folium
#     from folium.plugins import HeatMap


#     # Converta as colunas de latitude e longitude em uma lista de coordenadas
#     data = gdf[['LAT', 'LON']].values.tolist()

#     # Crie uma instância básica do mapa usando a média das latitudes e longitudes para centralizar
#     m = folium.Map(location=[gdf['LAT'].mean(), gdf['LON'].mean()], zoom_start=10)
#     # m = encounter.plot_gdf( gdf )

#     # Adicione os dados ao mapa usando o plugin HeatMap
#     HeatMap(data).add_to(m)
#     m.save("heatmap.html")
#     return m
# # %%

# def get_concat_gdf( list_encounters ):
#     flat_encounter = []
#     for i in range(len(list_encounters)):
#         for j in range(len(list_encounters[i])):
#             flat_encounter.append( list_encounters[i][j][0] )

#     return pd.concat( flat_encounter, ignore_index=False)

# # %%

# m = encounter.plot_gdf( filtered_gdf[ filtered_gdf["MMSI"].isin( unique_ships) ] )
# m.save("loitering.html")
# m

# # %%

# m = heat_map( filtered_gdf[ filtered_gdf["MMSI"].isin( unique_ships) ] )
# m.save("loitering.html")
# m

# # %%


# # %%

# # plot trajectories from points
# def plot_trajectory_lines( trajs ):
#     import folium

#     # lines_gdf.reset_index(inplace=True)

#     # start_point = Point(lines_gdf.iloc[0].geometry.coords[0])
#     # m = folium.Map(location=[start_point.y, start_point.x], zoom_start=10)
#     m = folium.Map(location=[27.29199, -90.96816], zoom_start=10)

#     # m = encounter.plot_gdf( lines_gdf )

#     for row in trajs:            
#         popup_content = f"{row.df['MMSI'].iloc[0]}"
#         coords = list(row.to_linestring().coords)
#         color = encounter.mmsi_to_color( row.df['MMSI'].iloc[0] )
            
#         folium.PolyLine(locations=[(lat, lon) for lon, lat in coords], 
#                     popup=popup_content,
#                     color=color,  # Define a cor do ponto
#                     weight=0.5
#         ).add_to(m)

#     return m

# # %%
