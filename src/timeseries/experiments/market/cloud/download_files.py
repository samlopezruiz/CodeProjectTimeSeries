import os

from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(
    os.path.dirname('D:\MEGA\ComputerScience\Machine_Learning\ml_cred'), \
    'timeseriesprediction-331120-1af388f95e96.json')


def download_google_storage_file(cloud_filepath):
    # Initialise a client
    storage_client = storage.Client("TimeSeriesPrediction")
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket('market_data_snp')
    # Create a blob object from the filepath
    blob = bucket.blob(cloud_filepath)
    # Download the file to a destination
    filepath = os.path.join('data', cloud_filepath)
    base_path = os.path.dirname(filepath)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    blob.download_to_filename(filepath)


if __name__ == '__main__':
    download_google_storage_file("regime/regime_ESc_r_ESc_macd_T10Y2Y_VIX.z")
    download_google_storage_file('split/split_ES_minute_60T_dwn_smpl_2015-01_to_2021-06_g12week_r25.z')
