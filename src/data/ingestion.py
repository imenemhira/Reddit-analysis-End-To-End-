from pathlib import Path 
import pandas as pd 
import logging 


logging.basicConfig(
    level= logging.INFO,
    format= '%(asctime)s-%(levelname)s-%(message)s'
)
logger =logging.getLogger(__name__)

def load_data(file_path : Path)-> pd.DataFrame :
    logger.info('Starting data ingestion')
    if not file_path.exists():
        raise FileNotFoundError('file not found')
    logger.info('loading dataset',file_path)
    df=pd.read_csv(file_path)
    logger.info('Dataset loaded succefully')
    logger.info('Dataset shape : %s',df.shape)
    return df