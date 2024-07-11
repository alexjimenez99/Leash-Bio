
import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session                 import s3_input, Session
import io

from zipfile import ZipFile


# Retrieve Files From S3 Bucket
def load_s3_files(bucket_name  = 'leashbio-kaggle',
                  key          = 'leash-BELKA.zip',
                  batch_data   = True):
    
    session     = boto3.Session()
    s3          = session.client('s3')
    obj         = s3.get_object(Bucket=bucket_name, Key=key)    
    data        = obj['Body'].read()

    stream      = obj['Body']
    # Load the data into a bytes buffer
    data_stream = io.BytesIO(data)
    
    
    chunk_size  = 10 * 1024 * 1024
    max_read    = 5 * 1024 * 1024 * 1024  # 5 GB
    buffer      = io.BytesIO()
    
    dataframes  = {}
    # The 'data_stream' is our zip file loaded into memory
    with ZipFile(data_stream, 'r') as zip_ref:
        # List all files contained in the zip
        list_of_files = zip_ref.namelist()
        print("Files in zip:", list_of_files)
        
        total_read = 0
        # Optionally, process each file within the zip
        for file_name in list_of_files:
            # Open the file
            
            if file_name.endswith('.parquet'):
                with zip_ref.open(file_name) as file:
                     # Read the stream in chunks until we reach approximately 5 GB
                    print(file_name)
                    # If you need to process a parquet file, load it into pandas (example)
                    if batch_data:
                        batch_size = 100  # Adjust the batch size as needed

                        # Process each batch
                        for batch in read_parquet_in_batches(file_path, batch_size):
                            df = batch.to_pandas()
                            # Process your DataFrame here
                            
                        
                    else:
                        df    = pd.read_parquet(file)
                        dataframes[file_name] = df
                    
                    
                        
            else:
                pass
                    # For other file types, read the contents directly
#                     content = file.read()
#                     print(f"Contents of {file_name}:", content[:100])  # Show first 100 characters


    return dataframes

# s3.meta.client.upload_file('leash-BELKA.zip', bucket_name, 'leash-BELKA.zip')

# Load with m5.12xlarge instance to load all the data into memory
# dataframes = load_s3_files()


