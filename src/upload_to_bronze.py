import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Load credentials from .env — never hardcode them here
load_dotenv()

account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
container_name = os.getenv("AZURE_CONTAINER_NAME")


# Connect to the storage account

blob_service_client = BlobServiceClient(
    account_url=f"https://{account_name}.blob.core.windows.net",
    credential=account_key
)

# Get a client for the file system (container) where you want to upload files
container_client = blob_service_client.get_container_client(container_name)

def upload_file(local_path, remote_path):
    with open(local_path, "rb") as f:
        data = f.read()
    blob_client = container_client.get_blob_client(remote_path)
    blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded {local_path} -> {remote_path}")

if __name__ == "__main__":
    upload_file("data/raw/diabetic_data.csv", "bronze/diabetic_data.csv")
    upload_file("data/raw/IDS_mapping.csv", "bronze/IDS_mapping.csv")