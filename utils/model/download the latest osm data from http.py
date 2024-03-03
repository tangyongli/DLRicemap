# download the latest osm data from https://www.openstreetmap.org/,location is defined by a bounding box
# the data is saved as a .osm file
# usage: python3 download_osm.py
# the bounding box is defined in the code
# the data is saved in the same directory as the script
# the file name is defined by the bounding box
<<<<<<<<<<<<<<  ✨ Codeium Command ⭐ >>>>>>>>>>>>>>>>
import requests

# Define the bounding box (south latitude, west longitude, north latitude, east longitude)
bounding_box = (50.7, 7.1, 50.8, 7.2)

def download_osm_data(bbox):
    """Download the latest OSM data within a bounding box and save as a .osm file."""
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    response = requests.get(url)
    if response.status_code == 200:
        file_name = f"osm_data_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.osm"
        with open(file_name, "wb") as file:
            file.write(response.content)
        print(f"Data successfully downloaded and saved as {file_name}")
    else:
        print(f"Failed to download data: {response.status_code}")

# Usage
if __name__ == "__main__":
    download_osm_data(bounding_box)
<<<<<<<  aa130982-8c38-48d3-b7cf-e018d8bf2d9a  >>>>>>>