import pycurl
from io import BytesIO
import os
import zipfile
from pathlib import Path

# Initialise Curl object
c = pycurl.Curl()

# change the range depending on which twic files I need to download
for i in range(1250, 1251):
    # notice the url is without 'www', this is beacuse the url has changed, I was getting a 301 error previously
    url = "http://theweekinchess.com/zips/twic" + str(i) + "g.zip"
    save_location_path = Path("/home/paluchasz/Desktop/chess_base_stuff/downloading_twic/twic" + str(i) + ".zip")
    unzipped_location_path = Path("/home/paluchasz/Desktop/chess_base_stuff/downloading_twic")
    # reason for using Path module is that now this code can be run on Windows as well where the paths use \ instead

    # downloading the file
    with open(save_location_path, 'wb') as f:

        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, f)
        c.perform()

    # unziping the file
    with zipfile.ZipFile(save_location_path, 'r') as myzip:
        myzip.extractall(unzipped_location_path)

    # removing old zip file
    os.remove(save_location_path)

c.close()
