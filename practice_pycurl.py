import pycurl
from io import BytesIO
import os
import zipfile

c = pycurl.Curl()

for i in range(1140, 1251):
    url = "http://theweekinchess.com/zips/twic" + str(i) + "c6.zip"
    path_to_save_location = "downloading_twic/twic" + str(i) + ".zip"
    unzipped_location = "downloading_twic"

    with open(path_to_save_location, 'wb') as f:

        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, f)
        c.perform()

    with zipfile.ZipFile(path_to_save_location, 'r') as myzip:
        myzip.extractall(unzipped_location)

    os.remove(path_to_save_location)

c.close()
