#!/usr/bin/python3
"""Run this script weekly with crontab in Bash."""

import json
import zipfile
from pathlib import Path

import pycurl

# path to the folder where we want to save the pgn
unzipped_location_path = Path("/home/paluchasz/Desktop/chessbase_stuff/downloading_twic")
# reason for using Path module is that now this code can be run on Windows as well where the paths use \ instead


def download_pgn(ids_list=None, start=None, end=None):
    """
    This function downloads zip files from TWIC website and then unzips them saving the pgn file and deleting the zip.
    Two possible inputs:
    Either pass in a list of integers corresponding to all the files you want to download e.g. [1200, 1201, 1210] OR
    pass in an integer start AND end corresponding to the first and last twic file you want to download
    Output: List of integers corresponding to files that failed to download
    """
    if start and end:
        ids_list = range(start, end + 1)

    # Initialise Curl object
    c = pycurl.Curl()

    undownloaded_files = []
    for i in ids_list:
        # notice the url is without 'www', this is because the url has changed, I was getting a 301 error previously
        url = "https://theweekinchess.com/zips/twic" + str(i) + "g.zip"
        name_of_zip = "twic" + str(i) + ".zip"
        save_location_path = unzipped_location_path / name_of_zip

        # downloading the file - taken from pycurl documentation
        with open(save_location_path, 'wb') as f:
            c.setopt(c.URL, url)
            c.setopt(c.WRITEDATA, f)
            c.perform()

        try:
            # unzipping the file - first line reads it (hence the 'r'), second line extracts it
            with zipfile.ZipFile(save_location_path, 'r') as myzip:
                myzip.extractall(unzipped_location_path)
        except zipfile.BadZipFile:
            print("BadZipFile error. Cannot read twic{}. This probably means that the zip file hasn't been uploaded to "
                  "the website yet or the url changed".format(i))
            undownloaded_files.append(i)

        # removing old zip file
        Path.unlink(save_location_path)

    # close Curl object
    c.close()

    return undownloaded_files


def check_if_download_needed():
    """
    This function goes through a json file "twic_number_to_be_downloaded" which contains a list of files to be
    downloaded. It leaves any ids that failed to download and also adds an integer corresponding to next weeks file id.
    """
    path1 = unzipped_location_path / "twic_number_to_be_downloaded_next.json"
    # path1 is the path to the json file which will be updated as files are downloaded weekly

    with open(path1, 'r') as file:
        file_ids = json.load(file)

    undownloaded_ids = download_pgn(ids_list=file_ids)
    undownloaded_ids.append(max(file_ids) + 1)  # Add next weeks file id

    with open(path1, 'w') as file:
        json.dump(undownloaded_ids, file)


if __name__ == '__main__':
    # download_pgn(start=1302, end=1304)  # example usage of first function
    check_if_download_needed()
