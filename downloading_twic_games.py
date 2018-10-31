#!/usr/bin/python3

import pycurl
from io import BytesIO
import os
import zipfile
from pathlib import Path
import fileinput

# This function downloads zip files from TWIC website and then unzips them saving the pgn
# file and deleting the zip.
def downloadPGNfile(start, end):
    # Initialise Curl object
    c = pycurl.Curl()

    # change the range depending on which twic files I need to download
    for i in range(start, end + 1):

        try:
            # notice the url is without 'www', this is beacuse the url has changed, I was getting a 301 error previously
            url = "http://theweekinchess.com/zips/twic" + str(i) + "g.zip"
            save_location_path = Path("/home/paluchasz/Desktop/chessbase_stuff/downloading_twic/twic" + str(i) + ".zip")
            unzipped_location_path = Path("/home/paluchasz/Desktop/chessbase_stuff/downloading_twic")
            # reason for using Path module is that now this code can be run on Windows as well where the paths use \ instead

            # downloading the file - taken from pycurl documentation
            with open(save_location_path, 'wb') as f:

                c.setopt(c.URL, url)
                c.setopt(c.WRITEDATA, f)
                c.perform()

            # unziping the file - first line reads it (hence the 'r'), second line extracts it
            with zipfile.ZipFile(save_location_path, 'r') as myzip:
                myzip.extractall(unzipped_location_path)

            # removing old zip file
            os.remove(save_location_path)

            return "success"

        except zipfile.BadZipFile as error:
            os.remove(save_location_path)
            return "BadZipFile error. This probably means that the zip file hasn't been uploaded to"
            " the website yet and so we cannot unzip this non-existent file"

    # close Curl object
    c.close()

# This function goes through a text file "twic_number_to_be_downloaded" line by line and then
# calls the other function to download the correct file. If this file has been downloaded successfuly
# we delete this number so that we don't download the same file next time. Finally, we write to the
# text file the next number which should be downloaded next time. We run this program weekly with
# crontab in Bash
def checkAndPerformDownloadIfNeccessary():
    path1 = Path("/home/paluchasz/Desktop/chessbase_stuff/downloading_twic/twic_number_to_be_downloaded_next.txt")
    # path1 is the path to the text document which whill be updated as files are downloaded weekly

    with open(path1, 'r+') as file: # r+ seems to be the read/write mode

        lines = file.readlines() # read the document into variable lines

        file.seek(0) # start at the beginning of file
        for line in lines:
            num = line
            if downloadPGNfile(int(num), int(num)) == "success":
                pass
            else:
                file.write(str(num))
                # if the pgn wasn't downloaded then we want to write the line again so that
                # once the file gets truncated below we still have this pgn to be downloaded
                # next time

        # truncate() deletes all lines in the file (except what we "wrote" in the for loop)
        file.truncate()

        # If the file who's number was last in the list was downloaded, then we want to add
        # the next number to the list so next time the program is executed something will be
        # tried to be downloaded and the file isn't just empty forever.
        next_zip = int(num) + 1
        file.write(str(next_zip))





checkAndPerformDownloadIfNeccessary()
