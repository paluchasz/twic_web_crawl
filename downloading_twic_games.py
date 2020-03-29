#!/usr/bin/python3

import os
import zipfile
from pathlib import Path

import pycurl

# path to the folder where we want to save the pgn
unzipped_location_path = Path("/home/paluchasz/Desktop/chessbase_stuff/downloading_twic")
# reason for using Path module is that now this code can be run on Windows as well where the paths use \ instead


def download_pgn(start, end):
    """
    This function downloads zip files from TWIC website and then unzips them saving the pgn
    file and deleting the zip.
    integer start: the first twic file you want to download
    integer end: the final twic file you want to download
    return True on success and False on failure
    """
    # Initialise Curl object
    c = pycurl.Curl()

    # change the range depending on which twic files I need to download
    for i in range(start, end + 1):

        try:
            # notice the url is without 'www', this is because the url has changed, I was getting a 301 error previously
            url = "https://theweekinchess.com/zips/twic" + str(i) + "g.zip"
            name_of_zip = "twic" + str(i) + ".zip"
            save_location_path = unzipped_location_path / name_of_zip

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

            return True

        except zipfile.BadZipFile as error:
            os.remove(save_location_path)
            print("BadZipFile error. This probably means that the zip file hasn't been uploaded to"
            " the website yet and so we cannot unzip this non-existent file")
            return False

    # close Curl object
    c.close()


def check_if_download_needed():
    """
    This function goes through a text file "twic_number_to_be_downloaded" line by line and then
    calls the other function to download the correct file. If this file has been downloaded successfuly
    we delete this number so that we don't download the same file next time. Finally, we write to the
    text file the next number which should be downloaded next time. We run this program weekly with
    crontab in Bash
    """
    path1 = unzipped_location_path / "twic_number_to_be_downloaded_next.txt"
    # path1 is the path to the text document which whill be updated as files are downloaded weekly

    with open(path1, 'r+') as file: # r+ seems to be the read/write mode

        lines = file.readlines() # read the document into variable lines
        print(lines) # has the form ['1268\n', '1269\n', ...] however sometimes the
        # \n is not present and then it starts writing to the same line so need to fix.

        file.seek(0) # start at the beginning of file
        for line in lines:
            num = line

            if download_pgn(int(num), int(num)):
                pass
            else:
                file.write(num)
                # if the pgn wasn't downloaded then we want to write the line again so that
                # once the file gets truncated below we still have this pgn to be downloaded
                # next time

        # truncate() deletes all lines in the file (except what we "wrote" in the for loop)
        file.truncate()

        # If the file who's number was last in the list was downloaded, then we want to add
        # the next number to the list so next time the program is executed something will be
        # tried to be downloaded and the file isn't just empty forever.
        next_zip = int(num) + 1

        file.write(str(next_zip) + '\n')
        # Need the \n to go to new line, as an example say we wrote manually 1268 in the text file, what
        # lines = file.readlines() gives us is ['1268\n']. If the program is not executed and downloading
        # this file fails on the next run of the program lines = ['1268\n', '1269' ]. This 1269 would
        # be on a new line, however because there is no \n the next time we execute it we would get
        # a nonsense line '12691270'


if __name__ == '__main__':
    check_if_download_needed()
