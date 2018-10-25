import pycurl
from io import BytesIO
import os
import zipfile

# def download_file(url):
#     buffer = BytesIO()
#     c = pycurl.Curl()
#     c.setopt(c.URL, url)
#     c.setopt(c.WRITEDATA, buffer)
#     c.perform()
#     c.close()
#
#     body = buffer.getvalue()
#     # Body is a byte string.
#     # We have to know the encoding in order to print it to a text file
#     # such as standard output.
#     return body.decode('iso-8859-1')

for i in range(1140, 1251):
    url = "http://theweekinchess.com/zips/twic" + str(i) + "c6.zip"
    path_to_save_location = "downloading_twic/twic" + str(i) + ".zip"

    with open(path_to_save_location, 'wb') as f:
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, f)
        c.perform()
        c.close()


# if __name__ == '__main__':
#     file = download_file('http://www.theweekinchess.com/zips/twic920c6.zip')
#     print(file)
#
