Downloading TWIC games 

The week in chess website uploads weekly chess games files which I should be adding to my Chessbase Twic database so I have the latest games available. Unfortunately, I haven't done this for about two years and now I have to download more than 100 zip files. 

Downloading and unzipping:

I managed to do this using the pycurl library. The urls have a common pattern with only the number increasing by 1 each week so I managed to do this using a for loop. Importantly, the url doesn't contain "www", when I was putting this in I was getting a 301 (moved permanently) error that the url has been changed. So the url looks something like "http://theweekinchess.com/zips/twic1248g.zip" , I want the g.zip ending as I want a pgn file (reason later). 
With the previous version when I used the command $ time python3 practice_pycurl.py it took over five minutes to download all the zip files. Now I changed the code, by putting c = pycurl.Curl() and c.close() outside the for loop I made sure that the Curl object is only initialised and closed once thus halving the time to two minutes thirty seconds.
Finally, used the zip library to unzip each file and the os library to delete the old remaining zip file.

Reason for downloading pgn:

If I had downloaded the cbv chessbase files instead I did not see a way to combine them and thus I would have to append them to my Twic database individually. With the pgn files I can first do 	      $ cat * >twic1143-1250.pgn do merge all the files into one (if I want to merge just two files say then I can do $ cat file1 file2 >file1-2). Now, I wanted to remove all files except this one. To do this I need to first enable pattern-list option with $ shopt -s extglob and then use $ rm -v!(“filename”) command. 

Notes:

Documentation for pycurl: http://pycurl.io/docs/latest/quickstart.html

List of http status codes: https://en.wikipedia.org/wiki/List_of_HTTP_status_codes

Curl Object: http://pycurl.io/docs/latest/curlobject.html

Documentation for zipfile: https://docs.python.org/3/library/zipfile.html#zipfile-objects

Automating the process:

Firstly, I updated my script. The idea is to have a separate text document with numbers which represent which twic file I need to download. So for each number/line in document the function which downloads pgn will be called and after this was successful the number will be deleted from the text file. Finally, we write a new number for the file that needs to be downloaded next week. It took me ages to work out how I can read and write in a text file...

How do we know whether download ‘was successful’? - Well, in my function to download PGN I added an exception. This would catch a BadZipFile error if the program tried to download a file which was not uploaded to the website yet. I am assuming more things can go wrong but I guess I will find out later?

How did I set up to execute it weekly? - I wrote a cronjob for this. Have to write $ crontab -e (use -l if you want to see a list of cron jobs) in terminal and then I used: 
30 12 * * 2 /home/paluchasz/My_projects/downloading_twic_games.py. 
I also added a shebang line in Python which I am not sure if it is necessary: #!/usr/bin/python3
