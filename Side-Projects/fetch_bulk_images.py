#!/usr/bin/python

# get_images.py is a script to mass download images from a website. 
# Used for training a Haar classifier for image recognition
# By Anton

import urllib2

user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
header = { 'User-Agent' : user_agent }

for num in range(106000, 150000):
    url = "https://images.pexels.com/photos/%s/pexels-photo-%s.jpeg?h=350&auto=compress" % (str(num), str(num))
    print url
    req = urllib2.Request(url, headers=header)
    filename = "stock%s.jpeg" % str(num)

    # Try for a 202, and count the 404s so we can go to the next batch of pictures
    try:
        u = urllib2.urlopen(req)
    except urllib2.URLError, e:
        if e.code == 404:
            print "No file detected at URL..."
    else:
        # Save the file if it is a 202
        f = open(filename, 'wb')
        meta = u.info()
        if meta:
            try:
                file_size = int(u.headers['content-length'])
                print "Downloading: %s Bytes: %s" % (filename, file_size)

                file_size_dl = 0
                block_sz = 8192
                while True:
                    buffer = u.read(block_sz)
                    if not buffer:
                        break

                    file_size_dl += len(buffer)
                    f.write(buffer)
                    status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                    status = status + chr(8)*(len(status)+1)
                    print status,

                f.close()
                print "W00tW00t!! Another photo you say!!!"
                counter+=1
            except:
                print 'Caught an error...'
                continue
