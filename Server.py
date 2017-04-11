import os
import pickle
import random
import socket
import sys
import hashlib

HOST = 'localhost'
PORT = 50007


data = []
labels = []
save_size = 100000


def getHash(filename):
    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            h_data = f.read(BUF_SIZE)
            if not h_data:
                break
            sha1.update(h_data)
    return "{0}".format(sha1.hexdigest())


def saveDataAndLabels(sub_data, sub_labels):
    id = str(random.randint(0, 100000))
    data_file_name = 'tmp/data_filename' + id + '.pickle'
    with open(data_file_name, 'wb') as handle:
        pickle.dump(sub_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    labels_file_name = 'tmp/labels_filename' + id + '.pickle'
    with open(labels_file_name, 'wb') as handle:
        pickle.dump(sub_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    hash_data = getHash(data_file_name)
    hash_labels = getHash(labels_file_name)
    os.rename(data_file_name, 'trainingdata/data_' + hash_data + '.pickle')
    os.rename(labels_file_name, 'trainingdata/labels_' + hash_labels + '.pickle')


def runServer():

    while 1:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen(1)
        conn, addr = s.accept()
        print 'Connected by', addr
        all =[]
        while 1:
            recv = conn.recv(4096)
            if not recv:
                break
            all.append(recv)
        data_arr = pickle.loads(''.join(all))
        #print "data: ", data_arr[0]
        #print "labels: ", data_arr[1]
        data = data + data_arr[0]
        labels = labels + data_arr[1]
        conn.close()
        s.close()

        print len(data)
        print len(labels)
        while len(labels) > save_size:
            sub_labels = labels[:save_size]
            sub_data = data[:save_size]

            saveDataAndLabels(sub_labels, sub_data)

            del data[:save_size]
            del labels[:save_size]

