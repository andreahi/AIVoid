import socket, pickle

import time


def send_data(arr1):
    HOST = 'localhost'
    PORT1 = 50007

    for i in range(100):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((HOST, PORT1))
            data_string = pickle.dumps(arr1)
            s.send(data_string)
            break
        except Exception as e:
            print "connection failed ", str(i)
            time.sleep(1)
            pass
        finally:
            s.close()



