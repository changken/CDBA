import os
import sys
import cv2
import romp
import bev
import json
import torch
import socket
import warnings
import numpy as np
from base64 import b64decode
from multiprocessing import Process, Value, Manager, set_start_method
set_start_method('spawn', force=True)
model_path = os.path.join(os.path.expanduser("~"), '.romp')
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def packData(data):
    length = str(len(data))
    length = length.zfill(8)
    return length.encode('utf-8') + data
    
def getAxisAngle(poses_list):
    AxisAngles = []
    for poses in poses_list:
        AxisAngle = []
        for i in range(0, len(poses), 3):
            theta = np.sqrt(poses[i]**2 + poses[i+1]**2 + poses[i+2]**2)
            w = poses[i:i+3]/(theta+1e-6)
            for key in w:
                AxisAngle.append(key)
            AxisAngle.append(theta)
        AxisAngles.append(AxisAngle)
    return AxisAngles

def convert_cam_to_3d_trans(cams, weight=2.):
    (s, tx, ty) = cams[:, 0], cams[:, 1], cams[:, 2]
    depth, dx, dy = 1./s, tx/s, ty/s
    trans3d = np.stack([dx, dy, depth], 1)*weight
    return trans3d

def decodeImg(img_string):
    img_bytes = b64decode(img_string.encode('utf-8'))
    img_encode = np.fromstring(img_bytes, np.uint8)
    img_array = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)
    return img_array

# 過濾多餘的人體姿態 或 跳號
def take_two(e):
    return e[1]

def reordering(source_track_ids, source_data, target_track_ids):
    t = sorted(source_track_ids)
    # 如果沒有排序 # 就排序
    if t != source_track_ids or len(source_track_ids) != len(target_track_ids) or t != target_track_ids:
        # 如果排序後跟target_track_ids不一樣
        #if t != target_track_ids:
        modify_arr = [(k, v) for (k, v) in enumerate(source_track_ids)]
        modify_arr = sorted(modify_arr, key=take_two)
        print(modify_arr)


        if len(source_track_ids) < len(target_track_ids):
            target_data = [None] * len(source_track_ids)
            for i in range(len(source_track_ids)):
                target_data[i] = source_data[modify_arr[i][0]]
        else: 
            target_data = [None] * len(target_track_ids)
            for i in range(len(target_track_ids)):
                target_data[i] = source_data[modify_arr[i][0]]

        return target_data
    
    return source_data

def run_romp(client, receiver, sender, run, mode, gpu, num_of_person):
    # settings = bev.main.default_settings
    settings = romp.main.default_settings
    settings.webcam_id = 0
    settings.mode = mode
    # settings.smooth_coeff = 1
    settings.temporal_optimize = True
    settings.smooth_coeff = 0.2

    if gpu == 'True' and torch.cuda.is_available():
        print(gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        settings.gpu = 0
    else:
        settings.onnx = True
        settings.gpu = -1
    # settings.show_largest = True
    settings.show_largest = False
    # romp_model = bev.BEV(settings)
    romp_model = romp.ROMP(settings)

    while run.value:
        if not receiver.empty():
            try:
                image = np.array(receiver.get(), dtype=np.uint8)
            except:
                continue

            if mode == 'webcam':
                while not receiver.empty():
                    receiver.get()  # clean the redundant input frames that is not able to process
            outputs_all = romp_model(image)

            if outputs_all is None:
                client.send('none'.encode('utf-8'))
                continue
            
            poses = getAxisAngle(outputs_all['smpl_thetas'])

            target_track_ids = [i for i in range(1,num_of_person+1)]

            poses_arr = reordering(outputs_all['track_ids'].tolist(), poses, target_track_ids)

            '''
            track_ids = outputs_all['track_ids']
            poses_arr = [[] for _ in range(len(track_ids))]
            for i, v in enumerate(track_ids):
                poses_arr[v-1] = poses[i]
            '''
            #poses = np.array(poses)
            if 'cam_trans' not in outputs_all:
                trans = convert_cam_to_3d_trans(outputs_all['cam']).tolist()
            else:
                trans = outputs_all['cam_trans'].tolist()
            #trans = np.array(trans)

            outputs = {'poses': poses_arr, 'trans': trans, 'dims': np.array(poses_arr).shape[0]}
            print("original poses", np.array(poses).shape)
            print("poses_arr", np.array(poses_arr).shape)
            print("trans", np.array(trans).shape)
            print("original dims", np.array(poses).shape[0])
            print("dims", np.array(poses_arr).shape[0])

            try:
                sender.put(outputs)
            except:
                continue

    print('romp finished')

def send_poses(client, sender, run):
    while run.value:
        if not sender.empty():
            client.send(json.dumps(sender.get(),cls = NumpyEncoder).encode('utf-8'))
    print('send finished')
        
def recv_data(client, address):
    receiver = Manager().Queue()
    sender = Manager().Queue()
    run = Value('i', 1)

    headerSize = 8
    dataBuffer = b''
    
    while True:
        data = client.recv(32768) # enlarge
        if data:
            dataBuffer += data
            while True:
                if len(dataBuffer) < headerSize:
                    break
                bodySize = int(dataBuffer[:headerSize])
                if len(dataBuffer) < headerSize + bodySize :
                    break

                body = dataBuffer[headerSize:headerSize + bodySize]
                body = json.loads(body.decode('utf-8'))
                type = body['type']

                if type == 'init':
                    gpu = body['gpu']
                    mode = body['mode']
                    num_of_person = body['num_of_person']
                    # Create a new process for ROMP to process the image
                    p = Process(target=run_romp, args=(client, receiver, sender, run, mode, gpu, num_of_person))
                    print('Started ROMP process')
                    p.start()
                    # Create a new process to send the poses to the client
                    p = Process(target=send_poses, args=(client, sender, run))
                    print('Started sending process')
                    p.start()

                if type == 'image':
                    image = decodeImg(body['content'])
                    receiver.put(image)

                dataBuffer = dataBuffer[headerSize + bodySize:]

                if type == 'done':
                    run.value = 0
                    p.join()
                    print('Transfer done. Connection closed by', address)
                    break
            
        if type == 'done':
            break    
          
if __name__ == '__main__':
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', int(sys.argv[1])))
    server.listen(5)
    print('Waiting for connection on port {}'.format(sys.argv[1]))

    while True:
        client, address = server.accept()
        # Create a new process to handle the connection
        p = Process(target=recv_data, args=(client, address))
        p.start()