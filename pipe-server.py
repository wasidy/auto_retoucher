# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 20:18:08 2024

@author: Pussy
"""
import socket
import pickle
import numpy as np
from scripts.pipelines import SDXLPipeLine


class MyData:
    def __init__(self, prompt, negative_prompt, steps, strength, image):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.strength = strength
        self.image = image


if __name__ == '__main__':
    print('Loading model')
    sdxl = SDXLPipeLine('models/juggernautXL_v7FP16VAEFix.safetensors')
    print('Model loaded')
    sd_socket = socket.socket()
    ip = 'localhost'
    port = 5080

    sd_socket.bind((ip, port))

    while True:
        print('Waiting for connection')
        data_for_gen = MyData(None, None, None, None, None)
        recieved_data = bytearray()
        sd_socket.listen(10)
        conn, addr = sd_socket.accept()
        print(f'Connected: {addr}')

        while True:
            d = conn.recv(1024)
            recieved_data = recieved_data + d

            if not d:
                break

        data = pickle.loads(recieved_data, encoding='bytes')

        image = np.array(data.image, dtype=np.uint8)
        print(f'Images size: {image.shape}\nStrength: {data.strength}\nPrompt: {data.prompt}\nNegative_prompt: {data.negative_prompt}\nSteps: {data.steps}')
        images = []
        images.append(image)

        result = sdxl.generate_batch_images(images,
                                            data.strength,
                                            data.prompt,
                                            data.negative_prompt,
                                            data.steps)

        print('Generated')
        send_data = pickle.dumps(result)
        conn.sendall(send_data)
        conn.close()
