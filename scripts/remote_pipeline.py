# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 20:31:10 2024

@author: Wasidy
"""

import socket
import pickle
import numpy as np


class MyData:
    def __init__(self, prompt=None, negative_prompt=None, steps=None, strength=None, image=None):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.strength = strength
        self.image = image


class RemoteSDXLPipeLine():
    ''' This class for sending image to server. For debugging '''

    def __init__(self, ip='localhost', port=5080):
        self.ip = ip
        self.port = port

    def generate_single_image(self, image, strength, prompt='', negative_prompt='', steps=50):

        return self.generate_batch_images([image], strength, prompt,
                                          negative_prompt, steps, batch_size=1)[0]

    def send_and_recieve(self, img, strength, prompt,
                         negative_prompt, steps):

        data = MyData(prompt, negative_prompt, steps, strength, img)

        pickled_data = pickle.dumps(data)
        recieved_data = bytearray()

        sock = socket.socket()
        sock.connect((self.ip, self.port))

        # Sending image and data for generation
        sock.sendall(pickled_data)
        sock.shutdown(1)

        # Recieving generated data
        while True:
            d = sock.recv(1024)
            recieved_data = recieved_data + d

            if not d:
                break

        image = pickle.loads(recieved_data, encoding='bytes')
        sock.shutdown(1)
        sock.close()

        return image

    def generate_batch_images(self, images, strength, prompt='',
                              negative_prompt='', steps=50, batch_size=1):
        ''' Generating images from list PIL or numpy '''

        generated_images = []
        

        if images:
            next_elem = True
            iter_img = iter(images)
        else:
            return None

        while next_elem:
            
            batch_images = []

            for i in range(batch_size):
                try:
                    temp_image = next(iter_img)
                    batch_images.append(temp_image)
                except StopIteration:
                    next_elem = False
                    break

            if batch_images:
               
                outputs = self.send_and_recieve(batch_images[0], strength,
                                                prompt, negative_prompt,
                                                steps)

                for im in outputs:
                    generated_images.append(np.array(im))
        print(f'Tiles generated: {len(generated_images)}')
        return generated_images


if __name__ == '__main__':
    s = RemoteSDXLPipeLine()
    from PIL import Image
    
    img = Image.open('C:/images/000.png')
    img = np.array(img, dtype=np.uint8)
    i = []
    i.append(img)
    i.append(img)
    i.append(img)

    a = s.generate_batch_images(i, 50, 'Hello world', 'tits', 50)
    
    