import glob
import os
import cv2
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

from vae import VAE
from show_results import latent_image_scatter, show_zspace, tsne_embedding_image


parser = argparse.ArgumentParser(prog='main.py')
parser.add_argument('--data_path', type=str, required=True, help='The path of image data.')
parser.add_argument('--save_path', type=str, required=True, help='The path of folder to save results (trained model, latent space plots).')
parser.add_argument('--image_size', type=int, default=64, help='The size of image.')
parser.add_argument('--latent_dim', type=int, default=256, help='Latent space dimension.')
parser.add_argument('--beta', type=int, default=1, help='For beta VAE.')
parser.add_argument('--epochs', type=int, default=30, help='The number of epochs.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
opt = parser.parse_args()


if __name__=="__main__":
    digit_size = opt.image_size
    img_shape = (digit_size, digit_size, 3)
    latent_dim = opt.latent_dim
    beta = opt.beta

    X = []
    for img in glob.glob(opt.data_path+"/*.jpg"):
        x = np.array(Image.open(img))
        x = cv2.resize(x, (digit_size, digit_size))
        X.append(x)
    X = np.array(X)

    X_train, X_test = train_test_split(X, test_size=0.3, random_state=1, shuffle=True)

    # VAE Model
    VAE_model = VAE(img_shape,latent_dim,beta)
    vae = VAE_model.get_vae()

    history = vae.fit(
        x=X_train/255, 
        y=X_train/255,
        shuffle=True,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
       )
    vae.save(opt.save_path+'/vae.h5')

    n_epoch = len(history.history['loss'])
    plt.figure()
    plt.plot(range(n_epoch), history.history['loss'], label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend() 
    plt.savefig(opt.save_path+"/VAE_history.jpg")
    plt.show(),plt.clf(),plt.close()

