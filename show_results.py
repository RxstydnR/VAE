import os
import itertools
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from scipy.stats import norm
from numpy.matlib import repmat

def imscatter(x, y, imgs, ax=None, zoom=1): 
    if ax is None: 
        ax = plt.gca() 
    
    x,y = np.atleast_1d(x, y) 
    artists = [] 
    for img,x0, y0 in zip(imgs, x, y): 
        im = OffsetImage(img, zoom=zoom) 
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False) 
        artists.append(ax.add_artist(ab)) 
    ax.update_datalim(np.column_stack([x, y])) 
    ax.autoscale() 
    return artists 


def latent_image_scatter(imgs,x,y):
    """ Representing a plot point as an image.

    Args:
        imgs (arr): image data
        x (arr): x axis values
        y (arr): y axis values
    """
    fig, ax = plt.subplots(dpi=200) 
    imscatter(x, y, imgs, ax=ax,  zoom=.25) 
    ax.plot(x, y, 'ko',alpha=0)
    # plt.savefig('cactus_plot.png',dpi=200, transparent=False)
    # plt.show(), plt.clf(), plt.close()
    return
    

def tsne_embedding(X_encoded, tsne_dim, save_path):
    """ Compress the z representation with tSNE and visualize it as a plot.

    Args:
        X_encoded (arr): z compressed by encoder
        tsne_dim (int): tSNE dimention
        save_path (str): Path to save folder
    """
    save_path = save_path+"/tsne"
    os.makedirs(save_path, exist_ok=True)

    tsne_embedded = TSNE(n_components=tsne_dim).fit_transform(X_encoded)
    
    for v in itertools.combinations(range(tsne_dim), 2):
        plt.figure(figsize=(6,6))
        plt.scatter(tsne_embedded[:,v[0]], tsne_embedded[:,v[1]], c=None)
        plt.xlabel("z{}".format(v[0]), fontsize=15)
        plt.ylabel("z{}".format(v[1]), fontsize=15)
        plt.savefig(save_path+"/z{}z{}".format(str(v[0]),str(v[1])), dpi=200, transparent=False)
        plt.show(), plt.clf(), plt.close()


def tsne_embedding_image(X, X_encoded, tsne_dim, save_path):
    """ Compress the z representation with tSNE and visualize it as each image.

    Args:
        X (arr): Raw Image data
        X_encoded (arr): z compressed by encoder
        tsne_dim (int): tSNE dimention
        save_path (str): Path to save folder
    """
    save_path = save_path+"/tsne_image"
    os.makedirs(save_path, exist_ok=True)

    tsne_embedded = TSNE(n_components=tsne_dim).fit_transform(X_encoded)
    
    for v in itertools.combinations(range(tsne_dim), 2):
        plt.figure(figsize=(6,6))
        latent_image_scatter(X, tsne_embedded[:,v[0]], tsne_embedded[:,v[1]])
        plt.xlabel("z{}".format(v[0]), fontsize=15)
        plt.ylabel("z{}".format(v[1]), fontsize=15)
        plt.savefig(save_path+"/z{}z{}".format(str(v[0]),str(v[1])), dpi=200, transparent=False)
        plt.show(), plt.clf(), plt.close()


def show_zspace(decoder, n, digit_size, z_dim, z_range, save_path):
    """ Visualize image embedding information for all combinations of z

    Args:
        decoder (model): Decoder of VAE
        n (int): Number of images to be visualized in a row
        digit_size (int): Size of image
        z_dim (int): Latent space dimention
        z_range (list): The width of the z representation to be used (generating values of the latent variable z from the Gaussian distribution.)
        save_path (str): Path to save folder
    """
    save_path = save_path+"/show_zspace"
    os.makedirs(save_path,exist_ok=True)

    x_min,x_max,y_min,y_max = z_range

    # We use Scipy's ppf function to convert the linear space coordinates to generate the value of the latent variable z. 
    # (Since the latent space is preceded by a Gaussian distribution.)
    # Scipyのppf関数を使って線型空間座標を変換し、潜在変数zの値を生成する (潜在空間の前はガウス分布であるため)
    grid_x = norm.ppf(np.linspace(x_min, x_max, n)) 
    grid_y = norm.ppf(np.linspace(y_min, y_max, n))

    figure = np.zeros((digit_size * n, digit_size * n, 3))

    for v in list(itertools.combinations(range(z_dim),2)):
        for j, yj in enumerate(grid_y):
            for i, xi in enumerate(grid_x):
                z_sample = np.zeros(z_dim)[np.newaxis,:]
                z_sample[0][v[0]] = xi
                z_sample[0][v[1]] = yj
                
                # z_sample = repmat(z_sample,batch_size,1)
                # x_decoded = decoder.predict(z_sample, batch_size=batch_size, verbose=0)
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded.reshape(digit_size, digit_size, 3)
                
                # i * digit_size: (i + 1) * digit_size ← Set the position of the image width.
                # j * digit_size: (j + 1) * digit_size ← Set the position of the image height.
                figure[(n-j-1) * digit_size: (n-j) * digit_size, i * digit_size: (i + 1) * digit_size] = digit
        plt.figure(figsize=(10,10))
        plt.imshow(figure)
        plt.xlabel("z{}".format(v[0]), fontsize=15)
        plt.ylabel("z{}".format(v[1]), fontsize=15)
        plt.savefig(save_path+"/z{}&z{}".format(str(v[0]),str(v[1])))
        plt.show(), plt.clf(), plt.close()


from mpl_toolkits.mplot3d import Axes3D
def plot_3d(X,save_path):
    """ plot 3D scatter of Latent Space.

    Args:
        X (arr): Latent space info. This must be (?,3).
        save_path (str): Path to save folder.
    """
    assert X.shape[1]==3,"X dim must be 3."
    save_path = save_path+"/3d_scatter"
    os.makedirs(save_path,exist_ok=True)
    
    fig = plt.figure()
    ax = Axes3D(fig)

    #軸ラベル
    ax.set_xlabel("z{}".format(0), fontsize=15)
    ax.set_ylabel("z{}".format(1), fontsize=15)
    ax.set_zlabel("z{}".format(2), fontsize=15)
    
    ax.plot(X[:,0],X[:,1],X[:,2], marker="o", linestyle='None')
    
    plt.savefig(save_path+"/3d_plot.jpg")
    plt.show(), plt.clf(), plt.close()