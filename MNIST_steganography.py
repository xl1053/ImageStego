"""
Image Steganography Tests
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from scipy.io import savemat, loadmat


def generate_data():
    """
    MNIST dataset
    Data preprocessing: 60000 training data, 10000 testing data. [-1,1]
    """
    (ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True,)
    
    # Initialize sentences and labels lists
    training_imgs = []
    training_labels = []
    testing_imgs = []
    testing_labels = []
    
    # Loop over all training examples and save the sentences and labels
    for s,l in ds_train:
        training_imgs.append(s.numpy()/127.5-1.)
        training_labels.append(l.numpy())
    
    # Loop over all test examples and save the sentences and labels
    for s,l in ds_test:
        testing_imgs.append(s.numpy()/127.5-1.)
        testing_labels.append(l.numpy())
    
    # Convert labels lists to numpy array
    training_imgs_final = np.array(training_imgs).reshape(60000, 28*28)
    testing_imgs_final = np.array(testing_imgs).reshape(10000, 28*28)
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)
    
    del training_imgs, testing_imgs, training_labels, testing_labels, ds_train, ds_test, ds_info
    
    return (training_imgs_final, training_labels_final), (testing_imgs_final, testing_labels_final)



def generate_FMNIST_data():
    """
    FMNIST dataset
    Data preprocessing: 60000 training data, 10000 testing data. [-1,1]
    """    
    (ds_train, ds_test), ds_info = tfds.load('FashionMNIST', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True,)
    
    # Initialize sentences and labels lists
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    
    # Loop over all training examples and save the sentences and labels
    for s,l in ds_train:
        train_imgs.append(s.numpy().astype(np.float32)/127.5-1.)
        train_labels.append(l.numpy())
    
    # Loop over all test examples and save the sentences and labels
    for s,l in ds_test:
        test_imgs.append(s.numpy().astype(np.float32)/127.5-1.)
        test_labels.append(l.numpy())
    
    # Convert labels lists to numpy array
    fmnist_train_imgs = np.array(train_imgs).reshape(60000, 28*28)
    fmnist_test_imgs = np.array(test_imgs).reshape(10000, 28*28)
    fmnist_train_labels = np.array(train_labels)
    fmnist_test_labels = np.array(test_labels)
    
    return (fmnist_train_imgs, fmnist_train_labels), (fmnist_test_imgs, fmnist_test_labels)


def img_ortho_proj(image, null_basis):
    """
    Find the orthogonal projection of an image onto Span{given basis}
    """  
    coeff = np.linalg.lstsq(null_basis, image.reshape(28*28, 1))[0]
    img_proj = null_basis.dot(coeff.reshape(coeff.size, 1)).reshape(28, 28)
    
    return img_proj


def plot_images(imgs, titles=None):
    """
    Given images(tuple) and titles(tuple), plot them in a row.
    """
    N = len(imgs)
    if titles != None and len(titles) != N:
        print("Number of images and titles do not match.")
        titles = None
    
    if N == 1:
        fig = plt.imshow(imgs[0].reshape(28, 28), cmap=plt.cm.gray_r, vmin=-1, vmax=1)
        if titles != None:
            plt.title(titles[0])
    else:
        fig, axes = plt.subplots(1, N, figsize=(N*3, 3))
        for k in range(N):
            ax = axes[k]
            ax.imshow(imgs[k].reshape(28, 28), cmap=plt.cm.gray_r, vmin=-1, vmax=1) 
            if titles != None:
                ax.set_title(titles[k]) 
    plt.show()
    
    return 1


def create_stego_img(cover_img, hidden_img, null_basis):
    cover_img = cover_img.reshape(1, 28*28)
    hidden_img = hidden_img.reshape(1, 28*28)
    if cover_img.max() > 1.:
        cover_img = (cover_img - cover_img.min()) / (cover_img.max() - cover_img.min()) * 2. - 1.
    if hidden_img.max() > 1.:
        hidden_img = (hidden_img - hidden_img.min()) / (hidden_img.max() - hidden_img.min()) * 2. - 1.        
    
    cover_proj = cover_img.dot(null_basis).dot(null_basis.transpose()).reshape(1, 28*28)
    hidden_proj = hidden_img.dot(null_basis).dot(null_basis.transpose()).reshape(1, 28*28)
    hidden_perp = hidden_img - hidden_proj
    
    stego_img = cover_proj * 0.7 + hidden_perp * 0.3
    stego_img = stego_img / abs(stego_img).max()
    
    stego_img_255 = (stego_img + 1.) * (255. / 2.)
    stego_img_255 = stego_img.astype(int)
    
    if stego_img_255.max() > 255:
        stego_img_255[np.where(stego_img_255>255)] = 255
    
    return stego_img_255, stego_img


def create_examples(imgs1, imgs2, model, null_basis):
    im1 = imgs1[np.random.randint(0, 60000), :]
    im2 = imgs2[np.random.randint(0, 60000), :]
    _, im3 = create_stego_img(im1, im2, null_basis)
    
    preds = model.predict(np.row_stack((im1, im2, im3)))
    classes =  np.argmax(preds, axis=1)
    
    fig = plot_images((im1, im2, im3),
               ('Cover\npredicted as '+str(classes[0]), 'Hidden\npredicted as '+str(classes[1]), 'Stego\npredicted as '+str(classes[2]), ))
    return fig



if __name__ == '__main__':
    #train_MNIST, test_MNIST = generate_data()
    data_MNIST = loadmat('DataSets\data_MNIST.mat')
    #train_FMNIST, test_FMNIST = generate_FMNIST_data()
    data_FMNIST = loadmat('DataSets\data_FMNIST.mat')
    # load the model trained with 10x dataset (784, 32, 16, 10)
    model = keras.models.load_model("TrainedModels\MNIST_10x_NN")
    #model.summary()
    null_basis = null_space(model.weights[0].numpy().transpose())
    create_examples(data_MNIST['train_X'], data_MNIST['train_X'], model, null_basis)
    
    input("Press Any Key To Exit")






