import PySimpleGUI as sg
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
from scipy.linalg import null_space
from scipy.io import savemat, loadmat
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def img_ortho_proj(image, null_basis):
    """
    Find the orthogonal projection of an image onto Span{given basis}
    """  
    coeff = np.linalg.lstsq(null_basis, image.reshape(28*28, 1))[0]
    img_proj = null_basis.dot(coeff.reshape(coeff.size, 1)).reshape(28, 28)
    
    return img_proj


#def plot_images(imgs, titles=None):
    #"""
    #Given images(tuple) and titles(tuple), plot them in a row.
    #"""
    #N = len(imgs)
    #if len(titles) != N:
        #print("Number of images and titles do not match.")
        #titles = None
        
    #fig, axes = plt.subplots(1, N, figsize=(3*N, 3))
    #for k in range(N):
        #ax = axes[k]
        #ax.imshow(imgs[k].reshape(28, 28), cmap=plt.cm.gray_r, vmin=-1, vmax=1) 
        #if titles != None:
            #ax.set_title(titles[k]) 
    ##plt.show()
    
    #return fig
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
        fig, axes = plt.subplots(1, N, figsize=(N*3, 4))
        for k in range(N):
            ax = axes[k]
            ax.imshow(imgs[k].reshape(28, 28), cmap=plt.cm.gray_r, vmin=-1, vmax=1) 
            if titles != None:
                ax.set_title(titles[k]) 
    #plt.show()
    
    return fig

#def create_stego_img(cover_img, hidden_img, null_basis):
    #cover_img = cover_img.reshape(1, 28*28)
    #hidden_img = hidden_img.reshape(1, 28*28)
    #if cover_img.max() > 1.:
        #cover_img = (cover_img - cover_img.min()) / (cover_img.max() - cover_img.min()) * 2. - 1.
    #if hidden_img.max() > 1.:
        #hidden_img = (hidden_img - hidden_img.min()) / (hidden_img.max() - hidden_img.min()) * 2. - 1.        
    
    #cover_proj = cover_img.dot(null_basis).dot(null_basis.transpose()).reshape(1, 28*28)
    #hidden_proj = hidden_img.dot(null_basis).dot(null_basis.transpose()).reshape(1, 28*28)
    #hidden_perp = hidden_img - hidden_proj
    
    #stego_img = cover_proj / abs(cover_proj).max() * 0.75 + hidden_perp / abs(hidden_perp).max() * 0.25
    #stego_img = (stego_img - stego_img.min()) / (stego_img.max() - stego_img.min()) * 255.
    #stego_img = stego_img.astype(int)
    
    #if stego_img.max() > 255:
        #stego_img[np.where(stego_img>255)] = 255
    #stego_img_std = stego_img / (255. / 2.) - 1.
    
    #return stego_img, stego_img_std
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
    
    stego_img = cover_proj * 0.65 + hidden_perp * 0.35
    scalars = [0.65 / abs(stego_img).max(), 0.35 / abs(stego_img).max()]
    stego_img = stego_img / abs(stego_img).max()
    
    stego_img_255 = (stego_img + 1.) * (255. / 2.)
    stego_img_255 = stego_img.astype(int)
    
    if stego_img_255.max() > 255:
        stego_img_255[np.where(stego_img_255>255)] = 255
    
    return stego_img_255, stego_img, scalars


def create_examples(imgs1, imgs2, model, null_basis):
    im1 = imgs1[np.random.randint(0, 60000), :]
    im2 = imgs2[np.random.randint(0, 60000), :]
    _, im3, scalars = create_stego_img(im1, im2, null_basis)
    
    preds = model.predict(np.row_stack((im1, im2, im3)))
    classes =  np.argmax(preds, axis=1)
    
    fig = plot_images((im1, im2, im3),
               ('Cover\npredicted as '+str(classes[0])+'\nrescaled w/ '+str(round(scalars[0], 2)),
                'Hidden\npredicted as '+str(classes[1])+'\nrescaled w/ '+str(round(scalars[1], 2)),
                'Stego\npredicted as '+str(classes[2]), ))
    return fig


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def main():
    #train_MNIST, test_MNIST = generate_data()
    data_MNIST = loadmat('DataSets\data_MNIST.mat')
    #train_FMNIST, test_FMNIST = generate_FMNIST_data()
    data_FMNIST = loadmat('DataSets\data_FMNIST.mat')
    # load the model trained with 10x dataset (784, 32, 16, 10)
    model = keras.models.load_model("TrainedModels\MNIST_10x_NN")
    #model.summary()
    null_basis = null_space(model.weights[0].numpy().transpose())

    layout = [
        [sg.Button("Hide numbers with numbers", size=(30, 3)), sg.Button("Hide numbers with clothes", size=(30, 3))],
        [sg.Canvas(size=(900, 400), key='-CANVAS-')],
    ]

    window = sg.Window('Examples of hiding numbers', layout, finalize=True, element_justification='center')
    fig = create_examples(data_MNIST['train_X'], data_MNIST['train_X'], model, null_basis)
    canvas = FigureCanvasTkAgg(fig, master=window['-CANVAS-'].TKCanvas) 
    canvas.draw()
    canvas.get_tk_widget().pack()

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'Hide numbers with numbers':
            if canvas is not None:
                canvas.get_tk_widget().pack_forget()
            if fig is not None:
                plt.close(fig)
            fig = create_examples(data_MNIST['train_X'], data_MNIST['train_X'], model, null_basis)
            canvas = FigureCanvasTkAgg(fig, master=window['-CANVAS-'].TKCanvas)   
            canvas.draw()
            canvas.get_tk_widget().pack()  
        elif event == 'Hide numbers with clothes':
            if canvas is not None:
                canvas.get_tk_widget().pack_forget()
            if fig is not None:
                plt.close(fig )            
            fig = create_examples(data_FMNIST['train_X'], data_MNIST['train_X'], model, null_basis)
            canvas = FigureCanvasTkAgg(fig, master=window['-CANVAS-'].TKCanvas) 
            canvas.draw()
            canvas.get_tk_widget().pack()
            
    window.close()
 

#main()
if __name__ == '__main__':
    main()
