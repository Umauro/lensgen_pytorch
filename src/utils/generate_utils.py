import numpy as np
import matplotlib.pyplot as plt

def save_image(lens_image,path,image_id):
    """
        Save lens image as matplotlib imshow figure

        Params:
            - lens_image (numpy array): a lens array
            - path (str): folder path
            - image_id (str): image identificator
    """
    image_shape = lens_image.shape
    fig = plt.figure(figsize=(1,1))
    axes = plt.Axes(fig,[0.,0.,1.,1.])
    axes.set_axis_off()
    fig.add_axes(axes)
    axes.imshow(lens_image,cmap='hot')
    plt.savefig('{}/{}.png'.format(path,image_id),dpi=image_shape[0])
    plt.close()