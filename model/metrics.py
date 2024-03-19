import numpy as np

def tensor2im(image_tensor, imtype=np.float32, min_max=(-1, 1)):
    image_numpy = image_tensor.squeeze().cpu().float().numpy()
    image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = len(image_numpy.shape)
    if n_dim == 4:
        nc, nd, nh, nw = image_numpy.shape
        image_numpy = np.transpose(image_numpy[:, int(nd / 2)], (1, 2, 0))
        image_numpy -= np.amin(image_numpy)
        image_numpy /= np.amax(image_numpy)
    elif n_dim == 3:
        nc, nh, nw = image_numpy.shape
        tmp = np.zeros((nh, nw, 3))
        #It selects all rows and columns (:) in the first two dimensions and the first two channels (:2) in the third dimension. 
        tmp[:, :, :2] = image_numpy.transpose(1, 2, 0)
        image_numpy = tmp
        image_numpy -= np.amin(image_numpy)
        image_numpy = (image_numpy / np.amax(image_numpy))
    elif n_dim == 2:
        nh, nw = image_numpy.shape
        image_numpy = image_numpy.reshape(nh, nw, 1)
        image_numpy = np.tile(image_numpy, (1, 1, 3))

    image_numpy = image_numpy * 255.0
    return image_numpy.astype(imtype)


def tensor2im_batch(image_tensor, imtype=np.float32, min_max=(-1, 1)):
    # Assuming image_tensor is a PyTorch tensor with shape (batch_size, channels, height, width)
    batch_size = image_tensor.size(0)
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = len(image_numpy.shape)-1 #otherwise 2D data with batch size > 1 is considered 4D
    if n_dim == 4:
        nc, nd, nh, nw = image_numpy.shape
        image_numpy = np.transpose(image_numpy[:, :, int(nd / 2)], (0, 2, 3, 1))
        image_numpy -= np.amin(image_numpy, axis=(1, 2), keepdims=True)
        image_numpy /= np.amax(image_numpy, axis=(1, 2), keepdims=True)
    elif n_dim == 3:
        nb, nc, nh, nw = image_numpy.shape
        tmp = np.zeros((nh, nw, nc, nb))
        tmp[:, :, :, :] = np.transpose(image_numpy, (2, 3, 1, 0))
        image_numpy = tmp
        image_numpy -= np.amin(image_numpy, axis=(0, 1), keepdims=True)
        image_numpy /= np.amax(image_numpy, axis=(0, 1), keepdims=True)
    elif n_dim == 2:
        nh, nw = image_numpy.shape
        image_numpy = image_numpy.reshape(batch_size, nh, nw, 1)
        image_numpy = np.tile(image_numpy, (1, 1, 1, 3))

    image_numpy = image_numpy * 255.0
    return image_numpy.astype(imtype)
