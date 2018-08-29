import h5py
import numpy as np

from scipy.misc import imresize


def write_file_hdf5(file_path, X, y, percentage=1.0, img_key="train_img", label_key="train_labels"):
    """
    Write a hdf5 file with X in train_img and y in train_labels in the specified file_path.

    :param file_path: Path (with file name) to which the file should be written.
    :param X: Image data. it's better if it is a numpy
    :param y: Image labels.
    :param percentage: Percentage of the original data that should be written in the output file.
    :return:
    """
    max_idx = int(X.shape[0] * percentage)
    h5f = h5py.File(file_path, 'w')
    h5f.create_dataset(img_key, data=X[0:max_idx], chunks=(100, X.shape[1], X.shape[2], X.shape[3]))
    h5f.create_dataset(label_key, data=y[0:max_idx], chunks=(100, y.shape[1]))
    h5f.close()


def reduce_resolution(X, size):
    """
    Resize a numpy array X with images to a different resolution

    :param X: Image shape: (Nb_images, WIDTH, HEIGHT, Channels)
    :param size:    int, float or tuple
                    int - Percentage of current size.
                    float - Fraction of current size.
                    tuple - Size of the output image.
    :return: X resized.
    """
    return np.array([imresize(x, size=size) for x in X])


def scale_locations(y_locations, size_output=None, size_input=None):
    """
    Resize a numpy array y with instrument locations to a different resoltution give by size_output.
    New positions will be rounded by numpy functions.
    ref: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.around.html#numpy.around

    :param y_locations: numpy array containing locations for instrument. shape: (nb_instruments, x_pos, y_pos)
    :param size_output: int, float or tuple
                        int - Percentage of current size.
                        float - Fraction of current size.
                        tuple - Size of the output image. (x', y') = scale*(x, y) -> scale = (x', y')/(x,y)
    :param size_input: original size of the image. Only necessary if size_output is a tuple

    :return: y_location resized.
    """
    if isinstance(size_output, (int, long)):
        size_output = float(size_output / 100.0)
    if isinstance(size_output, float):
        return np.round(y_locations * size_output)

    if isinstance(size_output, tuple):
        if not size_input:
            raise ("parameter 'size_input' not given.")
        scale = np.array(size_output, dtype=float) / np.array(size_input, dtype=float)
        return np.round(y_locations * scale)

    return None


def read_data_file(file_path="../data/data_20_25pct.hdf5", img_key=u'train_img', labels_key=u'train_labels'):
    f = h5py.File(file_path, 'r')

    X = np.array(f[img_key])
    X = np.reshape(X, [X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]])
    y = np.array(f[labels_key])

    f.close()
    return X, y

def location_labels_to_counter(labels):
    """
    Produces a n-hot vector.

    Originally:
      - If the tool indicating entry is 1111: its a rightclasper instrument.
      - If the tool indicating entry is 1110: its a leftclasper instrument.
      - If the tool indicating entry is 1100: its a rightscissor instrument.
      - If the tool indicating entry is 1000: its a leftscissor instrument.
    Mapped into:
      -rightclasper: 0.
      -leftclasper:  1.
      -rightscissor: 2.
      -leftscissor:  3.

    :param  labels: labels containning processed data according to Ibrar implemetation...
    :return: numpy vector with shape (nb_items, 4)

    """
    y = np.zeros(shape=(labels.shape[0], 4))
    rightclasper_id = 1111.
    leftclasper_id = 1110.
    rightscissor_id = 1100.
    leftscissor_id = 1000.

    for idx, label in enumerate(labels):
        if rightclasper_id in label:
            y[idx][0] = 1
        if leftclasper_id in label:
            y[idx][1] = 1
        if rightscissor_id in label:
            y[idx][2] = 1
        if leftscissor_id in label:
            y[idx][3] = 1

    return y



def original_labels_to_counter(labels, is_onehot=True):
    """
        Efficiency could be improved using some sort of numpy thing, but I don't know how to od
        Classes:
            0 - Contains nothing
            1 - Contains one instrument (right or left)
            2 - Contains two instrument

        :param labels: "original" labels from Endoscopic Vision Challange (when we first made it wrong)
        :param is_onehot: if true: classes will be one-hot vector.
                          otherwise: classes will be one single value
        :return: if is_onehot: numpy vector with shape (nb_items, 3)
                    otherwise: numpy vector with shape (nb_items, 1)
        """
    y = None
    if is_onehot:
        y = np.zeros(shape=(labels.shape[0], 3))
    else:
        y = np.zeros(shape=(labels.shape[0], 1))
    for idx, label in enumerate(labels):
        if label.shape[0] == 14:
            # Contains 2 instruments
            if is_onehot:
                y[idx][2] = 1.0
            else:
                y[idx] = 2.0
        elif label[0] != -1:
            # Contains 1 instrumment
            if is_onehot:
                y[idx][1] = 1.0
            else:
                y[idx] = 1.0
        elif is_onehot:
            y[idx][0] = 1.0
    return y


def shuffle(X, y):
    assert X.shape[0] == y.shape[0]
    perm = np.arange(y.shape[0])
    np.random.shuffle(perm)
    return X[perm], y[perm]

def p_map_generator(label,image_size = (576,720),sigma2=10):
    """
    Input : 
    label : FUll labels array containing labels of all the samples
    sigma2 : variance of gaussian

    Output: 
    p_maps : if no tool is present then from unifrom distribution otherwise from gaussian distribution
    of size (576, 720, no_joints, no_tools)
    """
    
    if len(label) == 0: # This image sample containg no labels.
        #p_maps = 1.0/(576*720-0)*np.ones(image_size)
        print("do something later on")
        
    else:
        no_tools = label.shape[0]/13
        p_maps = np.zeros((576,720,6,2))
        tool = 0
        if no_tools == 1:
            if label[0] == 1111:
                tool = 0
                # A right-clasper is present in the image, put uniform distributions for joints
                # of left-clasper at position 1 of p_maps
                # For the missing tool create joint maps from uniform distribution
                op_tool = 1
            elif label[0] == 1110:
                tool = 1
                # A left-clasper is present in the image, put uniform distributions for joints
                # of right-clasper at position 0 of p_maps
                # For the missing tool create joint maps from uniform distribution
                op_tool = 0

            # Generate gaussian maps for the joints of present tool
            for joint in range(1,12,2):
                location = [label[joint],label[joint+1]] # location = x,y in real world coordinates
                p_maps[:,:,joint/2, tool] = gaussian2d_map(image_size, location, sigma2) # 0th,1st,2,3,4,5

            # Generate uniform distribution maps for the joints of absent tool
            for joint in range(6):
                p_maps[:,:,joint, op_tool] = 1.0 / (576 * 720 - 0) * np.ones(image_size)

        # If both the tools are present, then generate gaussian maps for joints of both the tools
        elif no_tools == 2:
            for tool in range(no_tools):
                for joint in range(1,12,2):
                    location = [label[tool*13+joint],label[tool*13+joint+1]] # location = x,y in real world coordinates
                    #print("location:",location)
                    p_maps[:,:,joint/2, tool ] = gaussian2d_map(image_size, location, sigma2) # 0th,1st,2,3,4,5
        

    return p_maps

def gaussian2d_map(size, location,sigma2):
    x = np.array(range(size[1])) # Real world X coordinates are actually 720
    y = np.array(range(size[0])) # Y are 576
    X, Y = np.meshgrid(x, y)
    # Mean vector and covariance matrix
    mu = np.array(location)
    Sigma = sigma2*np.eye(2)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    p_map = multivariate_gaussian(pos, mu, Sigma)

    return p_map 

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / float(N)


if __name__ == "__main__":
    #EXAMPLE
    file_path = '../data/train_generated/train_images_position.hdf5'
    # file_path = "../data/train_generated/train_images_128.hdf5"

    f = h5py.File(file_path, 'r') #open file
    X = np.array(f[u'train_img'])
    y = np.array(f[u'train_labels'])
    f.close()#close file

    y = location_labels_to_counter(y)
    X = reduce_resolution(X, size=25)
    write_file_hdf5("../data/chuncked/data_chuncked_25size_onlylabels_onehot.hdf5", X, y)

