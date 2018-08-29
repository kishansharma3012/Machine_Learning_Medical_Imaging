import h5py
import numpy as np

from scipy.misc import imresize
import tensorflow as tf

def write_file_hdf5(file_path, X, y_cl, y_loc, percentage=1.0, img_key="train_img", label_key_c="train_class_labels",
                    label_key_l = "train_loc_labels"):
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
    h5f.create_dataset(img_key, data=X[0:max_idx], chunks=(100, ) + X.shape[1:])
    h5f.create_dataset(label_key_l, data=y_loc[0:max_idx], chunks=(100, ) + y_loc.shape[1:])
    h5f.create_dataset(label_key_c, data=y_cl[0:max_idx], chunks=(100, ) + y_cl.shape[1:])
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


def shuffle(X, y_1, y_2):
    assert X.shape[0] == y_1.shape[0] ==y_2.shape[0]
    perm = np.arange(y_1.shape[0])
    np.random.shuffle(perm)
    return X[perm], y_1[perm], y_2[perm]


def p_map_generator(labels, max_tools, image_input_size=(576, 720), image_output_size=None, sigma2=10):

    output = np.zeros( (labels.shape[0], ) + image_input_size + (6, max_tools))
    if image_output_size:
        output = np.zeros((labels.shape[0],) + image_output_size + (6, max_tools))

    for idx, label in enumerate(labels):
        gaussians = _p_map_generator(label, max_tools, image_size=image_input_size, sigma2=sigma2)
        if image_output_size:
            for joint in range(6):
                for tool in range(max_tools):
                    resized_g = imresize(gaussians[:,:,joint,tool], size=image_output_size)
                    output[idx,:,:,joint, tool] = resized_g
        else:
            output[idx] = gaussians
    return output


def _p_map_generator(label, max_tools, image_size=(576, 720), sigma2=10):
    """
    Input :
    label : label of size (13,) and (26, )
    sigma2 : variance

    Output:
    p_maps : if no tool is present then from unifrom distribution otherwise from gaussian distribution
    of size (576, 720, no_joints, no_tools)
    """

    if ((label[0] != 1111) and (label[0] != 1110) and (label[0] != 1100) and (label[0] != 1000)):
        p_maps = 1.0 / (576 * 720 - 0) * np.ones(image_size)

    else:
        if (label.shape[0] > 13):
            no_tools = 2
        else:
            no_tools = 1
        p_maps = np.zeros((576, 720, 6, max_tools))
        for tool in range(max_tools):
            for joint in range(1, 12, 2):
                if tool >= no_tools:
                    p_maps[:, :,int((joint+1)/2 - 1), tool] = 1.0 / (576 * 720 - 0) * np.ones(image_size)
                else:
                    location = [label[tool * 13 + joint], label[tool * 13 + joint + 1]]
                    p_maps[:, :, int((joint+1)/2 - 1), tool] = gaussian2d_map(image_size, location, sigma2)

    return p_maps


def gaussian2d_map(size, location, sigma2):
    x = np.array(range(size[1]))
    y = np.array(range(size[0]))
    X, Y = np.meshgrid(x, y)
    # Mean vector and covariance matrix
    mu = np.array(location)
    Sigma = sigma2 * np.eye(2)

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
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


""" Imge Utils """
def normalize_tensor(tensor_val, max_val=255.):
    return tf.multiply(tf.div(
        tf.subtract(
            tensor_val,
            tf.reduce_min(tensor_val)
        ),
        tf.add(tf.subtract(
            tf.reduce_max(tensor_val),
            tf.reduce_min(tensor_val)
        ), 0.0001)
    ), max_val)

def get_images_max_xy(gaussian, k=1, args=None):
    """
    TODO: it doesn't work properly yet

    :param gaussian:
    :param k:
    :return:
    """
    #k represents the amount of points to consider

    # input size should be (N, 12, W, H)
    input_shape = tf.shape(gaussian)
    gaussian = tf.reshape(gaussian, [-1, input_shape[1], input_shape[2], input_shape[3]])
    output_gaussians = tf.Variable(tf.zeros(shape=(args.nb_maps*args.width*args.height, 1), dtype=tf.float32),
                                   name="a_____out_gauss", trainable=False) #(N, w, h, 1)

    input_shape = tf.shape(gaussian)
    amount_gaussians = input_shape[1]

    # gaussian = tf.transpose(gaussian, [0,2,3,1])
    flat_logits_true = tf.reshape(gaussian, [-1, input_shape[2]*input_shape[3]*input_shape[1]], name="a_____out_gauss") #(W*H, 12)
    values, index = tf.nn.top_k(flat_logits_true, k=k)  # [-1, k]
    index = index[0] #[k]
    data = tf.expand_dims( 255.*np.ones((k), dtype=np.float32), 1)

    output_gaussians = tf.scatter_update(output_gaussians, index, data, name="a_____out_gauss")
    output_gaussians = tf.reshape(output_gaussians, [1,args.nb_maps, args.width, args.height, 1])
    output_gaussians = tf.Print(output_gaussians, [tf.shape(output_gaussians)], "output_gaussians.shape")

    # tf.summary.image("image")
    return output_gaussians


def get_max_xy(gaussian, k=1, dtype=tf.float64, verbose=False):
    # input size should be (N, 12, W, H) or (N, 12, W, H, 1)
    input_shape = tf.shape(gaussian)

    flat_logits_true = tf.reshape(gaussian, [-1, input_shape[1], input_shape[2]*input_shape[3] ])# (N, 12*W*H)
    index = tf.argmax(flat_logits_true,2, output_type=tf.int32)
    if verbose:
        print(index)
        values, index2 = tf.nn.top_k(flat_logits_true, k=k, name="variable")
        index = tf.Print(index, [index, index2, tf.shape(index), tf.shape(index2)], "index")
    #max_x = tf.cast(tf.divide(index, input_shape[2]), dtype)
    #max_y = tf.cast(tf.floormod(index, input_shape[2]), dtype)
    #values, index = tf.nn.top_k(flat_logits_true, k=k, name="variable") #[-1, 12]

    max_x = tf.cast(tf.divide(index, input_shape[2]), dtype)
    max_y = tf.cast(tf.floormod(index, input_shape[2]), dtype)

    return max_x, max_y # [-1,12,K], if k=1: [-1,12, 1]


def _get_images_bb_max_xy(gaussian, gaussian_idx_list, k=1, elem_idx=0, expand_last=False, args=None):
    # input size should be (N, 12, W, H, 1) or (N, 12, W, H) if expand_last
    shape = tf.constant([args.width, args.height], dtype=tf.float64)  # [2]
    pt_x_list, pt_y_list = get_max_xy(gaussian, k=k)

    gaussian_bb_list = None

    for gaussian_idx in gaussian_idx_list:
        pt_x = tf.cast(pt_x_list[elem_idx][gaussian_idx], dtype=tf.float64)
        pt_y = tf.cast(pt_y_list[elem_idx][gaussian_idx], dtype=tf.float64)

        location_list = tf.concat([pt_x, pt_y], 0)
        location_list = tf.reshape(location_list, [k, 2])
        location_list /= shape

        bb_bottom = location_list - tf.div(tf.constant([1.0, 1.0], dtype=tf.float64), shape)
        bb_top    = location_list + tf.div(tf.constant([10.0, 10.0], dtype=tf.float64),  shape)
        bb = tf.cast(tf.concat([bb_bottom, bb_top], 1), dtype=tf.float32, name="bb/"+str(gaussian_idx))
        bb = tf.reshape(bb, [1, k, 4])

        data = tf.cast(tf.identity(gaussian[elem_idx][gaussian_idx]), dtype=tf.float32)
        data = tf.expand_dims(data, 0)
        if expand_last:
            data = tf.expand_dims(data, -1)

        image = tf.get_variable("variable-1/"+str(gaussian_idx), [1,   args.width, args.height, 1], trainable=False,
                                    initializer=tf.zeros_initializer,
                                    dtype=tf.float32)
        image = tf.concat([image, data], 0, name="bb-concat/"+str(gaussian_idx))
        image = tf.expand_dims(tf.reduce_sum(image, 0), 0)

        img_bounding_boxes = tf.image.draw_bounding_boxes(image, bb) # 224, 224, 1
        yield img_bounding_boxes

def get_images_bb_max_xy(gaussian, gaussian_idx_list, k=1, elem_idx=0, expand_last=False, args=None):
    #aux = tf.stack(list(_get_images_bb_max_xy(gaussian, gaussian_idx_list, k, elem_idx, expand_last, args)))

    return gaussian#tf.transpose(aux, [1,0,2,3,4])

def produce_sum_maps(tensor_gaussians, args):
    """
        Expects: [-1, width, height, 6, 1] or [-1, width, height, 12]

    """
    y_conv_reshape = tf.reshape(tensor_gaussians, [-1, args.width, args.height, args.nb_maps])  # N, Width, height, Ins.*Joints
    y_conv_reshape = tf.transpose(y_conv_reshape, [0, 3, 1, 2])  # N, Ins.*Joints, Width, height, 1
    y_conv_reshape = tf.expand_dims(y_conv_reshape, 4)

    y_conv_sum = tf.reduce_sum(y_conv_reshape, 1)

    return y_conv_sum, y_conv_reshape

def add_imgs_to_summary(image_list, idx_list, base_name, max_outputs=1, family=None):
    # image_list should be [-1, <idx_list>, width, height, channel]
    if isinstance(idx_list, int):
        idx = idx_list
        if family:
            tf.summary.image(base_name + "_" + str(idx), image_list[:, idx], max_outputs=max_outputs, family=family)
        else:
            tf.summary.image(base_name + "_" + str(idx), image_list[:, idx], max_outputs=max_outputs)

    for idx in idx_list:
        if family:
            tf.summary.image(base_name+"_"+str(idx), image_list[:, idx], max_outputs=max_outputs, family=family)
        else:
            tf.summary.image(base_name+"_"+str(idx), image_list[:, idx], max_outputs=max_outputs)


if __name__ == "__main__":
    # EXAMPLE
    file_path = '/home/mlmi/Desktop/4_tools_dataset/train_images_224_4_tools_10_joints.hdf5'
    #file_path = "./data/train_class+loc_images_224_10ptc_chuncked.hdf5"

    f = h5py.File(file_path, 'r')  # open file
    print (f.keys())
    X = np.array(f[u'images']) #f[u'train_img']
    y_1 = np.array(f[u'classilabels']) # f[u'train_class_labels']
    y_2 = np.array(f[u'localilabels']) # f[u'train_loc_labels']
    f.close()  # close file
   
    #print(X[0],y_1[0],y_2[0])
    #X,y_1,y_2=shuffle(X,y_1,y_2)
    #print(X[0],y_1[0],y_2[0])
    #print(X[0],y_1[0],y_2[0])
    # y_new = np.zeros((376, 56, 56, 6,2))
    # print y_new.shape, y.shape
    # for joint in range(6):
    #     for item in range(2):
    #         y_new[:,:,:,joint, item] =  reduce_resolution(y[:,:,:,joint, item], size=0.25)
    #         print (item, joint)
    # y = y_new
    # y = reduce_resolution(y, size=0.25)
    # y = p_map_generator(y, image_output_size=(112,112), max_tools=2)

    print(X.shape, y_1.shape, y_2.shape)
    write_file_hdf5("./data/train_images_224_4tools_10joints_chuncked.hdf5", X, y_1,y_2)#, percentage=0.1)
