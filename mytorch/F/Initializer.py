import cupy as cp

def xavier_init(input_size, output_size):
    return cp.random.normal(0.0, cp.sqrt(2/(input_size + output_size)), (input_size, output_size), dtype=cp.float32)

def he_init_conv2d(filter_shape):
    # filter shape (output_depth, input_depth, filter_height, filter_width)
    fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3] # number of input units, product of: input depth, filter height, filter width
    stddev = cp.sqrt(2.0 / fan_in) # standard deviation of normal distribution

    return cp.random.normal(loc=0, scale=stddev, size=filter_shape, dtype=cp.float32)
