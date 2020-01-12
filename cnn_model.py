def cnn_model(x_ph, dropout_rate):
    ################################################################################ 
    # - x_ph: one-hot encoded DNA input data / placeholder of shape [?, 1, 101, 4]
    # - dropout_rate: dropout rate (1 - keep probability) of dropout layer
    # Return values:
    # - y_hat_op: class probabilities operator (our predicted y, sigmoid(z_op)) of shape [?, 2]
    # - z_op: unscaled log probabilities operator (output of last matrix multiplication,
    #      before activation function) of shape [?, 2]
    # - weights: a list of all tf.Variable weight matrices
    ################################################################################
    # x_ph: [batch, in_height, in_width, in_channels] -> [?, 1, 101, 4]
    # conv layer filter: [filter_height, filter_width, in_channels, out_channels] -> [1, 11, 4, 32]
    conv_filter_height = 1
    conv_filter_width = 11
    in_channels = 4
    out_channels = 32
    fc_units = 64
    max_pool_stride = 2
    
    # one convolutional layer with 32 kernels (filters) of width 11 and height 1, 
    # step size of 1 and ReLU activation. The input should be padded so that the 
    # output has the same size as the original input.
    w1 = tf.Variable(tf.truncated_normal([conv_filter_height, conv_filter_width,in_channels, out_channels], stddev = 0.01))
    b1 = tf.Variable(tf.constant(0.1, shape = [out_channels]))
    conv_op = tf.nn.conv2d(x_ph, w1, strides = [1,1,1,1], padding='SAME')
    conv_op = tf.nn.bias_add(conv_op, b1)
    conv_op = tf.nn.relu(conv_op)
    
    # One pooling layer with pooling stride of two
    p_strides = [1, max_pool_stride, max_pool_stride, 1]
    max_pool = tf.nn.max_pool(conv_op, ksize = p_strides, strides = p_strides, padding='SAME')
    
    # flatten the tensor to prepare it to the next layer
    flat1, num_features = flatten_tensor(max_pool)
    
    # One fully connected layer with 64 units
    w2 = tf.Variable(tf.truncated_normal([num_features, fc_units], stddev = 0.01))
    b2 = tf.Variable(tf.constant(0.1, shape = [fc_units]))
    dense_op = tf.matmul(flat1, w2)
    dense_op = tf.nn.bias_add(dense_op, b2)
    dense_op = tf.nn.relu(dense_op)
    
    # One dropout layer 
    dropout_op = tf.nn.dropout(dense_op, keep_prob = 1-dropout_rate)
    
    # One fully connected layer with 2 units
    w3 = tf.Variable(tf.truncated_normal([fc_units, 2], stddev = 0.01))
    b3 = tf.Variable(tf.constant(0.1, shape = [2]))
    final_dense_op = tf.matmul(dropout_op, w3)
    final_dense_op = tf.nn.bias_add(final_dense_op, b3)
    
    # The operator in the computation graph for the 
    # non-normalized class predictions
    z_op = final_dense_op
    
    # The normalized class predictions ([0, 1]), that 
    # can be interpreted as probabililties
    y_hat_op = tf.nn.softmax(z_op)
    
    # weights that are saved and used in the
    # l2 regularized loss function
    weights = [w1, w2, w3]

    return y_hat_op, z_op, weights
