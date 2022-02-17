import tensorflow as tf

patch_size = 23
half_patch_size = int(patch_size/2)

def decode_image(image_bytes):
    image = tf.image.decode_png(image_bytes)
    image = tf.cast(image, tf.float32)
    image = tf.multiply(image, 1.0 / 255.)
    return image

def patch_example_parser(record):
    keys_to_features = {
        "image/is_background": tf.FixedLenFeature((), tf.int64, default_value=None),
        "image/rgb_color": tf.FixedLenFeature([3], tf.float32, default_value=None),
        "image/encoded": tf.FixedLenFeature((), tf.string, default_value=None),
        'image/height': tf.FixedLenFeature((), tf.int64, default_value=None),
        'image/width': tf.FixedLenFeature((), tf.int64, default_value=None),
    }

    parsed = tf.parse_single_example(record, keys_to_features)
    
    # Perform additional preprocessing on the parsed data.
    image = decode_image(parsed["image/encoded"])    
    
    is_background = tf.cast(parsed["image/is_background"], tf.int32)
    rgb_color = parsed["image/rgb_color"]
    
    # features = {"image": image}
    
    return image, rgb_color