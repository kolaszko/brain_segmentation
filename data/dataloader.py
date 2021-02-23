import tensorflow as tf
import os


def brain_dataset(path, image_size=None, augment_function=None, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    def get_subdirectory_files(subdir):
        sub_path = os.path.join(path, subdir)
        return sorted([os.path.join(dp, f) for dp, dn, fn in
                       os.walk(os.path.expanduser(sub_path), followlinks=True) for f in
                       fn if f.endswith('.png')])

    def parse_function(image_path, labels_path):
        image = _read_image(image_path)
        labels = _read_labels(labels_path)

        if augment_function:
            image, labels = augment_function(image, labels)

        if image_size:
            image = tf.image.resize(image, image_size)
            labels = tf.image.resize(labels, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image, labels

    data_files = get_subdirectory_files('images/images')
    target_files = get_subdirectory_files('labels/labels')

    assert len(data_files) == len(target_files)

    ds = tf.data.Dataset.from_tensor_slices((data_files, target_files)) \
        .shuffle(len(data_files)) \
        .map(parse_function, num_parallel_calls)
    return ds, len(data_files)


@tf.function
def _read_image(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, 3)
    x = tf.image.convert_image_dtype(x, tf.float32)
    return x


@tf.function
def _read_labels(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, 1)
    x = tf.cast(x, tf.int32)
    return x
