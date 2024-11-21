import glob
import os
import random
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from tensorflow.python.platform import gfile

# 禁用 TensorFlow v2 行为
tf.disable_v2_behavior()

# 定义模型与数据路径常量
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = './images/model/'
MODEL_FILE = 'tensorflow_inception_graph.pb'
CACHE_DIR = './images/tmp/bottleneck/'
INPUT_DATA = './images/flower_photos/'
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

def create_image_lists(testing_percentage, validation_percentage):
    """
    从数据目录中读取所有图片，并随机划分为训练集、验证集和测试集。
    """
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, f'*.{extension}')
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        label_name = dir_name.lower()
        training_images, testing_images, validation_images = [], [], []

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = random.uniform(0, 100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result

def get_image_path(image_lists, image_dir, label_name, index, category):
    """
    获取指定类别图片的路径。
    """
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    return os.path.join(image_dir, sub_dir, base_name)

def get_bottleneck_path(image_lists, label_name, index, category):
    """
    获取指定类别图片的瓶颈文件路径。
    """
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    """
    使用 Inception-v3 模型计算图片的瓶颈特征向量。
    """
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    return np.squeeze(bottleneck_values)

def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    """
    获取或计算图片的瓶颈特征向量，并缓存到文件中。
    """
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        try:
            image_data = gfile.FastGFile(image_path, 'rb').read()
        except Exception as e:
            print(f"Error reading file {image_path}: {e}")
            return None
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        with open(bottleneck_path, 'w') as f:
            f.write(','.join(map(str, bottleneck_values)))
    else:
        with open(bottleneck_path, 'r') as f:
            bottleneck_values = [float(x) for x in f.read().split(',')]
    return bottleneck_values

def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    """
    随机获取一批缓存的瓶颈特征向量。
    """
    bottlenecks, ground_truths = [], []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category,
                                              jpeg_data_tensor, bottleneck_tensor)
        if bottleneck is None:
            continue
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

def main(_):
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[
    #     BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    jpeg_data_tensor, bottleneck_tensor = tf.import_graph_def(graph_def, return_elements=[
        JPEG_DATA_TENSOR_NAME, BOTTLENECK_TENSOR_NAME])

    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            # if i % 100 == 0:
            validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
            validation_accuracy = sess.run(evaluation_step, feed_dict={
                bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
            print(f"Step {i}: Validation accuracy = {validation_accuracy:.2f}")

if __name__ == '__main__':
    tf.app.run()
