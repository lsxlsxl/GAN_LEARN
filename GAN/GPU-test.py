import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽TensorFlow日志
import time
import numpy as np
import tensorflow as tf

Dec = tf.config.list_physical_devices('GPU') != []

if Dec == True:
    print("GPU可用")
    print("比较GPU和cpu的性能差距...")
    # 创建大型单精度浮点矩阵（10000x10000）
    matrix = np.random.rand(10000, 10000).astype(np.float32)

    # 预热：避免首次计算的初始化时间干扰
    print("执行预热计算...")
    with tf.device('/GPU:0'):
        _ = tf.matmul(matrix, matrix)  # GPU预热
    with tf.device('/CPU:0'):
        _ = tf.matmul(matrix, matrix)  # CPU预热

    # CPU计算
    start_cpu = time.time()
    with tf.device('/CPU:0'):
        result_cpu = tf.matmul(matrix, matrix)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # GPU计算
    start_gpu = time.time()
    with tf.device('/GPU:0'):
        result_gpu = tf.matmul(matrix, matrix)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    # 输出结果
    print(f"CPU 矩阵乘法耗时: {cpu_time:.2f}秒")
    print(f"GPU 矩阵乘法耗时: {gpu_time:.2f}秒")
    print(f"GPU加速比: {cpu_time / gpu_time:.1f}倍")
else:
    print("GPU不可用")