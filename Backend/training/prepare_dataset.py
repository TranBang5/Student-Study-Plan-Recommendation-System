import tensorflow as tf

def prepare_dataset(interactions):
    dataset = tf.data.Dataset.from_generator(
        lambda: interactions,
        output_signature={
            'student_id': tf.TensorSpec(shape=(), dtype=tf.string),
            'item_id': tf.TensorSpec(shape=(), dtype=tf.string),
            'interests': tf.TensorSpec(shape=(3,), dtype=tf.float32),  # Giả sử 3 subjects
            'academic_level': tf.TensorSpec(shape=(1,), dtype=tf.float32),
            'learning_style': tf.TensorSpec(shape=(1,), dtype=tf.float32),
            'item_type': tf.TensorSpec(shape=(1,), dtype=tf.float32),
            'features': tf.TensorSpec(shape=(None,), dtype=tf.float32)  # Số chiều phụ thuộc vào item_type
        }
    )
    return dataset.batch(32).cache().prefetch(tf.data.AUTOTUNE)