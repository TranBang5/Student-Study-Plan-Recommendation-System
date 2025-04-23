import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import List, Dict

class MultiTowerRecommenderModel(tfrs.models.Model):
    def __init__(self, student_vocab: List[str], course_vocab: List[str], tutor_vocab: List[str], resource_vocab: List[str], embedding_dim: int = 64):
        super().__init__()

        # Student Tower
        self.student_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=student_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(student_vocab) + 1, embedding_dim)
        ])
        self.student_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='relu')
        ])

        # Course Tower
        self.course_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=course_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(course_vocab) + 1, embedding_dim)
        ])
        self.course_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='relu')
        ])

        # Tutor Tower
        self.tutor_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=tutor_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(tutor_vocab) + 1, embedding_dim)
        ])
        self.tutor_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='relu')
        ])

        # Resource Tower
        self.resource_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=resource_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(resource_vocab) + 1, embedding_dim)
        ])
        self.resource_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='relu')
        ])

        # Retrieval tasks for each item type
        self.course_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(course_vocab).batch(128).map(self.course_embedding)
            )
        )
        self.tutor_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(tutor_vocab).batch(128).map(self.tutor_embedding)
            )
        )
        self.resource_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(resource_vocab).batch(128).map(self.resource_embedding)
            )
        )

    def call(self, inputs):
        # Student embedding
        student_emb = self.student_embedding(inputs['student_id'])
        student_features = tf.concat([
            student_emb,
            inputs['interests'],
            inputs['academic_level'],
            inputs['learning_style']
        ], axis=1)
        student_vector = self.student_dense(student_features)

        # Course embedding
        course_emb = self.course_embedding(inputs['item_id'])
        course_features = tf.concat([course_emb, inputs['features']], axis=1)
        course_vector = self.course_dense(course_features)

        # Tutor embedding
        tutor_emb = self.tutor_embedding(inputs['item_id'])
        tutor_features = tf.concat([tutor_emb, inputs['features']], axis=1)
        tutor_vector = self.tutor_dense(tutor_features)

        # Resource embedding
        resource_emb = self.resource_embedding(inputs['item_id'])
        resource_features = tf.concat([resource_emb, inputs['features']], axis=1)
        resource_vector = self.resource_dense(resource_features)

        return {
            'student_vector': student_vector,
            'course_vector': course_vector,
            'tutor_vector': tutor_vector,
            'resource_vector': resource_vector
        }

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        outputs = self(inputs)
        student_vector = outputs['student_vector']
        
        # Compute loss for each item type based on item_type
        course_mask = tf.equal(inputs['item_type'], 1.0)
        tutor_mask = tf.equal(inputs['item_type'], 2.0)
        resource_mask = tf.equal(inputs['item_type'], 3.0)

        course_loss = tf.where(
            course_mask,
            self.course_task(student_vector, outputs['course_vector'], compute_metrics=not training),
            0.0
        )
        tutor_loss = tf.where(
            tutor_mask,
            self.tutor_task(student_vector, outputs['tutor_vector'], compute_metrics=not training),
            0.0
        )
        resource_loss = tf.where(
            resource_mask,
            self.resource_task(student_vector, outputs['resource_vector'], compute_metrics=not training),
            0.0
        )

        return tf.reduce_mean(course_loss + tutor_loss + resource_loss)

    def save_model(self, path: str):
        self.save_weights(path)

    def load_model(self, path: str):
        self.load_weights(path)