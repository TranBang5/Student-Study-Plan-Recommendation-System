import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, StringLookup
from tensorflow.keras.models import Model

class TwoTowerModel:
    def __init__(self, student_vocab, item_vocab):
        self.student_vocab = student_vocab
        self.item_vocab = item_vocab
        self.embedding_dim = 32

    def build_student_tower(self):
        # Student inputs
        student_inputs = {
            'student_id': Input(name='student_id', shape=(1,), dtype=tf.string),
            'interests': Input(name='interests', shape=(None,), dtype=tf.string),
            'academic_level': Input(name='academic_level', shape=(1,), dtype=tf.string),
            'learning_style': Input(name='learning_style', shape=(1,), dtype=tf.string)
        }

        # Embeddings
        student_id_vectorized = StringLookup(
            vocabulary=self.student_vocab['student_id']
        )(student_inputs['student_id'])
        student_id_embedding = Embedding(
            input_dim=len(self.student_vocab['student_id']) + 1,
            output_dim=self.embedding_dim
        )(student_id_vectorized)

        interests_vectorized = StringLookup(
            vocabulary=self.student_vocab['interests'],
            output_mode='multi_hot'
        )(student_inputs['interests'])
        interests_embedding = Dense(self.embedding_dim)(interests_vectorized)

        academic_level_vectorized = StringLookup(
            vocabulary=self.student_vocab['academic_level']
        )(student_inputs['academic_level'])
        academic_level_embedding = Embedding(
            input_dim=len(self.student_vocab['academic_level']) + 1,
            output_dim=self.embedding_dim
        )(academic_level_vectorized)

        learning_style_vectorized = StringLookup(
            vocabulary=self.student_vocab['learning_style']
        )(student_inputs['learning_style'])
        learning_style_embedding = Embedding(
            input_dim=len(self.student_vocab['learning_style']) + 1,
            output_dim=self.embedding_dim
        )(learning_style_vectorized)

        # Concatenate embeddings
        student_embeddings = Concatenate()([
            student_id_embedding,
            interests_embedding,
            academic_level_embedding,
            learning_style_embedding
        ])
        student_output = Dense(self.embedding_dim, activation='relu')(student_embeddings)

        return student_inputs, student_output

    def build_item_tower(self):
        # Item inputs (resources, courses, tutors)
        resource_inputs = {
            'resource_id': Input(name='resource_id', shape=(1,), dtype=tf.string),
            'topic': Input(name='topic', shape=(1,), dtype=tf.string),
            'type': Input(name='type', shape=(1,), dtype=tf.string),
            'level': Input(name='level', shape=(None,), dtype=tf.string)
        }

        course_inputs = {
            'course_id': Input(name='course_id', shape=(1,), dtype=tf.string),
            'subject': Input(name='subject', shape=(1,), dtype=tf.string),
            'level': Input(name='level', shape=(1,), dtype=tf.string),
            'format': Input(name='format', shape=(1,), dtype=tf.string)
        }

        tutor_inputs = {
            'tutor_id': Input(name='tutor_id', shape=(1,), dtype=tf.string),
            'expertise': Input(name='expertise', shape=(None,), dtype=tf.string),
            'experience_years': Input(name='experience_years', shape=(1,), dtype=tf.float32),
            'teaching_style': Input(name='teaching_style', shape=(1,), dtype=tf.string)
        }

        # Resource embeddings
        resource_id_vectorized = StringLookup(
            vocabulary=self.item_vocab['resource_id']
        )(resource_inputs['resource_id'])
        resource_id_embedding = Embedding(
            input_dim=len(self.item_vocab['resource_id']) + 1,
            output_dim=self.embedding_dim
        )(resource_id_vectorized)

        topic_vectorized = StringLookup(
            vocabulary=self.item_vocab['topic']
        )(resource_inputs['topic'])
        topic_embedding = Embedding(
            input_dim=len(self.item_vocab['topic']) + 1,
            output_dim=self.embedding_dim
        )(topic_vectorized)

        type_vectorized = StringLookup(
            vocabulary=self.item_vocab['type']
        )(resource_inputs['type'])
        type_embedding = Embedding(
            input_dim=len(self.item_vocab['type']) + 1,
            output_dim=self.embedding_dim
        )(type_vectorized)

        level_vectorized = StringLookup(
            vocabulary=self.item_vocab['level'],
            output_mode='multi_hot'
        )(resource_inputs['level'])
        level_embedding = Dense(self.embedding_dim)(level_vectorized)

        resource_embeddings = Concatenate()([
            resource_id_embedding,
            topic_embedding,
            type_embedding,
            level_embedding
        ])
        resource_output = Dense(self.embedding_dim, activation='relu')(resource_embeddings)

        # Course embeddings
        course_id_vectorized = StringLookup(
            vocabulary=self.item_vocab['course_id']
        )(course_inputs['course_id'])
        course_id_embedding = Embedding(
            input_dim=len(self.item_vocab['course_id']) + 1,
            output_dim=self.embedding_dim
        )(course_id_vectorized)

        subject_vectorized = StringLookup(
            vocabulary=self.item_vocab['subject']
        )(course_inputs['subject'])
        subject_embedding = Embedding(
            input_dim=len(self.item_vocab['subject']) + 1,
            output_dim=self.embedding_dim
        )(subject_vectorized)

        course_level_vectorized = StringLookup(
            vocabulary=self.item_vocab['level']
        )(course_inputs['level'])
        course_level_embedding = Embedding(
            input_dim=len(self.item_vocab['level']) + 1,
            output_dim=self.embedding_dim
        )(course_level_vectorized)

        format_vectorized = StringLookup(
            vocabulary=self.item_vocab['format']
        )(course_inputs['format'])
        format_embedding = Embedding(
            input_dim=len(self.item_vocab['format']) + 1,
            output_dim=self.embedding_dim
        )(format_vectorized)

        course_embeddings = Concatenate()([
            course_id_embedding,
            subject_embedding,
            course_level_embedding,
            format_embedding
        ])
        course_output = Dense(self.embedding_dim, activation='relu')(course_embeddings)

        # Tutor embeddings
        tutor_id_vectorized = StringLookup(
            vocabulary=self.item_vocab['tutor_id']
        )(tutor_inputs['tutor_id'])
        tutor_id_embedding = Embedding(
            input_dim=len(self.item_vocab['tutor_id']) + 1,
            output_dim=self.embedding_dim
        )(tutor_id_vectorized)

        expertise_vectorized = StringLookup(
            vocabulary=self.item_vocab['expertise'],
            output_mode='multi_hot'
        )(tutor_inputs['expertise'])
        expertise_embedding = Dense(self.embedding_dim)(expertise_vectorized)

        experience_embedding = Dense(self.embedding_dim)(
            tf.expand_dims(tutor_inputs['experience_years'], axis=-1)
        )

        teaching_style_vectorized = StringLookup(
            vocabulary=self.item_vocab['teaching_style']
        )(tutor_inputs['teaching_style'])
        teaching_style_embedding = Embedding(
            input_dim=len(self.item_vocab['teaching_style']) + 1,
            output_dim=self.embedding_dim
        )(teaching_style_vectorized)

        tutor_embeddings = Concatenate()([
            tutor_id_embedding,
            expertise_embedding,
            experience_embedding,
            teaching_style_embedding
        ])
        tutor_output = Dense(self.embedding_dim, activation='relu')(tutor_embeddings)

        return (
            [resource_inputs, course_inputs, tutor_inputs],
            [resource_output, course_output, tutor_output]
        )

    def build_model(self):
        student_inputs, student_output = self.build_student_tower()
        item_inputs, item_outputs = self.build_item_tower()

        # Compute dot product for recommendation
        resource_score = tf.keras.layers.Dot(axes=1)([student_output, item_outputs[0]])
        course_score = tf.keras.layers.Dot(axes=1)([student_output, item_outputs[1]])
        tutor_score = tf.keras.layers.Dot(axes=1)([student_output, item_outputs[2]])

        # Combine inputs
        inputs = {**student_inputs, **item_inputs[0], **item_inputs[1], **item_inputs[2]}
        outputs = [resource_score, course_score, tutor_score]

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

if __name__ == "__main__":
    # Example vocabularies (replace with actual data)
    student_vocab = {
        'student_id': [f's{i+1}' for i in range(67)],
        'interests': ['math', 'literature', 'physics', 'chemistry', 'english', 'history', 'geography', 'biology'],
        'academic_level': [f'grade_{i}' for i in range(6, 13)],
        'learning_style': ['in_person', 'online']
    }
    item_vocab = {
        'resource_id': [f'r{i+1}' for i in range(25)],
        'topic': ['math', 'literature', 'physics', 'chemistry', 'english', 'history', 'geography', 'biology'],
        'type': ['paper', 'digital'],
        'level': [f'grade_{i}' for i in range(6, 13)],
        'course_id': [f'c{i+1}' for i in range(96)],
        'subject': ['math', 'literature', 'physics', 'chemistry', 'english', 'history', 'geography', 'biology'],
        'format': ['in_person', 'online'],
        'tutor_id': [f't{i+1}' for i in range(11)],
        'expertise': ['math', 'literature', 'physics', 'chemistry', 'english', 'history', 'geography', 'biology'],
        'teaching_style': ['in_person', 'online']
    }

    # Build and summarize model
    model = TwoTowerModel(student_vocab, item_vocab)
    two_tower_model = model.build_model()
    two_tower_model.summary()