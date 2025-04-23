from typing import List, Dict
from ..models.two_tower import MultiTowerRecommenderModel
from ..models.entities import StudentProfile, Course, Tutor, Resource
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

class Recommender:
    def __init__(self, model: MultiTowerRecommenderModel, courses: List[Course], tutors: List[Tutor], resources: List[Resource]):
        self.model = model
        self.courses = courses
        self.tutors = tutors
        self.resources = resources

        # Initialize BruteForce index for each item type
        self.course_index = tfrs.layers.factorized_top_k.BruteForce(self.model.student_dense)
        self.tutor_index = tfrs.layers.factorized_top_k.BruteForce(self.model.student_dense)
        self.resource_index = tfrs.layers.factorized_top_k.BruteForce(self.model.student_dense)

        # Index courses
        course_ids = tf.constant([c.course_id for c in courses], dtype=tf.string)
        course_features = tf.constant([self._preprocess_item(c, 'course')['features'] for c in courses], dtype=tf.float32)
        self.course_index.index_from_dataset(
            tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(course_ids),
                tf.data.Dataset.from_tensor_slices({
                    'item_id': course_ids,
                    'features': course_features
                }).map(self.model.course_embedding).map(self.model.course_dense)
            ))
        )

        # Index tutors
        tutor_ids = tf.constant([t.tutor_id for t in tutors], dtype=tf.string)
        tutor_features = tf.constant([self._preprocess_item(t, 'tutor')['features'] for t in tutors], dtype=tf.float32)
        self.tutor_index.index_from_dataset(
            tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(tutor_ids),
                tf.data.Dataset.from_tensor_slices({
                    'item_id': tutor_ids,
                    'features': tutor_features
                }).map(self.model.tutor_embedding).map(self.model.tutor_dense)
            ))
        )

        # Index resources
        resource_ids = tf.constant([r.resource_id for r in resources], dtype=tf.string)
        resource_features = tf.constant([self._preprocess_item(r, 'resource')['features'] for r in resources], dtype=tf.float32)
        self.resource_index.index_from_dataset(
            tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(resource_ids),
                tf.data.Dataset.from_tensor_slices({
                    'item_id': resource_ids,
                    'features': resource_features
                }).map(self.model.resource_embedding).map(self.model.resource_dense)
            ))
        )

    def _preprocess_student(self, student: StudentProfile) -> Dict:
        interests_vector = [1.0 if s in student.interests else 0.0 for s in ['Math', 'Science', 'Literature']]
        academic_level = {'Beginner': 1.0, 'Intermediate': 2.0, 'Advanced': 3.0}.get(student.academic_level, 0.0)
        learning_style = {'Visual': 1.0, 'Auditory': 2.0, 'Kinesthetic': 3.0}.get(student.learning_style, 0.0)
        return {
            'student_id': tf.convert_to_tensor([student.student_id], dtype=tf.string),
            'interests': tf.convert_to_tensor([interests_vector], dtype=tf.float32),
            'academic_level': tf.convert_to_tensor([[academic_level]], dtype=tf.float32),
            'learning_style': tf.convert_to_tensor([[learning_style]], dtype=tf.float32)
        }

    def _preprocess_item(self, item: any, item_type: str) -> Dict:
        features = []
        item_id = ''
        if item_type == 'course':
            item_id = item.course_id
            features = [
                {'Math': 1.0, 'Science': 2.0, 'Literature': 3.0}.get(item.subject, 0.0),
                {'Beginner': 1.0, 'Intermediate': 2.0, 'Advanced': 3.0}.get(item.difficulty, 0.0),
                {'online': 1.0, 'offline': 2.0, 'hybrid': 3.0}.get(item.format, 0.0)
            ]
        elif item_type == 'tutor':
            item_id = item.tutor_id
            expertise_vector = [1.0 if s in item.expertise else 0.0 for s in ['Math', 'Science', 'Literature']]
            features = expertise_vector + [
                {'Interactive': 1.0, 'Lecture': 2.0}.get(item.teaching_style, 0.0),
                item.rating
            ]
        else:  # resource
            item_id = item.resource_id
            features = [
                {'Math': 1.0, 'Science': 2.0, 'Literature': 3.0}.get(item.subject, 0.0),
                {'video': 1.0, 'pdf': 2.0, 'article': 3.0}.get(item.format, 0.0),
                {'Beginner': 1.0, 'Intermediate': 2.0, 'Advanced': 3.0}.get(item.difficulty, 0.0)
            ]
        return {
            'item_id': tf.convert_to_tensor([item_id], dtype=tf.string),
            'item_type': tf.convert_to_tensor([{'course': 1.0, 'tutor': 2.0, 'resource': 3.0}[item_type]], dtype=tf.float32),
            'features': features
        }

    def recommend_courses(self, student: StudentProfile, top_k: int = 5) -> List[Course]:
        student_input = self._preprocess_student(student)
        student_vector = self.model.student_dense(tf.concat([
            self.model.student_embedding(student_input['student_id']),
            student_input['interests'],
            student_input['academic_level'],
            student_input['learning_style']
        ], axis=1))
        scores, course_ids = self.course_index(student_vector)
        top_course_ids = course_ids[0, :top_k].numpy().astype(str)
        return [c for c in self.courses if c.course_id in top_course_ids]

    def recommend_tutors(self, student: StudentProfile, top_k: int = 3) -> List[Tutor]:
        student_input = self._preprocess_student(student)
        student_vector = self.model.student_dense(tf.concat([
            self.model.student_embedding(student_input['student_id']),
            student_input['interests'],
            student_input['academic_level'],
            student_input['learning_style']
        ], axis=1))
        scores, tutor_ids = self.tutor_index(student_vector)
        top_tutor_ids = tutor_ids[0, :top_k].numpy().astype(str)
        return [t for t in self.tutors if t.tutor_id in top_tutor_ids]

    def recommend_resources(self, student: StudentProfile, top_k: int = 5) -> List[Resource]:
        student_input = self._preprocess_student(student)
        student_vector = self.model.student_dense(tf.concat([
            self.model.student_embedding(student_input['student_id']),
            student_input['interests'],
            student_input['academic_level'],
            student_input['learning_style']
        ], axis=1))
        scores, resource_ids = self.resource_index(student_vector)
        top_resource_ids = resource_ids[0, :top_k].numpy().astype(str)
        return [r for r in self.resources if r.resource_id in top_resource_ids]