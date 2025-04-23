from mysql.connector import Error
from ..database.db import Database
import json

def fetch_interactions(db_config):
    db = Database(db_config)
    cursor = db.conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT i.student_id, i.item_id, i.interaction_type,
                   s.interests, s.academic_level, s.learning_style
            FROM interactions i
            JOIN students s ON i.student_id = s.student_id
        """)
        interactions = []
        for row in cursor.fetchall():
            academic_level = {'Beginner': 1.0, 'Intermediate': 2.0, 'Advanced': 3.0}.get(row['academic_level'], 0.0)
            learning_style = {'Visual': 1.0, 'Auditory': 2.0, 'Kinesthetic': 3.0}.get(row['learning_style'], 0.0)
            item_type = {'course_enroll': 1.0, 'tutor_hire': 2.0, 'resource_use': 3.0}.get(row['interaction_type'], 0.0)
            interests = json.loads(row['interests'])
            interests_vector = [1.0 if s in interests else 0.0 for s in ['Math', 'Science', 'Literature']]
            item_features = get_item_features(db, row['item_id'], row['interaction_type'])
            interactions.append({
                'student_id': row['student_id'],
                'item_id': row['item_id'],
                'interests': interests_vector,
                'academic_level': [academic_level],
                'learning_style': [learning_style],
                'item_type': [item_type],
                'features': item_features
            })
        return interactions
    finally:
        cursor.close()

def get_item_features(db, item_id, interaction_type):
    cursor = db.conn.cursor(dictionary=True)
    try:
        if interaction_type == 'course_enroll':
            cursor.execute("SELECT subject, difficulty, format FROM courses WHERE course_id = %s", (item_id,))
            row = cursor.fetchone()
            return [
                {'Math': 1.0, 'Science': 2.0, 'Literature': 3.0}.get(row['subject'], 0.0),
                {'Beginner': 1.0, 'Intermediate': 2.0, 'Advanced': 3.0}.get(row['difficulty'], 0.0),
                {'online': 1.0, 'offline': 2.0, 'hybrid': 3.0}.get(row['format'], 0.0)
            ]
        elif interaction_type == 'tutor_hire':
            cursor.execute("SELECT expertise, teaching_style, rating FROM tutors WHERE tutor_id = %s", (item_id,))
            row = cursor.fetchone()
            expertise = json.loads(row['expertise'])
            expertise_vector = [1.0 if s in expertise else 0.0 for s in ['Math', 'Science', 'Literature']]
            return expertise_vector + [
                {'Interactive': 1.0, 'Lecture': 2.0}.get(row['teaching_style'], 0.0),
                row['rating']
            ]
        else:  # resource_use
            cursor.execute("SELECT subject, format, difficulty FROM resources WHERE resource_id = %s", (item_id,))
            row = cursor.fetchone()
            return [
                {'Math': 1.0, 'Science': 2.0, 'Literature': 3.0}.get(row['subject'], 0.0),
                {'video': 1.0, 'pdf': 2.0, 'article': 3.0}.get(row['format'], 0.0),
                {'Beginner': 1.0, 'Intermediate': 2.0, 'Advanced': 3.0}.get(row['difficulty'], 0.0)
            ]
    finally:
        cursor.close()