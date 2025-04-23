import mysql.connector
from mysql.connector import Error
from .entities import StudentProfile, Course, Tutor, Resource, StudyPlan
import json

class Database:
    def __init__(self, conn_string: dict):
        try:
            self.conn = mysql.connector.connect(**conn_string)
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            raise

    def save_student(self, student: StudentProfile):
        cursor = self.conn.cursor(dictionary=True)
        try:
            query = """
                INSERT INTO students (student_id, interests, academic_level, learning_style, past_courses, performance_history)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    interests = VALUES(interests),
                    academic_level = VALUES(academic_level),
                    learning_style = VALUES(learning_style),
                    past_courses = VALUES(past_courses),
                    performance_history = VALUES(performance_history)
            """
            cursor.execute(query, (
                student.student_id,
                json.dumps(student.interests),
                student.academic_level,
                student.learning_style,
                json.dumps(student.past_courses),
                json.dumps(student.performance_history)
            ))
            self.conn.commit()
        finally:
            cursor.close()

    def get_student(self, student_id: str) -> StudentProfile:
        cursor = self.conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM students WHERE student_id = %s", (student_id,))
            data = cursor.fetchone()
            if data:
                return StudentProfile(
                    student_id=data['student_id'],
                    interests=json.loads(data['interests']),
                    academic_level=data['academic_level'],
                    learning_style=data['learning_style'],
                    past_courses=json.loads(data['past_courses']),
                    performance_history=json.loads(data['performance_history'])
                )
            return None
        finally:
            cursor.close()

    def get_all_courses(self) -> list[Course]:
        cursor = self.conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM courses")
            return [Course(**row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_all_tutors(self) -> list[Tutor]:
        cursor = self.conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM tutors")
            rows = cursor.fetchall()
            return [Tutor(
                tutor_id=row['tutor_id'],
                name=row['name'],
                expertise=json.loads(row['expertise']),
                rating=row['rating'],
                teaching_style=row['teaching_style'],
                availability=json.loads(row['availability'])
            ) for row in rows]
        finally:
            cursor.close()

    def get_all_resources(self) -> list[Resource]:
        cursor = self.conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM resources")
            return [Resource(**row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def save_study_plan(self, plan: StudyPlan):
        cursor = self.conn.cursor(dictionary=True)
        try:
            query = """
                INSERT INTO study_plans (plan_id, student_id, courses, tutors, resources, schedule, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                plan.plan_id,
                plan.student_id,
                json.dumps(plan.courses),
                json.dumps(plan.tutors),
                json.dumps(plan.resources),
                json.dumps(plan.schedule),
                plan.created_at
            ))
            self.conn.commit()
        finally:
            cursor.close()