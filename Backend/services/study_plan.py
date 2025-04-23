from models.entities import StudentProfile, Course, Tutor, Resource, StudyPlan
from datetime import datetime, timedelta
import uuid
import numpy as np

class StudyPlanService:
    @staticmethod
    def create_study_plan(
        student: StudentProfile,
        selected_courses: List[str],
        selected_tutors: List[str],
        selected_resources: List[str],
        duration_weeks: int = 4
    ) -> StudyPlan:
        schedule = {}
        start_date = datetime.now()
        
        for week in range(duration_weeks):
            week_start = start_date + timedelta(weeks=week)
            for course_id in selected_courses[:2]:
                session_date = week_start + timedelta(days=np.random.randint(0, 5))
                session_key = session_date.strftime('%Y-%m-%d')
                if session_key not in schedule:
                    schedule[session_key] = []
                schedule[session_key].append(f"Course: {course_id}")
            
            if selected_tutors:
                tutor_id = selected_tutors[0]
                session_date = week_start + timedelta(days=np.random.randint(0, 5))
                session_key = session_date.strftime('%Y-%m-%d')
                if session_key not in schedule:
                    schedule[session_key] = []
                schedule[session_key].append(f"Tutor session: {tutor_id}")
            
            for resource_id in selected_resources[:2]:
                session_date = week_start + timedelta(days=np.random.randint(0, 5))
                session_key = session_date.strftime('%Y-%m-%d')
                if session_key not in schedule:
                    schedule[session_key] = []
                schedule[session_key].append(f"Resource: {resource_id}")
        
        return StudyPlan(
            plan_id=str(uuid.uuid4()),
            student_id=student.student_id,
            courses=selected_courses,
            tutors=selected_tutors,
            resources=selected_resources,
            schedule=schedule,
            created_at=datetime.now()
        )