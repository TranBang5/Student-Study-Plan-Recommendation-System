from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict

@dataclass
class StudentProfile:
    student_id: str
    interests: List[str]
    academic_level: str
    learning_style: str
    past_courses: List[str]
    performance_history: Dict[str, float]

@dataclass
class Course:
    course_id: str
    title: str
    subject: str
    difficulty: str
    duration_hours: int
    format: str

@dataclass
class Tutor:
    tutor_id: str
    name: str
    expertise: List[str]
    rating: float
    teaching_style: str
    availability: List[str]

@dataclass
class Resource:
    resource_id: str
    title: str
    subject: str
    format: str
    difficulty: str
    estimated_time_minutes: int

@dataclass
class StudyPlan:
    plan_id: str
    student_id: str
    courses: List[str]  # List of course_ids
    tutors: List[str]  # List of tutor_ids
    resources: List[str]  # List of resource_ids
    schedule: Dict[str, List[str]]
    created_at: datetime