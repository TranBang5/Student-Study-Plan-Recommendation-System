from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Student:
    student_id: str
    interests: List[str]
    academic_level: str
    learning_style: str

    @classmethod
    def from_dict(cls, data: dict) -> 'Student':
        return cls(
            student_id=data['student_id'],
            interests=data['interests'].split(',') if isinstance(data['interests'], str) else data['interests'],
            academic_level=data['academic_level'],
            learning_style=data['learning_style']
        )

@dataclass
class Course:
    course_id: str
    subject: str
    level: str
    format: str

    @classmethod
    def from_dict(cls, data: dict) -> 'Course':
        return cls(
            course_id=data['course_id'],
            subject=data['subject'],
            level=data['level'],
            format=data['format']
        )

@dataclass
class Tutor:
    tutor_id: str
    expertise: List[str]
    experience_years: int
    teaching_style: str

    @classmethod
    def from_dict(cls, data: dict) -> 'Tutor':
        return cls(
            tutor_id=data['tutor_id'],
            expertise=data['expertise'].split(',') if isinstance(data['expertise'], str) else data['expertise'],
            experience_years=int(data['experience_years']),
            teaching_style=data['teaching_style']
        )

@dataclass
class Resource:
    resource_id: str
    topic: str
    type: str
    level: List[str]

    @classmethod
    def from_dict(cls, data: dict) -> 'Resource':
        return cls(
            resource_id=data['resource_id'],
            topic=data['topic'],
            type=data['type'],
            level=data['level'].split(',') if isinstance(data['level'], str) else data['level']
        )

@dataclass
class Interaction:
    interaction_id: str
    student_id: str
    item_id: str
    interaction_type: str
    schedule: List[str]

    @classmethod
    def from_dict(cls, data: dict) -> 'Interaction':
        schedule = data['schedule'].split('|') if isinstance(data['schedule'], str) and '|' in data['schedule'] else [data['schedule']]
        return cls(
            interaction_id=data['interaction_id'],
            student_id=data['student_id'],
            item_id=data['item_id'],
            interaction_type=data['interaction_type'],
            schedule=schedule
        )
