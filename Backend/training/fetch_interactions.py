import pandas as pd
import csv

def load_data():
    # Read CSV files with proper quoting
    students = pd.read_csv('/app/data/students.csv', quoting=csv.QUOTE_NONNUMERIC)
    courses = pd.read_csv('/app/data/courses.csv', quoting=csv.QUOTE_NONNUMERIC)
    tutors = pd.read_csv('/app/data/tutors.csv', quoting=csv.QUOTE_NONNUMERIC)
    resources = pd.read_csv('/app/data/resources.csv', quoting=csv.QUOTE_NONNUMERIC)
    interactions = pd.read_csv('/app/data/interactions.csv', quoting=csv.QUOTE_NONNUMERIC)

    # Split comma-separated fields into lists
    students['interests'] = students['interests'].str.split(',')
    resources['level'] = resources['level'].str.split(',')
    tutors['expertise'] = tutors['expertise'].str.split(',')
    interactions['schedule'] = interactions['schedule'].apply(
        lambda x: x.split('|') if '|' in x else [x]
    )

    return {
        'students': students,
        'courses': courses,
        'tutors': tutors,
        'resources': resources,
        'interactions': interactions
    }

def get_student_data(student_id, data):
    # Get student information
    student = data['students'][data['students']['student_id'] == student_id]
    if student.empty:
        return None
    return {
        'student_id': student_id,
        'interests': student['interests'].iloc[0],
        'academic_level': student['academic_level'].iloc[0],
        'learning_style': student['learning_style'].iloc[0]
    }

def get_recommendation_inputs(student_id, data):
    # Prepare inputs for recommendation model
    student_data = get_student_data(student_id, data)
    if not student_data:
        return None

    # Filter resources based on academic_level
    student_level = student_data['academic_level']
    matching_resources = data['resources'][
        data['resources']['level'].apply(lambda x: student_level in x)
    ]

    # Filter courses and tutors based on level and interests
    matching_courses = data['courses'][
        (data['courses']['level'] == student_level) &
        (data['courses']['subject'].isin(student_data['interests']))
    ]
    matching_tutors = data['tutors'][
        data['tutors']['expertise'].apply(
            lambda x: any(subject in x for subject in student_data['interests'])
        )
    ]

    return {
        'student': student_data,
        'resources': matching_resources.to_dict('records'),
        'courses': matching_courses.to_dict('records'),
        'tutors': matching_tutors.to_dict('records')
    }

if __name__ == "__main__":
    data = load_data()
    print("Students:\n", data['students'].head())
    print("\nResources:\n", data['resources'].head())
    print("\nSample recommendation inputs for s1:")
    print(get_recommendation_inputs('s1', data))