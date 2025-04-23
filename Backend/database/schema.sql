CREATE TABLE students (
    student_id VARCHAR(50) PRIMARY KEY,
    interests JSON,
    academic_level VARCHAR(50),
    learning_style VARCHAR(50),
    past_courses JSON,
    performance_history JSON
);

CREATE TABLE courses (
    course_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(200),
    subject VARCHAR(100),
    difficulty VARCHAR(50),
    duration_hours INT,
    format VARCHAR(50)
);

CREATE TABLE tutors (
    tutor_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    expertise JSON,
    rating FLOAT,
    teaching_style VARCHAR(50),
    availability JSON
);

CREATE TABLE resources (
    resource_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(200),
    subject VARCHAR(100),
    format VARCHAR(50),
    difficulty VARCHAR(50),
    estimated_time_minutes INT
);

CREATE TABLE study_plans (
    plan_id VARCHAR(50) PRIMARY KEY,
    student_id VARCHAR(50),
    courses JSON,
    tutors JSON,
    resources JSON,
    schedule JSON,
    created_at DATETIME,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);