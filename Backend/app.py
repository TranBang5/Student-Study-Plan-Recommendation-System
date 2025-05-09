from flask import Flask, request, jsonify, render_template, redirect, url_for
from .database.db import Database
from .services.recommender import Recommender
from .services.study_plan import StudyPlanService
from .models.two_tower import MultiTowerRecommenderModel
from .models.entities import StudentProfile, Course, Tutor, Resource, StudyPlan
import json

app = Flask(__name__)

# Initialize database and model
db_config = {
    'host': 'db',
    'user': 'user',
    'password': 'password',
    'database': 'education'
}
db = Database(db_config)
courses = db.get_all_courses()
tutors = db.get_all_tutors()
resources = db.get_all_resources()
course_vocab = [c.course_id for c in courses]
tutor_vocab = [t.tutor_id for t in tutors]
resource_vocab = [r.resource_id for r in resources]
student_vocab = []  # Populate with student IDs from DB if needed

model = MultiTowerRecommenderModel(
    student_vocab=student_vocab,
    course_vocab=course_vocab,
    tutor_vocab=tutor_vocab,
    resource_vocab=resource_vocab
)
model.load_model('backend/models/weights/multi_tower_model')
recommender = Recommender(model, courses, tutors, resources)

# Web routes
@app.route('/')
def index():
    return redirect(url_for('profile'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
        student = StudentProfile(
            student_id=request.form['student_id'],
            interests=request.form['interests'].split(','),
            academic_level=request.form['academic_level'],
            learning_style=request.form['learning_style'],
            past_courses=request.form['past_courses'].split(','),
            performance_history=json.loads(request.form['performance_history'])
        )
        db.save_student(student)
        return redirect(url_for('recommendations', student_id=student.student_id))
    return render_template('profile.html')

@app.route('/recommendations/<student_id>', methods=['GET', 'POST'])
def recommendations(student_id):
    student = db.get_student(student_id)
    if not student:
        return "Student not found", 404
    courses = recommender.recommend_courses(student)
    tutors = recommender.recommend_tutors(student)
    resources = recommender.recommend_resources(student)
    
    if request.method == 'POST':
        selected_courses = request.form.getlist('courses')
        selected_tutors = request.form.getlist('tutors')
        selected_resources = request.form.getlist('resources')
        duration_weeks = int(request.form.get('duration_weeks', 4))
        plan = StudyPlanService.create_study_plan(
            student,
            selected_courses,
            selected_tutors,
            selected_resources,
            duration_weeks
        )
        db.save_study_plan(plan)
        return redirect(url_for('study_plan', plan_id=plan.plan_id))
    
    return render_template('recommendations.html', student_id=student_id, courses=courses, tutors=tutors, resources=resources)

@app.route('/study_plan/<plan_id>')
def study_plan(plan_id):
    cursor = db.conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM study_plans WHERE plan_id = %s", (plan_id,))
        data = cursor.fetchone()
        if data:
            plan = StudyPlan(
                plan_id=data['plan_id'],
                student_id=data['student_id'],
                courses=json.loads(data['courses']),
                tutors=json.loads(data['tutors']),
                resources=json.loads(data['resources']),
                schedule=json.loads(data['schedule']),
                created_at=data['created_at']
            )
            return render_template('study_plan.html', plan=plan)
        return "Plan not found", 404
    finally:
        cursor.close()

# API routes
@app.route('/api/profile', methods=['POST'])
def save_profile():
    data = request.json
    student = StudentProfile(
        student_id=data['student_id'],
        interests=data['interests'],
        academic_level=data['academic_level'],
        learning_style=data['learning_style'],
        past_courses=data['past_courses'],
        performance_history=data['performance_history']
    )
    db.save_student(student)
    return jsonify({"message": "Profile saved"})

@app.route('/api/recommendations/<student_id>', methods=['GET'])
def get_recommendations(student_id):
    student = db.get_student(student_id)
    if not student:
        return jsonify({"error": "Student not found"}), 404
    courses = recommender.recommend_courses(student)
    tutors = recommender.recommend_tutors(student)
    resources = recommender.recommend_resources(student)
    return jsonify({
        "courses": [vars(c) for c in courses],
        "tutors": [vars(t) for t in tutors],
        "resources": [vars(r) for r in resources]
    })

@app.route('/api/study_plan', methods=['POST'])
def create_study_plan():
    data = request.json
    student = db.get_student(data['student_id'])
    if not student:
        return jsonify({"error": "Student not found"}), 404
    plan = StudyPlanService.create_study_plan(
        student,
        data['selected_courses'],
        data['selected_tutors'],
        data['selected_resources'],
        data.get('duration_weeks', 4)
    )
    db.save_study_plan(plan)
    return jsonify(vars(plan))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
