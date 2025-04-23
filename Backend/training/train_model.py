import tensorflow as tf
import tensorflow_recommenders as tfrs
from ..models.two_tower import MultiTowerRecommenderModel
from .fetch_interactions import fetch_interactions
from .prepare_dataset import prepare_dataset

def train_model(db_config, model_path='backend/models/weights/multi_tower_model'):
    # Lấy dữ liệu
    interactions = fetch_interactions(db_config)
    
    # Tạo vocab
    student_vocab = list(set([i['student_id'] for i in interactions]))
    course_vocab = list(set([i['item_id'] for i in interactions if i['item_type'][0] == 1.0]))
    tutor_vocab = list(set([i['item_id'] for i in interactions if i['item_type'][0] == 2.0]))
    resource_vocab = list(set([i['item_id'] for i in interactions if i['item_type'][0] == 3.0]))
    
    # Khởi tạo mô hình
    model = MultiTowerRecommenderModel(
        student_vocab=student_vocab,
        course_vocab=course_vocab,
        tutor_vocab=tutor_vocab,
        resource_vocab=resource_vocab
    )
    
    # Biên dịch mô hình
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    
    # Chuẩn bị dataset
    train_dataset, test_dataset = prepare_dataset(interactions)
    
    # Huấn luyện
    model.fit(train_dataset, epochs=10, verbose=1)
    
    # Đánh giá
    metrics = model.evaluate(test_dataset, return_dict=True)
    print("Evaluation metrics:", metrics)
    
    # Lưu mô hình
    model.save_model(model_path)
    return model

if __name__ == '__main__':
    db_config = {
        'host': 'db',
        'user': 'user',
        'password': 'password',
        'database': 'education'
    }
    train_model(db_config)