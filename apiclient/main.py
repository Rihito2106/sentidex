import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib import — required for Docker/server env

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path so apiclient can import from services/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from services.helpers import preprocess_comment
from services.model_service import load_model_and_vectorizer
from services.visualization_service import generate_sentiment_chart, generate_wordcloud_image, generate_trend_chart
from services.analytics_service import generate_advanced_analytics_dashboard

# Matplotlib / Seaborn styling
plt.style.use('dark_background')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)

dotenv.load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

app = Flask(__name__)
CORS(app, origins=["*"])

# ── Option A: Load via MLflow Model Registry (uncomment if using MLflow on EC2) ──
# import mlflow
# import mlflow.sklearn
# import pickle
#
# def load_model_and_vectorizer_mlflow(model_name, model_version, vectorizer_path):
#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
#     with open(vectorizer_path, 'rb') as file:
#         vectorizer = pickle.load(file)
#     return model, vectorizer
#
# model_name = "sentidex-lgbm"
# model_version = 1

# ── Option B: Load pkl files directly (default — simpler, used for Docker deploy) ──
# Paths are relative to project root (WORKDIR /app in Docker, or sentidex/ locally)
models_dir = os.path.join(parent_dir, 'models')
vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
model_path = os.path.join(models_dir, 'lgbm_model.pkl')

model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return "Welcome to the Sentidex YouTube Sentiment Analysis API!"


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments.toarray()).tolist()

        return jsonify([{"comment": c, "sentiment": s}
                        for c, s in zip(comments, predictions)])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.get_json()
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments    = [item['text']      for item in comments_data]
        timestamps  = [item['timestamp'] for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments  = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments.toarray()).tolist()
        predictions = [str(pred) for pred in predictions]

        response = [{"comment": c, "sentiment": s, "timestamp": t}
                    for c, s, t in zip(comments, predictions, timestamps)]
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        img_io = generate_sentiment_chart(sentiment_counts)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        img_io = generate_wordcloud_image(comments)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        img_io = generate_trend_chart(sentiment_data)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


@app.route('/generate_advanced_analytics', methods=['POST'])
def generate_advanced_analytics():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')
        comments_data  = data.get('comments')

        if not sentiment_data or not comments_data:
            return jsonify({"error": "Insufficient data provided"}), 400

        img_io = generate_advanced_analytics_dashboard(sentiment_data, comments_data)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_advanced_analytics: {e}")
        return jsonify({"error": f"Advanced analytics generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)