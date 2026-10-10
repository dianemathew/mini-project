from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('churn_model.joblib')
scaler = joblib.load('scaler.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    avg_session_duration = float(request.form['avg_session_duration'])
    posts_created = float(request.form['posts_created'])
    likes_given = float(request.form['likes_given'])
    comments_made = float(request.form['comments_made'])
    shares = float(request.form['shares'])
    watch_time = float(request.form['watch_time'])
    notifications_clicked = float(request.form['notifications_clicked'])
    recommended_content_clicks = float(request.form['recommended_content_clicks'])
    ads_clicked = float(request.form['ads_clicked'])
    engagement_score = float(request.form['engagement_score'])
    habit_score = float(request.form['habit_score'])
    age_group_25_34 = float(request.form['age_group_25_34'])
    age_group_35_44 = float(request.form['age_group_35_44'])
    age_group_45 = float(request.form['age_group_45'])

    input_data = np.array([[
        avg_session_duration,
        posts_created,
        likes_given,
        comments_made,
        shares,
        watch_time,
        notifications_clicked,
        recommended_content_clicks,
        ads_clicked,
        engagement_score,
        habit_score,
        age_group_25_34,
        age_group_35_44,
        age_group_45
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        result = "The user is likely to churn."
    else:
        result = "The user is not likely to churn."

    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)