import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .excellent { background-color: #d4edda; color: #155724; }
    .good { background-color: #d1ecf1; color: #0c5460; }
    .average { background-color: #fff3cd; color: #856404; }
    .below-average { background-color: #f8d7da; color: #721c24; }
    .poor { background-color: #f5c6cb; color: #721c24; }
    .very-poor { background-color: #f1b0b7; color: #721c24; }
    .critical { background-color: #dc3545; color: white; }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('score_predictor_model.pkl')
        le_dept = joblib.load('label_encoder_dept.pkl')
        le_time = joblib.load('label_encoder_time.pkl')
        le_target = joblib.load('label_encoder_target.pkl')
        return model, le_dept, le_time, le_target
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run 'train_model.py' first.")
        st.stop()

model, le_dept, le_time, le_target = load_model_artifacts()

# Performance band styling
def get_band_class(band):
    band_classes = {
        'Excellent': 'excellent',
        'Good': 'good',
        'Average': 'average',
        'Below Average': 'below-average',
        'Poor': 'poor',
        'Very Poor': 'very-poor',
        'Critical': 'critical'
    }
    return band_classes.get(band, 'average')

# Header
st.markdown('<h1 class="main-header">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict student performance band based on test-taking patterns")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/student-male.png", width=100)
    st.title("Input Features")
    st.markdown("---")
    
    # Department selection
    department = st.selectbox(
        "📚 Department",
        options=['Microbiology - Morning Class', 'Microbiology - Evening Class'],
        help="Select the student's department"
    )
    
    # Submission hour
    submission_hour = st.slider(
        "🕐 Submission Hour",
        min_value=6,
        max_value=21,
        value=9,
        help="Hour of test submission (24-hour format)"
    )
    
    # Time period (auto-calculated)
    if 6 <= submission_hour <= 9:
        time_period = 'Early Morning'
        time_emoji = '🌅'
    elif 10 <= submission_hour <= 13:
        time_period = 'Mid Morning'
        time_emoji = '☀️'
    elif 14 <= submission_hour <= 17:
        time_period = 'Afternoon'
        time_emoji = '🌤️'
    else:
        time_period = 'Evening'
        time_emoji = '🌙'
    
    st.info(f"{time_emoji} **Time Period:** {time_period}")
    
    # Number of attempts
    num_attempts = st.number_input(
        "🔄 Number of Attempts",
        min_value=1,
        max_value=10,
        value=1,
        help="How many times has the student attempted the test?"
    )
    
    st.markdown("---")
    predict_button = st.button("🔮 Predict Performance", use_container_width=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'Department': [le_dept.transform([department])[0]],
            'submission_hour': [submission_hour],
            'time_of_day': [le_time.transform([time_period])[0]],
            'num_attempts': [num_attempts]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        predicted_band = le_target.inverse_transform([prediction])[0]
        
        # Display prediction
        st.markdown("## 🎯 Prediction Result")
        band_class = get_band_class(predicted_band)
        
        st.markdown(f"""
            <div class="prediction-box {band_class}">
                <h2>Predicted Performance Band</h2>
                <h1>{predicted_band}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence scores
        st.markdown("### 📊 Confidence Scores")
        
        prob_df = pd.DataFrame({
            'Performance Band': le_target.classes_,
            'Probability': prediction_proba * 100
        }).sort_values('Probability', ascending=False)
        
        fig = px.bar(
            prob_df,
            x='Probability',
            y='Performance Band',
            orientation='h',
            color='Probability',
            color_continuous_scale='RdYlGn',
            text=prob_df['Probability'].apply(lambda x: f'{x:.1f}%')
        )
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Confidence (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if predicted_band in ['Critical', 'Very Poor', 'Poor']:
            st.error("""
                **⚠️ High Risk Student**
                - Immediate intervention required
                - Schedule one-on-one tutoring sessions
                - Review fundamental concepts
                - Consider remedial classes
            """)
        elif predicted_band in ['Below Average', 'Average']:
            st.warning("""
                **⚡ Moderate Performance**
                - Additional practice recommended
                - Group study sessions may help
                - Focus on weak areas
                - Regular progress monitoring
            """)
        else:
            st.success("""
                **✅ Strong Performance**
                - Student is on track
                - Encourage continued effort
                - Consider advanced materials
                - Peer tutoring opportunities
            """)

with col2:
    st.markdown("## 📈 Model Insights")
    
    # Feature importance (mock data - replace with actual from model)
    st.markdown("### Feature Importance")
    importance_data = {
        'Feature': ['Submission Hour', 'Department', 'Time Period', 'Attempts'],
        'Importance': [0.45, 0.30, 0.15, 0.10]
    }
    importance_df = pd.DataFrame(importance_data)
    
    fig_importance = px.pie(
        importance_df,
        values='Importance',
        names='Feature',
        hole=0.4
    )
    fig_importance.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Statistics
    st.markdown("### 📊 Dataset Statistics")
    st.metric("Total Students", "199")
    st.metric("Pass Rate", "17.59%")
    st.metric("Average Score", "34.07%")
