"""
=============================================================================
DIAMOND PRICE PREDICTOR - STREAMLIT APP
=============================================================================
A web application to predict diamond prices using a trained Gradient Boosting model.
Features: sidebar inputs, price prediction with confidence bands, and model card.
"""

import streamlit as st
import numpy as np
import joblib
import json

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="💎 Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD MODEL AND METADATA
# =============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and metadata."""
    model = joblib.load('/Users/Marcy_Student/Desktop/Marcy_Lab/DA2025_Lectures/Mod7/diamond_model.joblib')
    with open('/Users/Marcy_Student/Desktop/Marcy_Lab/DA2025_Lectures/Mod7/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, metadata

try:
    model, metadata = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# =============================================================================
# ENCODING MAPS
# =============================================================================
CUT_OPTIONS = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
COLOR_OPTIONS = ['J', 'I', 'H', 'G', 'F', 'E', 'D']  # J=worst, D=best
CLARITY_OPTIONS = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

CUT_MAP = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
COLOR_MAP = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
CLARITY_MAP = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

# =============================================================================
# SIDEBAR - INPUT CONTROLS
# =============================================================================
st.sidebar.title("💎 Diamond Features")
st.sidebar.markdown("---")

# Carat input
carat = st.sidebar.slider(
    "Carat (Weight)",
    min_value=0.2,
    max_value=5.0,
    value=1.0,
    step=0.01,
    help="Diamond weight in carats. 1 carat = 0.2 grams"
)

# Cut input
cut = st.sidebar.selectbox(
    "Cut Quality",
    options=CUT_OPTIONS,
    index=4,  # Default to Ideal
    help="Quality of the diamond's cut (Fair → Ideal)"
)

# Color input
color = st.sidebar.selectbox(
    "Color Grade",
    options=COLOR_OPTIONS,
    index=6,  # Default to D (best)
    help="Diamond color grade (J=yellowish → D=colorless)"
)

# Clarity input
clarity = st.sidebar.selectbox(
    "Clarity Grade",
    options=CLARITY_OPTIONS,
    index=7,  # Default to IF (best)
    help="Diamond clarity grade (I1=included → IF=internally flawless)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Presets")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("💍 Engagement Ring", use_container_width=True):
        st.session_state['preset'] = 'engagement'
with col2:
    if st.button("💰 Budget Option", use_container_width=True):
        st.session_state['preset'] = 'budget'

# =============================================================================
# MAIN CONTENT
# =============================================================================
st.title("💎 Diamond Price Predictor")
st.markdown("Predict diamond prices using machine learning based on the 4 C's: **Carat**, **Cut**, **Color**, and **Clarity**.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Card", "📋 Documentation"])

# =============================================================================
# TAB 1: PREDICTION
# =============================================================================
with tab1:
    if model_loaded:
        # Encode inputs
        cut_encoded = CUT_MAP[cut]
        color_encoded = COLOR_MAP[color]
        clarity_encoded = CLARITY_MAP[clarity]
        
        # Create feature array
        features = np.array([[carat, cut_encoded, color_encoded, clarity_encoded]])
        
        # Make prediction (log scale)
        log_price_pred = model.predict(features)[0]
        
        # Convert to actual price
        price_pred = np.exp(log_price_pred)
        
        # Calculate confidence band (90% interval)
        lower_offset = metadata['prediction_intervals']['lower_bound_offset']
        upper_offset = metadata['prediction_intervals']['upper_bound_offset']
        
        price_lower = np.exp(log_price_pred + lower_offset)
        price_upper = np.exp(log_price_pred + upper_offset)
        
        # Display prediction
        st.markdown("### Predicted Price")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.metric(
                label="Estimated Price",
                value=f"${price_pred:,.2f}",
                delta=None
            )
            
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px;'>
                <p style='margin: 0; color: #666;'>90% Confidence Interval</p>
                <p style='margin: 0; font-size: 1.2em;'>
                    <strong>${price_lower:,.2f}</strong> — <strong>${price_upper:,.2f}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature summary
        st.markdown("### Your Diamond")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background-color: #e8f4ea; border-radius: 10px;'>
                <p style='margin: 0; color: #666; font-size: 0.9em;'>Carat</p>
                <p style='margin: 0; font-size: 1.5em; font-weight: bold;'>{carat}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background-color: #e8f0f4; border-radius: 10px;'>
                <p style='margin: 0; color: #666; font-size: 0.9em;'>Cut</p>
                <p style='margin: 0; font-size: 1.5em; font-weight: bold;'>{cut}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background-color: #f4f0e8; border-radius: 10px;'>
                <p style='margin: 0; color: #666; font-size: 0.9em;'>Color</p>
                <p style='margin: 0; font-size: 1.5em; font-weight: bold;'>{color}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background-color: #f4e8f0; border-radius: 10px;'>
                <p style='margin: 0; color: #666; font-size: 0.9em;'>Clarity</p>
                <p style='margin: 0; font-size: 1.5em; font-weight: bold;'>{clarity}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Warning for edge cases
        if carat < 0.3 or carat > 4.5:
            st.warning("⚠️ **Note:** Your carat value is near the edge of the training data range. Predictions may be less accurate.")
    else:
        st.error("Model not loaded. Please check that model files exist.")

# =============================================================================
# TAB 2: MODEL CARD
# =============================================================================
with tab2:
    st.markdown("## 📊 Model Card")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Information")
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | **Model Type** | Gradient Boosting Regressor |
        | **Target** | Log(Price) → Price |
        | **Training Samples** | {metadata['data_stats']['n_samples']:,} |
        | **Test R²** | {metadata['metrics']['test_r2']:.4f} |
        | **CV R² (5-fold)** | {metadata['metrics']['cv_mean']:.4f} ± {metadata['metrics']['cv_std']:.4f} |
        """)
        
        st.markdown("### Features Used")
        st.markdown("""
        | Feature | Description | Encoding |
        |---------|-------------|----------|
        | **Carat** | Diamond weight | Numeric (0.2 - 5.0) |
        | **Cut** | Cut quality | Fair(0) → Ideal(4) |
        | **Color** | Color grade | J(0) → D(6) |
        | **Clarity** | Clarity grade | I1(0) → IF(7) |
        """)
    
    with col2:
        st.markdown("### Feature Importance")
        
        importance = metadata['feature_importance']
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            label = feat.replace('_encoded', '').title()
            st.progress(imp, text=f"{label}: {imp:.1%}")
        
        st.markdown("### Sanity Checks")
        checks = metadata['sanity_checks']
        
        st.markdown(f"""
        | Check | Status | Value |
        |-------|--------|-------|
        | VIF (Multicollinearity) | {'✅ Pass' if checks['vif_pass'] else '❌ Fail'} | All < 5 |
        | Overfitting (Gap < 0.02) | {'✅ Pass' if checks['overfit_pass'] else '❌ Fail'} | {checks['overfit_gap']:.4f} |
        | CV Stability (Std < 0.01) | {'✅ Pass' if checks['cv_stable'] else '❌ Fail'} | {metadata['metrics']['cv_std']:.4f} |
        | Residual Bias (~0) | {'✅ Pass' if abs(checks['residual_mean']) < 0.01 else '❌ Fail'} | {checks['residual_mean']:.6f} |
        """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Known Limitations")
    
    for limitation in metadata['known_limitations']:
        st.markdown(f"- {limitation}")

# =============================================================================
# TAB 3: DOCUMENTATION
# =============================================================================
with tab3:
    st.markdown("## 📋 Documentation")
    
    st.markdown("""
    ### How It Works
    
    This app uses a **Gradient Boosting Regressor** trained on the classic Seaborn Diamonds dataset 
    to predict diamond prices based on four key characteristics.
    
    #### The 4 C's of Diamonds
    
    1. **Carat** - The weight of the diamond. 1 carat = 200 milligrams. This is the most 
       important factor in determining price (accounts for ~95% of predictions).
    
    2. **Cut** - How well the diamond has been cut and polished. Grades from Fair to Ideal.
       A better cut means better light reflection and sparkle.
    
    3. **Color** - Diamond color grades range from D (colorless) to J (light yellow).
       Colorless diamonds are more rare and valuable.
    
    4. **Clarity** - Measures internal flaws (inclusions) and surface defects (blemishes).
       Grades from I1 (visible inclusions) to IF (internally flawless).
    
    ---
    
    ### Technical Details
    
    #### Data Preprocessing
    - **Cleaning**: Removed 20 diamonds with zero dimensions (measurement errors)
    - **Log Transform**: Applied `log(price)` to handle right-skewed price distribution
    - **Encoding**: Ordinal encoding for categorical features (preserves natural ordering)
    
    #### Model Training
    - **Algorithm**: Gradient Boosting with regularization to prevent overfitting
    - **Hyperparameters**: 100 trees, max_depth=5, learning_rate=0.1
    - **Validation**: 5-fold cross-validation with stratified train-test split
    
    #### Confidence Intervals
    - Based on empirical residual distribution (5th and 95th percentiles)
    - Represents 90% prediction interval
    - Note: These are approximate, not theoretical confidence bounds
    
    ---
    
    ### API Reference
    
    ```python
    # Load the model
    import joblib
    model = joblib.load('diamond_model.joblib')
    
    # Make a prediction
    import numpy as np
    features = np.array([[carat, cut_encoded, color_encoded, clarity_encoded]])
    log_price = model.predict(features)[0]
    price = np.exp(log_price)
    ```
    
    #### Encoding Reference
    
    | Cut | Code | | Color | Code | | Clarity | Code |
    |-----|------|-|-------|------|-|---------|------|
    | Fair | 0 | | J | 0 | | I1 | 0 |
    | Good | 1 | | I | 1 | | SI2 | 1 |
    | Very Good | 2 | | H | 2 | | SI1 | 2 |
    | Premium | 3 | | G | 3 | | VS2 | 3 |
    | Ideal | 4 | | F | 4 | | VS1 | 4 |
    | | | | E | 5 | | VVS2 | 5 |
    | | | | D | 6 | | VVS1 | 6 |
    | | | | | | | IF | 7 |
    """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>Built with Streamlit • Model: Gradient Boosting • Data: Seaborn Diamonds Dataset</p>",
    unsafe_allow_html=True
)