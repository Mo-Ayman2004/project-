import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import pickle
import chardet

st.set_page_config(
    page_title="Customer Sales Prediction",
    page_icon="📊",
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
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-high {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .prediction-medium {
        background: linear-gradient(135deg, #ffd43b, #fcc419);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    .prediction-low {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏢 Customer Sales Prediction Dashboard</div>', unsafe_allow_html=True)

# Detect encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

# Load data
@st.cache_data
def load_data():
    try:
        file_path = r"C:\Users\Media\Desktop\Original Data.csv"
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252', 'cp1252']
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                st.success(f"✅ Successfully loaded with {enc} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                continue

        if df is None:
            df = pd.read_csv(file_path, encoding='latin-1', errors='replace')
            st.warning("⚠ Loaded with error replacement - some characters may be replaced")

        # Convert date columns
        date_columns = ['Order Date', 'Ship Date']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                except Exception as e:
                    st.warning(f"⚠ Could not convert {col}: {e}")

        from pandas.tseries.holiday import USFederalHolidayCalendar

        # Create features
        if 'Order Date' in df.columns:
            df['Month'] = df['Order Date'].dt.month
            df['Quarter'] = df['Order Date'].dt.quarter
            df['Year'] = df['Order Date'].dt.year
            df['Day_of_Week'] = df['Order Date'].dt.dayofweek
            df['Is_Weekend'] = df['Day_of_Week'].isin([5,6]).astype(int)

        # Holiday feature
        try:
            if 'Order Date' in df.columns:
                cal = USFederalHolidayCalendar()
                holidays = cal.holidays(start=df['Order Date'].min(), end=df['Order Date'].max())
                df['Is_Holiday'] = df['Order Date'].isin(holidays).astype(int)
        except Exception as e:
            st.warning(f"⚠ Could not create holiday features: {e}")

        # Season
        if 'Month' in df.columns:
            df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12,1,2] else
                                            'Spring' if x in [3,4,5] else
                                            'Summer' if x in [6,7,8] else
                                            'Fall')

        # Total sales
        if all(col in df.columns for col in ['Sales','Quantity']):
            df['Total_sales'] = df['Sales'] * df['Quantity']
        else:
            st.error("❌ Missing 'Sales' or 'Quantity' columns")

        # Drop unnecessary columns
        columns_to_drop = ['Row ID','Order ID','Customer ID','Postal Code','Product ID']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

        # Numeric columns
        numeric_columns = ["Sales", "Discount", "Profit", "Shipping Cost"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)

        # Promotion
        if 'Discount' in df.columns:
            df['Promotion_Flag'] = (df['Discount']>0).astype(int)
            avg_discount = df['Discount'].mean()
            df['Promotion_Above_Avg'] = (df['Discount']>avg_discount).astype(int)

        # Additional date features
        if 'Order Date' in df.columns:
            df['Day_of_Month'] = df['Order Date'].dt.day

        # Label encoding
        categorical_columns = ["Order Priority","Ship Mode","Segment","Market","Category","Sub-Category","Region","Season"]
        encoder = LabelEncoder()
        for col in categorical_columns:
            if col in df.columns:
                try:
                    df[col+"_Encoded"] = encoder.fit_transform(df[col].astype(str))
                except:
                    continue
        # Drop original categorical
        original_cat_cols = ["Order Priority","Ship Mode","Segment","Market","Category","Sub-Category","Region"]
        df.drop(columns=[col for col in original_cat_cols if col in df.columns], inplace=True, errors='ignore')

        # Drop final unnecessary columns
        final_drop = ['Customer Name','Product Name','Order Date','Ship Date','City','State','Country','Sales','Region_Encoded','Segment_Encoded']
        df.drop(columns=[col for col in final_drop if col in df.columns], inplace=True, errors='ignore')

        # Handle missing
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['float64','int64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

        # Prepare X,y
        if 'Total_sales' in df.columns:
            X = df.drop('Total_sales', axis=1)
            y = df['Total_sales']
            for col in X.columns:
                if X[col].dtype=='object':
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        X[col].fillna(X[col].mean(), inplace=True)
                    except:
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            return df, X, y
        else:
            st.error("❌ 'Total_sales' column not found")
            return None, None, None

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = r"C:\Users\Media\Desktop\xgboost_sales_model (1).pkl"
        with open(model_path,'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error("Please make sure the model file exists at the specified path")
        return None

# Load data & model
df, X, y = load_data()
best_model = load_model()

if df is None:
    st.error("Failed to load data. Check data file path.")
    st.stop()

correct_feature_order = list(best_model.feature_names_in_) if best_model is not None and hasattr(best_model,'feature_names_in_') else list(X.columns)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview","Model Info","Sales Prediction"])

# --- Pages ---
if page=="Data Overview":
    st.header("📊 Data Overview")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    with col2:
        st.subheader("Data Statistics")
        st.dataframe(df.describe())
        st.subheader("Features used for prediction")
        for col in X.columns:
            st.write(f"- {col}")

    st.subheader("Sales Distribution Analysis")
    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(10,6))
        ax.hist(df['Total_sales'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Total Sales')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Total Sales')
        st.pyplot(fig)
    with col2:
        monthly_sales = df.groupby('Month')['Total_sales'].mean()
        fig,ax = plt.subplots(figsize=(10,6))
        monthly_sales.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Sales')
        ax.set_title('Average Sales by Month')
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif page=="Model Info":
    st.header("🤖 Model Information")
    if best_model is not None:
        st.success("✅ XGBoost Model Loaded Successfully!")
        st.subheader("Model Details")
        st.write(f"*Model Type:* {type(best_model).__name__}")
        st.subheader("Feature Order Required by Model")
        st.write(correct_feature_order)
        if hasattr(best_model,'feature_importances_'):
            st.subheader("Feature Importance")
            feat_imp = pd.DataFrame({'feature':correct_feature_order,'importance':best_model.feature_importances_}).sort_values('importance',ascending=True)
            fig,ax=plt.subplots(figsize=(10,8))
            ax.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance for Sales Prediction')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
            st.subheader("Top 5 Features")
            top5 = feat_imp.tail(5)
            for idx,row in top5.iterrows():
                st.write(f"{row['feature']}: {row['importance']:.4f}")
        else:
            st.info("Feature importance not available")
    else:
        st.error("❌ Model failed to load")

elif page=="Sales Prediction":
    st.header("🔮 Sales Prediction")
    if best_model is None:
        st.error("Load the model first")
        st.stop()
    st.markdown("### Enter Customer/Order Details")
    with st.form("prediction_form"):
        col1,col2 = st.columns(2)
        with col1:
            Quantity = st.slider("Quantity",1,20,3)
            Discount = st.slider("Discount Rate",0.0,1.0,0.1,0.01)
            Profit = st.number_input("Expected Profit",-1000.0,10000.0,50.0,10.0)
            Shipping_Cost = st.number_input("Shipping Cost",0.0,1000.0,25.0,5.0)
        with col2:
            Month = st.selectbox("Month",options=list(range(1,13)),index=5)
            Year = st.selectbox("Year", options=[2011,2012,2013,2014,2015], index=3)
            Day_of_Week = st.selectbox("Day of Week", options=list(range(7)), index=2)
            Is_Weekend = st.selectbox("Is Weekend", options=[0,1], index=0)
            Day_of_Month = st.slider("Day of Month",1,31,15)
        col3,col4 = st.columns(2)
        with col3:
            Ship_Mode_Encoded = st.selectbox("Ship Mode", options=[0,1,2,3], index=1)
            Market_Encoded = st.selectbox("Market", options=[0,1,2,3,4], index=3)
        with col4:
            Category_Encoded = st.selectbox("Category", options=[0,1,2], index=1)
            Sub_Category_Encoded = st.slider("Sub-Category Code",0,16,8)
            Order_Priority_Encoded = st.selectbox("Order Priority", options=[0,1,2,3], index=2)
        submitted = st.form_submit_button("Predict Sales", use_container_width=True)
        if submitted:
            input_dict = {
                'Quantity': Quantity, 'Discount': Discount, 'Profit': Profit, 'Shipping Cost': Shipping_Cost,
                'Month': Month, 'Year': Year, 'Day_of_Week': Day_of_Week, 'Is_Weekend': Is_Weekend,
                'Day_of_Month': Day_of_Month, 'Ship Mode_Encoded': Ship_Mode_Encoded,
                'Market_Encoded': Market_Encoded, 'Category_Encoded': Category_Encoded,
                'Sub-Category_Encoded': Sub_Category_Encoded, 'Order Priority_Encoded': Order_Priority_Encoded
            }
            input_data_ordered=[]
            for feat in correct_feature_order:
                if feat in input_dict:
                    input_data_ordered.append(input_dict[feat])
                else:
                    default_value = X[feat].mean() if feat in X.columns else 0
                    input_data_ordered.append(default_value)
                    st.warning(f"⚠ Using default value for missing feature: {feat}")
            input_df = pd.DataFrame([input_data_ordered], columns=correct_feature_order)
            try:
                prediction = best_model.predict(input_df)[0]
                st.markdown("### 📊 Prediction Result")
                sales_75 = df['Total_sales'].quantile(0.75)
                sales_25 = df['Total_sales'].quantile(0.25)
                if prediction>sales_75:
                    cls="prediction-high"; cat="High Sales"; emoji="💰"
                elif prediction>sales_25:
                    cls="prediction-medium"; cat="Medium Sales"; emoji="📈"
                else:
                    cls="prediction-low"; cat="Low Sales"; emoji="📉"
                c1,c2,c3 = st.columns([1,2,1])
                with c2:
                    st.markdown(f'<div class="{cls}"><h2>{emoji} {cat}</h2><h1>${prediction:,.2f}</h1><p>Predicted Total Sales Amount</p></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.error("Check required features and order")

st.markdown("---")
st.markdown("### 💡 About This Dashboard")
st.markdown("""
This dashboard provides sales prediction capabilities using a pre-trained XGBoost model.
- *Data Overview*: Explore the dataset and understand feature distributions
- *Model Info*: Learn about the trained model and feature importance
- *Sales Prediction*: Predict total sales based on customer and order features

*Note*: The model uses historical sales data to forecast future sales performance.
""")
