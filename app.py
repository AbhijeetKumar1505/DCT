import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO, BytesIO
import base64
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os
import re
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Data Cleaning Tool",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #4B5D67;
    }
    .stButton button {
        background-color: #4B9CD3;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #357AB7;
    }
    .st-emotion-cache-16txtl3 {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'cleaning_history' not in st.session_state:
    st.session_state.cleaning_history = []
if 'data_history' not in st.session_state:
    st.session_state.data_history = []
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'custom_value' not in st.session_state:
    st.session_state.custom_value = ""

# Functions for data cleaning operations
def remove_duplicates(df):
    """Remove duplicate rows from the dataframe"""
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_after = len(df)
    return df, f"Removed {rows_before - rows_after} duplicate rows"

def fill_missing_values(df, method, columns):
    """Fill missing values using the specified method"""
    for col in columns:
        if method == "Mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
        elif method == "Median":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
        elif method == "Mode":
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif method == "Zero":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
        elif method == "Forward Fill":
            df[col] = df[col].ffill()
        elif method == "Backward Fill":
            df[col] = df[col].bfill()
        elif method == "Custom Value":
            df[col] = df[col].fillna(st.session_state.custom_value)
    
    return df, f"Filled missing values in {', '.join(columns)} using {method}"

def remove_outliers(df, columns, method, threshold=1.5):
    """Remove outliers from the specified columns"""
    rows_before = len(df)
    
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        if method == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == "Z-Score":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores <= threshold]
            
    rows_after = len(df)
    return df, f"Removed {rows_before - rows_after} outliers from {', '.join(columns)} using {method}"

def standardize_column(df, columns):
    """Standardize columns (z-score normalization)"""
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std if std != 0 else df[col]
    
    return df, f"Standardized columns: {', '.join(columns)}"

def normalize_column(df, columns):
    """Normalize columns to range [0, 1]"""
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df, f"Normalized columns: {', '.join(columns)}"

def rename_columns(df, rename_dict):
    """Rename columns based on the provided dictionary"""
    df = df.rename(columns=rename_dict)
    return df, f"Renamed columns: {', '.join([f'{old} → {new}' for old, new in rename_dict.items()])}"

def convert_column_types(df, column_types):
    """Convert column data types"""
    for col, new_type in column_types.items():
        try:
            if new_type == "Numeric":
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif new_type == "String":
                df[col] = df[col].astype(str)
            elif new_type == "Category":
                df[col] = df[col].astype('category')
            elif new_type == "Datetime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            return df, f"Error converting {col} to {new_type}: {e}"
    return df, f"Converted types for columns: {', '.join([f'{col} to {new_type}' for col, new_type in column_types.items()])}"

def clean_text_columns(df, columns, lowercase=True, remove_special=True, trim=True):
    """Clean text columns"""
    for col in columns:
        if pd.api.types.is_string_dtype(df[col]):
            if trim:
                df[col] = df[col].str.strip()
            if lowercase:
                df[col] = df[col].str.lower()
            if remove_special:
                df[col] = df[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    return df, f"Cleaned text in columns: {', '.join(columns)}"

def encode_columns(df, columns, method):
    """Encode categorical columns"""
    if method == "One-Hot":
        df = pd.get_dummies(df, columns=columns, prefix=columns)
        return df, f"One-hot encoded columns: {', '.join(columns)}"
    elif method == "Label":
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col].astype(str))
        return df, f"Label encoded columns: {', '.join(columns)}"

def create_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display: inline-block; padding: 0.5em 1em; color: white; background-color: #4CAF50; text-decoration: none; border-radius: 4px;">{text}</a>'
    return href

def create_log_download_link(history, filename):
    """Generate a download link for the cleaning history log"""
    log_content = "\n".join([f"{i+1}. {action}" for i, action in enumerate(history)])
    b64 = base64.b64encode(log_content.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}" style="display: inline-block; padding: 0.5em 1em; color: white; background-color: #4CAF50; text-decoration: none; border-radius: 4px;">Download Cleaning Log</a>'
    return href

def ensure_consistent_types(df):
    """Ensure all object-type columns are converted to strings"""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

@st.cache_data
def load_data(file):
    """Cache file loading for performance"""
    if file.name.endswith('.csv'):
        return pd.read_csv(file, encoding='utf-8', low_memory=False)
    return pd.read_excel(file, engine='openpyxl')

def main():
    st.title("🧹 Enhanced Data Cleaning Tool")
    
    # Sidebar
    st.sidebar.header("Operations")
    
    # File upload section
    st.sidebar.subheader("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None and (st.session_state.file_name != uploaded_file.name or st.session_state.data is None):
        st.session_state.file_name = uploaded_file.name
        try:
            data = load_data(uploaded_file)
            st.session_state.data = data
            st.session_state.original_data = data.copy()
            st.session_state.cleaning_history = []
            st.session_state.data_history = []
            st.sidebar.success(f"Successfully loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading file: {e}. Please check the file format and try again.")
    
    # If data is loaded, show data statistics and cleaning options
    if st.session_state.data is not None:
        data = ensure_consistent_types(st.session_state.data)
        
        # Data overview
        st.header("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", data.shape[0])
        col2.metric("Columns", data.shape[1])
        col3.metric("Missing Values", data.isna().sum().sum())
        col4.metric("Duplicate Rows", data.duplicated().sum())
        
        # Preview data
        with st.expander("Preview Data", expanded=True):
            st.dataframe(data.head(100), use_container_width=True)
            
        # Column information
        with st.expander("Column Information"):
            col_stats = pd.DataFrame({
                'Type': data.dtypes,
                'Unique Values': data.nunique(),
                'Missing Values': data.isna().sum(),
                'Missing (%)': (data.isna().sum() / len(data) * 100).round(2)
            })
            st.dataframe(col_stats, use_container_width=True)
        
        # Data visualization
        with st.expander("Data Visualization"):
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
            
            viz_col1, viz_col2 = st.columns(2)
            if numeric_cols:
                with viz_col1:
                    st.subheader("Numeric Column Distribution")
                    sel_num_col = st.selectbox("Select numeric column", numeric_cols, key="num_viz")
                    fig = px.histogram(data, x=sel_num_col, nbins=20)
                    st.plotly_chart(fig, use_container_width=True)
            
            if categorical_cols:
                with viz_col2:
                    st.subheader("Categorical Column Distribution")
                    sel_cat_col = st.selectbox("Select categorical column", categorical_cols, key="cat_viz")
                    value_counts = data[sel_cat_col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': sel_cat_col, 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # Advanced visualizations
        with st.expander("Advanced Visualizations"):
            st.subheader("Correlation Heatmap")
            if numeric_cols:
                corr = data[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Scatter Plot")
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis", numeric_cols, index=1, key="scatter_y")
                fig = px.scatter(data, x=x_col, y=y_col)
                st.plotly_chart(fig, use_container_width=True)
        
        # Cleaning operations
        st.sidebar.subheader("2. Select Cleaning Operations")
        
        cleaning_option = st.sidebar.selectbox(
            "Choose operation", 
            ["Remove Duplicates", "Handle Missing Values", "Remove Outliers", 
             "Standardize Data", "Normalize Data", "Rename Columns", "Drop Columns",
             "Convert Data Types", "Clean Text Columns", "Encode Categorical Variables"],
            key="cleaning_option"
        )
        
        # Handle different cleaning operations
        if cleaning_option == "Remove Duplicates":
            st.sidebar.markdown("Remove duplicate rows from the data")
            if st.sidebar.button("Remove Duplicates"):
                st.session_state.data_history.append(data.copy())
                data, message = remove_duplicates(data)
                st.session_state.data = data
                st.session_state.cleaning_history.append(message)
                st.experimental_rerun()
        
        elif cleaning_option == "Handle Missing Values":
            cols_with_missing = data.columns[data.isna().any()].tolist()
            
            if cols_with_missing:
                st.sidebar.markdown("Columns with missing values:")
                selected_cols = st.sidebar.multiselect("Select columns", cols_with_missing, default=cols_with_missing, key="missing_cols")
                
                fill_method = st.sidebar.selectbox(
                    "Fill method", 
                    ["Mean", "Median", "Mode", "Zero", "Forward Fill", "Backward Fill", "Custom Value"],
                    key="fill_method"
                )
                
                if fill_method == "Custom Value":
                    st.session_state.custom_value = st.sidebar.text_input("Enter custom value", key="custom_value_input")
                
                if st.sidebar.button("Fill Missing Values") and selected_cols:
                    st.session_state.data_history.append(data.copy())
                    data, message = fill_missing_values(data, fill_method, selected_cols)
                    st.session_state.data = data
                    st.session_state.cleaning_history.append(message)
                    st.experimental_rerun()
            else:
                st.sidebar.info("No missing values found in the data")
        
        elif cleaning_option == "Remove Outliers":
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.sidebar.multiselect("Select columns", numeric_cols, key="outlier_cols")
                outlier_method = st.sidebar.selectbox("Method", ["IQR", "Z-Score"], key="outlier_method")
                
                if outlier_method == "IQR":
                    threshold = st.sidebar.slider("IQR Threshold", 1.0, 3.0, 1.5, 0.1, key="iqr_threshold")
                else:
                    threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="zscore_threshold")
                
                if st.sidebar.button("Remove Outliers") and selected_cols:
                    st.session_state.data_history.append(data.copy())
                    data, message = remove_outliers(data, selected_cols, outlier_method, threshold)
                    st.session_state.data = data
                    st.session_state.cleaning_history.append(message)
                    st.experimental_rerun()
            else:
                st.sidebar.info("No numeric columns found in the data")
        
        elif cleaning_option == "Standardize Data":
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.sidebar.multiselect("Select columns to standardize", numeric_cols, key="std_cols")
                
                if st.sidebar.button("Standardize") and selected_cols:
                    st.session_state.data_history.append(data.copy())
                    data, message = standardize_column(data, selected_cols)
                    st.session_state.data = data
                    st.session_state.cleaning_history.append(message)
                    st.experimental_rerun()
            else:
                st.sidebar.info("No numeric columns found in the data")
        
        elif cleaning_option == "Normalize Data":
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.sidebar.multiselect("Select columns to normalize", numeric_cols, key="norm_cols")
                
                if st.sidebar.button("Normalize") and selected_cols:
                    st.session_state.data_history.append(data.copy())
                    data, message = normalize_column(data, selected_cols)
                    st.session_state.data = data
                    st.session_state.cleaning_history.append(message)
                    st.experimental_rerun()
            else:
                st.sidebar.info("No numeric columns found in the data")
        
        elif cleaning_option == "Rename Columns":
            col_rename = {}
            for col in data.columns:
                new_name = st.sidebar.text_input(f"Rename '{col}'", col, key=f"rename_{col}")
                if new_name != col:
                    col_rename[col] = new_name
            
            if st.sidebar.button("Rename Columns") and col_rename:
                st.session_state.data_history.append(data.copy())
                data, message = rename_columns(data, col_rename)
                st.session_state.data = data
                st.session_state.cleaning_history.append(message)
                st.experimental_rerun()
        
        elif cleaning_option == "Drop Columns":
            selected_cols = st.sidebar.multiselect("Select columns to drop", data.columns, key="drop_cols")
            
            if st.sidebar.button("Drop Columns") and selected_cols:
                st.session_state.data_history.append(data.copy())
                data = data.drop(columns=selected_cols)
                message = f"Dropped columns: {', '.join(selected_cols)}"
                st.session_state.data = data
                st.session_state.cleaning_history.append(message)
                st.experimental_rerun()
        
        elif cleaning_option == "Convert Data Types":
            selected_cols = st.sidebar.multiselect("Select columns to convert", data.columns, key="convert_cols")
            type_options = ["Numeric", "String", "Category", "Datetime"]
            column_types = {}
            
            for col in selected_cols:
                new_type = st.sidebar.selectbox(f"Type for {col}", type_options, key=f"type_{col}")
                column_types[col] = new_type
            
            if st.sidebar.button("Convert Types") and column_types:
                st.session_state.data_history.append(data.copy())
                data, message = convert_column_types(data, column_types)
                st.session_state.data = data
                st.session_state.cleaning_history.append(message)
                st.experimental_rerun()
        
        elif cleaning_option == "Clean Text Columns":
            text_cols = data.select_dtypes(include=['object']).columns.tolist()
            if text_cols:
                selected_cols = st.sidebar.multiselect("Select text columns", text_cols, key="text_cols")
                lowercase = st.sidebar.checkbox("Convert to lowercase", value=True, key="lowercase")
                remove_special = st.sidebar.checkbox("Remove special characters", value=True, key="remove_special")
                trim = st.sidebar.checkbox("Trim whitespace", value=True, key="trim")
                
                if st.sidebar.button("Clean Text") and selected_cols:
                    st.session_state.data_history.append(data.copy())
                    data, message = clean_text_columns(data, selected_cols, lowercase, remove_special, trim)
                    st.session_state.data = data
                    st.session_state.cleaning_history.append(message)
                    st.experimental_rerun()
            else:
                st.sidebar.info("No text columns found in the data")
        
        elif cleaning_option == "Encode Categorical Variables":
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                selected_cols = st.sidebar.multiselect("Select columns to encode", categorical_cols, key="encode_cols")
                encode_method = st.sidebar.selectbox("Encoding method", ["One-Hot", "Label"], key="encode_method")
                
                if st.sidebar.button("Encode Columns") and selected_cols:
                    st.session_state.data_history.append(data.copy())
                    data, message = encode_columns(data, selected_cols, encode_method)
                    st.session_state.data = data
                    st.session_state.cleaning_history.append(message)
                    st.experimental_rerun()
            else:
                st.sidebar.info("No categorical columns found in the data")
        
        # Batch cleaning
        st.sidebar.subheader("3. Batch Cleaning")
        batch_operations = st.sidebar.multiselect(
            "Select multiple operations to apply",
            ["Remove Duplicates", "Handle Missing Values", "Remove Outliers", "Standardize Data", "Normalize Data"],
            key="batch_ops"
        )
        
        if batch_operations:
            batch_params = {}
            for op in batch_operations:
                if op == "Handle Missing Values":
                    cols_with_missing = data.columns[data.isna().any()].tolist()
                    batch_params[op] = {
                        "columns": st.sidebar.multiselect(f"Columns for {op}", cols_with_missing, key=f"batch_missing_cols_{op}"),
                        "method": st.sidebar.selectbox(f"Fill method for {op}", ["Mean", "Median", "Mode", "Zero", "Forward Fill", "Backward Fill"], key=f"batch_fill_{op}")
                    }
                elif op == "Remove Outliers":
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    batch_params[op] = {
                        "columns": st.sidebar.multiselect(f"Columns for {op}", numeric_cols, key=f"batch_outlier_cols_{op}"),
                        "method": st.sidebar.selectbox(f"Method for {op}", ["IQR", "Z-Score"], key=f"batch_outlier_method_{op}"),
                        "threshold": st.sidebar.slider(f"Threshold for {op}", 1.0, 5.0, 1.5, key=f"batch_outlier_threshold_{op}")
                    }
                elif op in ["Standardize Data", "Normalize Data"]:
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    batch_params[op] = {
                        "columns": st.sidebar.multiselect(f"Columns for {op}", numeric_cols, key=f"batch_cols_{op}")
                    }
            
            if st.sidebar.button("Apply Batch Operations"):
                for op in batch_operations:
                    st.session_state.data_history.append(data.copy())
                    if op == "Remove Duplicates":
                        data, message = remove_duplicates(data)
                    elif op == "Handle Missing Values" and batch_params[op]["columns"]:
                        data, message = fill_missing_values(data, batch_params[op]["method"], batch_params[op]["columns"])
                    elif op == "Remove Outliers" and batch_params[op]["columns"]:
                        data, message = remove_outliers(data, batch_params[op]["columns"], batch_params[op]["method"], batch_params[op]["threshold"])
                    elif op == "Standardize Data" and batch_params[op]["columns"]:
                        data, message = standardize_column(data, batch_params[op]["columns"])
                    elif op == "Normalize Data" and batch_params[op]["columns"]:
                        data, message = normalize_column(data, batch_params[op]["columns"])
                    st.session_state.cleaning_history.append(message)
                st.session_state.data = data
                st.experimental_rerun()
        
        # Undo operation
        st.sidebar.subheader("4. Undo")
        if st.session_state.data_history and st.sidebar.button("Undo Last Operation"):
            st.session_state.data = st.session_state.data_history.pop()
            if st.session_state.cleaning_history:
                st.session_state.cleaning_history.pop()
            st.experimental_rerun()
        
        # History of cleaning operations
        if st.session_state.cleaning_history:
            with st.expander("Cleaning History", expanded=True):
                for i, action in enumerate(st.session_state.cleaning_history):
                    st.write(f"{i+1}. {action}")
        
        # Reset and download options
        st.sidebar.subheader("5. Final Actions")
        col1, col2, col3 = st.sidebar.columns(3)
        
        with col1:
            if st.button("Reset Data"):
                st.session_state.data = st.session_state.original_data.copy()
                st.session_state.cleaning_history = []
                st.session_state.data_history = []
                st.experimental_rerun()
        
        with col2:
            filename = f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            download_button_html = create_download_link(data, filename, "Download Data")
            st.markdown(download_button_html, unsafe_allow_html=True)
        
        with col3:
            log_filename = f"cleaning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            log_download_html = create_log_download_link(st.session_state.cleaning_history, log_filename)
            st.markdown(log_download_html, unsafe_allow_html=True)
    
    else:
        # Instructions for first-time users
        st.markdown("""
        ## Welcome to the Enhanced Data Cleaning Tool
        
        This tool helps you clean and prepare your data for analysis. Follow these steps:
        
        1. **Upload Data**: Use the sidebar to upload a CSV or Excel file
        2. **Explore Data**: View statistics and visualizations of your data
        3. **Clean Data**: Apply various cleaning operations to improve data quality
        4. **Download Results**: Get your cleaned data and cleaning log for further analysis
        
        ### Features:
        
        - Remove duplicates and outliers
        - Handle missing values with multiple methods
        - Standardize or normalize numeric columns
        - Convert data types and clean text columns
        - Encode categorical variables
        - Batch process multiple operations
        - Undo last operation
        - Export cleaning history
        - Advanced visualizations (correlation heatmap, scatter plots)
        
        ### Supported File Formats:
        
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        """)

if __name__ == "__main__":
    main()