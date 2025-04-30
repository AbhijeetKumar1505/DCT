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
    page_title="Data Cleaning Studio",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = "light"  # Default theme

def set_theme(theme):
    st.session_state.theme = theme
    st.rerun()

# Generate theme-based styles
def get_theme_css():
    if st.session_state.theme == "dark":
        return """
        <style>
            body {
                color: #f0f0f0;
                background-color: #1e1e1e;
            }
            .main {
                padding: 2rem;
                color: #f0f0f0;
            }
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
                background-color: #1e1e1e;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #f0f0f0 !important;
            }
            .stButton > button {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 0.5rem 1rem;
                width: 100%;
            }
            .stButton > button:hover {
                background-color: #0b7dda;
            }
            /* Enhanced sidebar */
            [data-testid="stSidebar"] {
                background-color: #2d2d2d;
                color: #f0f0f0;
                border-right: 1px solid #444;
            }
            [data-testid="stSidebar"] > div {
                padding: 2rem 1rem;
                color: #f0f0f0;
            }
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2, 
            [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label {
                color: #f0f0f0 !important;
                margin-top: 1.5rem;
                margin-bottom: 0.8rem;
                font-weight: 600;
            }
            /* Add separator lines between sidebar sections */
            [data-testid="stSidebar"] h2 {
                border-top: 1px solid #444;
                padding-top: 1.5rem;
                margin-top: 2rem;
            }
            /* First heading doesn't need top border */
            [data-testid="stSidebar"] h2:first-of-type {
                border-top: none;
                margin-top: 0;
            }
            /* Format sidebar selects and inputs */
            [data-testid="stSidebar"] .stSelectbox,
            [data-testid="stSidebar"] .stMultiSelect {
                margin-bottom: 1rem;
            }
            [data-testid="stSidebar"] .stButton > button {
                margin-top: 0.5rem;
                margin-bottom: 1.5rem;
            }
            [data-testid="stExpander"] {
                border: 1px solid #444;
                border-radius: 8px;
                margin-bottom: 1rem;
                background-color: #2d2d2d;
            }
            [data-testid="stMetric"] {
                background-color: #2d2d2d;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.2);
                color: #f0f0f0;
            }
            [data-testid="stMetricLabel"] {
                color: #f0f0f0 !important;
                font-weight: 600;
            }
            [data-testid="stMetricValue"] {
                color: #f0f0f0 !important;
            }
            /* Ensure all text has good contrast */
            .stMarkdown, p, li, td, th, span, label, div {
                color: #f0f0f0 !important;
            }
            .card {
                background-color: #2d2d2d;
                border-radius: 8px;
                padding: 1.5rem;
                height: 100%;
            }
            .highlight-card {
                background-color: #253529;
                border-radius: 8px;
                padding: 1.5rem;
                margin-top: 1rem;
                border-left: 5px solid #388E3C;
            }
            .info-card {
                background-color: #253246;
                border-radius: 8px;
                padding: 1rem;
                margin-top: 1rem;
            }
            .metric-container {
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 1rem;
            }
        </style>
        """
    else:  # Light theme
        return """
        <style>
            body {
                color: #1A1A1A;
                background-color: #ffffff;
            }
            .main {
                padding: 2rem;
                color: #1A1A1A;
            }
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
                background-color: #ffffff;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #1A1A1A !important;
            }
            .stButton > button {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 0.5rem 1rem;
                width: 100%;
            }
            .stButton > button:hover {
                background-color: #0b7dda;
            }
            /* Enhanced sidebar */
            [data-testid="stSidebar"] {
                background-color: #f1f3f6;
                color: #1A1A1A;
                border-right: 1px solid #ddd;
            }
            [data-testid="stSidebar"] > div {
                padding: 2rem 1rem;
                color: #1A1A1A;
            }
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2, 
            [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label {
                color: #1A1A1A !important;
                margin-top: 1.5rem;
                margin-bottom: 0.8rem;
                font-weight: 600;
            }
            /* Add separator lines between sidebar sections */
            [data-testid="stSidebar"] h2 {
                border-top: 1px solid #ddd;
                padding-top: 1.5rem;
                margin-top: 2rem;
            }
            /* First heading doesn't need top border */
            [data-testid="stSidebar"] h2:first-of-type {
                border-top: none;
                margin-top: 0;
            }
            /* Format sidebar selects and inputs */
            [data-testid="stSidebar"] .stSelectbox,
            [data-testid="stSidebar"] .stMultiSelect {
                margin-bottom: 1rem;
            }
            [data-testid="stSidebar"] .stButton > button {
                margin-top: 0.5rem;
                margin-bottom: 1.5rem;
            }
            [data-testid="stExpander"] {
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-bottom: 1rem;
                background-color: #ffffff;
            }
            [data-testid="stMetric"] {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                color: #1A1A1A;
            }
            [data-testid="stMetricLabel"] {
                color: #1A1A1A !important;
                font-weight: 600;
            }
            [data-testid="stMetricValue"] {
                color: #1A1A1A !important;
            }
            /* Ensure all text has good contrast */
            .stMarkdown, p, li, td, th, span, label, div {
                color: #1A1A1A !important;
            }
            .card {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 1.5rem;
                height: 100%;
            }
            .highlight-card {
                background-color: #e8f5e9;
                border-radius: 8px;
                padding: 1.5rem;
                margin-top: 1rem;
                border-left: 5px solid #388E3C;
            }
            .info-card {
                background-color: #e3f2fd;
                border-radius: 8px;
                padding: 1rem;
                margin-top: 1rem;
            }
            .metric-container {
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 1rem;
            }
            /* Add space between sidebar sections */
            [data-testid="stSidebar"] .block-container {
                padding-top: 0;
            }
        </style>
        """

# Apply current theme's CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

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
    href = f'''
    <a href="data:file/csv;base64,{b64}" download="{filename}" 
       style="display: inline-block; 
              padding: 0.5em 1em; 
              color: white; 
              background-color: #388E3C; 
              text-decoration: none; 
              border-radius: 4px;
              text-align: center;
              width: 100%;
              font-weight: 500;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        {text}
    </a>
    '''
    return href

def create_log_download_link(history, filename):
    """Generate a download link for the cleaning history log"""
    log_content = "\n".join([f"{i+1}. {action}" for i, action in enumerate(history)])
    b64 = base64.b64encode(log_content.encode()).decode()
    href = f'''
    <a href="data:text/plain;base64,{b64}" download="{filename}" 
       style="display: inline-block; 
              padding: 0.5em 1em; 
              color: white; 
              background-color: #7B1FA2; 
              text-decoration: none; 
              border-radius: 4px;
              text-align: center;
              width: 100%;
              font-weight: 500;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        📋 Download Cleaning Log
    </a>
    '''
    return href

def ensure_consistent_types(df):
    """Ensure all object-type columns are properly converted to formats compatible with Arrow"""
    for col in df.select_dtypes(include=['object']).columns:
        # Convert any complex objects to strings to ensure Arrow compatibility
        df[col] = df[col].astype(str)
    
    # Explicitly make a copy to ensure any underlying references are broken
    df = df.copy()
    return df

@st.cache_data
def load_data(file):
    """Cache file loading for performance"""
    if file.name.endswith('.csv'):
        return pd.read_csv(file, encoding='utf-8', low_memory=False)
    return pd.read_excel(file, engine='openpyxl')

def enable_dataframe_editing(df):
    """Enable direct editing of the dataframe through Streamlit's experimental data editor"""
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        height=400,
        column_config={
            col: st.column_config.Column(
                f"{col}",
                help=f"Type: {df[col].dtype}",
                width="auto"
            ) for col in df.columns
        },
        disabled=False,
        hide_index=False,
    )
    return edited_df

def suggest_data_transformations(df):
    """Suggest intelligent data transformations based on the current state of the data"""
    suggestions = []
    
    # Check for empty columns
    empty_cols = [col for col in df.columns if df[col].isna().sum() == len(df)]
    if empty_cols:
        suggestions.append({
            "title": "Remove Empty Columns",
            "description": f"Found {len(empty_cols)} columns with no values",
            "columns": empty_cols
        })
    
    # Check for mostly empty columns
    mostly_empty = [col for col in df.columns if df[col].isna().mean() > 0.7 and df[col].isna().mean() < 1.0]
    if mostly_empty:
        suggestions.append({
            "title": "Fix Mostly Empty Columns",
            "description": f"Found {len(mostly_empty)} columns with >70% missing values",
            "columns": mostly_empty
        })
    
    # Check for columns with consistent patterns
    text_cols = df.select_dtypes(include=['object']).columns
    date_pattern_cols = []
    numeric_pattern_cols = []
    
    for col in text_cols:
        # Sample non-null values to check patterns
        sample = df[col].dropna().sample(min(10, len(df[col].dropna()))).astype(str)
        
        # Check if it looks like dates
        date_like = all(re.search(r'\d{1,4}[/-]\d{1,2}[/-]\d{1,4}', str(val)) for val in sample)
        if date_like:
            date_pattern_cols.append(col)
        
        # Check if it's numeric stored as text
        numeric_like = all(re.search(r'^-?\d+(\.\d+)?$', str(val).strip()) for val in sample)
        if numeric_like:
            numeric_pattern_cols.append(col)
    
    if date_pattern_cols:
        suggestions.append({
            "title": "Convert to Datetime",
            "description": f"Found {len(date_pattern_cols)} columns with date-like patterns",
            "columns": date_pattern_cols
        })
    
    if numeric_pattern_cols:
        suggestions.append({
            "title": "Convert to Numeric",
            "description": f"Found {len(numeric_pattern_cols)} columns with numeric values stored as text",
            "columns": numeric_pattern_cols
        })
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        suggestions.append({
            "title": "Fix Duplicate Column Names",
            "description": "Detected duplicate column names which can cause issues",
            "columns": []
        })
    
    return suggestions

def generate_formula_from_text(text, df):
    """Generate a Python formula based on natural language input"""
    # This is a simple rule-based implementation
    # In a real application, you might use a more sophisticated NLP approach
    
    text = text.lower().strip()
    formula = None
    error = None
    
    # Sample column names for suggestions
    sample_cols = list(df.columns[:3])
    col_list = ", ".join([f"'{col}'" for col in sample_cols])
    
    try:
        # Create new column or calculate sum
        if "create" in text and "column" in text:
            # Extract column name if provided
            match = re.search(r'(?:called|named)\s+["\']?([a-zA-Z0-9_\s]+)["\']?', text)
            new_col_name = match.group(1).strip() if match else "new_column"
            
            # Determine operation
            if "sum" in text or "add" in text:
                # Try to find column names in the text
                cols_to_sum = []
                for col in df.columns:
                    if col.lower() in text:
                        cols_to_sum.append(col)
                
                if cols_to_sum:
                    formula = f"df['{new_col_name}'] = {' + '.join([f"df['{col}']" for col in cols_to_sum])}"
                else:
                    formula = f"df['{new_col_name}'] = df[{col_list}].sum(axis=1)"
                    
            elif "average" in text or "mean" in text:
                formula = f"df['{new_col_name}'] = df.mean(axis=1)"
                
            elif "multiply" in text:
                formula = f"df['{new_col_name}'] = df[{col_list}].prod(axis=1)"
                
            elif "uppercase" in text or "upper case" in text:
                formula = f"df['{new_col_name}'] = df[{col_list[0]}].str.upper()"
                
            elif "lowercase" in text or "lower case" in text:
                formula = f"df['{new_col_name}'] = df[{col_list[0]}].str.lower()"
                
            else:
                formula = f"df['{new_col_name}'] = 0  # Replace with your calculation"
                
        # Filter data
        elif "filter" in text or "where" in text:
            if "greater than" in text or ">" in text:
                match = re.search(r'(\w+)\s+(?:is\s+)?(?:greater than|>)\s+(\d+)', text)
                if match:
                    col, value = match.groups()
                    if col in df.columns:
                        formula = f"df = df[df['{col}'] > {value}]"
                    else:
                        formula = f"df = df[df[{col_list}] > {value}]"
                
            elif "less than" in text or "<" in text:
                match = re.search(r'(\w+)\s+(?:is\s+)?(?:less than|<)\s+(\d+)', text)
                if match:
                    col, value = match.groups()
                    if col in df.columns:
                        formula = f"df = df[df['{col}'] < {value}]"
                    else:
                        formula = f"df = df[df[{col_list}] < {value}]"
                        
            elif "equal" in text or "=" in text:
                match = re.search(r'(\w+)\s+(?:is\s+)?(?:equal to|=)\s+["\']?([a-zA-Z0-9_\s]+)["\']?', text)
                if match:
                    col, value = match.groups()
                    if col in df.columns:
                        formula = f"df = df[df['{col}'] == '{value}']"
                    else:
                        formula = f"df = df[df[{col_list}] == '{value}']"
            
            else:
                formula = f"df = df[df[{col_list}] > 0]  # Replace with your filter condition"
        
        # Sort data
        elif "sort" in text or "order" in text:
            if "ascending" in text:
                formula = f"df = df.sort_values(by={col_list}, ascending=True)"
            elif "descending" in text:
                formula = f"df = df.sort_values(by={col_list}, ascending=False)"
            else:
                formula = f"df = df.sort_values(by={col_list})"
                
        # Replace values
        elif "replace" in text:
            if "with" in text:
                match = re.search(r'replace\s+["\']?([a-zA-Z0-9_\s]+)["\']?\s+with\s+["\']?([a-zA-Z0-9_\s]+)["\']?', text)
                if match:
                    old_val, new_val = match.groups()
                    formula = f"df = df.replace('{old_val}', '{new_val}')"
                else:
                    formula = f"df = df.replace('old_value', 'new_value')"
            else:
                formula = f"df = df.replace('old_value', 'new_value')"
                
        # Fill missing values
        elif ("fill" in text and "missing" in text) or "null" in text or "na" in text:
            if "mean" in text:
                formula = "df = df.fillna(df.mean())"
            elif "median" in text:
                formula = "df = df.fillna(df.median())"
            elif "zero" in text:
                formula = "df = df.fillna(0)"
            else:
                formula = "df = df.fillna('Unknown')"
                
        # Group by operations
        elif "group" in text or "aggregate" in text:
            match = re.search(r'group\s+by\s+["\']?([a-zA-Z0-9_\s]+)["\']?', text)
            if match:
                group_col = match.group(1).strip()
                if "sum" in text:
                    formula = f"df = df.groupby('{group_col}').sum().reset_index()"
                elif "mean" in text or "average" in text:
                    formula = f"df = df.groupby('{group_col}').mean().reset_index()"
                elif "count" in text:
                    formula = f"df = df.groupby('{group_col}').count().reset_index()"
                else:
                    formula = f"df = df.groupby('{group_col}').agg('sum').reset_index()"
            else:
                formula = f"df = df.groupby({col_list}).sum().reset_index()"
                
        # If we couldn't match anything
        if not formula:
            formula = "# Could not generate a formula from your input"
            error = "Could not understand the request. Please try rephrasing or use a different command."
            
    except Exception as e:
        formula = "# Error generating formula"
        error = f"Error: {str(e)}"
        
    return formula, error

def main():
    # Main layout with title and theme toggle
    header_col1, header_col2 = st.columns([6, 1])
    
    with header_col1:
        st.title("🧮 Data Cleaning Studio")
    
    with header_col2:
        # Theme toggle in a more visible position with unique keys
        current_theme = st.session_state.theme
        if current_theme == "light":
            if st.button("🌙", help="Switch to Dark Mode", key="dark_mode_header"):
                set_theme("dark")
        else:
            if st.button("☀️", help="Switch to Light Mode", key="light_mode_header"):
                set_theme("light")
    
    # Sidebar with branding
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h3 style="margin-bottom: 0.5rem;">🧮 Data Cleaning Studio</h3>
        <div style="height: 2px; background: #2196F3; margin: 0.5rem auto 1.5rem auto; width: 50px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data operations section
    st.sidebar.markdown("## 📊 Data Operations")
    
    # File upload section
    uploaded_file = st.sidebar.file_uploader("Upload Data (CSV or Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None and (st.session_state.file_name != uploaded_file.name or st.session_state.data is None):
        st.session_state.file_name = uploaded_file.name
        try:
            data = load_data(uploaded_file)
            # Ensure data types are compatible with Arrow from the beginning
            data = ensure_consistent_types(data)
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
        
        # Cleaning operations in sidebar
        st.sidebar.markdown("## 🧹 Cleaning Tools")
        
        cleaning_option = st.sidebar.selectbox(
            "Select cleaning operation", 
            ["Remove Duplicates", "Handle Missing Values", "Remove Outliers", 
             "Standardize Data", "Normalize Data", "Rename Columns", "Drop Columns",
             "Convert Data Types", "Clean Text Columns", "Encode Categorical Variables"],
            key="cleaning_option"
        )
        
        # Handle different cleaning operations based on selection
        if cleaning_option == "Remove Duplicates":
            st.sidebar.markdown("Remove duplicate rows from the data")
            if st.sidebar.button("Apply Operation", key="btn_remove_dupes"):
                st.session_state.data_history.append(data.copy())
                data, message = remove_duplicates(data)
                st.session_state.data = data
                st.session_state.cleaning_history.append(message)
                st.rerun()
        
        # Data overview
        st.header("📊 Data Overview")
        
        # Create metrics using the metric-container class
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Rows", f"{data.shape[0]:,}")
        col2.metric("🔢 Columns", data.shape[1])
        col3.metric("❓ Missing Values", f"{data.isna().sum().sum():,}")
        col4.metric("🔄 Duplicate Rows", f"{data.duplicated().sum():,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Preview data
        with st.expander("Preview Data", expanded=True):
            st.dataframe(data.head(100), use_container_width=True)
            
        # Add interactive data editing
        with st.expander("🧠 Interactive Data Editor", expanded=False):
            st.markdown("<h3 style='color: #2C3E50;'>Edit Your Data Directly</h3>", unsafe_allow_html=True)
            st.markdown("""
            <p style='color: #2C3E50;'>
            Make changes directly to your data like in a spreadsheet. Changes will be automatically applied.
            You can:
            <ul>
                <li>Edit any cell value</li>
                <li>Add new rows</li>
                <li>Sort columns</li>
                <li>Filter data</li>
            </ul>
            </p>
            """, unsafe_allow_html=True)
            
            # Create two tabs for editing and AI suggestions
            edit_tab, ai_tab = st.tabs(["✏️ Edit Spreadsheet", "🤖 AI Suggestions"])
            
            with edit_tab:
                edited_data = enable_dataframe_editing(data)
                
                if st.button("Apply Changes"):
                    st.session_state.data_history.append(data.copy())
                    message = "Applied manual edits to the data"
                    st.session_state.data = edited_data
                    st.session_state.cleaning_history.append(message)
                    st.success("Changes applied successfully!")
                    st.rerun()
            
            with ai_tab:
                st.markdown("<h4 style='color: #2C3E50;'>Smart Data Transformation Suggestions</h4>", unsafe_allow_html=True)
                
                suggestions = suggest_data_transformations(data)
                
                if not suggestions:
                    st.info("No suggestions available for your data at this time.")
                
                for i, suggestion in enumerate(suggestions):
                    with st.container():
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.markdown(f"**{suggestion['title']}**: {suggestion['description']}")
                        
                        with cols[1]:
                            if st.button(f"Apply", key=f"apply_suggestion_{i}"):
                                st.session_state.data_history.append(data.copy())
                                
                                if suggestion['title'] == "Remove Empty Columns":
                                    data = data.drop(columns=suggestion['columns'])
                                    message = f"Removed {len(suggestion['columns'])} empty columns"
                                
                                elif suggestion['title'] == "Fix Mostly Empty Columns":
                                    # Ask what to do with mostly empty columns
                                    fix_choice = st.radio(
                                        "Choose action for mostly empty columns:",
                                        ["Drop columns", "Fill with default values"],
                                        key=f"fix_choice_{i}"
                                    )
                                    
                                    if fix_choice == "Drop columns":
                                        data = data.drop(columns=suggestion['columns'])
                                        message = f"Removed {len(suggestion['columns'])} mostly empty columns"
                                    else:
                                        for col in suggestion['columns']:
                                            if pd.api.types.is_numeric_dtype(data[col]):
                                                data[col] = data[col].fillna(0)
                                            else:
                                                data[col] = data[col].fillna("Unknown")
                                        message = f"Filled {len(suggestion['columns'])} mostly empty columns"
                                
                                elif suggestion['title'] == "Convert to Datetime":
                                    for col in suggestion['columns']:
                                        data[col] = pd.to_datetime(data[col], errors='coerce')
                                    message = f"Converted {len(suggestion['columns'])} columns to datetime"
                                
                                elif suggestion['title'] == "Convert to Numeric":
                                    for col in suggestion['columns']:
                                        data[col] = pd.to_numeric(data[col], errors='coerce')
                                    message = f"Converted {len(suggestion['columns'])} columns to numeric"
                                
                                elif suggestion['title'] == "Fix Duplicate Column Names":
                                    # Make column names unique by adding suffixes
                                    data.columns = pd.Series(data.columns).map(lambda x: f"{x}_{pd.Series(data.columns).value_counts()[x]}" if pd.Series(data.columns).value_counts()[x] > 1 else x)
                                    message = "Fixed duplicate column names"
                                
                                st.session_state.data = data
                                st.session_state.cleaning_history.append(message)
                                st.success(f"Applied: {message}")
                                st.rerun()
                    
                    st.markdown("---")
                
                # Manual formula input section
                st.markdown("<h4 style='color: #2C3E50;'>Apply Custom Formula</h4>", unsafe_allow_html=True)
                
                # Create tabs for code and natural language
                code_tab, nl_tab = st.tabs(["Code Formula", "Natural Language"])
                
                with code_tab:
                    st.markdown("""
                    <p style='color: #2C3E50;'>
                    Enter a formula using Python syntax to transform your data. Use 'df' to refer to the dataframe.
                    </p>
                    <p style='color: #2C3E50;'>
                    Examples:
                    <ul>
                        <li><code>df['new_col'] = df['col1'] + df['col2']</code> - Add a new column</li>
                        <li><code>df['col'] = df['col'].str.upper()</code> - Convert text to uppercase</li>
                        <li><code>df = df[df['value'] > 100]</code> - Filter rows</li>
                    </ul>
                    </p>
                    """, unsafe_allow_html=True)
                    
                    formula = st.text_area("Enter formula:", height=100, key="code_formula")
                    
                    if st.button("Execute Formula", key="exec_code_formula") and formula:
                        try:
                            # Store original dataframe
                            st.session_state.data_history.append(data.copy())
                            
                            # Create a safe local namespace with limited imports
                            local_vars = {'df': data, 'pd': pd, 'np': np}
                            
                            # Execute the formula
                            exec(formula, {}, local_vars)
                            
                            # Get the modified dataframe
                            modified_df = local_vars['df']
                            
                            # Update session state
                            st.session_state.data = modified_df
                            message = f"Applied custom formula: {formula}"
                            st.session_state.cleaning_history.append(message)
                            st.success("Formula executed successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error executing formula: {str(e)}")
                
                with nl_tab:
                    st.markdown("""
                    <p style='color: #2C3E50;'>
                    Describe what you want to do with your data in plain English, and we'll generate a formula for you.
                    </p>
                    <p style='color: #2C3E50;'>
                    Examples:
                    <ul>
                        <li>"Create a new column called total that adds price and tax"</li>
                        <li>"Filter where age is greater than 30"</li>
                        <li>"Sort by price in descending order"</li>
                        <li>"Replace missing values with the mean"</li>
                        <li>"Group by category and calculate the sum"</li>
                    </ul>
                    </p>
                    """, unsafe_allow_html=True)
                    
                    nl_query = st.text_area("Describe what you want to do:", height=100, key="nl_query", 
                                           placeholder="E.g., Create a new column called 'full_name' that combines first_name and last_name")
                    
                    if st.button("Generate & Execute", key="gen_nl_formula") and nl_query:
                        # Generate formula from natural language
                        generated_formula, error = generate_formula_from_text(nl_query, data)
                        
                        if error:
                            st.error(error)
                        else:
                            # Display the generated formula
                            st.code(generated_formula, language="python")
                            
                            try:
                                # Store original dataframe
                                st.session_state.data_history.append(data.copy())
                                
                                # Create a safe local namespace with limited imports
                                local_vars = {'df': data, 'pd': pd, 'np': np}
                                
                                # Execute the formula
                                exec(generated_formula, {}, local_vars)
                                
                                # Get the modified dataframe
                                modified_df = local_vars['df']
                                
                                # Show preview of changes
                                st.markdown("### Preview of changes:")
                                st.dataframe(modified_df.head(5))
                                
                                if st.button("Apply Changes", key="apply_nl_changes"):
                                    # Update session state
                                    st.session_state.data = modified_df
                                    message = f"Applied formula from natural language: {nl_query}"
                                    st.session_state.cleaning_history.append(message)
                                    st.success("Changes applied successfully!")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error executing generated formula: {str(e)}")
                                st.code(generated_formula, language="python")
                                st.markdown("You can modify the formula and execute it manually in the Code Formula tab.")
        
        # Column information
        with st.expander("Column Information"):
            col_stats = pd.DataFrame({
                'Type': data.dtypes,
                'Unique Values': data.nunique(),
                'Missing Values': data.isna().sum(),
                'Missing (%)': (data.isna().sum() / len(data) * 100).round(2)
            })
            # Ensure column statistics are compatible with Arrow
            col_stats = col_stats.reset_index().rename(columns={'index': 'Column'})
            # Convert all columns to string or numeric types for Arrow compatibility
            for col in col_stats.columns:
                if col_stats[col].dtype == 'object':
                    col_stats[col] = col_stats[col].astype(str)
            
            # Format the Missing % column
            if 'Missing (%)' in col_stats.columns:
                col_stats['Missing (%)'] = col_stats['Missing (%)'].apply(lambda x: f"{x}%" if pd.notnull(x) else "0%")
            
            # Add visual indicators for missing values
            if 'Missing Values' in col_stats.columns:
                max_missing = col_stats['Missing Values'].astype(float).max()
                if max_missing > 0:
                    col_stats['Completeness'] = col_stats['Missing Values'].astype(float).apply(
                        lambda x: '🟢 Complete' if x == 0 else 
                        f"🟠 {100 - (x / len(data) * 100):.1f}% Complete" if x < len(data) * 0.5 else 
                        f"🔴 {100 - (x / len(data) * 100):.1f}% Complete"
                    )
                else:
                    col_stats['Completeness'] = "🟢 Complete"
            
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
        
        # History of cleaning operations - moved from sidebar to main panel
        if st.session_state.cleaning_history:
            with st.expander("🔄 Cleaning History", expanded=False):
                st.markdown("### Actions Performed")
                for i, action in enumerate(st.session_state.cleaning_history):
                    st.markdown(f"**{i+1}.** {action}")
                    
                # Add a download button for the log in the expander too
                st.markdown("<br>", unsafe_allow_html=True)
                log_filename = f"cleaning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                log_download_html = create_log_download_link(st.session_state.cleaning_history, log_filename)
                st.markdown(log_download_html, unsafe_allow_html=True)
        
        # Cleaning operations
        st.sidebar.markdown("## 🔄 Batch Processing")
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
                st.rerun()
        
        # Undo operation - simplified
        st.sidebar.markdown("## ⏪ History")
        if st.session_state.data_history and st.sidebar.button("↩️ Undo Last Operation"):
            st.session_state.data = st.session_state.data_history.pop()
            if st.session_state.cleaning_history:
                st.session_state.cleaning_history.pop()
            st.rerun()
        
        # Reset and download options
        st.sidebar.markdown("## 💾 Export & Reset")
        
        # Create two columns for buttons
        reset_col, download_col = st.sidebar.columns(2)
        
        with reset_col:
            if st.button("🔄 Reset Data", help="Reset to original data", key="reset_data_btn"):
                st.session_state.data = st.session_state.original_data.copy()
                st.session_state.cleaning_history = []
                st.session_state.data_history = []
                st.rerun()
        
        with download_col:
            filename = f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            download_button_html = create_download_link(data, filename, "💾 Download CSV")
            st.markdown(download_button_html, unsafe_allow_html=True)
        
        # Simplified download log button
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        log_filename = f"cleaning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_download_html = create_log_download_link(st.session_state.cleaning_history, log_filename)
        st.sidebar.markdown(log_download_html, unsafe_allow_html=True)
    
    else:
        # Instructions for first-time users
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1>✨ Your Data Transformation Workspace</h1>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">Powerful tools for cleaning, transforming, and analyzing your data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>📊 How to Use</h3>
                <ol style="margin-top: 1rem;">
                    <li><b>Upload Data</b>: Use the sidebar to upload a CSV or Excel file</li>
                    <li><b>Explore Data</b>: View statistics and visualizations</li>
                    <li><b>Clean Data</b>: Apply various cleaning operations</li>
                    <li><b>Edit Directly</b>: Use the spreadsheet editor to make changes</li>
                    <li><b>Download Results</b>: Get your cleaned data and history log</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>✨ Features</h3>
                <ul style="margin-top: 1rem; columns: 2;">
                    <li>Remove duplicates</li>
                    <li>Handle missing values</li>
                    <li>Remove outliers</li>
                    <li>Standardize data</li>
                    <li>Normalize data</li>
                    <li>Clean text columns</li>
                    <li>Encode categories</li>
                    <li>Direct spreadsheet editing</li>
                    <li>Natural language formulas</li>
                    <li>Smart data suggestions</li>
                    <li>Advanced visualizations</li>
                    <li>Batch operations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Highlight new features
        st.markdown("""
        <div class="highlight-card">
            <h3 style="margin-bottom: 0.5rem;">🚀 Powerful Spreadsheet Features</h3>
            <div>
                <p><b>Direct Spreadsheet Editing</b>: Edit your data directly in the app like in Excel or Google Sheets</p>
                <p><b>Smart AI Suggestions</b>: Get intelligent recommendations for cleaning and transforming your data</p>
                <p><b>Natural Language Formulas</b>: Describe what you want to do in plain English, and we'll generate the code</p>
                <p style="margin-top: 0.5rem;"><i>Find these features in the "Interactive Data Editor" section after loading your data!</i></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # File format info
        st.markdown("""
        <div class="info-card">
            <h4 style="margin-bottom: 0.5rem;">📁 Supported File Formats</h4>
            <div style="display: flex; gap: 2rem;">
                <div>
                    <p><b>CSV</b> (.csv)</p>
                </div>
                <div>
                    <p><b>Excel</b> (.xlsx, .xls)</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get started prompt
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <p style="font-size: 1.2rem;">👈 Use the sidebar on the left to get started!</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()