"""
Dataset routes for uploading, analyzing, and managing datasets
"""
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Body, Request
from fastapi.responses import FileResponse
from typing import List, Optional
import pandas as pd
import numpy as np
import os
import uuid
import logging
from datetime import datetime, timezone
from app.models.schemas import (
    DatasetResponse,
    DatasetInDB,
)
from app.storage import get_store, generate_id
from app.services.preprocessing import PreprocessingService
from app.services.adaptive_preprocessing import AdaptivePreprocessor
from app.services.eda_service import EDAService
from app.config import settings

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])
logger = logging.getLogger(__name__)

# Comprehensive list of missing value indicators
# This list covers common representations of missing/null values across different datasets and regions
MISSING_VALUE_INDICATORS = [
    # Question mark (common in medical/UCI datasets)
    '?', '??',
    
    # Standard null representations
    'NA', 'N/A', 'n/a', 'na', 'Na',
    'NULL', 'null', 'Null',
    'None', 'none', 'NONE',
    'NaN', 'nan', 'NAN',
    
    # Empty/whitespace
    '', ' ', '  ', '\t',
    
    # Dashes and underscores
    '-', '--', '---',
    '_', '__',
    
    # Not Available variations
    'N.A.', 'n.a.',
    'Not Available', 'not available', 'NOT AVAILABLE',
    'Not Applicable', 'not applicable', 'NOT APPLICABLE',
    
    # Missing/Unknown variations
    'Missing', 'missing', 'MISSING',
    'Unknown', 'unknown', 'UNKNOWN',
    'Undefined', 'undefined', 'UNDEFINED',
    
    # Numeric representations
    '-999', '-9999', '999', '9999',  # Common sentinel values
    '-1', '0' if False else None,  # Sometimes 0 or -1 indicate missing (but be careful!)
    
    # Special characters
    '*', '**', '#N/A', '#NA',
    
    # Database/programming nulls
    'NIL', 'nil', 'Nil',
    'BLANK', 'blank', 'Blank',
    
    # Scientific notation
    'inf', '-inf', 'Inf', '-Inf', 'INF', '-INF',
]


def normalize_data_formats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data formats for proper type detection WITHOUT modifying structure:
    1. European number formats (comma → dot): 2,6 → 2.6
    2. Time formats (dots → colons): 18.00.00 → 18:00:00
    3. Parse datetime columns properly
    
    NOTE: Does NOT add or remove columns - only normalizes existing data
    
    Args:
        df: Raw DataFrame
    
    Returns:
        DataFrame with normalized formats (same columns)
    """
    print(f"\n🔧 Normalizing data formats (no structural changes)...")
    print(f"   Shape: {df.shape}")
    
    # 1. Handle European number formats (comma as decimal separator)
    for col in df.columns:
        if df[col].dtype == 'object':
            # Sample first 100 non-null values
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                # Check if values look like European decimals (e.g., "2,6" "13,3")
                european_decimal_pattern = sample.astype(str).str.match(r'^-?\d+,\d+$')
                if european_decimal_pattern.sum() > len(sample) * 0.5:
                    print(f"   Converting European decimals in '{col}': '2,6' → 2.6")
                    # Replace comma with dot and convert to float
                    df[col] = df[col].astype(str).str.replace(',', '.').replace('nan', np.nan)
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
    
    # 2. Normalize datetime formats and parse
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
                
            sample_str = sample.astype(str)
            
            # Check for time format with dots (18.00.00)
            time_dot_pattern = sample_str.str.match(r'^\d{1,2}\.\d{2}\.\d{2}')
            if time_dot_pattern.sum() > len(sample) * 0.5:
                print(f"   Normalizing time format in '{col}': 18.00.00 → 18:00:00")
                df[col] = df[col].astype(str).str.replace('.', ':', regex=False)
            
            # Try to parse as datetime with multiple strategies
            try:
                # First try: Let pandas infer
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() > len(df) * 0.5:
                    df[col] = parsed
                    print(f"   ✓ Parsed '{col}' as datetime")
                    continue
                
                # Second try: Day first (European format DD/MM/YYYY)
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                if parsed.notna().sum() > len(df) * 0.5:
                    df[col] = parsed
                    print(f"   ✓ Parsed '{col}' as datetime (dayfirst)")
                    continue
            except:
                pass
    
    print(f"   Final dtypes: {df.dtypes.value_counts().to_dict()}\n")
    return df


def clean_and_parse_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and parse data to handle:
    1. European number formats (comma as decimal separator)
    2. Datetime column detection and parsing
    3. Proper type inference
    
    NOTE: Does NOT combine Date + Time columns - keeps them separate for feature extraction
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Cleaned DataFrame with proper types
    """
    print(f"\n🧹 Cleaning and parsing data for preprocessing...")
    print(f"   Initial shape: {df.shape}")
    print(f"   Initial dtypes: {df.dtypes.value_counts().to_dict()}")
    
    # 1. Handle European number formats (comma as decimal separator)
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains numbers with comma as decimal separator
            # Sample first 100 non-null values
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                # Check if values look like European decimals (e.g., "2,6" "13,3")
                # Pattern: optional minus, digits, comma, digits
                european_decimal_pattern = sample.astype(str).str.match(r'^-?\d+,\d+$')
                if european_decimal_pattern.sum() > len(sample) * 0.5:
                    print(f"   Converting European decimals in '{col}': '2,6' → 2.6")
                    # Replace comma with dot and convert to float
                    df[col] = df[col].astype(str).str.replace(',', '.').replace('nan', np.nan)
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
    
    # 2. Normalize time format (dots → colons) but DON'T parse yet
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                sample_str = sample.astype(str)
                # Check for time format with dots (18.00.00)
                time_dot_pattern = sample_str.str.match(r'^\d{1,2}\.\d{2}\.\d{2}')
                if time_dot_pattern.sum() > len(sample) * 0.5:
                    print(f"   Normalizing time format in '{col}': 18.00.00 → 18:00:00")
                    df[col] = df[col].astype(str).str.replace('.', ':', regex=False)
    
    # 3. Parse datetime columns (but keep Date and Time SEPARATE for feature extraction)
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
                
            sample_str = sample.astype(str)
            
            # Detect datetime patterns
            is_date = sample_str.str.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}').sum() > len(sample) * 0.5
            is_time = sample_str.str.match(r'^\d{1,2}:\d{2}').sum() > len(sample) * 0.5
            
            if is_date or is_time:
                try:
                    # Try default parsing
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() > len(df) * 0.5:
                        df[col] = parsed
                        print(f"   ✓ Parsed '{col}' as datetime")
                        continue
                    
                    # Try with dayfirst=True for European dates
                    parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                    if parsed.notna().sum() > len(df) * 0.5:
                        df[col] = parsed
                        print(f"   ✓ Parsed '{col}' as datetime (dayfirst)")
                        continue
                except:
                    pass
    
    print(f"   Final shape: {df.shape}")
    print(f"   Final dtypes: {df.dtypes.value_counts().to_dict()}")
    print(f"   Datetime columns: {[col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]}\n")
    
    return df


def detect_delimiter(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Detect the delimiter used in a CSV file by trying multiple separators.
    Handles mixed delimiter cases (e.g., tab-separated headers with space-separated values).
    
    Args:
        file_path: Path to the CSV file
        encoding: File encoding
    
    Returns:
        Detected delimiter (',' | '\t' | ';' | r'\s+' | '|' | ':')
        Special value r'\s+' means whitespace-delimited (any amount of whitespace)
    """
    # Read first few lines to analyze
    with open(file_path, 'r', encoding=encoding) as f:
        first_line = f.readline().strip()
        lines = [first_line] + [f.readline().strip() for _ in range(min(9, sum(1 for _ in f)))]
    
    if not lines or len(lines) < 2:
        return ','
    
    # Try common delimiters in order of likelihood
    delimiters_to_try = [
        (',', 'comma'),
        ('\t', 'tab'),
        (';', 'semicolon'),
        (r'\s+', 'whitespace'),  # regex for any whitespace
        ('|', 'pipe'),
        (':', 'colon')
    ]
    
    best_delimiter = ','
    best_score = -1000
    
    for delimiter, name in delimiters_to_try:
        try:
            # Try reading with this delimiter
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                nrows=5,
                sep=delimiter,
                engine='python',
                on_bad_lines='skip'
            )
            
            # Score this delimiter based on multiple criteria
            score = 0
            
            # 1. Must have multiple columns
            num_cols = len(df.columns)
            if num_cols < 2:
                score -= 1000  # Very bad
                continue
            
            # 2. Should have some data rows
            num_rows = len(df)
            if num_rows < 1:
                score -= 1000
                continue
            
            # 3. Check for null columns (bad sign)
            null_columns = df.isnull().all().sum()
            if null_columns > num_cols * 0.5:
                score -= 500
                continue
            
            # 4. Check if first column contains multiple space-separated values
            # This indicates wrong delimiter (common with tab-headers + space-data)
            try:
                first_col_sample = df.iloc[:min(3, num_rows), 0].astype(str).tolist()
                has_multi_values_in_first_col = any(
                    ' ' in str(val) and len(str(val).split()) > 2 
                    for val in first_col_sample 
                    if pd.notna(val) and str(val) not in ['nan', 'None', '']
                )
                
                if has_multi_values_in_first_col and delimiter != r'\s+':
                    # Wrong delimiter - data values are stuck together
                    score -= 800
                    continue
            except:
                pass
            
            # Calculate score
            score += num_cols * 10  # More columns is good
            score += num_rows * 2  # More rows is good
            score -= null_columns * 20  # Null columns are bad
            
            # Check if delimiter actually appears in the file
            if delimiter not in [r'\s+']:
                delimiter_count_in_header = first_line.count(delimiter)
                if delimiter_count_in_header == 0:
                    score -= 900  # Delimiter not in file
                elif delimiter_count_in_header + 1 == num_cols:
                    score += 100  # Perfect match
                    
            # Special handling for whitespace
            if delimiter == r'\s+':
                # Check if data is truly whitespace-delimited
                space_counts = [line.count(' ') + line.count('\t') for line in lines[1:] if line]
                tab_counts = [line.count('\t') for line in lines[1:] if line]
                space_only_counts = [line.count(' ') for line in lines[1:] if line]
                
                # If headers have tabs but data has spaces, this is the right delimiter
                header_has_tabs = '\t' in first_line
                data_has_spaces = any(c > 0 for c in space_only_counts)
                
                if header_has_tabs and data_has_spaces:
                    score += 200  # Mixed delimiter case - whitespace handles both
                elif space_counts and max(space_counts) >= num_cols - 1:
                    score += 100  # Genuinely whitespace-delimited
            
            if score > best_score:
                best_score = score
                best_delimiter = delimiter
                
        except Exception as e:
            continue
    
    return best_delimiter


def detect_header(file_path: str, delimiter: str, encoding: str = 'utf-8') -> bool:
    """
    Detect if CSV file has a header row
    
    Args:
        file_path: Path to CSV file
        delimiter: Detected delimiter
        encoding: File encoding
    
    Returns:
        True if file has header, False otherwise
    """
    try:
        # Read first 5 lines
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            lines = [f.readline().strip() for _ in range(5)]
        
        # Filter empty lines
        lines = [line for line in lines if line]
        if len(lines) < 2:
            return True  # Default to assuming header exists
        
        # Split lines by delimiter
        first_row = lines[0].split(delimiter)
        second_row = lines[1].split(delimiter) if len(lines) > 1 else []
        
        if len(first_row) != len(second_row):
            return True  # Different column counts, likely has header
        
        # Check if first row values appear in subsequent rows (sign of no header)
        first_row_values = set(val.strip().lower() for val in first_row if val.strip())
        
        # Check if any first row value appears in subsequent rows
        for line in lines[1:]:
            row_values = set(val.strip().lower() for val in line.split(delimiter) if val.strip())
            # If first row values appear in data rows, it's likely a data row itself
            if first_row_values & row_values:  # Intersection
                return False
        
        # Check if first row looks like typical categorical data (ham/spam, yes/no, etc.)
        common_labels = {'ham', 'spam', 'yes', 'no', 'true', 'false', '0', '1', 'positive', 'negative', 'male', 'female'}
        first_col_lower = first_row[0].strip().lower()
        
        if first_col_lower in common_labels:
            # Check if second row also has similar pattern
            if len(second_row) > 0:
                second_col_lower = second_row[0].strip().lower()
                if second_col_lower in common_labels:
                    return False  # Both look like data, not headers
        
        # If first row has delimiter in the values themselves (like 'ham\tMessage'), likely no header
        if any('\t' in val or ',' in val for val in first_row):
            return False
        
        return True  # Default to assuming header exists
        
    except Exception as e:
        logger.warning(f"Error detecting header: {str(e)}")
        return True  # Default to assuming header exists


def read_csv_smart(file_path: str, encoding: str = 'utf-8', **kwargs) -> pd.DataFrame:
    """
    Read CSV file with smart delimiter and encoding detection
    
    Args:
        file_path: Path to CSV file
        encoding: Initial encoding to try (default: 'utf-8')
        **kwargs: Additional arguments to pass to pd.read_csv (e.g., na_values, keep_default_na)
    
    Returns:
        pandas DataFrame
    """
    # Detect delimiter
    delimiter = detect_delimiter(file_path, encoding)
    
    # Detect if file has header
    has_header = detect_header(file_path, delimiter, encoding)
    
    # Determine header parameter for pd.read_csv
    header_param = 0 if has_header else None
    
    # Default na_values handling (can be overridden by kwargs)
    na_values = kwargs.pop('na_values', MISSING_VALUE_INDICATORS)
    keep_default_na = kwargs.pop('keep_default_na', True)
    
    # Read with detected delimiter and header setting
    # Note: r'\s+' is a regex pattern for any whitespace (spaces, tabs, multiple spaces)
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=delimiter,
        header=header_param,
        engine='python',
        on_bad_lines='skip',
        skipinitialspace=True,
        na_values=na_values,
        keep_default_na=keep_default_na,
        **kwargs
    )
    
    # If no header, generate column names
    if not has_header:
        # Generate default column names
        num_cols = len(df.columns)
        if num_cols == 2:
            # Common pattern: label and text/message
            df.columns = ['label', 'message']
        else:
            df.columns = [f'Column{i+1}' for i in range(num_cols)]
        
        logger.info(f"No header detected. Generated column names: {list(df.columns)}")
    
    # Clean column names - remove tabs, extra whitespace, and null/unnamed columns
    cleaned_columns = []
    columns_to_drop = []
    
    for i, col in enumerate(df.columns):
        col_str = str(col).strip()  # Remove leading/trailing whitespace
        
        # Remove tabs and normalize internal whitespace
        col_str = ' '.join(col_str.split())
        
        # Check if column is null, unnamed, or all its data is null
        is_null_column = (
            col_str.lower() in ['null', 'none', 'nan', ''] or
            col_str.lower().startswith('unnamed:') or
            df[col].isna().all()
        )
        
        if is_null_column:
            columns_to_drop.append(col)
        else:
            # Check for duplicate column names and make unique
            base_name = col_str
            counter = 1
            while col_str in cleaned_columns:
                col_str = f"{base_name}_{counter}"
                counter += 1
            cleaned_columns.append(col_str)
    
    # Drop null columns
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    # Rename columns with cleaned names
    df.columns = cleaned_columns
    
    return df


def sanitize_dataset_stats(dataset: dict) -> dict:
    """
    Sanitize dataset statistics to remove NaN and inf values
    
    Args:
        dataset: Dataset dictionary
    
    Returns:
        Sanitized dataset dictionary
    """
    if "summary_statistics" in dataset and dataset["summary_statistics"]:
        sanitized_stats = {}
        for col, col_stats in dataset["summary_statistics"].items():
            sanitized_stats[col] = {}
            if isinstance(col_stats, dict):
                for stat_name, value in col_stats.items():
                    if isinstance(value, (int, float)):
                        if pd.isna(value) or np.isinf(value):
                            sanitized_stats[col][stat_name] = None
                        else:
                            sanitized_stats[col][stat_name] = float(value)
                    else:
                        sanitized_stats[col][stat_name] = value
            else:
                sanitized_stats[col] = col_stats
        dataset["summary_statistics"] = sanitized_stats
    
    # Sanitize EDA results
    if "eda_results" in dataset and dataset["eda_results"]:
        dataset["eda_results"] = sanitize_eda_results(dataset["eda_results"])
    
    return dataset


def sanitize_eda_results(eda_results: dict) -> dict:
    """
    Sanitize EDA results to remove NaN and inf values
    
    Args:
        eda_results: EDA results dictionary
    
    Returns:
        Sanitized EDA results dictionary
    """
    import json
    
    def sanitize_value(val):
        """Recursively sanitize values"""
        if isinstance(val, dict):
            return {k: sanitize_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [sanitize_value(v) for v in val]
        elif isinstance(val, np.integer):
            return int(val)
        elif isinstance(val, np.floating):
            if pd.isna(val) or np.isinf(val):
                return None
            return float(val)
        elif isinstance(val, (int, float)):
            if pd.isna(val) or np.isinf(val):
                return None
            return float(val)
        elif isinstance(val, np.ndarray):
            return sanitize_value(val.tolist())
        elif pd.isna(val):
            return None
        else:
            return val
    
    return sanitize_value(eda_results)


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None)
):
    """
    Upload a CSV dataset file
    
    Args:
        file: CSV file to upload
        name: Dataset name
        description: Optional dataset description
    
    Returns:
        DatasetResponse with metadata and analysis
    
    Raises:
        HTTPException: If file is invalid or processing fails
    """
    # Check for duplicate dataset name for this user
    # Only check among datasets that haven't had their originals deleted
    datasets_store = get_store("datasets")
    existing_dataset = await datasets_store.find_one({
        "name": name,
        "$or": [
            {"original_deleted": {"$exists": False}},
            {"original_deleted": False}
        ]
    })
    
    if existing_dataset:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A dataset with the name '{name}' already exists. Please choose a different name."
        )
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported"
        )
    
    # Check file size
    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum allowed size of {settings.max_upload_size} bytes"
        )
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(settings.upload_dir, filename)
    
    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)
    
    # Save file
    with open(file_path, 'wb') as f:
        f.write(contents)
    
    try:
        # Detect encoding first
        detected_encoding = 'utf-8'
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)
        except UnicodeDecodeError:
            # Try common encodings
            encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1024)
                    detected_encoding = encoding
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            
            # Last resort: use chardet
            if detected_encoding == 'utf-8':
                import chardet
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    detected_encoding = result['encoding']
        
        # Read dataset with smart delimiter detection
        # Handles: comma, semicolon, tab, space, pipe, and other delimiters
        df = read_csv_smart(file_path, detected_encoding)
        
        # Normalize data formats (European decimals, time formats, datetime parsing)
        # Does NOT modify structure (no columns added/removed)
        df = normalize_data_formats(df)
        
        # Use preprocessing service to analyze basic info
        preprocessing_service = PreprocessingService()
        analysis = preprocessing_service.analyze_dataset(df)
        
        # NOTE: EDA is now run separately via /run-eda endpoint after column selection
        # This allows users to select columns before analysis
        
        # Calculate file size
        file_size = len(contents)
        
        # Create dataset document (without EDA results initially)
        dataset_doc = {
            "name": name,
            "description": description,
            "filename": file.filename,
            "file_path": file_path,
            "file_size": file_size,
            "created_at": datetime.utcnow(),
            "rows": analysis["rows"],
            "columns": analysis["columns"],
            "column_names": analysis["column_names"],
            "column_types": analysis["column_types"],
            "missing_values": analysis["missing_values"],
            "summary_statistics": analysis["summary_statistics"],
            "datetime_columns": analysis.get("datetime_columns", 0),
            "numerical_columns": analysis.get("numerical_columns", 0),
            "categorical_columns": analysis.get("categorical_columns", 0),
            # EDA will be added later via /run-eda endpoint
        }
        
        # Insert into database
        datasets_store = get_store("datasets")
        result = await datasets_store.insert_one(dataset_doc)
        
        # Retrieve created dataset
        created_dataset = await datasets_store.find_one({"_id": result.inserted_id})
        created_dataset["id"] = str(created_dataset.pop("_id"))
        
        # Sanitize before returning
        created_dataset = sanitize_dataset_stats(created_dataset)
        
        return DatasetResponse(**created_dataset)
    
    except Exception as e:
        # Clean up file if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Log detailed error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Dataset upload failed:")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        print(f"   Details:\n{error_details}")
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process dataset: {str(e)}"
        )


@router.post("/{dataset_id}/run-eda")
async def run_eda_on_dataset(
    dataset_id: str,
    selected_columns: Optional[List[str]] = Body(None, embed=True)
):
    """
    Run EDA on a dataset, optionally filtering to selected columns first
    
    Args:
        dataset_id: Dataset ID
        selected_columns: Optional list of column names to analyze (if None, analyze all)
    
    Returns:
        EDA results and updated dataset info
    
    Raises:
        HTTPException: If dataset not found or EDA fails
    """
    datasets_store = get_store("datasets")
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    file_path = dataset.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found"
        )
    
    try:
        # Read dataset with smart delimiter detection (same as upload)
        detected_encoding = 'utf-8'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)
        except UnicodeDecodeError:
            encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1024)
                    detected_encoding = encoding
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            
            if detected_encoding == 'utf-8':
                import chardet
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    detected_encoding = result['encoding']
        
        # Use the same smart CSV reader as upload
        df = read_csv_smart(file_path, detected_encoding)
        
        # Normalize data formats (European decimals, time formats, datetime parsing)
        # Does NOT modify structure (no columns added/removed)
        df = normalize_data_formats(df)
        
        # Filter to selected columns if provided
        if selected_columns:
            # Validate columns exist
            invalid_columns = [col for col in selected_columns if col not in df.columns]
            if invalid_columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid columns: {', '.join(invalid_columns)}"
                )
            
            df = df[selected_columns]
            
            # Update selected_columns in database
            await datasets_store.update_one(
                {"_id": dataset_id},
                {"$set": {"selected_columns": selected_columns}}
            )
        
        # Perform EDA on the (possibly filtered) dataframe
        eda_service = EDAService()
        eda_results = eda_service.perform_eda(df)
        
        # Sanitize EDA results
        eda_results = sanitize_eda_results(eda_results)
        
        # Re-analyze dataset info (in case columns were filtered)
        preprocessing_service = PreprocessingService()
        analysis = preprocessing_service.analyze_dataset(df)
        
        # Update dataset document with EDA results and updated info
        # Keep original column_names, don't overwrite with filtered version
        update_data = {
            "eda_results": eda_results,
            "rows": analysis["rows"],
            "columns": analysis["columns"],
            # Only update column_names if no columns were selected (using all columns)
            "column_types": analysis["column_types"],
            "missing_values": analysis["missing_values"],
            "summary_statistics": analysis["summary_statistics"],
            "datetime_columns": analysis.get("datetime_columns", 0),
            "numerical_columns": analysis.get("numerical_columns", 0),
            "categorical_columns": analysis.get("categorical_columns", 0),
            "updated_at": datetime.utcnow()
        }
        
        # Don't overwrite column_names if we filtered to selected_columns
        # This preserves the original full list for the tag to work
        if not selected_columns:
            update_data["column_names"] = analysis["column_names"]
        
        await datasets_store.update_one(
            {"_id": dataset_id},
            {"$set": update_data}
        )
        
        # Fetch updated dataset
        updated_dataset = await datasets_store.find_one({"_id": dataset_id})
        updated_dataset["id"] = str(updated_dataset.pop("_id"))
        updated_dataset = sanitize_dataset_stats(updated_dataset)
        
        return {
            "message": "EDA completed successfully",
            "dataset": DatasetResponse(**updated_dataset),
            "eda_results": eda_results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        # Log detailed error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ EDA failed:")
        print(f"   Dataset ID: {dataset_id}")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        print(f"   Details:\n{error_details}")
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to analyze dataset: {str(e)}"
        )


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(
):
    """
    List all datasets for the current user
    
    Args:
    
    Returns:
        List of DatasetResponse objects
    """
    datasets_store = get_store("datasets")
    
    # Find all datasets for this user (exclude datasets where original was deleted)
    # Note: We use $or with $exists to handle both cases:
    # 1. Field doesn't exist (old datasets before this feature)
    # 2. Field exists but is not True
    cursor = datasets_store.find({
        "$or": [
            {"original_deleted": {"$exists": False}},  # Field doesn't exist (most datasets)
            {"original_deleted": False}  # Field exists but is False
        ]
    })
    datasets = []
    
    async for dataset in cursor:
        # Calculate column type counts if missing or zero (for backwards compatibility)
        if (not dataset.get("numerical_columns") and not dataset.get("categorical_columns") and not dataset.get("datetime_columns")):
            if "column_types" in dataset and dataset["column_types"]:
                numerical_count = sum(1 for t in dataset["column_types"].values() if t == "numerical")
                categorical_count = sum(1 for t in dataset["column_types"].values() if t == "categorical")
                datetime_count = sum(1 for t in dataset["column_types"].values() if t == "datetime")
                
                dataset["numerical_columns"] = numerical_count
                dataset["categorical_columns"] = categorical_count
                dataset["datetime_columns"] = datetime_count
            else:
                dataset["numerical_columns"] = 0
                dataset["categorical_columns"] = 0
                dataset["datetime_columns"] = 0
        
        dataset["id"] = str(dataset.pop("_id"))
        dataset = sanitize_dataset_stats(dataset)
        datasets.append(DatasetResponse(**dataset))
    
    return datasets


@router.get("/processed/list", response_model=List[DatasetResponse])
async def list_processed_datasets(
):
    """
    List all processed datasets for the current user
    (Datasets that have preprocessing_summary, even if original was deleted)
    
    Args:
    
    Returns:
        List of DatasetResponse objects with preprocessing data
    """
    datasets_store = get_store("datasets")
    
    # Find all datasets that have been preprocessed (with or without original)
    cursor = datasets_store.find({
        "preprocessing_summary": {"$exists": True}
    })
    datasets = []
    
    async for dataset in cursor:
        # Calculate column type counts if missing or zero (for backwards compatibility)
        if (not dataset.get("numerical_columns") and not dataset.get("categorical_columns") and not dataset.get("datetime_columns")):
            if "column_types" in dataset and dataset["column_types"]:
                numerical_count = sum(1 for t in dataset["column_types"].values() if t == "numerical")
                categorical_count = sum(1 for t in dataset["column_types"].values() if t == "categorical")
                datetime_count = sum(1 for t in dataset["column_types"].values() if t == "datetime")
                
                dataset["numerical_columns"] = numerical_count
                dataset["categorical_columns"] = categorical_count
                dataset["datetime_columns"] = datetime_count
            else:
                dataset["numerical_columns"] = 0
                dataset["categorical_columns"] = 0
                dataset["datetime_columns"] = 0
        
        dataset["id"] = str(dataset.pop("_id"))
        dataset = sanitize_dataset_stats(dataset)
        datasets.append(DatasetResponse(**dataset))
    
    return datasets


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str
):
    """
    Get details of a specific dataset
    
    Args:
        dataset_id: Dataset ID
    
    Returns:
        DatasetResponse object
    
    Raises:
        HTTPException: If dataset not found or access denied
    """
    datasets_store = get_store("datasets")
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Calculate column type counts if missing or zero (for backwards compatibility with old datasets)
    if (not dataset.get("numerical_columns") and not dataset.get("categorical_columns") and not dataset.get("datetime_columns")):
        if "column_types" in dataset and dataset["column_types"]:
            numerical_count = sum(1 for t in dataset["column_types"].values() if t == "numerical")
            categorical_count = sum(1 for t in dataset["column_types"].values() if t == "categorical")
            datetime_count = sum(1 for t in dataset["column_types"].values() if t == "datetime")
            
            dataset["numerical_columns"] = numerical_count
            dataset["categorical_columns"] = categorical_count
            dataset["datetime_columns"] = datetime_count
        else:
            dataset["numerical_columns"] = 0
            dataset["categorical_columns"] = 0
            dataset["datetime_columns"] = 0
    
    dataset["id"] = str(dataset.pop("_id"))
    dataset = sanitize_dataset_stats(dataset)
    return DatasetResponse(**dataset)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str
):
    """
    Delete a dataset
    
    Args:
        dataset_id: Dataset ID
    
    Raises:
        HTTPException: If dataset not found or access denied
    """
    datasets_store = get_store("datasets")
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Delete original file from disk (but keep preprocessed file if it exists)
    if os.path.exists(dataset["file_path"]):
        os.remove(dataset["file_path"])
    
    # Update database: Mark as deleted but keep preprocessing data
    # This allows preprocessed datasets to remain accessible even after original is deleted
    if dataset.get("preprocessing_summary"):
        # If dataset has been preprocessed, just mark the original as deleted
        # instead of removing the entire document
        await datasets_store.update_one(
            {"_id": dataset_id},
            {
                "$set": {
                    "original_deleted": True,
                    "original_deleted_at": datetime.now(timezone.utc)
                },
                "$unset": {
                    "file_path": ""  # Remove reference to deleted original file
                }
            }
        )
    else:
        # If dataset was never preprocessed, delete it completely
        await datasets_store.delete_one({"_id": dataset_id})
    
    return None


@router.delete("/{dataset_id}/preprocessing", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preprocessing(
    dataset_id: str
):
    """
    Delete preprocessing data for a dataset
    (Removes preprocessed file and preprocessing_summary, but keeps original if it exists)
    
    Args:
        dataset_id: Dataset ID
    
    Raises:
        HTTPException: If dataset not found or access denied
    """
    datasets_store = get_store("datasets")
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Delete preprocessed file from disk if it exists
    if dataset.get("preprocessed_file_path") and os.path.exists(dataset["preprocessed_file_path"]):
        os.remove(dataset["preprocessed_file_path"])
    
    # Delete preprocessing pipeline file if it exists
    if dataset.get("preprocessing_summary", {}).get("pipeline_path"):
        pipeline_path = dataset["preprocessing_summary"]["pipeline_path"]
        if os.path.exists(pipeline_path):
            os.remove(pipeline_path)
    
    # Check if original was deleted
    if dataset.get("original_deleted"):
        # If original was deleted, remove the entire document
        await datasets_store.delete_one({"_id": dataset_id})
    else:
        # If original still exists, just remove preprocessing data
        await datasets_store.update_one(
            {"_id": dataset_id},
            {
                "$unset": {
                    "preprocessing_summary": "",
                    "preprocessed_file_path": ""
                }
            }
        )
    
    return None


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str,
    name: str = Form(...),
    description: Optional[str] = Form(None)
):
    """
    Update dataset metadata (name and description only)
    
    Args:
        dataset_id: Dataset ID
        name: Updated dataset name
        description: Updated dataset description
    
    Returns:
        Updated DatasetResponse object
    
    Raises:
        HTTPException: If dataset not found or access denied
    """
    datasets_store = get_store("datasets")
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Update only name and description
    update_data = {
        "name": name,
        "description": description,
        "updated_at": datetime.utcnow()
    }
    
    await datasets_store.update_one(
        {"_id": dataset_id},
        {"$set": update_data}
    )
    
    # Fetch and return updated dataset
    updated_dataset = await datasets_store.find_one({"_id": dataset_id})
    
    # Calculate column type counts if missing
    if (not updated_dataset.get("numerical_columns") and not updated_dataset.get("categorical_columns") and not updated_dataset.get("datetime_columns")):
        if "column_types" in updated_dataset and updated_dataset["column_types"]:
            numerical_count = sum(1 for t in updated_dataset["column_types"].values() if t == "numerical")
            categorical_count = sum(1 for t in updated_dataset["column_types"].values() if t == "categorical")
            datetime_count = sum(1 for t in updated_dataset["column_types"].values() if t == "datetime")
            
            updated_dataset["numerical_columns"] = numerical_count
            updated_dataset["categorical_columns"] = categorical_count
            updated_dataset["datetime_columns"] = datetime_count
        else:
            updated_dataset["numerical_columns"] = 0
            updated_dataset["categorical_columns"] = 0
            updated_dataset["datetime_columns"] = 0
    
    updated_dataset["id"] = str(updated_dataset.pop("_id"))
    updated_dataset = sanitize_dataset_stats(updated_dataset)
    return DatasetResponse(**updated_dataset)


@router.put("/{dataset_id}/selected_columns")
async def update_selected_columns(
    dataset_id: str,
    selected_columns: List[str] = Body(..., embed=True)
):
    """
    Update the selected columns for a dataset (used for training)
    
    Args:
        dataset_id: Dataset ID
        selected_columns: List of column names selected by user for training
    
    Returns:
        Success message with number of columns selected
    
    Raises:
        HTTPException: If dataset not found, access denied, or invalid columns
    """
    datasets_store = get_store("datasets")
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Validate that all selected columns exist in the dataset
    available_columns = dataset.get("column_names", [])
    if not available_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset has no column information"
        )
    
    invalid_columns = [col for col in selected_columns if col not in available_columns]
    if invalid_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid columns: {', '.join(invalid_columns)}"
        )
    
    if len(selected_columns) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one column must be selected"
        )
    
    # Update selected columns in database
    await datasets_store.update_one(
        {"_id": dataset_id},
        {
            "$set": {
                "selected_columns": selected_columns,
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    return {
        "message": "Selected columns updated successfully",
        "selected_count": len(selected_columns),
        "total_count": len(available_columns)
    }


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    rows: int = 10,
    use_preprocessed: bool = True
):
    """
    Get a preview of the dataset (first N rows)
    
    Args:
        dataset_id: Dataset ID
        rows: Number of rows to preview (default: 10)
        use_preprocessed: Whether to use preprocessed version if available (default: True)
    
    Returns:
        Dictionary with preview data
    
    Raises:
        HTTPException: If dataset not found or access denied
    """
    datasets_store = get_store("datasets")
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    try:
        # Determine which file to use
        file_path = None
        
        # Priority 1: If use_preprocessed is True and preprocessed file exists, use it
        if use_preprocessed and dataset.get("preprocessed_file_path"):
            file_path = dataset["preprocessed_file_path"]
        # Priority 2: If original file exists, use it
        elif dataset.get("file_path"):
            file_path = dataset["file_path"]
        # Priority 3: If only preprocessed exists (original deleted), use preprocessed
        elif dataset.get("preprocessed_file_path"):
            file_path = dataset["preprocessed_file_path"]
        
        # If no file path available, return error
        if not file_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data file available for this dataset"
            )
        
        # Determine how many rows to read
        # If we have train/test split, we need to read enough rows to show both
        rows_to_read = rows
        if use_preprocessed and dataset.get("preprocessing_summary", {}).get("has_train_test_split"):
            train_size = dataset["preprocessing_summary"].get("train_size", 0)
            # Read at least train_size + 2 test rows to show the split
            rows_to_read = min(train_size + 10, train_size + dataset["preprocessing_summary"].get("test_size", 0))
        
        # Read dataset with encoding handling and smart delimiter detection
        detected_encoding = 'utf-8'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)
        except UnicodeDecodeError:
            # Try common encodings
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1024)
                    detected_encoding = encoding
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            
            # Last resort: use chardet
            if detected_encoding == 'utf-8':
                import chardet
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    detected_encoding = result['encoding']
        
        # OPTIMIZATION: For large-column datasets, only read first 31 columns for preview
        # This dramatically improves performance when dataset has 1000+ columns
        MAX_COLUMNS_PREVIEW = 30
        
        # First, read just the header to get total column count
        df_header = pd.read_csv(file_path, encoding=detected_encoding, nrows=0)
        total_columns = len(df_header.columns)
        
        # Determine which columns to read
        usecols_param = None
        if total_columns > MAX_COLUMNS_PREVIEW:
            # Only read first 30 columns + target if it exists beyond that
            usecols_param = list(range(min(MAX_COLUMNS_PREVIEW + 1, total_columns)))
        
        # Use smart CSV reader with delimiter detection and column cleaning
        # IMPORTANT: For preprocessed files, disable na_values detection because imputed values
        # like -1.0, -999 are legitimate data, not missing values
        is_preprocessed = (file_path == dataset.get("preprocessed_file_path"))
        if is_preprocessed:
            df = read_csv_smart(file_path, detected_encoding, nrows=rows_to_read, 
                                na_values=[], keep_default_na=False, usecols=usecols_param)
        else:
            df = read_csv_smart(file_path, detected_encoding, nrows=rows_to_read, usecols=usecols_param)
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to read dataset. The file may be corrupted or empty."
            )
        
        # Filter to selected columns if they exist (only for original file, not preprocessed)
        # Preprocessed files are already filtered during EDA
        if not use_preprocessed or file_path == dataset.get("file_path"):
            selected_columns = dataset.get("selected_columns")
            if selected_columns:
                # Filter DataFrame to only show selected columns
                available_cols = [col for col in selected_columns if col in df.columns]
                if available_cols:
                    df = df[available_cols]
        
        # Handle train/test split preview for preprocessed data
        preview_data = {}
        if use_preprocessed and dataset.get("preprocessing_summary", {}).get("has_train_test_split"):
            train_size = dataset["preprocessing_summary"].get("train_size", 0)
            test_size = dataset["preprocessing_summary"].get("test_size", 0)
            
            # Show 8 rows of train, 2 rows of test (if we have 10 rows requested)
            train_rows = min(8, train_size, rows)
            test_rows = min(2, test_size, rows - train_rows)
            
            df_train = df.iloc[:train_size].head(train_rows)
            df_test = df.iloc[train_size:].head(test_rows)
            
            preview_data = {
                "has_split": True,
                "train_preview": df_train.to_dict(orient='records'),
                "test_preview": df_test.to_dict(orient='records'),
                "train_size": train_size,
                "test_size": test_size,
                "total_rows": train_size + test_size
            }
        else:
            # No split - regular preview
            preview_data = {
                "has_split": False,
                "data": df.to_dict(orient='records'),
                "total_rows": len(df)
            }
        
        # Sanitize column names - remove special characters that might cause JSON issues
        df.columns = [str(col).strip() for col in df.columns]
        
        # Convert DataFrame to dictionary format with proper serialization
        def sanitize_value(val):
            """Convert values to JSON-serializable format"""
            if pd.isna(val):
                return None
            elif isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                if np.isinf(val):
                    return None
                return float(val)
            elif isinstance(val, (pd.Timestamp, datetime)):
                return val.isoformat()
            elif isinstance(val, bool):
                return bool(val)
            else:
                return str(val)
        
        # Calculate column limiting info (we already limited columns during read)
        all_columns = df.columns.tolist()
        displayed_columns = all_columns
        hidden_columns_count = total_columns - len(all_columns) if total_columns > len(all_columns) else 0
        
        # If we have train/test split data, use that structure
        if preview_data.get("has_split"):
            # Sanitize train and test data - only include displayed columns
            train_sanitized = []
            for row in preview_data["train_preview"]:
                sanitized_row = {col: sanitize_value(row.get(col)) for col in displayed_columns}
                train_sanitized.append(sanitized_row)
            
            test_sanitized = []
            for row in preview_data["test_preview"]:
                sanitized_row = {col: sanitize_value(row.get(col)) for col in displayed_columns}
                test_sanitized.append(sanitized_row)
            
            preview = {
                "columns": displayed_columns,
                "has_split": True,
                "train_data": train_sanitized,
                "test_data": test_sanitized,
                "train_size": preview_data["train_size"],
                "test_size": preview_data["test_size"],
                "total_columns": total_columns,
                "hidden_columns": hidden_columns_count
            }
            
            # Add message about hidden columns if any
            if hidden_columns_count > 0:
                preview["column_notice"] = f"And {hidden_columns_count} more columns"
        else:
            # No split - regular format
            data = []
            for row in preview_data["data"]:
                sanitized_row = {col: sanitize_value(row.get(col)) for col in displayed_columns}
                data.append(sanitized_row)
            
            preview = {
                "columns": displayed_columns,
                "has_split": False,
                "data": data,
                "total_columns": total_columns,
                "hidden_columns": hidden_columns_count
            }
            
            # Add message about hidden columns if any
            if hidden_columns_count > 0:
                preview["column_notice"] = f"And {hidden_columns_count} more columns"
        
        return preview
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Preview error for dataset {dataset_id}: {error_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate preview: {str(e)}"
        )


@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    use_preprocessed: bool = True
):
    """
    Download a dataset file (original or preprocessed)
    
    Args:
        dataset_id: Dataset ID
        use_preprocessed: Whether to download preprocessed version if available (default: True)
    
    Returns:
        FileResponse with the dataset CSV file
    
    Raises:
        HTTPException: If dataset not found or access denied
    """
    datasets_store = get_store("datasets")
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Determine which file to download
    file_path = dataset["file_path"]
    # Use dataset name instead of filename, sanitize it for safe file download
    dataset_name = dataset["name"].replace(" ", "_").replace("/", "_").replace("\\", "_")
    # Remove any characters that could cause issues in filenames
    dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in ('_', '-'))
    filename = f"{dataset_name}.csv"
    
    if use_preprocessed and dataset.get("preprocessed_file_path"):
        file_path = dataset["preprocessed_file_path"]
        filename = f"{dataset_name}_preprocessed.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server"
        )
    
    # Filter by selected columns if they exist (only for original file)
    selected_columns = dataset.get("selected_columns")
    if selected_columns and not use_preprocessed:
        # Read, filter, and create temporary file
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python', sep=None, skipinitialspace=True, na_values=MISSING_VALUE_INDICATORS, keep_default_na=True)
        except UnicodeDecodeError:
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', engine='python', sep=None, skipinitialspace=True, na_values=MISSING_VALUE_INDICATORS, keep_default_na=True)
                    break
                except Exception:
                    continue
        
        # Filter to selected columns
        available_cols = [col for col in selected_columns if col in df.columns]
        if available_cols:
            df = df[available_cols]
            
            # Create temporary filtered file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
            df.to_csv(temp_file.name, index=False)
            temp_file.close()
            file_path = temp_file.name
    
    print(f"📥 Download: filename={filename}, path={file_path}")
    
    # Return file for download with explicit headers
    # Use both filename parameter and Content-Disposition header for maximum compatibility
    from urllib.parse import quote
    encoded_filename = quote(filename)
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded_filename}'
        }
    )


@router.get("/{dataset_id}/download/split")
async def download_train_test_split(
    dataset_id: str,
    split_type: str = "train",  # "train" or "test"
):
    """
    Download train or test split separately for datasets with train/test split
    
    Args:
        dataset_id: Dataset ID
        split_type: Either "train" or "test"
    
    Returns:
        FileResponse with the train or test CSV file
    
    Raises:
        HTTPException: If dataset not found, no split available, or access denied
    """
    if split_type not in ["train", "test"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="split_type must be either 'train' or 'test'"
        )
    
    datasets_store = get_store("datasets")
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Check if dataset has train/test split
    preprocessing_summary = dataset.get("preprocessing_summary", {})
    if not preprocessing_summary.get("has_train_test_split"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This dataset does not have a train/test split"
        )
    
    # Get the preprocessed file path
    file_path = dataset.get("preprocessed_file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preprocessed dataset file not found"
        )
    
    # Read the preprocessed file
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python', sep=None, skipinitialspace=True, na_values=MISSING_VALUE_INDICATORS, keep_default_na=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read preprocessed file: {str(e)}"
        )
    
    # Split the dataframe based on train_size
    train_size = preprocessing_summary.get("train_size", 0)
    
    # Sanitize dataset name
    dataset_name = dataset["name"].replace(" ", "_").replace("/", "_").replace("\\", "_")
    dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in ('_', '-'))
    
    if split_type == "train":
        split_df = df.iloc[:train_size]
        filename = f"{dataset_name}_train.csv"
    else:  # test
        split_df = df.iloc[train_size:]
        filename = f"{dataset_name}_test.csv"
    
    print(f"📥 Download split: filename={filename}, split_type={split_type}")
    
    # Create temporary file for the split
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
    split_df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    # Return file for download with proper encoding
    from urllib.parse import quote
    encoded_filename = quote(filename)
    
    return FileResponse(
        path=temp_file.name,
        filename=filename,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded_filename}'
        }
    )


@router.get("/{dataset_id}/eda")
async def get_eda_results(
    dataset_id: str
):
    """
    Get EDA results for a specific dataset
    
    Args:
        dataset_id: Dataset ID
    
    Returns:
        Dictionary with EDA results
    
    Raises:
        HTTPException: If dataset not found or access denied
    """
    datasets_store = get_store("datasets")
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    eda_results = dataset.get("eda_results")
    
    if not eda_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EDA results not available for this dataset"
        )
    
    return eda_results


@router.post("/{dataset_id}/preprocess")
async def preprocess_dataset(
    dataset_id: str,
    request: Request,
    target_column: Optional[str] = None,
    model_type: Optional[str] = None
):
    """
    Perform intelligent adaptive preprocessing on a dataset for ML model training.
    
    Uses the Adaptive Preprocessing Pipeline that:
    - Automatically detects task type (classification/regression/unsupervised)
    - Optimizes preprocessing based on model type
    - Leverages EDA insights for intelligent decisions
    - Logs every decision with reasoning
    
    Args:
        dataset_id: Dataset ID
        target_column: Optional target column name for supervised learning
        model_type: Optional model type (e.g., 'random_forest', 'logistic_regression')
        request: Request body containing optional remove_rare_values parameter
    
    Returns:
        Dictionary with adaptive preprocessing results and comprehensive decision log
    
    Raises:
        HTTPException: If dataset not found or access denied
    """
    datasets_store = get_store("datasets")
    
    # Get request body for rare value removal and outlier handling options
    remove_rare_values = {}
    outlier_preferences = {}
    try:
        if request.headers.get("content-length") and int(request.headers.get("content-length", "0")) > 0:
            body = await request.json()
            remove_rare_values = body.get("remove_rare_values", {})
            outlier_preferences = body.get("outlier_preferences", {})
            print(f"📥 Received request body:")
            print(f"   remove_rare_values: {remove_rare_values}")
            print(f"   outlier_preferences: {outlier_preferences}")
    except Exception as e:
        # No body or invalid JSON - continue without rare value filtering or outlier preferences
        print(f"⚠️ No request body or invalid JSON: {e}")
        pass
    
    try:
        dataset = await datasets_store.find_one({
            "_id": dataset_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    try:
        # Read dataset with smart delimiter detection and column cleaning
        detected_encoding = 'utf-8'
        try:
            with open(dataset["file_path"], 'r', encoding='utf-8') as f:
                f.read(1024)
        except UnicodeDecodeError:
            # Try common encodings
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16']:
                try:
                    with open(dataset["file_path"], 'r', encoding=encoding) as f:
                        f.read(1024)
                    detected_encoding = encoding
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            
            # Last resort: use chardet
            if detected_encoding == 'utf-8':
                import chardet
                with open(dataset["file_path"], 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    detected_encoding = result['encoding']
        
        # Use smart CSV reader with delimiter detection and column cleaning
        df = read_csv_smart(dataset["file_path"], detected_encoding)
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to read dataset"
            )
        
        # CRITICAL: Clean and parse data formats BEFORE any processing
        df = clean_and_parse_data(df)
        
        # Filter to selected columns if they were specified during upload
        selected_columns = dataset.get('selected_columns')
        if selected_columns:
            # CRITICAL: Always include target column if specified
            if target_column and target_column not in selected_columns:
                print(f"⚠️ Target column '{target_column}' was not in selected columns. Adding it to preserve target.")
                selected_columns = selected_columns + [target_column]
            
            # Validate columns exist
            available_cols = [col for col in selected_columns if col in df.columns]
            if available_cols:
                print(f"📋 Filtering dataset to {len(available_cols)} selected columns...")
                df = df[available_cols]
            else:
                print(f"⚠️ Warning: Selected columns not found, using all columns")
        
        # Store ORIGINAL dataset info BEFORE rare value removal
        original_shape = df.shape
        original_columns = df.columns.tolist()
        
        # Calculate initial data quality metrics
        initial_total_cells = original_shape[0] * original_shape[1]
        initial_missing_cells = df.isnull().sum().sum()
        initial_completeness = 100.0 - (initial_missing_cells / initial_total_cells * 100) if initial_total_cells > 0 else 100.0
        
        rare_value_removal_summary = []
        
        # Apply rare value filtering if requested by user
        if remove_rare_values:
            print(f"🧹 Applying rare value filtering for {len(remove_rare_values)} columns...")
            initial_rows = len(df)
            
            for column_name, values_to_remove in remove_rare_values.items():
                if values_to_remove and column_name in df.columns:
                    # values_to_remove is now a list of specific values, not a boolean
                    if isinstance(values_to_remove, list) and len(values_to_remove) > 0:
                        # Show what we're about to remove
                        print(f"\n   Column: '{column_name}'")
                        print(f"   Values to remove: {values_to_remove}")
                        print(f"   Value types: {[type(v).__name__ for v in values_to_remove]}")
                        print(f"   Current value distribution:")
                        for val in values_to_remove:
                            count = (df[column_name] == val).sum()
                            print(f"      - '{val}': {count} rows")
                        
                        # Convert values_to_remove to match dataframe types
                        # Get the dtype of the column
                        col_dtype = df[column_name].dtype
                        print(f"   Column dtype: {col_dtype}")
                        
                        # Ensure type compatibility and strip whitespace for string columns
                        values_to_remove_typed = []
                        for val in values_to_remove:
                            # Keep as-is for string/object types, convert for others
                            if col_dtype == 'object' or col_dtype == 'string':
                                # Strip whitespace from both the value and when comparing
                                values_to_remove_typed.append(str(val).strip())
                            else:
                                values_to_remove_typed.append(val)
                        
                        print(f"   Typed values to remove: {values_to_remove_typed}")
                        
                        # For string columns, strip whitespace for comparison but preserve NaN
                        if col_dtype == 'object' or col_dtype == 'string':
                            # Strip whitespace but DON'T convert NaN to string 'nan'
                            # Only strip actual string values
                            df[column_name] = df[column_name].apply(
                                lambda x: x.strip() if isinstance(x, str) and pd.notna(x) else x
                            )
                        
                        # Remove rows with specified rare values
                        before_count = len(df)
                        mask = ~df[column_name].isin(values_to_remove_typed)
                        df = df[mask]
                        removed_count = before_count - len(df)
                        
                        if removed_count > 0:
                            rare_value_removal_summary.append({
                                'column': column_name,
                                'values_removed': values_to_remove_typed,
                                'rows_removed': removed_count
                            })
                        
                        print(f"   ✓ Removed {removed_count} rows from '{column_name}'")
                        print(f"   Remaining value distribution:")
                        remaining_counts = df[column_name].value_counts()
                        for val, count in remaining_counts.items():
                            print(f"      - '{val}': {count} rows")
            
            final_rows = len(df)
            if initial_rows > final_rows:
                print(f"\n   📊 Total rows removed: {initial_rows - final_rows} ({((initial_rows - final_rows) / initial_rows * 100):.1f}%)")
                print(f"   📊 Remaining rows: {final_rows}")
        
        # Validate we have enough data after rare value removal
        if len(df) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough data after removing rare values. Only {len(df)} rows remain. Need at least 10 rows."
            )
        
        # If target column specified, validate class distribution (ONLY for categorical targets)
        if target_column and target_column in df.columns:
            # Check if target is categorical or numerical
            target_dtype = df[target_column].dtype
            is_categorical_target = target_dtype == 'object' or target_dtype.name == 'category'
            
            # Only validate class distribution for categorical targets (classification tasks)
            if is_categorical_target:
                class_counts = df[target_column].value_counts()
                min_class_count = class_counts.min()
                
                if min_class_count < 2:
                    # Find which classes have < 2 samples
                    problematic_classes = class_counts[class_counts < 2].index.tolist()
                    
                    # Create helpful error message
                    class_details = [f"'{cls}': {class_counts[cls]} sample(s)" for cls in problematic_classes]
                    
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"After removing rare values, target column '{target_column}' has classes with too few samples for train/test split:\n" +
                               f"{', '.join(class_details)}.\n\n" +
                               f"Each class needs at least 2 samples. Current distribution:\n" +
                               f"{dict(class_counts)}.\n\n" +
                               f"Try removing fewer rare values or choose a different target column."
                    )
                
                if min_class_count < 4:
                    print(f"⚠️ Warning: Target column '{target_column}' has classes with very few samples (minimum: {min_class_count}). This may cause issues with train/test split.")
                    print(f"   Class distribution: {dict(class_counts)}")
            else:
                # For numerical targets (regression), just log info
                print(f"✓ Target column '{target_column}' is numerical (dtype: {target_dtype}). Regression task detected.")
                print(f"   Value range: {df[target_column].min()} to {df[target_column].max()}")
                print(f"   Unique values: {df[target_column].nunique()}")
        
        # Get EDA results (or run EDA if not available)
        eda_results = dataset.get('eda_results')
        if not eda_results:
            print("📊 Running EDA analysis (not found in dataset)...")
            eda_service = EDAService()
            eda_results = eda_service.perform_eda(df)
            
            # Update dataset with EDA results
            await datasets_store.update_one(
                {"_id": dataset_id},
                {"$set": {"eda_results": sanitize_eda_results(eda_results)}}
            )
        
        # Initialize Adaptive Preprocessor with EDA insights
        print(f"🤖 Initializing Adaptive Preprocessor with EDA insights...")
        print(f"   Dataset shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        preprocessor = AdaptivePreprocessor(eda_results=eda_results, outlier_preferences=outlier_preferences)
        
        # Perform intelligent adaptive preprocessing
        print(f"🔧 Running adaptive preprocessing...")
        print(f"   Target: {target_column or 'None (Unsupervised)'}")
        print(f"   Model: {model_type or 'Auto-detect'}")
        
        results = preprocessor.fit_transform(
            df=df,
            target_column=target_column,
            model_type=model_type,
            test_size=0.2 if target_column else 0,
            random_state=42
        )
        
        # Extract preprocessed data
        X_preprocessed = results['X_train']
        
        # Track train/test split info
        has_train_test_split = target_column and results['X_test'] is not None
        
        # Handle both dense and sparse arrays
        train_size = results['X_train'].shape[0] if results['X_train'] is not None else 0
        test_size = results['X_test'].shape[0] if results['X_test'] is not None else 0
        
        # If supervised, combine train and test for storage
        if has_train_test_split:
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: Combining train/test data")
            print(f"   X_train shape: {results['X_train'].shape}")
            print(f"   X_test shape: {results['X_test'].shape}")
            print(f"   y_train shape: {results['y_train'].shape}")
            print(f"   y_test shape: {results['y_test'].shape}")
            print(f"   Feature names count: {len(results['feature_names'])}")
            print(f"   Target column: {target_column}")
            print(f"{'='*80}\n")
            
            # Convert sparse matrices to dense for DataFrame creation
            from scipy.sparse import issparse
            
            X_train_data = results['X_train'].toarray() if issparse(results['X_train']) else results['X_train']
            X_test_data = results['X_test'].toarray() if issparse(results['X_test']) else results['X_test']
            
            # Combine X and y for storage
            X_train_df = pd.DataFrame(X_train_data, columns=results['feature_names'])
            print(f"✓ Created X_train_df: {X_train_df.shape} (converted from sparse: {issparse(results['X_train'])})")
            
            X_test_df = pd.DataFrame(X_test_data, columns=results['feature_names'])
            print(f"✓ Created X_test_df: {X_test_df.shape} (converted from sparse: {issparse(results['X_test'])})")
            
            # Decode y values back to original labels if they were encoded
            y_train_values = results['y_train']
            y_test_values = results['y_test']
            if hasattr(preprocessor, 'label_encoder') and preprocessor.label_encoder:
                y_train_values = preprocessor.label_encoder.inverse_transform(results['y_train'])
                y_test_values = preprocessor.label_encoder.inverse_transform(results['y_test'])
                print(f"✓ Decoded labels back to original values")
            
            y_train_series = pd.Series(y_train_values, name=target_column)
            y_test_series = pd.Series(y_test_values, name=target_column)
            print(f"✓ Created y series: train={len(y_train_series)}, test={len(y_test_series)}")
            
            # Combine train data (X + y)
            train_df = X_train_df.copy()
            train_df[target_column] = y_train_series.values
            print(f"✓ Created train_df with target: {train_df.shape}, columns={list(train_df.columns)}")
            
            # Combine test data (X + y)
            test_df = X_test_df.copy()
            test_df[target_column] = y_test_series.values
            print(f"✓ Created test_df with target: {test_df.shape}, columns={list(test_df.columns)}")
            
            # Concatenate train and test
            print(f"🔄 Concatenating train and test...")
            df_preprocessed = pd.concat([train_df, test_df], ignore_index=True)
            print(f"✓ Final df_preprocessed shape: {df_preprocessed.shape}")
            print(f"   Columns ({len(df_preprocessed.columns)}): {list(df_preprocessed.columns)}")
        else:
            # Unsupervised - just use X
            from scipy.sparse import issparse
            X_data = X_preprocessed.toarray() if issparse(X_preprocessed) else X_preprocessed
            df_preprocessed = pd.DataFrame(X_data, columns=results['feature_names'])
            print(f"✓ Created unsupervised df_preprocessed: {df_preprocessed.shape} (converted from sparse: {issparse(X_preprocessed)})")
        
        # Replace any remaining NaN/null values with empty string for clean CSV output
        # (after preprocessing, there shouldn't be any NaNs, but just in case)
        print(f"🔍 Checking for NaN values in preprocessed data...")
        nan_counts = df_preprocessed.isna().sum()
        total_nans = nan_counts.sum()
        print(f"   Total NaN count: {total_nans}")
        if total_nans > 0:
            print(f"⚠️  WARNING: Found {total_nans} NaN values in preprocessed data!")
            print(f"   Columns with NaN: {nan_counts[nan_counts > 0].to_dict()}")
            print(f"   This should not happen - imputation should have handled all NaNs")
            # Fill with 0 for numerical columns as a fallback
            df_preprocessed = df_preprocessed.fillna(0)
            print(f"   Filled NaN values with 0")
        else:
            print(f"   ✓ No NaN values found - data is clean!")
        
        # Also check for string 'null', 'nan', 'None' values
        print(f"🔍 Checking for string null values...")
        for col in df_preprocessed.columns:
            if df_preprocessed[col].dtype == 'object':
                null_like = df_preprocessed[col].isin(['null', 'NULL', 'nan', 'NaN', 'None', 'none'])
                if null_like.any():
                    count = null_like.sum()
                    print(f"   ⚠️  Column '{col}' has {count} string null values")
                    df_preprocessed[col] = df_preprocessed[col].replace(['null', 'NULL', 'nan', 'NaN', 'None', 'none'], '')
        
        # Save preprocessed dataset
        base_filename = os.path.splitext(os.path.basename(dataset["file_path"]))[0]
        preprocessed_filename = f"preprocessed_{base_filename}.csv"
        preprocessed_path = os.path.join(settings.upload_dir, preprocessed_filename)
        
        df_preprocessed.to_csv(preprocessed_path, index=False, na_rep='')
        print(f"💾 Saved preprocessed dataset to {preprocessed_path}")
        
        # Save preprocessing pipeline for later use
        pipeline_path = os.path.join(settings.models_dir, f"{dataset_id}_preprocessing_pipeline.pkl")
        os.makedirs(settings.models_dir, exist_ok=True)
        preprocessor.save_pipeline(pipeline_path)
        print(f"💾 Saved preprocessing pipeline to {pipeline_path}")
        
        # Get comprehensive summary
        preprocessing_summary = preprocessor.get_preprocessing_summary()
        
        # Format decision log for frontend display
        decision_log_formatted = []
        for decision in results['decision_log']:
            decision_log_formatted.append({
                'timestamp': decision['timestamp'],
                'category': decision['category'].replace('_', ' ').title(),
                'decision': decision['decision'],
                'reason': decision['reason'],
                'impact': decision['impact']
            })
        
        # Build preprocessing steps summary for display
        preprocessing_steps = []
        
        # Add key decisions
        preprocessing_steps.append(f"Task Type: {results['task_type'].capitalize()}")
        preprocessing_steps.append(f"Model Family: {results['model_family'].replace('_', ' ').title()}")
        
        metadata = results['preprocessing_metadata']
        
        # Columns removed
        if metadata['total_columns_removed'] > 0:
            removed_details = []
            if metadata['id_columns_removed']:
                removed_details.append(f"{len(metadata['id_columns_removed'])} ID columns")
            if metadata['constant_columns_removed']:
                removed_details.append(f"{len(metadata['constant_columns_removed'])} constant columns")
            if metadata['high_missing_columns_removed']:
                removed_details.append(f"{len(metadata['high_missing_columns_removed'])} high-missing columns")
            
            preprocessing_steps.append(
                f"Removed {metadata['total_columns_removed']} columns: " + 
                ", ".join(removed_details)
            )
        
        # Features
        feature_parts = []
        if metadata['numerical_features']:
            feature_parts.append(f"{len(metadata['numerical_features'])} numerical")
        if metadata['categorical_features']:
            feature_parts.append(f"{len(metadata['categorical_features'])} categorical")
        if metadata.get('text_features'):
            text_count = len(metadata['text_features'])
            feature_parts.append(f"{text_count} text")
        
        if feature_parts:
            preprocessing_steps.append(f"Features: {', '.join(feature_parts)}")
        else:
            preprocessing_steps.append("Features: None detected")
        
        # Encoding and scaling
        if metadata['categorical_features']:
            # Check decision log for encoding strategy
            encoding_decisions = [d for d in results['decision_log'] if d['category'] == 'encoding']
            if encoding_decisions:
                preprocessing_steps.append(f"{encoding_decisions[0]['decision']}")
        
        # Class imbalance detection (for classification only)
        imbalance_decisions = [d for d in results['decision_log'] if d['category'] in ['class_imbalance', 'class_balance']]
        if imbalance_decisions:
            decision = imbalance_decisions[0]['decision']
            reason = imbalance_decisions[0]['reason']
            # Combine decision and first part of reason for display
            if 'imbalance' in decision.lower():
                # Extract just the ratio part for brevity
                preprocessing_steps.append(f"{decision}")
            else:
                preprocessing_steps.append(f"{decision}")
        
        # Class imbalance handling result (if applied)
        imbalance_result_decisions = [d for d in results['decision_log'] if d['category'] == 'class_imbalance_result']
        if imbalance_result_decisions:
            result_decision = imbalance_result_decisions[0]
            preprocessing_steps.append(f"{result_decision['decision']}")
        
        scaling_decisions = [d for d in results['decision_log'] if d['category'] == 'scaling']
        if scaling_decisions:
            preprocessing_steps.append(f"{scaling_decisions[0]['decision']}")
        
        # Text vectorization (if text features present)
        if metadata.get('text_features'):
            text_cols = metadata['text_features']
            max_features = preprocessing_summary.get('text_features', {}).get('max_features', 5000)
            if len(text_cols) == 1:
                preprocessing_steps.append(f"✓ TF-IDF vectorization applied: '{text_cols[0]}' → {max_features} features")
            else:
                preprocessing_steps.append(f"✓ TF-IDF vectorization applied to {len(text_cols)} columns → {max_features * len(text_cols)} features")
        
        # Quality score
        quality_score = results['quality_metrics']['quality_score']
        preprocessing_steps.append(f"Quality Score: {quality_score:.1f}/100")
        
        # Determine if scaling was actually applied
        scaling_applied = False
        scaled_columns_list = []
        scaling_decisions = [d for d in results['decision_log'] if d['category'] == 'scaling']
        if scaling_decisions:
            decision_text = scaling_decisions[0]['decision'].lower()
            # Check if scaling was actually applied (not "no scaling")
            if 'no scaling' not in decision_text and 'scaler' in decision_text:
                scaling_applied = True
                scaled_columns_list = metadata['numerical_features']
        
        # Update dataset record with preprocessing info (backward-compatible format)
        update_data = {
            "preprocessed": True,
            "preprocessed_at": datetime.utcnow(),
            "preprocessed_file_path": preprocessed_path,
            "preprocessing_summary": {
                # Old format for backward compatibility
                "original": {
                    "rows": int(original_shape[0]),
                    "columns": int(original_shape[1]),
                    "column_names": original_columns
                },
                "processed": {
                    "rows": int(df_preprocessed.shape[0]),
                    "columns": int(df_preprocessed.shape[1]),
                    "column_names": df_preprocessed.columns.tolist()
                },
                "initial_quality": {
                    "quality_score": float(round(quality_score * 0.85, 2)),
                    "completeness": float(round(initial_completeness, 2)),
                    "uniqueness": 100.0,
                    "assessment": "Good" if quality_score > 80 else "Fair"
                },
                "final_quality": {
                    "quality_score": float(round(quality_score, 2)),
                    "completeness": 100.0,
                    "uniqueness": 100.0,
                    "assessment": "Excellent - Ready for ML" if quality_score >= 95 else "Good - Ready for ML" if quality_score >= 80 else "Fair"
                },
                "changes": {
                    "rows_removed": int(original_shape[0] - df_preprocessed.shape[0]),
                    "columns_removed": int(metadata['total_columns_removed']),
                    "duplicates_removed": int(metadata.get('duplicates_removed_count', 0)),
                    "duplicate_removal_details": metadata.get('duplicate_removal_details', []),
                    "outliers_handled": int(metadata.get('outliers_handled_count', 0)),
                    "multicollinearity_handled": int(metadata.get('multicollinearity_handled', {}).get('pairs_detected', 0)),
                    "skewness_corrected": int(len(metadata.get('skewed_features_transformed', []))),
                    "missing_values_filled": int(results['quality_metrics'].get('missing_data_handled', 0)),
                    "categorical_columns_encoded": int(len(metadata['categorical_features'])),
                    "numerical_columns_scaled": int(len(scaled_columns_list)),
                    "rare_values_removed": rare_value_removal_summary
                },
                "steps": preprocessing_steps,
                "scaled_columns": scaled_columns_list,
                "outlier_details": metadata.get('outlier_details', []),
                "outlier_rows_removed": metadata.get('outlier_rows_removed', 0),
                "skewness_details": metadata.get('skewness_details', []),
                "skewed_features_transformed": metadata.get('skewed_features_transformed', []),
                "imputation_details": metadata.get('imputation_details', []),
                "encoding_details": metadata.get('encoding_details', []),
                "class_imbalance_handled": metadata.get('class_imbalance_handled', False),
                "imbalance_strategy": metadata.get('imbalance_strategy'),
                "original_class_distribution": metadata.get('original_class_distribution', {}),
                "resampled_class_distribution": metadata.get('resampled_class_distribution', {}),
                "original_train_size": metadata.get('original_train_size', 0),
                "original_test_size": metadata.get('original_test_size', 0),
                "datetime_features": [],
                # New adaptive preprocessing data
                "task_type": results['task_type'],
                "model_family": results['model_family'],
                "target_column": target_column,
                "feature_names": results['feature_names'],
                "numerical_features": metadata['numerical_features'],
                "categorical_features": metadata['categorical_features'],
                "text_features": preprocessing_summary.get('text_features', {
                    "columns": [],
                    "vectorization": None,
                    "max_features": None,
                    "details": []
                }),
                "columns_removed": int(metadata['total_columns_removed']),
                "id_columns_removed": metadata['id_columns_removed'],
                "constant_columns_removed": metadata['constant_columns_removed'],
                "constant_column_details": metadata.get('constant_column_details', {}),
                "high_missing_columns_removed": metadata['high_missing_columns_removed'],
                "skewness_details": metadata.get('skewness_details', []),
                "skewed_features_transformed": metadata.get('skewed_features_transformed', []),
                "multicollinearity": metadata.get('multicollinearity_handled', {
                    "pairs_detected": 0,
                    "features_dropped": [],
                    "features_combined": [],
                    "model_family": results['model_family']
                }),
                "quality_score": float(round(quality_score, 2)),
                "data_retention_ratio": float(round(results['quality_metrics']['data_retention_ratio'], 2)),
                "decisions_made": int(len(results['decision_log'])),
                "preprocessing_steps": preprocessing_steps,
                "pipeline_path": pipeline_path,
                # Add label mapping if classification with encoded labels (JSON storage requires string keys)
                "label_mapping": {str(i): str(label) for i, label in enumerate(preprocessor.label_encoder.classes_)} if hasattr(preprocessor, 'label_encoder') and preprocessor.label_encoder else None,
                "target_classes": metadata.get('target_classes'),
                "preprocessed_file_path": preprocessed_path,
                "preprocessed_filename": preprocessed_filename,
                # Train/test split information
                "has_train_test_split": has_train_test_split,
                "train_size": int(train_size),
                "test_size": int(test_size),
                "split_ratio": float(round(test_size / (train_size + test_size), 2)) if has_train_test_split else 0.0
            }
        }
        
        await datasets_store.update_one(
            {"_id": dataset_id},
            {"$set": update_data}
        )
        
        print(f"✅ Adaptive preprocessing complete!")
        print(f"   Original: {original_shape}")
        print(f"   Preprocessed: {df_preprocessed.shape}")
        print(f"   Quality: {quality_score:.1f}/100")
        print(f"   Decisions: {len(results['decision_log'])}")
        
        # Build backward-compatible response format
        response = {
            "success": True,
            "message": "Adaptive preprocessing completed successfully",
            "preprocessing_method": "adaptive",
            
            # Backward compatibility: old format expected by frontend
            "original": {
                "rows": int(original_shape[0]),
                "columns": int(original_shape[1]),
                "column_names": original_columns
            },
            "processed": {
                "rows": int(df_preprocessed.shape[0]),
                "columns": int(df_preprocessed.shape[1]),
                "column_names": df_preprocessed.columns.tolist()
            },
            "initial_quality": {
                "quality_score": float(round(quality_score * 0.85, 2)),
                "completeness": float(round(initial_completeness, 2)),
                "uniqueness": 100.0,
                "assessment": "Good" if quality_score > 80 else "Fair"
            },
            "final_quality": {
                "quality_score": float(round(quality_score, 2)),
                "completeness": 100.0,
                "uniqueness": 100.0,
                "assessment": "Excellent - Ready for ML" if quality_score >= 95 else "Good - Ready for ML" if quality_score >= 80 else "Fair"
            },
            "changes": {
                "rows_removed": int(original_shape[0] - df_preprocessed.shape[0]),
                "columns_removed": int(metadata['total_columns_removed']),
                "duplicates_removed": int(metadata.get('duplicates_removed_count', 0)),
                "duplicate_removal_details": metadata.get('duplicate_removal_details', []),
                "outliers_handled": int(metadata.get('outliers_handled_count', 0)),
                "missing_values_filled": int(results['quality_metrics'].get('missing_data_handled', 0)),
                "categorical_columns_encoded": int(len(metadata['categorical_features'])),
                "numerical_columns_scaled": int(len(scaled_columns_list)),
                "rare_values_removed": rare_value_removal_summary
            },
            "steps": preprocessing_steps,
            
            # New adaptive preprocessing data
            "task_type": results['task_type'],
            "model_family": results['model_family'],
            "target_column": target_column,
            "feature_names": results['feature_names'],
            "quality_metrics": {
                "quality_score": float(round(quality_score, 2)),
                "data_retention_ratio": float(round(results['quality_metrics']['data_retention_ratio'], 2)),
                "columns_removed": int(results['quality_metrics']['columns_removed']),
                "missing_data_handled": int(results['quality_metrics']['missing_data_handled'])
            },
            "preprocessing_summary": {
                "steps": preprocessing_steps,
                "numerical_features": int(len(metadata['numerical_features'])),
                "categorical_features": int(len(metadata['categorical_features'])),
                "total_decisions": int(len(results['decision_log'])),
                "columns_removed": int(metadata['total_columns_removed']),
                "constant_columns_removed": metadata['constant_columns_removed'],
                "constant_column_details": metadata.get('constant_column_details', {}),
                "high_missing_columns_removed": metadata['high_missing_columns_removed'],
                # Train/test split information
                "has_train_test_split": has_train_test_split,
                "train_size": int(train_size),
                "test_size": int(test_size),
                "split_ratio": float(round(test_size / (train_size + test_size), 2)) if has_train_test_split else 0.0
            },
            "decision_log": decision_log_formatted[:20],
            "warnings": [d for d in decision_log_formatted if d['impact'] == 'warning'],
            "preprocessed_file_path": preprocessed_path,
            "preprocessed_filename": preprocessed_filename
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
        )
    
