"""
Pydantic models for data validation and schema definition
Defines the structure for Datasets, Models, and Experiments
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DatasetBase(BaseModel):
    """Base dataset model"""
    name: str
    description: Optional[str] = None
    filename: str
    file_path: str
    file_size: Optional[int] = None


class DatasetCreate(DatasetBase):
    """Dataset creation model"""
    pass


class DatasetInDB(DatasetBase):
    """Dataset model as stored in database"""
    id: str = Field(alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]
    missing_values: Dict[str, Any]  # Changed from Dict[str, int] to Dict[str, Any]
    summary_statistics: Optional[Dict[str, Any]] = None
    eda_results: Optional[Dict[str, Any]] = None
    model_config = {"populate_by_name": True}


class DatasetResponse(BaseModel):
    """Dataset model for API responses"""
    id: str
    name: str
    description: Optional[str] = None
    filename: str
    file_path: Optional[str] = None  # Optional - may be deleted if original removed
    file_size: Optional[int] = None
    created_at: datetime
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]
    missing_values: Dict[str, Any]  # Changed from Dict[str, int] to Dict[str, Any]
    summary_statistics: Optional[Dict[str, Any]] = None
    eda_results: Optional[Dict[str, Any]] = None
    selected_columns: Optional[List[str]] = None  # Selected columns for analysis
    preprocessed: Optional[bool] = None
    preprocessed_at: Optional[datetime] = None
    preprocessing_summary: Optional[Dict[str, Any]] = None
    preprocessed_file_path: Optional[str] = None
    original_deleted: Optional[bool] = None  # Indicates if original was deleted
    original_deleted_at: Optional[datetime] = None  # When original was deleted


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration"""
    drop_columns: List[str] = []
    numerical_strategy: str = "mean"  # mean, median
    categorical_strategy: str = "mode"  # mode, constant
    scaling_method: str = "standard"  # standard, minmax
    encoding_method: str = "onehot"  # onehot, ordinal
    

class AdaptivePreprocessingConfig(BaseModel):
    """Adaptive preprocessing configuration with intelligent defaults"""
    use_adaptive: bool = True  # Use adaptive preprocessing
    use_eda_insights: bool = True  # Leverage EDA results for decisions
    force_drop_columns: List[str] = []  # Force drop these columns
    force_keep_columns: List[str] = []  # Force keep these columns
    imputation_overrides: Optional[Dict[str, str]] = None  # Column-specific imputation
    encoding_overrides: Optional[Dict[str, str]] = None  # Column-specific encoding
    disable_id_detection: bool = False  # Disable automatic ID column detection
    missing_threshold_override: Optional[float] = None  # Override missing threshold
    test_size: float = 0.2  # Train/test split ratio
    random_state: int = 42  # Random seed for reproducibility


class ModelType(str, Enum):
    """Supported model types"""
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


class ModelBase(BaseModel):
    """Base model definition"""
    model_config = {"protected_namespaces": ()}
    
    name: str
    model_type: ModelType
    description: Optional[str] = None


class ModelCreate(ModelBase):
    """Model creation request"""
    dataset_id: str
    target_column: str
    preprocessing_config: Optional[PreprocessingConfig] = None
    adaptive_preprocessing_config: Optional[AdaptivePreprocessingConfig] = None


class ModelInDB(ModelBase):
    """Model as stored in database"""
    model_config = {"protected_namespaces": (), "populate_by_name": True}
    
    id: str = Field(alias="_id")
    dataset_id: str
    target_column: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model_path: str
    preprocessing_path: str
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    
    # Feature information
    feature_names: List[str]
    feature_importance: Optional[Dict[str, float]] = None


class ModelResponse(ModelBase):
    """Model response for API"""
    id: str
    dataset_id: str
    target_column: str
    created_at: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    feature_names: List[str]
    feature_importance: Optional[Dict[str, float]] = None


class ExperimentBase(BaseModel):
    """Base experiment model"""
    name: str
    description: Optional[str] = None


class ExperimentCreate(ExperimentBase):
    """Experiment creation request"""
    dataset_id: str
    models: List[ModelCreate]


class ExperimentInDB(ExperimentBase):
    """Experiment as stored in database"""
    model_config = {"protected_namespaces": (), "populate_by_name": True}
    
    id: str = Field(alias="_id")
    dataset_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, running, completed, failed
    model_ids: List[str] = []
    preprocessing_steps: List[str] = []
    duration_seconds: Optional[float] = None


class ExperimentResponse(ExperimentBase):
    """Experiment response for API"""
    model_config = {"protected_namespaces": ()}
    
    id: str
    dataset_id: str
    created_at: datetime
    status: str
    model_ids: List[str]
    preprocessing_steps: List[str]
    duration_seconds: Optional[float] = None


# ==================== AUTOML SCHEMAS ====================

class AutoMLTaskType(str, Enum):
    """AutoML task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class AutoMLConfig(BaseModel):
    """AutoML training configuration"""
    n_trials: int = Field(default=75, ge=10, le=200, description="Number of Optuna trials per model")
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of cross-validation folds")
    test_size: float = Field(default=0.2, ge=0.1, le=0.4, description="Test set size")
    penalty_factor: float = Field(default=2.0, ge=1.0, le=5.0, description="Overfitting penalty factor")
    overfit_threshold_reject: float = Field(default=0.20, description="Threshold for model rejection")
    overfit_threshold_high: float = Field(default=0.10, description="Threshold for high penalty")
    models_to_train: Optional[List[str]] = Field(default=None, description="Specific models to train, or None for all")
    max_cpu_cores: int = Field(default=4, ge=1, le=32, description="Maximum CPU cores to use for training")


class AutoMLTrainRequest(BaseModel):
    """AutoML training request"""
    dataset_id: str
    target_column: str
    task_type: AutoMLTaskType
    name: str
    description: Optional[str] = None
    config: AutoMLConfig = Field(default_factory=AutoMLConfig)
    preprocessing_config: Optional[AdaptivePreprocessingConfig] = Field(default_factory=AdaptivePreprocessingConfig)


class ModelResultSchema(BaseModel):
    """Individual model result"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    model_type: str
    train_score: float
    cv_score: float
    cv_std: float
    test_score: float
    overfit_gap: float
    penalty: float
    generalization_score: float
    overfitting: bool
    rejected: bool
    rejection_reason: Optional[str] = None
    detailed_metrics: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    best_params: Dict[str, Any]
    n_trials: int
    optimization_time: float


class AutoMLResponse(BaseModel):
    """AutoML training response"""
    model_config = {"protected_namespaces": ()}
    
    id: str
    dataset_id: str
    name: str
    description: Optional[str] = None
    task_type: AutoMLTaskType
    training_id: Optional[str] = None  # For cancellation support
    
    # Best model info
    best_model_name: str
    best_model_type: str
    best_generalization_score: float
    best_cv_score: float
    best_test_score: float
    best_overfit_gap: float
    best_params: Dict[str, Any]
    
    # All models
    all_models: List[ModelResultSchema]
    total_models_evaluated: int
    models_rejected: int
    
    # Plots
    plot_paths: Dict[str, str]
    
    # Metadata
    feature_names: List[str]
    target_column: str
    preprocessing_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    training_duration: float
    
    # Paths
    model_path: str
    preprocessing_path: Optional[str] = None
    
    # Configuration
    config: Optional[Dict[str, Any]] = None
