"""
AutoML Routes - Generalization-focused model training with Optuna
=================================================================

This endpoint provides AutoML training with:
- Optuna hyperparameter optimization
- Overfitting detection and penalty system
- Model selection based on generalization score
- Comprehensive evaluation plots
- Real-time training orchestration via TrainingManager
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Optional
import pandas as pd
import numpy as np
import os
import uuid
from datetime import datetime
import traceback
import asyncio

from app.models.schemas import (
    AutoMLTrainRequest,
    AutoMLResponse,
    ModelResultSchema,
    UnsupervisedModelResultSchema,
    AutoMLTaskType,
    UnsupervisedSubtype,
)
from app.storage import get_store, generate_id
from app.services.automl_engine import AutoMLEngine, TaskType, ModelResult
from app.services.unsupervised_engine import (
    UnsupervisedEngine, UnsupervisedTaskType, UnsupervisedModelResult, UnsupervisedResult,
)
from app.services.automl_plots import AutoMLPlotter
from app.services.adaptive_preprocessing import AdaptivePreprocessor
from app.services.training_manager import TrainingManager, TrainingPhase
from app.config import settings

# Legacy global cancellation tracker (kept for backward compatibility)
active_trainings = {}
force_cancel_flag = {}

# Singleton training manager
training_manager = TrainingManager.get_instance()

router = APIRouter(prefix="/api/automl", tags=["AutoML"])


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization.
    Also sanitizes float nan/inf values which are not JSON-compliant."""
    import numpy as np
    import math

    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def convert_model_result_to_schema(result: ModelResult) -> ModelResultSchema:
    """Convert ModelResult dataclass to Pydantic schema"""
    return ModelResultSchema(
        model_name=result.model_name,
        model_type=result.model_type,
        train_score=result.train_score,
        cv_score=result.cv_score,
        cv_std=result.cv_std,
        test_score=result.test_score,
        overfit_gap=result.overfit_gap,
        penalty=result.penalty,
        generalization_score=result.generalization_score,
        overfitting=result.overfitting,
        rejected=result.rejected,
        rejection_reason=result.rejection_reason,
        detailed_metrics=result.detailed_metrics,
        feature_importance=result.feature_importance,
        best_params=result.best_params,
        n_trials=result.n_trials,
        optimization_time=result.optimization_time
    )


@router.post("/train/start", status_code=status.HTTP_200_OK)
async def start_training(
    request: AutoMLTrainRequest,
):
    """
    Initialize training and start it in a background thread.
    Returns training ID immediately for status polling.
    
    The frontend should poll /training-status to track progress,
    and when status shows phase=completed, fetch the model via /models/{model_id}.
    """
    global training_manager

    # Check if training is already running
    if training_manager._is_running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Training session '{training_manager._session_id}' is already running. Cancel it first."
        )

    # Create unique training ID
    training_id = generate_id()
    
    # Determine which models will be trained
    task_type_str = request.task_type.value if request.task_type else "classification"
    is_unsupervised = task_type_str in ("unsupervised", "clustering")
    models_to_train = request.config.models_to_train

    if not models_to_train:
        if is_unsupervised:
            # Determine unsupervised sub-task model list
            subtype = request.unsupervised_subtype
            if subtype and subtype.value == "dimensionality_reduction":
                models_to_train = list(UnsupervisedEngine.DIMENSIONALITY_REDUCTION_MODELS.keys())
            elif subtype and subtype.value == "anomaly_detection":
                models_to_train = list(UnsupervisedEngine.ANOMALY_DETECTION_MODELS.keys())
            else:
                # Default: clustering
                models_to_train = list(UnsupervisedEngine.CLUSTERING_MODELS.keys())
                # Add optional models
                try:
                    from hdbscan import HDBSCAN
                    models_to_train.append('hdbscan')
                except ImportError:
                    pass
        elif task_type_str == "classification":
            models_to_train = list(AutoMLEngine.CLASSIFICATION_MODELS.keys())
        elif task_type_str == "regression":
            models_to_train = list(AutoMLEngine.REGRESSION_MODELS.keys())
        else:
            models_to_train = list(AutoMLEngine.CLUSTERING_MODELS.keys())
    
    # Initialize TrainingManager session
    training_manager.start_session(
        session_id=training_id,
        model_names=models_to_train,
        n_trials=request.config.n_trials,
    )
    
    # Also set legacy tracking for backward compatibility
    active_trainings[training_id] = {"cancelled": False, "request": request}
    force_cancel_flag[training_id] = training_manager.force_cancel_flag

    print(f"\n🚀 Training initialized with ID: {training_id}")
    print(f"   Models: {models_to_train}")
    print(f"   Trials per model: {request.config.n_trials}")
    
    # Start background training thread
    training_manager.run_in_background(
        target=_run_training_background,
        args=(training_id, request),
    )

    return {"training_id": training_id, "models": models_to_train}


def _run_training_background(training_id: str, request: AutoMLTrainRequest):
    """
    Background training function - runs in a separate thread.
    Uses TrainingManager for state reporting and cooperative cancellation.
    
    This is a synchronous function (runs in threading.Thread) that performs:
    1. Dataset loading
    2. Preprocessing (if needed)
    3. AutoML training with Optuna
    4. Plot generation
    5. Model saving
    6. Database persistence
    """
    import asyncio

    global training_manager

    # Create a new event loop for this thread (needed for async storage calls)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _run_training_async(training_id, request)
        )
    except Exception as e:
        print(f"❌ Background training failed: {e}")
        traceback.print_exc()
        training_manager.add_log(f"Fatal error: {str(e)}")
        training_manager.finish_session(TrainingPhase.FAILED)
    finally:
        # Clean up legacy tracking
        if training_id in active_trainings:
            del active_trainings[training_id]
        if training_id in force_cancel_flag:
            del force_cancel_flag[training_id]
        loop.close()


async def _run_training_async(training_id: str, request: AutoMLTrainRequest):
    """Async inner function for background training - has access to async storage."""
    global training_manager

    datasets_store = get_store("datasets")
    automl_store = get_store("automl_models")

    # === STEP 1: Load Dataset ===
    training_manager.set_phase(TrainingPhase.PREPROCESSING)

    try:
        dataset = await datasets_store.find_one({"_id": request.dataset_id})
    except Exception:
        training_manager.add_log(f"Invalid dataset ID: {request.dataset_id}")
        training_manager.finish_session(TrainingPhase.FAILED)
        return

    if not dataset:
        training_manager.add_log(f"Dataset not found: {request.dataset_id}")
        training_manager.finish_session(TrainingPhase.FAILED)
        return

    start_time = datetime.now()

    try:
        # ===== LOAD & PREPROCESS (same logic as original /train endpoint) =====
        preprocessing_summary = dataset.get("preprocessing_summary", {})
        is_preprocessed = dataset.get("preprocessed", False) and dataset.get("preprocessed_file_path")
        label_encoder = None
        label_mapping = None
        preprocessing_pipeline = None
        preprocessing_metadata = {}
        preprocessing_method = "basic"

        if is_preprocessed and preprocessing_summary:
            training_manager.add_log("Loading preprocessed dataset")
            df = pd.read_csv(dataset["preprocessed_file_path"], sep=None, engine='python', skipinitialspace=True)

            has_train_test_split = preprocessing_summary.get("has_train_test_split", False)
            train_size = preprocessing_summary.get("train_size", 0)
            test_size = preprocessing_summary.get("test_size", 0)
            target_column = preprocessing_summary.get("target_column")

            if has_train_test_split and train_size > 0:
                df_train = df.iloc[:train_size].copy()
                df_test = df.iloc[train_size:train_size + test_size].copy()

                if target_column and target_column in df.columns:
                    X_train = df_train.drop(columns=[target_column])
                    y_train = df_train[target_column]
                    X_test = df_test.drop(columns=[target_column])
                    y_test = df_test[target_column]

                    if request.task_type == AutoMLTaskType.CLASSIFICATION and y_train.dtype == 'object':
                        from sklearn.preprocessing import LabelEncoder
                        label_encoder = LabelEncoder()
                        y_train_encoded = label_encoder.fit_transform(y_train)
                        y_test_encoded = label_encoder.transform(y_test)
                        label_mapping = {str(i): str(label) for i, label in enumerate(label_encoder.classes_)}
                        X_transformed = X_train.values
                        y = y_train_encoded
                        y_test = y_test_encoded
                    else:
                        X_transformed = X_train.values
                        y = y_train.values
                        y_test = y_test.values

                    feature_names = X_train.columns.tolist()
                else:
                    X_transformed = df_train.values
                    y = None
                    feature_names = df_train.columns.tolist()
                    X_test, y_test = None, None

                pipeline_path = preprocessing_summary.get("pipeline_path")
                if pipeline_path and os.path.exists(pipeline_path):
                    try:
                        import joblib
                        pipeline_data = joblib.load(pipeline_path)
                        preprocessing_pipeline = pipeline_data.get('pipeline') if isinstance(pipeline_data, dict) else pipeline_data
                    except Exception as e:
                        print(f"⚠️ Failed to load preprocessing pipeline: {e}")

                preprocessing_method = "already_preprocessed"
                preprocessing_metadata = {
                    "numerical_features": preprocessing_summary.get("numerical_features", []),
                    "categorical_features": preprocessing_summary.get("categorical_features", []),
                    "text_features": preprocessing_summary.get("text_features", {}).get("columns", []) if isinstance(preprocessing_summary.get("text_features"), dict) else []
                }
            else:
                if target_column and target_column in df.columns:
                    y_values = df[target_column]
                    if request.task_type == AutoMLTaskType.CLASSIFICATION and y_values.dtype == 'object':
                        from sklearn.preprocessing import LabelEncoder
                        label_encoder = LabelEncoder()
                        y_encoded = label_encoder.fit_transform(y_values)
                        label_mapping = {str(i): str(label) for i, label in enumerate(label_encoder.classes_)}
                        y = y_encoded
                    else:
                        y = y_values.values
                    X_transformed = df.drop(columns=[target_column]).values
                    feature_names = df.drop(columns=[target_column]).columns.tolist()
                else:
                    X_transformed = df.values
                    y = None
                    feature_names = df.columns.tolist()

                X_test, y_test = None, None
                pipeline_path = preprocessing_summary.get("pipeline_path")
                if pipeline_path and os.path.exists(pipeline_path):
                    try:
                        import joblib
                        pipeline_data = joblib.load(pipeline_path)
                        preprocessing_pipeline = pipeline_data.get('pipeline') if isinstance(pipeline_data, dict) else pipeline_data
                    except Exception as e:
                        print(f"⚠️ Failed to load preprocessing pipeline: {e}")
                preprocessing_method = "already_preprocessed"
                preprocessing_metadata = {
                    "numerical_features": preprocessing_summary.get("numerical_features", []),
                    "categorical_features": preprocessing_summary.get("categorical_features", []),
                    "text_features": preprocessing_summary.get("text_features", {}).get("columns", []) if isinstance(preprocessing_summary.get("text_features"), dict) else []
                }
        else:
            # Raw dataset
            training_manager.add_log("Loading raw dataset")
            df = pd.read_csv(dataset["file_path"], sep=None, engine='python', skipinitialspace=True)

            is_unsupervised = request.task_type in (AutoMLTaskType.UNSUPERVISED, AutoMLTaskType.CLUSTERING) and not request.target_column
            target_column = request.target_column if request.target_column else None

            if not is_unsupervised:
                if not target_column or target_column not in df.columns:
                    training_manager.add_log(f"Target column '{target_column}' not found")
                    training_manager.finish_session(TrainingPhase.FAILED)
                    return
            X_test, y_test = None, None

            # Preprocessing
            if is_unsupervised:
                # ─── Geometry-aware unsupervised preprocessing ───
                # Uses UnsupervisedPreprocessor which:
                #   1. Removes ID/constant columns
                #   2. Handles skewness with PowerTransformer
                #   3. Scales numeric with RobustScaler (not StandardScaler)
                #   4. OHE categorical WITHOUT scaling (preserves binary geometry)
                #   5. VarianceThreshold to remove near-zero-variance features
                #   6. Auto-PCA when dimensionality is too high
                training_manager.add_log("Unsupervised preprocessing (geometry-aware pipeline)")
                from ..services.unsupervised_preprocessing import UnsupervisedPreprocessor

                unsup_preprocessor = UnsupervisedPreprocessor(
                    max_onehot_cardinality=15,
                    variance_threshold=0.01,
                    pca_variance_threshold=0.95,
                    auto_pca=True,
                    auto_pca_trigger_features=50,
                    handle_skewness=True,
                    skewness_threshold=1.0,
                    id_column_heuristics=True,
                    log_callback=lambda msg: training_manager.add_log(f"[Preprocess] {msg}"),
                )

                X_transformed, feature_names, preprocessing_metadata = unsup_preprocessor.fit_transform(df)
                preprocessing_pipeline = unsup_preprocessor.get_pipeline_components()

                y = None
                preprocessing_method = "unsupervised_geometry_aware"

            elif request.preprocessing_config and request.preprocessing_config.use_adaptive:
                preprocessor = AdaptivePreprocessor(
                    eda_results=dataset.get("eda_results"),
                    outlier_preferences={}
                )
                preprocessing_results = preprocessor.fit_transform(
                    df=df, target_column=request.target_column,
                    model_type=None, test_size=0, random_state=42
                )
                X_transformed = preprocessing_results['X_train']
                y = preprocessing_results['y_train']
                feature_names = preprocessing_results['feature_names']
                preprocessing_pipeline = preprocessor.pipeline
                preprocessing_metadata = preprocessing_results.get('preprocessing_metadata', {})
                preprocessing_method = "adaptive"
            else:
                X = df.drop(columns=[request.target_column])
                y = df[request.target_column]
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                from sklearn.compose import ColumnTransformer
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                if request.task_type in [AutoMLTaskType.CLASSIFICATION, AutoMLTaskType.CLUSTERING]:
                    if y.dtype == 'object':
                        label_encoder = LabelEncoder()
                        y_encoded = label_encoder.fit_transform(y)
                        label_mapping = {str(i): str(label) for i, label in enumerate(label_encoder.classes_)}
                        y = y_encoded
                from sklearn.preprocessing import OneHotEncoder
                transformers = []
                if numeric_features:
                    transformers.append(('num', StandardScaler(), numeric_features))
                if categorical_features:
                    transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features))
                if transformers:
                    preprocessor_obj = ColumnTransformer(transformers=transformers)
                    X_transformed = preprocessor_obj.fit_transform(X)
                else:
                    X_transformed = X.values
                feature_names = []
                if numeric_features:
                    feature_names.extend(numeric_features)
                if categorical_features and transformers:
                    cat_encoder = transformers[1][1] if len(transformers) > 1 else transformers[0][1]
                    if hasattr(cat_encoder, 'get_feature_names_out'):
                        feature_names.extend(cat_encoder.get_feature_names_out(categorical_features).tolist())
                elif categorical_features:
                    feature_names.extend(categorical_features)
                y = y.values if hasattr(y, 'values') else y
                preprocessing_pipeline = None
                preprocessing_metadata = {}
                preprocessing_method = "basic"
                X_test, y_test = None, None

        training_manager.add_log(f"Preprocessed: {len(feature_names)} features, {len(X_transformed)} samples")

        # Check cancellation after preprocessing
        if training_manager.is_cancelled():
            training_manager.add_log("Cancelled during preprocessing")
            training_manager.finish_session(TrainingPhase.CANCELLED)
            return

        # ===== STEP 2: AUTOML TRAINING =====
        training_manager.set_phase(TrainingPhase.TRAINING)

        is_unsupervised_task = (
            request.task_type == AutoMLTaskType.UNSUPERVISED or
            (request.task_type == AutoMLTaskType.CLUSTERING and y is None) or
            (y is None and not request.target_column)
        )

        if is_unsupervised_task:
            # ──── UNSUPERVISED TRAINING PATH ────
            await _run_unsupervised_training(
                training_manager, request, X_transformed, feature_names,
                preprocessing_pipeline, preprocessing_metadata, preprocessing_method,
                start_time, is_preprocessed, datasets_store, automl_store,
            )
            return

        # ──── SUPERVISED TRAINING PATH (existing) ────
        if request.task_type == AutoMLTaskType.CLASSIFICATION:
            task_type = TaskType.CLASSIFICATION
        elif request.task_type == AutoMLTaskType.REGRESSION:
            task_type = TaskType.REGRESSION
        else:
            task_type = TaskType.CLUSTERING

        use_presplit_data = is_preprocessed and X_test is not None

        engine = AutoMLEngine(
            task_type=task_type,
            n_trials=request.config.n_trials,
            cv_folds=request.config.cv_folds,
            test_size=request.config.test_size,
            penalty_factor=request.config.penalty_factor,
            overfit_threshold_reject=request.config.overfit_threshold_reject,
            overfit_threshold_high=request.config.overfit_threshold_high,
            max_cpu_cores=getattr(request.config, 'max_cpu_cores', 4),
            verbose=True
        )

        # Callbacks for TrainingManager integration
        def on_model_start(model_name):
            training_manager.set_current_model(model_name)
            training_manager.clear_skip()  # Reset skip flag for new model

        def on_model_complete(model_name, score, model_status):
            training_manager.complete_model(model_name, score=score, status=model_status)

        def trial_progress_cb(trial_num, total_trials, best_score):
            training_manager.update_trial(trial_num, total_trials, best_score)

        # Build fit kwargs
        fit_kwargs = dict(
            X=X_transformed,
            y=y,
            feature_names=feature_names,
            model_subset=request.config.models_to_train,
            cancellation_callback=training_manager.cancellation_callback,
            force_cancel_flag=training_manager.force_cancel_flag,
            skip_callback=training_manager.is_skip_requested,
            on_model_start=on_model_start,
            on_model_complete=on_model_complete,
            trial_progress_callback=trial_progress_cb,
        )

        if use_presplit_data:
            X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
            y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
            fit_kwargs['X_test'] = X_test_array
            fit_kwargs['y_test'] = y_test_array

        automl_result = engine.fit(**fit_kwargs)

        training_duration = (datetime.now() - start_time).total_seconds()

        # Check if all models were cancelled
        if all(r.stability_status == "cancelled" for r in automl_result.all_models):
            training_manager.add_log("All models cancelled")
            training_manager.finish_session(TrainingPhase.CANCELLED)
            return

        # ===== STEP 3: GENERATE PLOTS =====
        training_manager.set_phase(TrainingPhase.PLOTTING)
        training_manager.add_log("Generating evaluation plots...")

        plots_dir = os.path.join(settings.models_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plotter = AutoMLPlotter(output_dir=plots_dir)

        if use_presplit_data:
            X_train_plot = X_transformed
            y_train_plot = y
            X_test_plot = X_test.values if hasattr(X_test, 'values') else X_test
            y_test_plot = y_test.values if hasattr(y_test, 'values') else y_test
        else:
            from sklearn.model_selection import train_test_split
            X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(
                X_transformed, y,
                test_size=request.config.test_size,
                random_state=42,
                stratify=y if task_type == TaskType.CLASSIFICATION else None
            )

        raw_plot_paths = plotter.generate_all_plots(
            automl_result=automl_result,
            X=X_train_plot, y=y_train_plot,
            X_test=X_test_plot, y_test=y_test_plot
        )

        # Store only filenames (not full paths) so the /plots/ endpoint works correctly
        plot_paths = {
            name: os.path.basename(path)
            for name, path in raw_plot_paths.items()
        }

        # ===== STEP 4: SAVE MODEL =====
        training_manager.set_phase(TrainingPhase.SAVING)
        training_manager.add_log("Saving model to disk...")

        model_id = generate_id()
        model_filename = f"{model_id}_automl_model.joblib"
        model_path = os.path.join(settings.models_dir, model_filename)
        os.makedirs(settings.models_dir, exist_ok=True)

        engine.save(
            model_path,
            label_encoder=label_encoder,
            label_mapping=label_mapping,
            preprocessing_pipeline=preprocessing_pipeline
        )

        preprocessing_path = None
        if preprocessing_pipeline:
            preprocessing_filename = f"{model_id}_preprocessing.joblib"
            preprocessing_path = os.path.join(settings.models_dir, preprocessing_filename)
            import joblib
            joblib.dump(preprocessing_pipeline, preprocessing_path)

        # ===== STEP 5: SAVE TO DATABASE =====
        best_model = automl_result.best_model
        automl_doc = {
            "_id": model_id,
            "dataset_id": request.dataset_id,
            "name": request.name,
            "description": request.description,
            "task_type": request.task_type.value,
            "best_model_name": best_model.model_name,
            "best_model_type": best_model.model_type,
            "best_generalization_score": float(best_model.generalization_score),
            "best_cv_score": float(best_model.cv_score),
            "best_test_score": float(best_model.test_score),
            "best_overfit_gap": float(best_model.overfit_gap),
            "best_params": convert_numpy_types(best_model.best_params),
            "best_detailed_metrics": convert_numpy_types(best_model.detailed_metrics),
            "best_feature_importance": convert_numpy_types(best_model.feature_importance),
            "all_models": [
                convert_numpy_types({
                    "model_name": r.model_name,
                    "model_type": r.model_type,
                    "train_score": r.train_score,
                    "cv_score": r.cv_score,
                    "cv_std": r.cv_std,
                    "test_score": r.test_score,
                    "overfit_gap": r.overfit_gap,
                    "penalty": r.penalty,
                    "generalization_score": r.generalization_score,
                    "overfitting": r.overfitting,
                    "rejected": r.rejected,
                    "rejection_reason": r.rejection_reason,
                    "detailed_metrics": r.detailed_metrics,
                    "feature_importance": r.feature_importance,
                    "best_params": r.best_params,
                    "n_trials": r.n_trials,
                    "optimization_time": r.optimization_time
                }) for r in automl_result.all_models
            ],
            "total_models_evaluated": automl_result.total_models_evaluated,
            "models_rejected": automl_result.models_rejected,
            "plot_paths": plot_paths,
            "feature_names": feature_names,
            "target_column": request.target_column,
            "label_mapping": label_mapping,
            "preprocessing_metadata": preprocessing_metadata,
            "created_at": datetime.utcnow(),
            "training_duration": training_duration,
            "model_path": model_path,
            "preprocessing_path": preprocessing_path,
            "config": request.config.dict(),
            "preprocessing_config": request.preprocessing_config.dict() if request.preprocessing_config else None
        }

        await automl_store.insert_one(automl_doc)

        # Mark session as completed, store result model_id for frontend retrieval
        training_manager.add_log(f"Training complete! Model ID: {model_id}")
        training_manager.add_log(f"Best model: {best_model.model_name} (gen_score: {best_model.generalization_score:.4f})")

        # Store the result model_id in manager so frontend can retrieve it
        with training_manager._state_lock:
            training_manager._result_model_id = model_id
            training_manager._training_duration = training_duration

        training_manager.finish_session(TrainingPhase.COMPLETED)
        print(f"✅ Background training completed! Model ID: {model_id}")

    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        traceback.print_exc()
        training_manager.add_log(f"Error: {str(e)}")
        training_manager.finish_session(TrainingPhase.FAILED)


# ─── Unsupervised Training Path ──────────────────────────────────────

async def _run_unsupervised_training(
    training_manager,
    request: AutoMLTrainRequest,
    X_transformed: np.ndarray,
    feature_names: List,
    preprocessing_pipeline,
    preprocessing_metadata: dict,
    preprocessing_method: str,
    start_time: datetime,
    is_preprocessed: bool,
    datasets_store,
    automl_store,
):
    """
    Unsupervised training path - handles clustering, dimensionality reduction, and anomaly detection.
    Called from _run_training_async when task is unsupervised with no target column.
    """
    import joblib

    # Determine sub-task type
    subtype_val = request.unsupervised_subtype.value if request.unsupervised_subtype else "clustering"
    if subtype_val == "dimensionality_reduction":
        task_subtype = UnsupervisedTaskType.DIMENSIONALITY_REDUCTION
    elif subtype_val == "anomaly_detection":
        task_subtype = UnsupervisedTaskType.ANOMALY_DETECTION
    else:
        task_subtype = UnsupervisedTaskType.CLUSTERING

    training_manager.add_log(f"Unsupervised mode: {task_subtype.value}")

    engine = UnsupervisedEngine(
        task_subtype=task_subtype,
        n_trials=request.config.n_trials,
        random_state=42,
        max_cpu_cores=getattr(request.config, 'max_cpu_cores', 4),
        verbose=True,
    )

    # Callbacks
    def on_model_start(model_name):
        training_manager.set_current_model(model_name)
        training_manager.clear_skip()

    def on_model_complete(model_name, score, model_status):
        training_manager.complete_model(model_name, score=score, status=model_status)

    def trial_progress_cb(trial_num, total_trials, best_score):
        training_manager.update_trial(trial_num, total_trials, best_score)

    unsupervised_result = engine.fit(
        X=X_transformed,
        feature_names=feature_names,
        model_subset=request.config.models_to_train,
        cancellation_callback=training_manager.cancellation_callback,
        force_cancel_flag=training_manager.force_cancel_flag,
        skip_callback=training_manager.is_skip_requested,
        on_model_start=on_model_start,
        on_model_complete=on_model_complete,
        trial_progress_callback=trial_progress_cb,
    )

    training_duration = (datetime.now() - start_time).total_seconds()

    # Check if all cancelled
    if all(r.stability_status == "cancelled" for r in unsupervised_result.all_models):
        training_manager.add_log("All models cancelled")
        training_manager.finish_session(TrainingPhase.CANCELLED)
        return

    # ===== PLOTS =====
    training_manager.set_phase(TrainingPhase.PLOTTING)
    training_manager.add_log("Generating unsupervised evaluation plots...")

    plots_dir = os.path.join(settings.models_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plotter = AutoMLPlotter(output_dir=plots_dir)

    raw_plot_paths = plotter.generate_unsupervised_plots(
        unsupervised_result=unsupervised_result,
        X=X_transformed,
        feature_names=feature_names,
    )

    plot_paths = {
        name: os.path.basename(path)
        for name, path in raw_plot_paths.items()
    }

    # ===== SAVE MODEL =====
    training_manager.set_phase(TrainingPhase.SAVING)
    training_manager.add_log("Saving unsupervised model to disk...")

    model_id = generate_id()
    model_filename = f"{model_id}_automl_model.joblib"
    model_path = os.path.join(settings.models_dir, model_filename)
    os.makedirs(settings.models_dir, exist_ok=True)

    engine.save(model_path, preprocessing_pipeline=preprocessing_pipeline)

    preprocessing_path = None
    if preprocessing_pipeline:
        preprocessing_filename = f"{model_id}_preprocessing.joblib"
        preprocessing_path = os.path.join(settings.models_dir, preprocessing_filename)
        joblib.dump(preprocessing_pipeline, preprocessing_path)

    # ===== EXPORT ARTIFACTS =====
    best = unsupervised_result.best_model

    # Save cluster labels CSV
    cluster_labels_list = None
    anomaly_labels_list = None
    explained_variance_data = None

    if best.labels is not None:
        cluster_labels_list = best.labels.tolist()
        labels_path = os.path.join(settings.models_dir, f"{model_id}_cluster_labels.csv")
        pd.DataFrame({'cluster_label': best.labels}).to_csv(labels_path, index=False)
        training_manager.add_log(f"Saved cluster labels to {os.path.basename(labels_path)}")

    if best.anomaly_labels is not None:
        anomaly_labels_list = best.anomaly_labels.tolist()
        anomaly_path = os.path.join(settings.models_dir, f"{model_id}_anomaly_labels.csv")
        pd.DataFrame({
            'anomaly_label': best.anomaly_labels,
            'anomaly_score': best.anomaly_scores if best.anomaly_scores is not None else 0,
        }).to_csv(anomaly_path, index=False)

    if best.explained_variance_ratio is not None:
        explained_variance_data = {
            'explained_variance_ratio': best.explained_variance_ratio.tolist(),
            'cumulative': np.cumsum(best.explained_variance_ratio).tolist(),
            'total_explained': float(np.sum(best.explained_variance_ratio)),
        }

    # Save evaluation metrics JSON
    metrics_path = os.path.join(settings.models_dir, f"{model_id}_evaluation_metrics.json")
    import json
    with open(metrics_path, 'w') as f:
        json.dump(convert_numpy_types(best.detailed_metrics), f, indent=2)

    # ===== SAVE TO DATABASE =====
    automl_doc = {
        "_id": model_id,
        "dataset_id": request.dataset_id,
        "name": request.name,
        "description": request.description,
        "task_type": "unsupervised",
        "unsupervised_subtype": task_subtype.value,
        "best_model_name": best.model_name,
        "best_model_type": best.model_type,
        "best_generalization_score": float(best.primary_score),
        "best_primary_score": float(best.primary_score),
        "best_cv_score": float(best.primary_score),  # No CV for unsupervised
        "best_test_score": float(best.primary_score),
        "best_overfit_gap": 0.0,
        "best_params": convert_numpy_types(best.best_params),
        "best_detailed_metrics": convert_numpy_types(best.detailed_metrics),
        "best_feature_importance": None,
        "all_models": [
            convert_numpy_types({
                "model_name": r.model_name,
                "model_type": r.model_type,
                "task_subtype": r.task_subtype.value if hasattr(r.task_subtype, 'value') else str(r.task_subtype),
                "primary_score": r.primary_score,
                "detailed_metrics": r.detailed_metrics,
                "best_params": r.best_params,
                "n_trials": r.n_trials,
                "optimization_time": r.optimization_time,
                "stability_status": r.stability_status,
                "rejected": r.rejected,
                "rejection_reason": r.rejection_reason,
            }) for r in unsupervised_result.all_models
        ],
        "total_models_evaluated": unsupervised_result.total_models_evaluated,
        "models_rejected": unsupervised_result.models_rejected,
        "plot_paths": plot_paths,
        "feature_names": feature_names,
        "target_column": None,
        "label_mapping": None,
        "preprocessing_metadata": preprocessing_metadata,
        "created_at": datetime.utcnow(),
        "training_duration": training_duration,
        "model_path": model_path,
        "preprocessing_path": preprocessing_path,
        "config": request.config.dict(),
        "preprocessing_config": request.preprocessing_config.dict() if request.preprocessing_config else None,
        "cluster_labels": cluster_labels_list,
        "anomaly_labels": anomaly_labels_list,
        "explained_variance": explained_variance_data,
        "n_samples": unsupervised_result.n_samples,
        "n_features": unsupervised_result.n_features,
    }

    await automl_store.insert_one(automl_doc)

    training_manager.add_log(f"Training complete! Model ID: {model_id}")
    training_manager.add_log(f"Best model: {best.model_name} (score: {best.primary_score:.4f})")

    with training_manager._state_lock:
        training_manager._result_model_id = model_id
        training_manager._training_duration = training_duration

    training_manager.finish_session(TrainingPhase.COMPLETED)
    print(f"✅ Unsupervised training completed! Model ID: {model_id}")


# ─── New Training Orchestration Endpoints ──────────────────────────────

@router.get("/training-status")
async def get_training_status():
    """
    Get real-time training status. Frontend should poll this every 1-2 seconds.
    
    Returns structured JSON with:
    - session_id, is_running, phase
    - total_elapsed_seconds, current_model, current_model_elapsed_seconds
    - completed_models, remaining_models
    - current_trial, total_trials
    - per-model detailed states
    - logs
    """
    status_data = training_manager.get_status()
    
    # Add result model_id if training is complete
    with training_manager._state_lock:
        status_data["result_model_id"] = getattr(training_manager, '_result_model_id', None)
        status_data["training_duration"] = getattr(training_manager, '_training_duration', None)
    
    return status_data


@router.post("/skip-model")
async def skip_current_model():
    """
    Skip the currently training model and move to the next one.
    Uses cooperative pattern - sets a flag that the Optuna study checks.
    """
    if not training_manager._is_running:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No training session is currently running"
        )
    
    current = training_manager._current_model_name
    training_manager.request_skip()
    
    return {
        "message": f"Skip requested for model: {current}",
        "skipped_model": current,
    }


@router.post("/cancel-training")
async def cancel_training_new():
    """
    Cancel the entire training session cooperatively.
    Sets the cancellation flag which is checked at every trial boundary
    and between model transitions.
    """
    if not training_manager._is_running:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No training session is currently running"
        )
    
    session_id = training_manager._session_id
    training_manager.request_cancel()
    
    # Also set legacy flags
    if session_id and session_id in active_trainings:
        active_trainings[session_id]["cancelled"] = True
    
    return {
        "message": "Training cancellation requested - models will stop cooperatively",
        "session_id": session_id,
    }


# ─── Legacy Cancel Endpoint ───────────────────────────────────────────

@router.post("/cancel/{training_id}")
async def cancel_training(
    training_id: str,
):
    """
    Cancel an ongoing training session (legacy endpoint - also triggers TrainingManager).
    
    Args:
        training_id: Training session ID
    
    Returns:
        Success message
    """
    # Trigger TrainingManager cancellation
    if training_manager._is_running:
        training_manager.request_cancel()

    # Legacy cancellation
    if training_id in active_trainings:
        active_trainings[training_id]["cancelled"] = True
        if training_id in force_cancel_flag:
            force_cancel_flag[training_id][0] = True
        return {"message": f"Training {training_id} cancelled"}
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Training session {training_id} not found"
    )


# ─── Model Management Endpoints ───────────────────────────────────────

@router.get("/models")
async def list_models():
    """List all trained AutoML models"""
    automl_store = get_store("automl_models")
    datasets_store = get_store("datasets")
    
    cursor = automl_store.find({})
    result = []
    
    async for doc in cursor:
        # Resolve dataset name
        dataset_name = None
        try:
            dataset = await datasets_store.find_one({"_id": doc.get("dataset_id")})
            if dataset:
                dataset_name = dataset.get("name", dataset.get("filename", "Unknown"))
        except Exception:
            pass
        
        result.append({
            "id": doc["_id"],
            "name": doc.get("name", "Unnamed"),
            "description": doc.get("description"),
            "task_type": doc.get("task_type"),
            "best_model_name": doc.get("best_model_name"),
            "best_model_type": doc.get("best_model_type"),
            "best_generalization_score": doc.get("best_generalization_score"),
            "best_cv_score": doc.get("best_cv_score"),
            "best_test_score": doc.get("best_test_score"),
            "best_overfit_gap": doc.get("best_overfit_gap"),
            "total_models_evaluated": doc.get("total_models_evaluated"),
            "models_rejected": doc.get("models_rejected"),
            "created_at": doc.get("created_at"),
            "training_duration": doc.get("training_duration"),
            "feature_names": doc.get("feature_names", []),
            "preprocessing_metadata": doc.get("preprocessing_metadata"),
            "all_models": doc.get("all_models", []),
            "dataset_name": dataset_name,
            "dataset_id": doc.get("dataset_id"),
            "unsupervised_subtype": doc.get("unsupervised_subtype"),
            "best_primary_score": doc.get("best_primary_score"),
        })
    
    return convert_numpy_types(result)


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get a specific AutoML model by ID"""
    automl_store = get_store("automl_models")
    datasets_store = get_store("datasets")
    
    doc = await automl_store.find_one({"_id": model_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    # Resolve dataset name
    dataset_name = None
    try:
        dataset = await datasets_store.find_one({"_id": doc.get("dataset_id")})
        if dataset:
            dataset_name = dataset.get("name", dataset.get("filename", "Unknown"))
    except Exception:
        pass
    
    response = {
        "id": doc["_id"],
        "name": doc.get("name", "Unnamed"),
        "description": doc.get("description"),
        "task_type": doc.get("task_type"),
        "dataset_id": doc.get("dataset_id"),
        "dataset_name": dataset_name,
        "best_model_name": doc.get("best_model_name"),
        "best_model_type": doc.get("best_model_type"),
        "best_generalization_score": doc.get("best_generalization_score"),
        "best_cv_score": doc.get("best_cv_score"),
        "best_test_score": doc.get("best_test_score"),
        "best_overfit_gap": doc.get("best_overfit_gap"),
        "best_params": doc.get("best_params"),
        "best_detailed_metrics": doc.get("best_detailed_metrics"),
        "best_feature_importance": doc.get("best_feature_importance"),
        "all_models": doc.get("all_models", []),
        "total_models_evaluated": doc.get("total_models_evaluated"),
        "models_rejected": doc.get("models_rejected"),
        "plot_paths": doc.get("plot_paths", {}),
        "feature_names": doc.get("feature_names", []),
        "target_column": doc.get("target_column"),
        "label_mapping": doc.get("label_mapping"),
        "preprocessing_metadata": doc.get("preprocessing_metadata"),
        "created_at": doc.get("created_at"),
        "training_duration": doc.get("training_duration"),
        "model_path": doc.get("model_path"),
        "preprocessing_path": doc.get("preprocessing_path"),
        "config": doc.get("config"),
        "preprocessing_config": doc.get("preprocessing_config"),
        "status": "completed",
        # Unsupervised-specific fields
        "unsupervised_subtype": doc.get("unsupervised_subtype"),
        "cluster_labels": doc.get("cluster_labels"),
        "anomaly_labels": doc.get("anomaly_labels"),
        "explained_variance": doc.get("explained_variance"),
        "n_samples": doc.get("n_samples"),
        "n_features": doc.get("n_features"),
        "best_primary_score": doc.get("best_primary_score"),
    }
    return convert_numpy_types(response)


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete an AutoML model and its associated files"""
    automl_store = get_store("automl_models")
    
    doc = await automl_store.find_one({"_id": model_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    # Delete model file
    model_path = doc.get("model_path")
    if model_path and os.path.exists(model_path):
        try:
            os.remove(model_path)
        except Exception as e:
            print(f"⚠️ Failed to delete model file: {e}")
    
    # Delete preprocessing file
    preprocessing_path = doc.get("preprocessing_path")
    if preprocessing_path and os.path.exists(preprocessing_path):
        try:
            os.remove(preprocessing_path)
        except Exception as e:
            print(f"⚠️ Failed to delete preprocessing file: {e}")
    
    # Delete plot files
    plot_paths = doc.get("plot_paths", {})
    for plot_name, plot_path in plot_paths.items():
        full_path = os.path.join(settings.models_dir, "plots", plot_path) if not os.path.isabs(plot_path) else plot_path
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except Exception as e:
                print(f"⚠️ Failed to delete plot {plot_name}: {e}")
    
    # Delete from database
    await automl_store.delete_one({"_id": model_id})
    
    return {"message": f"Model {model_id} deleted successfully"}


@router.get("/models/{model_id}/download")
async def download_model(model_id: str):
    """Download a trained AutoML model file"""
    automl_store = get_store("automl_models")
    
    doc = await automl_store.find_one({"_id": model_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    model_path = doc.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file not found on disk"
        )
    
    model_name = doc.get("name", "model")
    filename = f"{model_name}_{model_id}.joblib"
    
    return FileResponse(
        path=model_path,
        filename=filename,
        media_type="application/octet-stream"
    )


# ─── Export Deployment Package Endpoint ────────────────────────────────

@router.get("/models/{model_id}/export")
async def export_model(model_id: str):
    """
    Export a trained model as a complete deployment ZIP package.
    
    The package includes:
    - model_pipeline.pkl   : Combined preprocessing + model pipeline
    - feature_schema.json  : Input feature specifications
    - label_encoder.json   : Label encoding/decoding (classification only)
    - metadata.json        : Training metadata, scores, environment info
    - requirements.txt     : Minimal Python dependencies
    - README.md            : Usage documentation
    - example_inference.py : Standalone prediction script
    - deployment_guide.pdf : Professional PDF deployment guide
    """
    import shutil
    from app.services.model_export import export_model_package

    automl_store = get_store("automl_models")

    doc = await automl_store.find_one({"_id": model_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    model_path = doc.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file not found on disk"
        )

    try:
        zip_path = export_model_package(model_doc=doc, model_path=model_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build export package: {str(e)}"
        )

    model_name = doc.get("name", "model").replace(" ", "_")
    zip_filename = f"{model_name}_deployment_package.zip"

    # Stream the ZIP and clean up temp dir after response
    tmp_dir = os.path.dirname(zip_path)

    def iterfile():
        try:
            with open(zip_path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        iterfile(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_filename}"',
        },
    )


# ─── Prediction Endpoint ──────────────────────────────────────────────

@router.post("/models/{model_id}/predict")
async def predict(model_id: str, input_data: dict):
    """
    Make a prediction using a trained AutoML model.
    
    Args:
        model_id: The model ID
        input_data: Dictionary of feature_name -> value
    
    Returns:
        Prediction result with confidence and probabilities
    """
    import time
    import joblib
    
    automl_store = get_store("automl_models")
    
    doc = await automl_store.find_one({"_id": model_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    model_path = doc.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file not found on disk"
        )
    
    start_time = time.time()
    
    try:
        # Load saved model data
        model_data = joblib.load(model_path)
        best_model = model_data['best_model']
        label_mapping = model_data.get('label_mapping')
        label_encoder = model_data.get('label_encoder')
        preprocessing_pipeline = model_data.get('preprocessing_pipeline')
        
        feature_names = doc.get("feature_names", [])
        preprocessing_metadata = doc.get("preprocessing_metadata", {})
        text_features = preprocessing_metadata.get("text_features", [])
        categorical_features = preprocessing_metadata.get("categorical_features", [])
        numerical_features = preprocessing_metadata.get("numerical_features", [])
        
        # Build input dataframe
        if text_features or (preprocessing_pipeline is not None):
            # Use adaptive preprocessing pipeline
            raw_features = list(set(
                (numerical_features or []) +
                (categorical_features or []) +
                (text_features or [])
            ))
            if not raw_features:
                raw_features = list(input_data.keys())
            
            input_df = pd.DataFrame([{f: input_data.get(f, None) for f in raw_features}])
            
            if preprocessing_pipeline is not None:
                try:
                    X_input = preprocessing_pipeline.transform(input_df)
                except Exception as e:
                    print(f"⚠️ Pipeline transform failed: {e}, falling back to manual")
                    X_input = input_df.values
            else:
                X_input = input_df.values
        else:
            # Direct feature vector
            X_input = np.array([[
                float(input_data.get(f, 0)) if f not in categorical_features
                else input_data.get(f, 0)
                for f in feature_names
            ]])
        
        # Make prediction
        prediction_encoded = best_model.predict(X_input)[0]
        
        # Get confidence / probabilities
        confidence = None
        probabilities = None
        
        if hasattr(best_model, 'predict_proba'):
            try:
                proba = best_model.predict_proba(X_input)[0]
                confidence = float(max(proba))
                probabilities = {
                    str(class_label): float(prob) 
                    for class_label, prob in zip(best_model.classes_, proba)
                }
            except Exception:
                pass
        
        processing_time = time.time() - start_time
        
        # Decode prediction if we have label mapping
        if label_mapping and isinstance(prediction_encoded, (int, np.integer)):
            prediction = label_mapping.get(str(int(prediction_encoded)), str(prediction_encoded))
        elif isinstance(prediction_encoded, (np.integer, np.floating)):
            prediction = float(prediction_encoded)
        else:
            prediction = str(prediction_encoded)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "processing_time": f"{processing_time:.3f}s",
            "input_features": len(feature_names),
            "model_name": doc.get('name'),
            "model_type": doc.get('best_model_type'),
            "task_type": doc.get('task_type')
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ─── Encoding Info Endpoint ───────────────────────────────────────────

@router.get("/models/{model_id}/encoding-info")
async def get_encoding_info(model_id: str):
    """
    Get categorical encoding mappings for a model.
    Used by the frontend to show dropdown options for categorical features.
    """
    import joblib
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    automl_store = get_store("automl_models")
    
    doc = await automl_store.find_one({"_id": model_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    preprocessing_metadata = doc.get("preprocessing_metadata", {})
    feature_names = doc.get("feature_names", [])
    label_mapping = doc.get("label_mapping")
    numerical_features = preprocessing_metadata.get("numerical_features", [])
    categorical_features = preprocessing_metadata.get("categorical_features", [])
    text_features = preprocessing_metadata.get("text_features", [])
    
    # Try to extract categorical mappings from saved preprocessing pipeline
    categorical_mappings = {}
    
    preprocessing_path = doc.get("preprocessing_path")
    if preprocessing_path and os.path.exists(preprocessing_path):
        try:
            preprocessor = joblib.load(preprocessing_path)
            
            # Handle ColumnTransformer
            if isinstance(preprocessor, ColumnTransformer):
                for name, transformer, columns in preprocessor.transformers_:
                    # Handle old 'cat' name and new 'cat_ohe' / 'cat_ord' names
                    if name in ('cat', 'cat_ohe', 'cat_ord'):
                        # Navigate through pipeline steps to find encoder
                        encoder = None
                        if hasattr(transformer, 'named_steps'):
                            encoder = transformer.named_steps.get('encoder')
                        elif isinstance(transformer, (OrdinalEncoder, OneHotEncoder)):
                            encoder = transformer
                        
                        if encoder and hasattr(encoder, 'categories_'):
                            for i, col in enumerate(columns):
                                if i < len(encoder.categories_):
                                    categories = encoder.categories_[i]
                                    categorical_mappings[col] = {
                                        int(j): str(cat) for j, cat in enumerate(categories)
                                    }
        except Exception as e:
            print(f"⚠️ Failed to extract encoding info: {e}")
    
    # Also try from model file itself
    if not categorical_mappings:
        model_path = doc.get("model_path")
        if model_path and os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                pipeline = model_data.get('preprocessing_pipeline')
                if pipeline and isinstance(pipeline, ColumnTransformer):
                    for name, transformer, columns in pipeline.transformers_:
                        if name in ('cat', 'cat_ohe', 'cat_ord'):
                            encoder = None
                            if hasattr(transformer, 'named_steps'):
                                encoder = transformer.named_steps.get('encoder')
                            elif isinstance(transformer, (OrdinalEncoder, OneHotEncoder)):
                                encoder = transformer
                            
                            if encoder and hasattr(encoder, 'categories_'):
                                for i, col in enumerate(columns):
                                    if i < len(encoder.categories_):
                                        categories = encoder.categories_[i]
                                        categorical_mappings[col] = {
                                            int(j): str(cat) for j, cat in enumerate(categories)
                                        }
            except Exception as e:
                print(f"⚠️ Failed to extract encoding from model: {e}")
    
    return {
        "categorical_mappings": categorical_mappings,
        "feature_names": feature_names,
        "label_mapping": label_mapping,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "text_features": text_features,
    }


# ─── Plot Serving Endpoint ────────────────────────────────────────────

@router.get("/plots/{plot_path:path}")
async def get_plot(plot_path: str):
    """Serve a plot image file"""
    plots_dir = os.path.join(settings.models_dir, "plots")
    
    # Try the path as-is first (filename only)
    full_path = os.path.join(plots_dir, plot_path)
    
    # If not found, the path might already include the plots_dir prefix (legacy data)
    if not os.path.exists(full_path):
        # Try just the basename
        full_path = os.path.join(plots_dir, os.path.basename(plot_path))
    
    # Also try as absolute path (in case full path was stored)
    if not os.path.exists(full_path) and os.path.exists(plot_path):
        full_path = plot_path
    
    if not os.path.exists(full_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot not found: {plot_path}"
        )
    
    return FileResponse(
        path=full_path,
        media_type="image/png"
    )


@router.get("/system-info")
async def get_system_info():
    """Get system information like CPU count for frontend configuration"""
    import multiprocessing
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
    except ImportError:
        cpu_count = multiprocessing.cpu_count()
    
    return {
        "cpu_count": cpu_count or 4,
    }
