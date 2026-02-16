"""
Model training routes for training and managing ML models
"""
from fastapi import APIRouter, HTTPException, status
from typing import List
import pandas as pd
import os
from datetime import datetime
from app.models.schemas import (
    ModelCreate,
    ModelResponse,
    ModelType,
)
from app.storage import get_store, generate_id
from app.services.preprocessing import PreprocessingService
from app.services.model_training import ModelTrainingService
from app.config import settings

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.post("/train", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def train_model(
    model_request: ModelCreate,
):
    """
    Train a new machine learning model on a dataset
    
    Args:
        model_request: ModelCreate object with training configuration
    
    Returns:
        ModelResponse with training results and metrics
    
    Raises:
        HTTPException: If dataset not found or training fails
    """
    datasets_store = get_store("datasets")
    models_store = get_store("models")
    
    # Verify dataset exists
    try:
        dataset = await datasets_store.find_one({
            "_id": model_request.dataset_id
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
        # Load dataset
        df = pd.read_csv(dataset["file_path"], sep=None, engine='python', skipinitialspace=True)
        
        # Verify target column exists
        if model_request.target_column not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target column '{model_request.target_column}' not found in dataset"
            )
        
        # Step 1: Preprocessing
        preprocessing_service = PreprocessingService()
        
        config = model_request.preprocessing_config
        if config:
            X_transformed, y, preprocessing_steps = preprocessing_service.fit_transform(
                df=df,
                target_column=model_request.target_column,
                numerical_strategy=config.numerical_strategy,
                categorical_strategy=config.categorical_strategy,
                scaling_method=config.scaling_method,
                encoding_method=config.encoding_method
            )
        else:
            X_transformed, y, preprocessing_steps = preprocessing_service.fit_transform(
                df=df,
                target_column=model_request.target_column
            )
        
        # Step 2: Model Training
        training_service = ModelTrainingService(model_type=model_request.model_type)
        
        metrics = training_service.train_and_evaluate(
            X=X_transformed,
            y=y,
            feature_names=preprocessing_service.feature_names
        )
        
        # Step 3: Save model and preprocessing pipeline
        model_id = generate_id()
        model_filename = f"{model_id}_model.joblib"
        preprocessing_filename = f"{model_id}_preprocessing.joblib"
        
        model_path = os.path.join(settings.models_dir, model_filename)
        preprocessing_path = os.path.join(settings.models_dir, preprocessing_filename)
        
        os.makedirs(settings.models_dir, exist_ok=True)
        
        training_service.save_model(model_path)
        preprocessing_service.save_pipeline(preprocessing_path)
        
        # Step 4: Save model metadata to database
        model_doc = {
            "_id": model_id,
            "dataset_id": model_request.dataset_id,
            "name": model_request.name,
            "model_type": model_request.model_type.value,
            "description": model_request.description,
            "target_column": model_request.target_column,
            "created_at": datetime.utcnow(),
            "model_path": model_path,
            "preprocessing_path": preprocessing_path,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "confusion_matrix": metrics["confusion_matrix"],
            "feature_names": preprocessing_service.feature_names,
            "feature_importance": metrics.get("feature_importance")
        }
        
        await models_store.insert_one(model_doc)
        
        # Return response
        model_doc["id"] = str(model_doc.pop("_id"))
        model_doc["model_type"] = ModelType(model_doc["model_type"])
        
        return ModelResponse(**model_doc)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to train model: {str(e)}"
        )


@router.get("/", response_model=List[ModelResponse])
async def list_models(
    dataset_id: str = None,
):
    """
    List all models
    
    Args:
        dataset_id: Optional filter by dataset ID
    
    Returns:
        List of ModelResponse objects
    """
    models_store = get_store("models")
    
    # Build query
    query = {}
    if dataset_id:
        query["dataset_id"] = dataset_id
    
    # Find all models
    cursor = models_store.find(query)
    models = []
    
    async for model in cursor:
        model["id"] = str(model.pop("_id"))
        model["model_type"] = ModelType(model["model_type"])
        models.append(ModelResponse(**model))
    
    return models


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
):
    """
    Get details of a specific model
    
    Args:
        model_id: Model ID
    
    Returns:
        ModelResponse object
    
    Raises:
        HTTPException: If model not found
    """
    models_store = get_store("models")
    
    try:
        model = await models_store.find_one({
            "_id": model_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model ID format"
        )
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    model["id"] = str(model.pop("_id"))
    model["model_type"] = ModelType(model["model_type"])
    
    return ModelResponse(**model)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
):
    """
    Delete a trained model
    
    Args:
        model_id: Model ID
    
    Raises:
        HTTPException: If model not found
    """
    models_store = get_store("models")
    
    try:
        model = await models_store.find_one({
            "_id": model_id
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model ID format"
        )
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    # Delete model files from disk
    if os.path.exists(model["model_path"]):
        os.remove(model["model_path"])
    if os.path.exists(model["preprocessing_path"]):
        os.remove(model["preprocessing_path"])
    
    # Delete from database
    await models_store.delete_one({"_id": model_id})
    
    return None


@router.get("/{model_id}/compare")
async def compare_models(
    model_ids: List[str],
):
    """
    Compare multiple models side by side
    
    Args:
        model_ids: List of model IDs to compare
    
    Returns:
        Dictionary with comparison data
    """
    models_store = get_store("models")
    
    comparison = []
    
    for model_id in model_ids:
        try:
            model = await models_store.find_one({
                "_id": model_id
            })
            
            if model:
                comparison.append({
                    "id": str(model["_id"]),
                    "name": model["name"],
                    "model_type": model["model_type"],
                    "accuracy": model["accuracy"],
                    "precision": model["precision"],
                    "recall": model["recall"],
                    "f1_score": model["f1_score"],
                    "created_at": model["created_at"]
                })
        except Exception:
            continue
    
    return {"models": comparison}
