/**
 * Train AutoML Model Page
 * Step-by-step wizard for training AutoML models with generalization focus
 */
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  ArrowLeft,
  ChevronRight,
  ChevronLeft,
  Sparkles,
  Database,
  Settings,
  Play,
  Loader2,
  CheckCircle,
  AlertTriangle,
  Info,
  Zap,
  TrendingUp,
  Check,
  ChevronDown,
  ChevronUp,
  Clock,
  Target,
  TrendingDown,
  Award,
  X,
  SkipForward,
  Timer,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Textarea } from '../components/ui/textarea';
import { Checkbox } from '../components/ui/checkbox';
import { Slider } from '../components/ui/slider';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { Badge } from '../components/ui/badge';
import { Separator } from '../components/ui/separator';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '../components/ui/dropdown-menu';
import { datasetsAPI, automlAPI } from '../utils/api';

/** Format seconds to HH:MM:SS */
function formatElapsed(totalSeconds) {
  if (!totalSeconds || totalSeconds < 0) return '00:00:00';
  const hrs = Math.floor(totalSeconds / 3600);
  const mins = Math.floor((totalSeconds % 3600) / 60);
  const secs = Math.floor(totalSeconds % 60);
  return [hrs, mins, secs].map(v => String(v).padStart(2, '0')).join(':');
}

export default function TrainAutoMLModel() {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [processedDatasets, setProcessedDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [training, setTraining] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [skipping, setSkipping] = useState(false);
  const [currentTrainingId, setCurrentTrainingId] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const pollingRef = useRef(null);
  const abortControllerRef = useRef(null);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [expandedLogs, setExpandedLogs] = useState({});
  const [maxCpuCores, setMaxCpuCores] = useState(16);

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    dataset_id: '',
    config: {
      n_trials: 75,
      cv_folds: 5,
      penalty_factor: 2.0,
      overfit_threshold_reject: 0.20,
      overfit_threshold_high: 0.10,
      models_to_train: null, // null = all models
      max_cpu_cores: 4, // Limit CPU cores to prevent 100% usage
    },
  });

  const [selectedModels, setSelectedModels] = useState({
    classification: [],
    regression: [],
    clustering: [],
    unsupervised_clustering: [],
    unsupervised_dimensionality_reduction: [],
    unsupervised_anomaly_detection: [],
  });
  const [unsupervisedSubtype, setUnsupervisedSubtype] = useState('clustering');

  const unsupervisedSubtypes = [
    { id: 'clustering', name: 'Clustering', description: 'Group similar data points into clusters' },
    { id: 'dimensionality_reduction', name: 'Dimensionality Reduction', description: 'Reduce feature dimensions while preserving structure' },
    { id: 'anomaly_detection', name: 'Anomaly Detection', description: 'Identify outliers and unusual data points' },
  ];

  const availableModels = {
    classification: [
      { id: 'logistic_regression', name: 'Logistic Regression', description: 'Linear classifier' },
      { id: 'decision_tree', name: 'Decision Tree', description: 'Interpretable tree-based' },
      { id: 'random_forest', name: 'Random Forest', description: 'Ensemble of trees' },
      { id: 'gradient_boosting', name: 'Gradient Boosting', description: 'Boosted trees' },
      { id: 'xgboost', name: 'XGBoost', description: 'Optimized gradient boosting' },
      { id: 'lightgbm', name: 'LightGBM', description: 'Fast gradient boosting' },
      { id: 'svm', name: 'SVM', description: 'Support vector machine' },
      { id: 'knn', name: 'K-Nearest Neighbors', description: 'Non-parametric' },
      { id: 'adaboost', name: 'AdaBoost', description: 'Adaptive boosting' },
    ],
    regression: [
      { id: 'ridge', name: 'Ridge', description: 'L2 regularized linear' },
      { id: 'lasso', name: 'Lasso', description: 'L1 regularized linear' },
      { id: 'elastic_net', name: 'Elastic Net', description: 'L1+L2 regularized' },
      { id: 'decision_tree', name: 'Decision Tree', description: 'Tree regressor' },
      { id: 'random_forest', name: 'Random Forest', description: 'Ensemble regressor' },
      { id: 'gradient_boosting', name: 'Gradient Boosting', description: 'Boosted trees' },
      { id: 'xgboost', name: 'XGBoost', description: 'Optimized boosting' },
      { id: 'lightgbm', name: 'LightGBM', description: 'Fast boosting' },
      { id: 'svr', name: 'SVR', description: 'Support vector regression' },
      { id: 'knn', name: 'K-Nearest Neighbors', description: 'Local patterns' },
      { id: 'adaboost', name: 'AdaBoost', description: 'Adaptive boosting' },
    ],
    clustering: [
      { id: 'kmeans', name: 'K-Means', description: 'Centroid-based' },
      { id: 'dbscan', name: 'DBSCAN', description: 'Density-based' },
      { id: 'agglomerative', name: 'Agglomerative', description: 'Hierarchical' },
    ],
    unsupervised_clustering: [
      { id: 'kmeans', name: 'K-Means', description: 'Centroid-based clustering' },
      { id: 'mini_batch_kmeans', name: 'MiniBatch K-Means', description: 'Scalable K-Means' },
      { id: 'dbscan', name: 'DBSCAN', description: 'Density-based spatial clustering' },
      { id: 'agglomerative', name: 'Agglomerative', description: 'Hierarchical clustering' },
      { id: 'hdbscan', name: 'HDBSCAN', description: 'Hierarchical density-based (optional)' },
    ],
    unsupervised_dimensionality_reduction: [
      { id: 'pca', name: 'PCA', description: 'Principal Component Analysis' },
      { id: 'truncated_svd', name: 'Truncated SVD', description: 'Sparse-friendly SVD' },
      { id: 'umap', name: 'UMAP', description: 'Manifold learning (optional)' },
    ],
    unsupervised_anomaly_detection: [
      { id: 'isolation_forest', name: 'Isolation Forest', description: 'Tree-based anomaly detection' },
      { id: 'local_outlier_factor', name: 'Local Outlier Factor', description: 'Density-based outlier detection' },
    ],
  };

  useEffect(() => {
    fetchProcessedDatasets();
    // Fetch system CPU count for slider max
    automlAPI.getSystemInfo().then(info => {
      if (info?.cpu_count) {
        setMaxCpuCores(info.cpu_count);
        // Cap current value if it exceeds actual CPU count
        setFormData(prev => ({
          ...prev,
          config: {
            ...prev.config,
            max_cpu_cores: Math.min(prev.config.max_cpu_cores, info.cpu_count),
          },
        }));
      }
    }).catch(() => {});
  }, []);

  // Navigation guard: Prevent user from leaving page during training
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (training && currentTrainingId) {
        e.preventDefault();
        e.returnValue = 'Model training is in progress. If you leave, the training will be cancelled. Are you sure?';
        
        // Attempt to cancel training on page unload (browser refresh/close)
        // Using fetch with keepalive flag for reliability during page unload
        try {
          fetch(`http://localhost:8000/api/automl/cancel/${currentTrainingId}`, {
            method: 'POST',
            keepalive: true,
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({}),
          }).catch(() => {
            // Ignore errors during unload
          });
        } catch (err) {
          console.error('Failed to cancel training on unload:', err);
        }
        
        return e.returnValue;
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [training, currentTrainingId]);

  // Cancel training when component unmounts (in-app navigation)
  useEffect(() => {
    return () => {
      if (training && currentTrainingId) {
        // Cancel training when navigating away
        automlAPI.cancelTraining(currentTrainingId).catch(err => {
          console.error('Failed to cancel training on unmount:', err);
        });
        
        // Stop polling
        if (pollingRef.current) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
      }
    };
  }, [training, currentTrainingId]);

  const fetchProcessedDatasets = async () => {
    try {
      setLoading(true);
      console.log('Fetching processed datasets...');
      const datasets = await datasetsAPI.listProcessed();
      console.log('Datasets received:', datasets);
      // Filter to only show datasets that have been preprocessed
      const processed = datasets.filter(d => d.preprocessing_summary);
      console.log('Processed datasets:', processed);
      setProcessedDatasets(processed);
    } catch (err) {
      console.error('Error fetching datasets:', err);
      setError('Failed to fetch processed datasets');
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetSelect = (datasetId) => {
    const dataset = processedDatasets.find(d => d.id === datasetId);
    setSelectedDataset(dataset);
    
    // Autofill name and description from dataset (user can still change them)
    const autoName = dataset?.name ? `${dataset.name} - AutoML Model` : '';
    const autoDescription = dataset?.description || `AutoML model trained on ${dataset?.name || 'dataset'}`;
    
    setFormData({ 
      ...formData, 
      dataset_id: datasetId,
      name: formData.name || autoName,  // Only autofill if empty
      description: formData.description || autoDescription,  // Only autofill if empty
    });
  };

  const handleModelToggle = (modelId, taskType) => {
    setSelectedModels({
      ...selectedModels,
      [taskType]: selectedModels[taskType].includes(modelId)
        ? selectedModels[taskType].filter(id => id !== modelId)
        : [...selectedModels[taskType], modelId],
    });
  };

  const handleTrainModel = async () => {
    try {
      setTraining(true);
      setError(null);
      setTrainingStatus(null);

      // Get task type from preprocessing summary
      const taskType = selectedDataset?.preprocessing_summary?.task_type || 'classification';
      const targetColumn = selectedDataset?.preprocessing_summary?.target_column || '';
      const isUnsupervised = taskType === 'unsupervised';

      // Determine model key for unsupervised subtypes
      const modelKey = isUnsupervised ? `unsupervised_${unsupervisedSubtype}` : taskType;

      // Determine which models will be trained
      const modelsToTrain = selectedModels[modelKey]?.length > 0
        ? selectedModels[modelKey]
        : availableModels[modelKey]?.map(m => m.id) || [];

      // Initialize logs for display
      const initialLogs = modelsToTrain.map((modelId) => ({
        id: modelId,
        name: availableModels[modelKey]?.find(m => m.id === modelId)?.name || modelId,
      }));
      setTrainingLogs(initialLogs);

      // Prepare request data
      const requestData = {
        name: formData.name,
        description: formData.description,
        dataset_id: formData.dataset_id,
        target_column: isUnsupervised ? null : targetColumn,
        task_type: taskType,
        ...(isUnsupervised && { unsupervised_subtype: unsupervisedSubtype }),
        config: {
          ...formData.config,
          models_to_train:
            selectedModels[modelKey]?.length > 0
              ? selectedModels[modelKey]
              : null,
        },
        preprocessing_config: {
          use_adaptive: false,
          use_eda_insights: false,
        },
      };

      // Start training (returns immediately, runs in background)
      const startResponse = await automlAPI.startTraining(requestData);
      const trainingId = startResponse.training_id;

      setCurrentTrainingId(trainingId);

      // Start polling for status
      startStatusPolling();

    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start training');
      setTraining(false);
      setCancelling(false);
    }
  };

  const startStatusPolling = () => {
    // Clear any existing polling
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
    }

    pollingRef.current = setInterval(async () => {
      try {
        const status = await automlAPI.getTrainingStatus();
        setTrainingStatus(status);

        // Check if training completed
        if (!status.is_running) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;

          if (status.phase === 'completed' && status.result_model_id) {
            // Navigate to model detail
            const totalTime = (status.total_elapsed_seconds / 60).toFixed(1);
            setTimeout(() => {
              navigate(`/dashboard/models/${status.result_model_id}`, {
                state: {
                  successMessage: `Model trained successfully in ${totalTime} minutes!`,
                  trainingTime: totalTime,
                },
              });
            }, 500);
          } else if (status.phase === 'cancelled') {
            setTraining(false);
            setCancelling(false);
            setTrainingStatus(null);
            setCurrentTrainingId(null);
            setError(null);
          } else if (status.phase === 'failed') {
            const lastLog = status.logs?.[status.logs.length - 1] || 'Training failed';
            setError(lastLog);
            setTraining(false);
            setCancelling(false);
            setTrainingStatus(null);
            setCurrentTrainingId(null);
          }
        }
      } catch (err) {
        // Don't crash polling on transient errors
        console.warn('Status poll failed:', err);
      }
    }, 1500);
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  const handleCancelTraining = async () => {
    if (!currentTrainingId) return;

    try {
      setCancelling(true);
      await automlAPI.cancelTraining(currentTrainingId);
      // Polling will detect phase=cancelled and reset UI
    } catch (err) {
      console.error('Error cancelling training:', err);
      setCancelling(false);
      setError('Failed to cancel training: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleSkipModel = async () => {
    try {
      setSkipping(true);
      await automlAPI.skipModel();
      // Reset skipping after a short delay (polling will update status)
      setTimeout(() => setSkipping(false), 2000);
    } catch (err) {
      console.error('Error skipping model:', err);
      setSkipping(false);
    }
  };

  const canProceedStep1 = formData.name && formData.dataset_id;
  const taskType = selectedDataset?.preprocessing_summary?.task_type || 'classification';
  const isUnsupervised = taskType === 'unsupervised';
  const modelKey = isUnsupervised ? `unsupervised_${unsupervisedSubtype}` : taskType;

  const steps = [
    { number: 1, title: 'Dataset & Info', icon: Database },
    { number: 2, title: 'Model Selection', icon: Settings },
    { number: 3, title: 'Configuration', icon: Zap },
    { number: 4, title: 'Review & Train', icon: Play },
  ];

  return (
    <div className="space-y-6 w-full mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => navigate('/dashboard/models')}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
              <Sparkles className="h-8 w-8 text-primary" />
              Train AutoML Model
            </h1>
            <p className="text-muted-foreground">
              Generalization-focused training with Optuna optimization
            </p>
          </div>
        </div>
      </div>

      {/* Progress Steps */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            {steps.map((s, index) => (
              <React.Fragment key={s.number}>
                <div className="flex items-center gap-3">
                  <div
                    className={`flex items-center justify-center w-10 h-10 rounded-full border-2 ${
                      step >= s.number
                        ? 'bg-primary border-primary text-primary-foreground'
                        : 'border-muted-foreground/20 text-muted-foreground'
                    }`}
                  >
                    {step > s.number ? (
                      <CheckCircle className="h-5 w-5" />
                    ) : (
                      <s.icon className="h-5 w-5" />
                    )}
                  </div>
                  <div className="hidden sm:block">
                    <div className="text-sm font-medium">{s.title}</div>
                    <div className="text-xs text-muted-foreground">Step {s.number}</div>
                  </div>
                </div>
                {index < steps.length - 1 && (
                  <div className="flex-1 h-0.5 bg-muted-foreground/20 mx-4" />
                )}
              </React.Fragment>
            ))}
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Step 1: Dataset & Info */}
      {step === 1 && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Select Processed Dataset</CardTitle>
              <CardDescription>
                Choose from datasets that have been preprocessed and are ready for training
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-primary" />
                </div>
              ) : processedDatasets.length === 0 ? (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    No processed datasets found. Please preprocess a dataset first before training an AutoML model.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-2">
                  <Label>Dataset</Label>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" className="w-full justify-between">
                        {selectedDataset ? (
                          <span>{selectedDataset.name}</span>
                        ) : (
                          <span className="text-muted-foreground">Select a dataset...</span>
                        )}
                        <ChevronDown className="h-4 w-4 ml-2" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="w-[600px] max-h-[400px] overflow-y-auto" align="start">
                      {processedDatasets.map((dataset) => (
                        <DropdownMenuItem
                          key={dataset.id}
                          onClick={() => handleDatasetSelect(dataset.id)}
                          className="cursor-pointer p-4"
                        >
                          <div className="flex-1">
                            <div className="font-medium">{dataset.name}</div>
                            <div className="text-sm text-muted-foreground mt-1">
                              {dataset.rows} rows × {dataset.columns} columns
                            </div>
                            {dataset.preprocessing_summary && (
                              <div className="flex gap-2 mt-2">
                                <Badge variant="outline" className="text-xs">
                                  {dataset.preprocessing_summary.task_type || 'Unknown Task'}
                                </Badge>
                                {dataset.preprocessing_summary.target_column && (
                                  <Badge variant="outline" className="text-xs">
                                    Target: {dataset.preprocessing_summary.target_column}
                                  </Badge>
                                )}
                                <Badge variant="outline" className="text-xs bg-green-500/10">
                                  ✓ Preprocessed
                                </Badge>
                              </div>
                            )}
                          </div>
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Model Information</CardTitle>
              <CardDescription>
                Give your AutoML model a name and description (auto-filled from dataset, but you can change them)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Model Name *</Label>
                <Input
                  id="name"
                  placeholder="e.g., Fraud Detection Model"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Description (Optional)</Label>
                <Textarea
                  id="description"
                  placeholder="Describe what this model will predict..."
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  rows={3}
                />
              </div>
            </CardContent>
          </Card>

          {selectedDataset && selectedDataset.preprocessing_summary && (
            <Card>
              <CardHeader>
                <CardTitle>Dataset Information</CardTitle>
                <CardDescription>Preprocessing details for selected dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground mb-1">Task Type</div>
                    <Badge>{selectedDataset.preprocessing_summary.task_type}</Badge>
                  </div>
                  {selectedDataset.preprocessing_summary.target_column && (
                    <div>
                      <div className="text-muted-foreground mb-1">Target Column</div>
                      <div className="font-mono font-semibold">
                        {selectedDataset.preprocessing_summary.target_column}
                      </div>
                    </div>
                  )}
                  <div>
                    <div className="text-muted-foreground mb-1">Features</div>
                    <div className="font-semibold">
                      {selectedDataset.preprocessing_summary.final_shape?.[1] || selectedDataset.columns} features
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Step 2: Model Selection */}
      {step === 2 && (
        <div className="space-y-6">
          <Alert>
            <Info className="h-4 w-4" />
            <AlertTitle>Task Type: {taskType.charAt(0).toUpperCase() + taskType.slice(1)}</AlertTitle>
            <AlertDescription>
              {isUnsupervised
                ? 'Unsupervised learning — no target column needed. Select a subtype and models below.'
                : 'Based on your preprocessed dataset. Select specific models to train or leave empty to train all available models.'}
            </AlertDescription>
          </Alert>

          {/* Unsupervised Subtype Selector */}
          {isUnsupervised && (
            <Card>
              <CardHeader>
                <CardTitle>Unsupervised Learning Type</CardTitle>
                <CardDescription>
                  Choose what kind of unsupervised analysis to perform
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {unsupervisedSubtypes.map((subtype) => (
                    <div
                      key={subtype.id}
                      className={`border-2 rounded-lg p-4 cursor-pointer transition-colors ${
                        unsupervisedSubtype === subtype.id
                          ? 'border-primary bg-primary/5'
                          : 'border-muted hover:border-muted-foreground/30'
                      }`}
                      onClick={() => setUnsupervisedSubtype(subtype.id)}
                    >
                      <div className="font-medium">{subtype.name}</div>
                      <div className="text-sm text-muted-foreground mt-1">{subtype.description}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Select Models (Optional)</CardTitle>
              <CardDescription>
                Choose specific models or leave empty to train all {availableModels[modelKey]?.length || 0} models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {availableModels[modelKey]?.map((model) => (
                  <div
                    key={model.id}
                    className="flex items-start space-x-3 border rounded-lg p-3 hover:bg-accent/50 cursor-pointer"
                    onClick={() => handleModelToggle(model.id, modelKey)}
                  >
                    <Checkbox
                      checked={selectedModels[modelKey]?.includes(model.id) || false}
                      onCheckedChange={() => handleModelToggle(model.id, modelKey)}
                    />
                    <div className="flex-1">
                      <div className="font-medium text-sm">{model.name}</div>
                      <div className="text-xs text-muted-foreground">{model.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Step 3: Optuna Configuration */}
      {step === 3 && (
        <div className="space-y-6">
          <Alert>
            <TrendingUp className="h-4 w-4" />
            <AlertTitle>{isUnsupervised ? 'Unsupervised Optimization' : 'Generalization-Focused Training'}</AlertTitle>
            <AlertDescription>
              {isUnsupervised
                ? 'Optuna will optimize hyperparameters using unsupervised metrics (silhouette score, explained variance, etc.).'
                : 'Models are selected based on generalization score, not raw performance. This ensures models that truly generalize well are chosen.'}
            </AlertDescription>
          </Alert>

          <Card>
            <CardHeader>
              <CardTitle>Optuna Configuration</CardTitle>
              <CardDescription>
                Hyperparameter optimization settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Number of Trials per Model: {formData.config.n_trials}</Label>
                <Slider
                  value={[formData.config.n_trials]}
                  onValueChange={(value) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, n_trials: value[0] },
                    })
                  }
                  min={10}
                  max={200}
                  step={5}
                />
                <div className="text-xs text-muted-foreground">
                  More trials = better optimization but longer training time
                </div>
              </div>

              <div className="space-y-2">
                <Label>Cross-Validation Folds: {formData.config.cv_folds}</Label>
                <Slider
                  value={[formData.config.cv_folds]}
                  onValueChange={(value) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, cv_folds: value[0] },
                    })
                  }
                  min={2}
                  max={10}
                  step={1}
                />
                <div className="text-xs text-muted-foreground">
                  More folds = more robust CV score
                </div>
              </div>
            </CardContent>
          </Card>

          {!isUnsupervised && (
          <Card>
            <CardHeader>
              <CardTitle>Overfitting Control</CardTitle>
              <CardDescription>
                Configure penalty system for overfitting detection
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Penalty Factor: {formData.config.penalty_factor.toFixed(1)}</Label>
                <Slider
                  value={[formData.config.penalty_factor * 10]}
                  onValueChange={(value) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, penalty_factor: value[0] / 10 },
                    })
                  }
                  min={10}
                  max={50}
                  step={5}
                />
                <div className="text-xs text-muted-foreground">
                  Higher = stronger penalty for overfitting
                </div>
              </div>

              <div className="space-y-2">
                <Label>
                  Rejection Threshold: {formData.config.overfit_threshold_reject.toFixed(2)}
                </Label>
                <Slider
                  value={[formData.config.overfit_threshold_reject * 100]}
                  onValueChange={(value) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, overfit_threshold_reject: value[0] / 100 },
                    })
                  }
                  min={10}
                  max={40}
                  step={5}
                />
                <div className="text-xs text-muted-foreground">
                  Models with overfit gap above this threshold will be rejected
                </div>
              </div>

              <div className="space-y-2">
                <Label>
                  High Penalty Threshold: {formData.config.overfit_threshold_high.toFixed(2)}
                </Label>
                <Slider
                  value={[formData.config.overfit_threshold_high * 100]}
                  onValueChange={(value) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, overfit_threshold_high: value[0] / 100 },
                    })
                  }
                  min={5}
                  max={20}
                  step={1}
                />
                <div className="text-xs text-muted-foreground">
                  Higher penalty factor applied above this threshold
                </div>
              </div>
            </CardContent>
          </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Performance Settings</CardTitle>
              <CardDescription>
                Resource allocation for training
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>
                    Max CPU Cores: {formData.config.max_cpu_cores}
                  </Label>
                  <Badge variant="outline" className="text-xs">
                    {formData.config.max_cpu_cores <= Math.ceil(maxCpuCores / 4) ? 'Light' : formData.config.max_cpu_cores <= Math.ceil(maxCpuCores / 2) ? 'Balanced' : 'Maximum'}
                  </Badge>
                </div>
                <Slider
                  value={[formData.config.max_cpu_cores]}
                  onValueChange={(value) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, max_cpu_cores: value[0] },
                    })
                  }
                  min={1}
                  max={maxCpuCores}
                  step={1}
                />
                <div className="text-xs text-muted-foreground">
                  Controls parallel processing. Lower values reduce system load but increase training time. (System has {maxCpuCores} logical cores)
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Step 4: Review & Train */}
      {step === 4 && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Review Configuration</CardTitle>
              <CardDescription>
                Verify your settings before starting training
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Model Name</div>
                  <div className="font-medium">{formData.name}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Dataset</div>
                  <div className="font-medium">{selectedDataset?.name}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Task Type</div>
                  <Badge>{taskType}</Badge>
                  {isUnsupervised && (
                    <Badge variant="outline" className="ml-2">
                      {unsupervisedSubtype.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </Badge>
                  )}
                </div>
                {!isUnsupervised && selectedDataset?.preprocessing_summary?.target_column && (
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Target Column</div>
                    <div className="font-mono text-sm">{selectedDataset.preprocessing_summary.target_column}</div>
                  </div>
                )}
              </div>

              <Separator />

              <div>
                <div className="text-sm text-muted-foreground mb-2">Models to Train</div>
                <div className="flex flex-wrap gap-2">
                  {selectedModels[modelKey]?.length > 0 ? (
                    selectedModels[modelKey].map((modelId) => {
                      const model = availableModels[modelKey]?.find((m) => m.id === modelId);
                      return <Badge key={modelId} variant="outline">{model?.name}</Badge>;
                    })
                  ) : (
                    <Badge variant="outline">All {availableModels[modelKey]?.length || 0} models</Badge>
                  )}
                </div>
              </div>

              <Separator />

              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground mb-1">Optuna Trials</div>
                  <div className="font-semibold">{formData.config.n_trials}</div>
                </div>
                <div>
                  <div className="text-muted-foreground mb-1">CV Folds</div>
                  <div className="font-semibold">{formData.config.cv_folds}</div>
                </div>
                {!isUnsupervised && (
                  <>
                    <div>
                      <div className="text-muted-foreground mb-1">Penalty Factor</div>
                      <div className="font-semibold">{formData.config.penalty_factor.toFixed(1)}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground mb-1">Rejection Threshold</div>
                      <div className="font-semibold">{formData.config.overfit_threshold_reject.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground mb-1">High Penalty Threshold</div>
                      <div className="font-semibold">{formData.config.overfit_threshold_high.toFixed(2)}</div>
                    </div>
                  </>
                )}
                <div>
                  <div className="text-muted-foreground mb-1">Max CPU Cores</div>
                  <div className="font-semibold">{formData.config.max_cpu_cores}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Alert>
            <Info className="h-4 w-4" />
            <AlertTitle>Training Information</AlertTitle>
            <AlertDescription className="space-y-2">
              <p>
                Training may take 20-45 minutes depending on dataset size and number of models.
              </p>
              <p className="font-medium text-primary">
                The best model will be selected based on generalization score, not raw test scores.
              </p>
            </AlertDescription>
          </Alert>

          {training && (
            <Card className="border-primary">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Loader2 className="h-5 w-5 animate-spin text-primary" />
                      {cancelling ? 'Cancelling Training...' : (
                        trainingStatus?.phase === 'plotting' ? 'Generating Plots...' :
                        trainingStatus?.phase === 'saving' ? 'Saving Model...' :
                        trainingStatus?.phase === 'preprocessing' ? 'Preprocessing Data...' :
                        'Training in Progress...'
                      )}
                    </CardTitle>
                    <CardDescription>
                      {cancelling 
                        ? 'Stopping all models gracefully. This may take a moment...'
                        : trainingStatus?.phase === 'training'
                          ? `Training model ${(trainingStatus?.models_completed_count || 0) + 1} of ${trainingStatus?.total_models || '?'}`
                          : 'Processing...'}
                    </CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleSkipModel}
                      disabled={cancelling || skipping || trainingStatus?.phase !== 'training'}
                      title="Skip current model and move to next"
                    >
                      {skipping ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Skipping...
                        </>
                      ) : (
                        <>
                          <SkipForward className="h-4 w-4 mr-2" />
                          Skip Model
                        </>
                      )}
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleCancelTraining}
                      disabled={cancelling}
                    >
                      {cancelling ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Cancelling...
                        </>
                      ) : (
                        <>
                          <X className="h-4 w-4 mr-2" />
                          Cancel Training
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-5">
                {/* Time Display */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm">
                    <Timer className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Total Elapsed:</span>
                    <span className="font-mono font-semibold text-base">
                      {formatElapsed(trainingStatus?.total_elapsed_seconds)}
                    </span>
                  </div>
                  {trainingStatus?.current_model && (() => {
                    const currentModelState = trainingStatus?.models?.find(
                      m => m.model_name === trainingStatus.current_model
                    );
                    // Only show current model timer if it's actively training
                    return currentModelState?.status === 'training';
                  })() && (
                    <div className="flex items-center gap-2 text-sm">
                      <Clock className="w-4 h-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Current Model:</span>
                      <span className="font-mono font-semibold">
                        {formatElapsed(trainingStatus?.current_model_elapsed_seconds)}
                      </span>
                    </div>
                  )}
                </div>

                {/* Global Progress Bar */}
                {trainingStatus && (
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Overall Progress</span>
                      <span>
                        {trainingStatus.models_completed_count || 0} / {trainingStatus.total_models || 0} models
                      </span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2.5">
                      <div
                        className="bg-primary h-2.5 rounded-full transition-all duration-500"
                        style={{
                          width: `${trainingStatus.total_models
                            ? ((trainingStatus.models_completed_count || 0) / trainingStatus.total_models) * 100
                            : 0}%`,
                        }}
                      />
                    </div>
                  </div>
                )}

                {/* Current Model Trial Progress */}
                {trainingStatus?.current_model && trainingStatus.phase === 'training' && (
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span className="font-medium">
                        {trainingStatus.current_model.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                      <span>
                        Trial {trainingStatus.current_trial || 0} / {trainingStatus.total_trials || 0}
                      </span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{
                          width: `${trainingStatus.total_trials
                            ? ((trainingStatus.current_trial || 0) / trainingStatus.total_trials) * 100
                            : 0}%`,
                        }}
                      />
                    </div>
                  </div>
                )}

                {/* Per-Model Status Grid */}
                <div className="space-y-2">
                  <div className="text-sm font-medium">Models:</div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {(trainingStatus?.models || trainingLogs.map(l => ({
                      model_name: l.id,
                      status: 'pending',
                      current_trial: 0,
                      total_trials: formData.config.n_trials,
                      elapsed_seconds: 0,
                      best_score: null,
                    }))).map((model) => {
                      const displayName = (availableModels[modelKey]?.find(m => m.id === model.model_name)?.name)
                        || model.model_name?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
                        || 'Unknown';

                      const statusConfig = {
                        pending: { dot: 'bg-muted-foreground/30', text: 'text-muted-foreground' },
                        training: { dot: 'bg-primary animate-pulse', text: 'text-primary font-medium' },
                        completed: { dot: 'bg-green-500', text: 'text-green-600' },
                        skipped: { dot: 'bg-yellow-500', text: 'text-yellow-600' },
                        cancelled: { dot: 'bg-red-400', text: 'text-red-500' },
                        failed: { dot: 'bg-red-500', text: 'text-red-600' },
                      };
                      const cfg = statusConfig[model.status] || statusConfig.pending;

                      return (
                        <div
                          key={model.model_name}
                          className={`flex items-center justify-between p-2.5 border rounded-lg ${
                            model.status === 'training' ? 'bg-primary/5 border-primary/30' : 'bg-muted/20'
                          }`}
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            {model.status === 'completed' ? (
                              <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                            ) : model.status === 'skipped' ? (
                              <SkipForward className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                            ) : model.status === 'failed' || model.status === 'cancelled' ? (
                              <X className="w-4 h-4 text-red-500 flex-shrink-0" />
                            ) : (
                              <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${cfg.dot}`} />
                            )}
                            <span className={`text-sm truncate ${cfg.text}`}>{displayName}</span>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            {model.status === 'training' && (
                              <span className="text-xs text-muted-foreground font-mono">
                                {model.current_trial}/{model.total_trials}
                              </span>
                            )}
                            {model.best_score !== null && model.best_score !== undefined && model.best_score > -9999 && (
                              <Badge variant="outline" className="text-xs">
                                {model.best_score.toFixed(3)}
                              </Badge>
                            )}
                            {model.elapsed_seconds > 0 && (
                              <span className="text-xs text-muted-foreground font-mono">
                                {formatElapsed(model.elapsed_seconds)}
                              </span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Training Logs (collapsible) */}
                {trainingStatus?.logs?.length > 0 && (
                  <details className="text-xs">
                    <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                      Training Logs ({trainingStatus.logs.length})
                    </summary>
                    <div className="mt-2 max-h-40 overflow-y-auto bg-muted/30 rounded-lg p-2 font-mono space-y-0.5">
                      {trainingStatus.logs.map((log, i) => (
                        <div key={i} className="text-muted-foreground">{log}</div>
                      ))}
                    </div>
                  </details>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Navigation Buttons */}
      <div className="flex items-center justify-between pt-6 border-t">
        <Button
          variant="outline"
          onClick={() => setStep(step - 1)}
          disabled={step === 1 || training}
        >
          <ChevronLeft className="h-4 w-4 mr-2" />
          Previous
        </Button>

        {step < 4 ? (
          <Button
            onClick={() => setStep(step + 1)}
            disabled={step === 1 && !canProceedStep1}
          >
            Next
            <ChevronRight className="h-4 w-4 ml-2" />
          </Button>
        ) : (
          <Button onClick={handleTrainModel} disabled={training} size="lg">
            {training ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Training...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Start Training
              </>
            )}
          </Button>
        )}
      </div>
    </div>
  );
}
