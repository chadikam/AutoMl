/**
 * AutoML Model Detail Page
 * Display complete training results, plots, and model comparison
 */
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Download,
  Package,
  Trash2,
  Trophy,
  TrendingUp,
  TrendingDown,
  Layers,
  CheckCircle,
  AlertTriangle,
  Loader2,
  BarChart3,
  Target,
  Settings,
  Clock,
  Database,
  Zap,
  Award,
  Sparkles,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { Separator } from '../components/ui/separator';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import { automlAPI } from '../utils/api';

export default function AutoMLModelDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    fetchModelDetails();
  }, [id]);

  const fetchModelDetails = async () => {
    try {
      setLoading(true);
      const model = await automlAPI.getModel(id);
      setModel(model);
    } catch (err) {
      setError('Failed to load model details');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this model?')) return;

    try {
      setDeleting(true);
      await automlAPI.deleteModel(id);
      navigate('/dashboard/models/automl');
    } catch (err) {
      alert('Failed to delete model');
      setDeleting(false);
    }
  };

  const handleDownload = async () => {
    try {
      const blob = await automlAPI.downloadModel(id);
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${model.name.replace(/\s+/g, '_')}_model.pkl`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      alert('Failed to download model');
    }
  };

  const [exporting, setExporting] = useState(false);

  const handleExportPackage = async () => {
    setExporting(true);
    try {
      const blob = await automlAPI.exportModel(id);
      const url = window.URL.createObjectURL(new Blob([blob], { type: 'application/zip' }));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${model.name.replace(/\s+/g, '_')}_deployment_package.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert('Failed to export package. Please try again.');
    } finally {
      setExporting(false);
    }
  };

  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getOverfitBadge = (gap) => {
    if (gap > 0.20) {
      return <Badge variant="destructive">Severe Overfit</Badge>;
    } else if (gap > 0.10) {
      return <Badge className="bg-orange-500">High Overfit</Badge>;
    } else if (gap > 0.05) {
      return <Badge className="bg-yellow-500">Moderate Overfit</Badge>;
    } else {
      return <Badge className="bg-green-500">Good Generalization</Badge>;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center space-y-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto" />
          <p className="text-muted-foreground">Loading model details...</p>
        </div>
      </div>
    );
  }

  if (error || !model) {
    return (
      <Alert variant="destructive" className="max-w-2xl mx-auto mt-8">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error || 'Model not found'}</AlertDescription>
      </Alert>
    );
  }

  // API returns flat structure, not nested
  const isUnsupervised = model.task_type === 'unsupervised';
  const bestModel = isUnsupervised ? {
    model_name: model.best_model_name,
    model_type: model.best_model_type,
    primary_score: model.best_generalization_score ?? model.best_primary_score ?? 0,
    best_params: model.best_params,
    training_time: model.all_models?.find(m => m.model_name === model.best_model_name)?.optimization_time || 0,
  } : {
    model_name: model.best_model_name,
    model_type: model.best_model_type,
    generalization_score: model.best_generalization_score,
    cv_score: model.best_cv_score,
    test_score: model.best_test_score,
    overfit_gap: model.best_overfit_gap,
    best_params: model.best_params,
    training_time: model.all_models?.find(m => m.model_name === model.best_model_name)?.optimization_time || 0,
  };
  const allModels = model.all_models || [];
  
  // Default config values if not provided
  const config = model.config || {
    n_trials: 75,
    cv_folds: 5,
    penalty_factor: 2.0,
    overfit_threshold_reject: 0.20,
    overfit_threshold_high: 0.10,
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-4">
          <Button variant="ghost" size="icon" onClick={() => navigate('/dashboard/models/automl')}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold tracking-tight">{model.name}</h1>
              <Badge>{model.task_type}</Badge>
              {model.status === 'completed' && (
                <Badge variant="outline" className="bg-green-500/10 text-green-700">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Completed
                </Badge>
              )}
            </div>
            {model.description && (
              <p className="text-muted-foreground">{model.description}</p>
            )}
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                {formatDate(model.created_at)}
              </span>
              <span className="flex items-center gap-1">
                <Database className="h-4 w-4" />
                {model.dataset_name}
              </span>
              {model.target_column && (
                <span className="flex items-center gap-1">
                  <Target className="h-4 w-4" />
                  {model.target_column}
                </span>
              )}
              {model.unsupervised_subtype && (
                <Badge variant="outline">
                  {model.unsupervised_subtype.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </Badge>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={handleDownload}>
            <Download className="h-4 w-4 mr-2" />
            Download .pkl
          </Button>
          <Button variant="outline" onClick={handleExportPackage} disabled={exporting}>
            {exporting ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Package className="h-4 w-4 mr-2" />
            )}
            {exporting ? 'Building...' : 'Export Package'}
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={deleting}
          >
            {deleting ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Trash2 className="h-4 w-4 mr-2" />
            )}
            Delete
          </Button>
        </div>
      </div>

      {/* Best Model Highlight */}
      <Card className="border-primary shadow-lg">
        <CardHeader className="bg-primary/5">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary rounded-lg">
                <Trophy className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <CardTitle className="text-2xl">{bestModel.model_name}</CardTitle>
                <CardDescription>
                  {isUnsupervised
                    ? 'Best unsupervised model by primary score'
                    : 'Selected for best generalization performance'}
                </CardDescription>
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-primary">
                {(isUnsupervised ? bestModel.primary_score : bestModel.generalization_score)?.toFixed(4) ?? 'N/A'}
              </div>
              <div className="text-sm text-muted-foreground">
                {isUnsupervised ? 'Primary Score' : 'Generalization Score'}
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-6">
          {isUnsupervised ? (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
              {(() => {
                const bestModelData = allModels.find(m => m.model_name === bestModel.model_name) || {};
                const metrics = bestModelData.detailed_metrics || {};
                return (
                  <>
                    {metrics.silhouette_score != null && (
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Silhouette Score</div>
                        <div className="text-2xl font-semibold">{metrics.silhouette_score.toFixed(4)}</div>
                      </div>
                    )}
                    {metrics.davies_bouldin_score != null && (
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Davies-Bouldin</div>
                        <div className="text-2xl font-semibold">{metrics.davies_bouldin_score.toFixed(4)}</div>
                      </div>
                    )}
                    {metrics.calinski_harabasz_score != null && (
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Calinski-Harabasz</div>
                        <div className="text-2xl font-semibold">{metrics.calinski_harabasz_score.toFixed(2)}</div>
                      </div>
                    )}
                    {metrics.total_explained_variance != null && (
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Explained Variance</div>
                        <div className="text-2xl font-semibold">{(metrics.total_explained_variance * 100).toFixed(1)}%</div>
                      </div>
                    )}
                    {metrics.n_components != null && (
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Components</div>
                        <div className="text-2xl font-semibold">{metrics.n_components}</div>
                      </div>
                    )}
                    {metrics.outlier_ratio != null && (
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Outlier Ratio</div>
                        <div className="text-2xl font-semibold">{(metrics.outlier_ratio * 100).toFixed(1)}%</div>
                      </div>
                    )}
                    {metrics.n_outliers != null && (
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Outliers Found</div>
                        <div className="text-2xl font-semibold">{metrics.n_outliers}</div>
                      </div>
                    )}
                    <div>
                      <div className="text-sm text-muted-foreground mb-1">Training Time</div>
                      <div className="text-2xl font-semibold flex items-center gap-2">
                        <Clock className="h-5 w-5" />
                        {formatDuration(bestModel.training_time)}
                      </div>
                    </div>
                  </>
                );
              })()}
            </div>
          ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div>
              <div className="text-sm text-muted-foreground mb-1">CV Score</div>
              <div className="text-2xl font-semibold flex items-center gap-2">
                {bestModel.cv_score.toFixed(4)}
                <TrendingUp className="h-4 w-4 text-green-500" />
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">Test Score</div>
              <div className="text-2xl font-semibold">{bestModel.test_score.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">Overfit Gap</div>
              <div className="text-2xl font-semibold flex items-center gap-2">
                {bestModel.overfit_gap.toFixed(4)}
                {getOverfitBadge(bestModel.overfit_gap)}
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">Training Time</div>
              <div className="text-2xl font-semibold flex items-center gap-2">
                <Clock className="h-5 w-5" />
                {formatDuration(bestModel.training_time)}
              </div>
            </div>
          </div>
          )}
        </CardContent>
      </Card>

      {/* Selection Explanation Alert */}
      {isUnsupervised ? (
        <Alert>
          <Sparkles className="h-4 w-4" />
          <AlertTitle>Why This Model Was Selected</AlertTitle>
          <AlertDescription>
            <p>
              This model was selected based on its <strong>primary unsupervised score</strong>.
              {model.unsupervised_subtype === 'clustering' && ' For clustering, the silhouette score measures how well clusters are separated.'}
              {model.unsupervised_subtype === 'dimensionality_reduction' && ' For dimensionality reduction, explained variance measures how much information is preserved.'}
              {model.unsupervised_subtype === 'anomaly_detection' && ' For anomaly detection, separation quality between inliers and outliers is evaluated.'}
            </p>
          </AlertDescription>
        </Alert>
      ) : (
      <Alert>
        <Sparkles className="h-4 w-4" />
        <AlertTitle>Why This Model Was Selected</AlertTitle>
        <AlertDescription>
          <p className="mb-2">
            This model was selected based on its <strong>generalization score</strong>, not raw test performance. 
            The generalization score is calculated as: <code className="bg-muted px-1 py-0.5 rounded">cv_score - (overfit_gap × penalty_factor)</code>
          </p>
          <p className="text-sm">
            Penalty Factor: {config.penalty_factor.toFixed(1)} | 
            Overfit Gap: {bestModel.overfit_gap.toFixed(4)} | 
            Penalty Applied: {(bestModel.overfit_gap * config.penalty_factor).toFixed(4)}
          </p>
        </AlertDescription>
      </Alert>
      )}

      <Tabs defaultValue="plots" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="plots">
            <BarChart3 className="h-4 w-4 mr-2" />
            Plots
          </TabsTrigger>
          <TabsTrigger value="models">
            <Layers className="h-4 w-4 mr-2" />
            All Models
          </TabsTrigger>
          <TabsTrigger value="hyperparameters">
            <Settings className="h-4 w-4 mr-2" />
            Hyperparameters
          </TabsTrigger>
          <TabsTrigger value="config">
            <Zap className="h-4 w-4 mr-2" />
            Configuration
          </TabsTrigger>
        </TabsList>

        {/* Plots Tab */}
        <TabsContent value="plots" className="space-y-6">
          {model.plot_paths && Object.keys(model.plot_paths).length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(model.plot_paths).map(([plotName, plotPath]) => (
                <Card key={plotName}>
                  <CardHeader>
                    <CardTitle className="text-lg capitalize">
                      {plotName.replace(/_/g, ' ')}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <img
                      src={plotPath.startsWith('http') ? plotPath : `/api/automl/plots/${plotPath}`}
                      alt={plotName}
                      className="w-full rounded-lg border"
                    />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>No plots available for this model.</AlertDescription>
            </Alert>
          )}
        </TabsContent>

        {/* All Models Tab */}
        <TabsContent value="models" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Model Comparison</CardTitle>
              <CardDescription>
                All {allModels.length} models trained with their generalization scores
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[50px]">Rank</TableHead>
                      <TableHead>Model</TableHead>
                      {isUnsupervised ? (
                        <>
                          <TableHead className="text-right">Primary Score</TableHead>
                          <TableHead className="text-right">Trials</TableHead>
                          <TableHead className="text-right">Training Time</TableHead>
                          <TableHead>Status</TableHead>
                        </>
                      ) : (
                        <>
                          <TableHead className="text-right">Gen. Score</TableHead>
                          <TableHead className="text-right">CV Score</TableHead>
                          <TableHead className="text-right">Test Score</TableHead>
                          <TableHead className="text-right">Overfit Gap</TableHead>
                          <TableHead className="text-right">Training Time</TableHead>
                          <TableHead>Status</TableHead>
                        </>
                      )}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {allModels
                      .sort((a, b) => {
                        if (isUnsupervised) {
                          return (b.primary_score ?? b.generalization_score ?? 0) - (a.primary_score ?? a.generalization_score ?? 0);
                        }
                        return b.generalization_score - a.generalization_score;
                      })
                      .map((m, index) => (
                        <TableRow
                          key={m.model_name}
                          className={m.model_name === bestModel.model_name ? 'bg-primary/5' : ''}
                        >
                          <TableCell className="font-medium">
                            {index === 0 && (
                              <Award className="h-5 w-5 text-primary inline" />
                            )}
                            {index > 0 && index + 1}
                          </TableCell>
                          <TableCell className="font-medium">
                            {m.model_name}
                            {m.model_name === bestModel.model_name && (
                              <Badge variant="outline" className="ml-2">
                                Selected
                              </Badge>
                            )}
                          </TableCell>
                          {isUnsupervised ? (
                            <>
                              <TableCell className="text-right font-semibold text-primary">
                                {(m.primary_score ?? m.generalization_score ?? 0).toFixed(4)}
                              </TableCell>
                              <TableCell className="text-right">
                                {m.n_trials ?? '-'}
                              </TableCell>
                              <TableCell className="text-right">
                                {formatDuration(m.optimization_time)}
                              </TableCell>
                              <TableCell>
                                {m.rejected ? (
                                  <Badge variant="destructive">Rejected</Badge>
                                ) : (
                                  <Badge variant="outline" className="bg-green-500/10">
                                    Valid
                                  </Badge>
                                )}
                              </TableCell>
                            </>
                          ) : (
                            <>
                          <TableCell className="text-right font-semibold text-primary">
                            {m.generalization_score.toFixed(4)}
                          </TableCell>
                          <TableCell className="text-right">{m.cv_score.toFixed(4)}</TableCell>
                          <TableCell className="text-right">{m.test_score.toFixed(4)}</TableCell>
                          <TableCell className="text-right">
                            <div className="flex items-center justify-end gap-2">
                              {m.overfit_gap.toFixed(4)}
                              {m.overfit_gap > 0.10 ? (
                                <TrendingUp className="h-4 w-4 text-orange-500" />
                              ) : (
                                <TrendingDown className="h-4 w-4 text-green-500" />
                              )}
                            </div>
                          </TableCell>
                          <TableCell className="text-right">
                            {formatDuration(m.optimization_time)}
                          </TableCell>
                          <TableCell>
                            {m.overfit_gap > config.overfit_threshold_reject ? (
                              <Badge variant="destructive">Rejected</Badge>
                            ) : (
                              <Badge variant="outline" className="bg-green-500/10">
                                Valid
                              </Badge>
                            )}
                          </TableCell>
                            </>
                          )}
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          {/* Training Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <Layers className="h-8 w-8 mx-auto mb-2 text-primary" />
                  <div className="text-2xl font-bold">{allModels.length}</div>
                  <div className="text-sm text-muted-foreground">Models Trained</div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-500" />
                  <div className="text-2xl font-bold">
                    {isUnsupervised
                      ? allModels.filter(m => !m.rejected).length
                      : allModels.filter(m => m.overfit_gap <= config.overfit_threshold_reject).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Passed</div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <AlertTriangle className="h-8 w-8 mx-auto mb-2 text-red-500" />
                  <div className="text-2xl font-bold">
                    {isUnsupervised
                      ? allModels.filter(m => m.rejected).length
                      : allModels.filter(m => m.overfit_gap > config.overfit_threshold_reject).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Rejected</div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <Clock className="h-8 w-8 mx-auto mb-2 text-blue-500" />
                  <div className="text-2xl font-bold">
                    {formatDuration(model.training_duration)}
                  </div>
                  <div className="text-sm text-muted-foreground">Total Time</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Hyperparameters Tab */}
        <TabsContent value="hyperparameters" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Best Model Hyperparameters</CardTitle>
              <CardDescription>
                Optimized parameters found by Optuna for {bestModel.model_name}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {bestModel.best_params && Object.keys(bestModel.best_params).length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(bestModel.best_params).map(([param, value]) => (
                    <div key={param} className="border rounded-lg p-4">
                      <div className="text-sm text-muted-foreground mb-1 font-mono">{param}</div>
                      <div className="font-semibold">
                        {typeof value === 'number' ? value.toFixed(6) : String(value)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <Alert>
                  <AlertDescription>No hyperparameters available.</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Optuna Study Info */}
          <Card>
            <CardHeader>
              <CardTitle>Optimization Details</CardTitle>
              <CardDescription>Optuna hyperparameter optimization summary</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Trials per Model</div>
                    <div className="text-xl font-semibold">{config.n_trials}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">CV Folds</div>
                    <div className="text-xl font-semibold">{config.cv_folds}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">From Preprocessing</div>
                    <div className="text-xl font-semibold">
                      Pre-split
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Configuration Tab */}
        <TabsContent value="config" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Training Configuration</CardTitle>
              <CardDescription>Complete configuration used for this training run</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="font-semibold mb-3">Overfitting Control</h3>
                {isUnsupervised ? (
                  <Alert>
                    <AlertDescription>
                      Overfitting control is not applicable for unsupervised learning tasks.
                    </AlertDescription>
                  </Alert>
                ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="border rounded-lg p-3">
                    <div className="text-sm text-muted-foreground mb-1">Penalty Factor</div>
                    <div className="text-xl font-semibold">{config.penalty_factor}</div>
                  </div>
                  <div className="border rounded-lg p-3">
                    <div className="text-sm text-muted-foreground mb-1">Rejection Threshold</div>
                    <div className="text-xl font-semibold">
                      {config.overfit_threshold_reject}
                    </div>
                  </div>
                  <div className="border rounded-lg p-3">
                    <div className="text-sm text-muted-foreground mb-1">High Overfit Threshold</div>
                    <div className="text-xl font-semibold">
                      {config.overfit_threshold_high}
                    </div>
                  </div>
                </div>
                )}
              </div>

              <Separator />

              <div>
                <h3 className="font-semibold mb-3">Preprocessing</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between border rounded-lg p-3">
                    <span className="text-sm">Adaptive Preprocessing</span>
                    <Badge variant={model.preprocessing_config?.use_adaptive ? 'default' : 'outline'}>
                      {model.preprocessing_config?.use_adaptive ? 'Enabled' : 'Disabled'}
                    </Badge>
                  </div>
                  {model.preprocessing_config?.use_adaptive && (
                    <div className="flex items-center justify-between border rounded-lg p-3">
                      <span className="text-sm">EDA Insights</span>
                      <Badge variant={model.preprocessing_config?.use_eda_insights ? 'default' : 'outline'}>
                        {model.preprocessing_config?.use_eda_insights ? 'Enabled' : 'Disabled'}
                      </Badge>
                    </div>
                  )}
                </div>
              </div>

              <Separator />

              <div>
                <h3 className="font-semibold mb-3">Dataset Information</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="border rounded-lg p-3">
                    <div className="text-sm text-muted-foreground mb-1">Dataset</div>
                    <div className="font-medium">{model.dataset_name}</div>
                  </div>
                  {model.target_column && (
                    <div className="border rounded-lg p-3">
                      <div className="text-sm text-muted-foreground mb-1">Target Column</div>
                      <div className="font-mono text-sm">{model.target_column}</div>
                    </div>
                  )}
                  <div className="border rounded-lg p-3">
                    <div className="text-sm text-muted-foreground mb-1">Task Type</div>
                    <Badge>{model.task_type}</Badge>
                    {model.unsupervised_subtype && (
                      <Badge variant="outline" className="ml-2">{model.unsupervised_subtype}</Badge>
                    )}
                  </div>
                  <div className="border rounded-lg p-3">
                    <div className="text-sm text-muted-foreground mb-1">Dataset ID</div>
                    <div className="font-mono text-xs truncate">{model.dataset_id}</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
