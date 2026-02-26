/**
 * AutoML Models List Page
 * Displays all trained AutoML models with generalization scores
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Plus, 
  Search, 
  Filter, 
  Bot, 
  Calendar,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  XCircle,
  Loader2,
  Sparkles,
  BarChart3,
  LayoutGrid,
  List,
  ChevronLeft,
  ChevronRight,
  Trash2
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import { Alert, AlertDescription } from '../components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table';
import { Checkbox } from '../components/ui/checkbox';
import { automlAPI } from '../utils/api';

export default function AutoMLModels() {
  const navigate = useNavigate();
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterTaskType, setFilterTaskType] = useState('all');
  const [viewMode, setViewMode] = useState(() => {
    // Load view preference from localStorage
    return localStorage.getItem('automl-view-mode') || 'cards';
  });
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 8;
  const [selectedModels, setSelectedModels] = useState([]);
  const [deleting, setDeleting] = useState(false);

  // TODO: Re-enable in v2 after full validation
  // Feature flags fetched from backend
  const [featureFlags, setFeatureFlags] = useState({
    enable_unsupervised: false,
    enable_text_processing: false,
  });

  // Save view mode preference
  useEffect(() => {
    localStorage.setItem('automl-view-mode', viewMode);
  }, [viewMode]);

  useEffect(() => {
    fetchModels();
    // Fetch feature flags
    automlAPI.getFeatureFlags().then(flags => {
      if (flags) setFeatureFlags(flags);
    }).catch(() => {});
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await automlAPI.listModels();
      setModels(response);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch AutoML models');
      console.error('Error fetching models:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredModels = models.filter(model => {
    const matchesSearch = model.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterTaskType === 'all' || model.task_type === filterTaskType;
    return matchesSearch && matchesFilter;
  });

  // Pagination
  const totalPages = Math.ceil(filteredModels.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedModels = filteredModels.slice(startIndex, endIndex);

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, filterTaskType]);

  const getTaskTypeBadge = (taskType) => {
    const colors = {
      classification: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
      regression: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
      clustering: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
    };
    return colors[taskType] || 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300';
  };

  const formatDuration = (seconds) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
  };

  const handleSelectAll = (checked) => {
    if (checked) {
      setSelectedModels(paginatedModels.map(m => m.id));
    } else {
      setSelectedModels([]);
    }
  };

  const handleSelectModel = (modelId, checked) => {
    if (checked) {
      setSelectedModels(prev => [...prev, modelId]);
    } else {
      setSelectedModels(prev => prev.filter(id => id !== modelId));
    }
  };

  const handleBulkDelete = async () => {
    if (selectedModels.length === 0) return;
    
    if (!confirm(`Are you sure you want to delete ${selectedModels.length} model(s)? This action cannot be undone.`)) {
      return;
    }

    try {
      setDeleting(true);
      await Promise.all(selectedModels.map(id => automlAPI.deleteModel(id)));
      setSelectedModels([]);
      await fetchModels();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to delete models');
    } finally {
      setDeleting(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">AutoML Models</h1>
            <p className="text-muted-foreground">Generalization-focused model training</p>
          </div>
        </div>
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Sparkles className="h-8 w-8 text-primary" />
            AutoML Models
          </h1>
          <p className="text-muted-foreground">
            Trained models with Optuna optimization and generalization scoring
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => navigate('/dashboard/models/test')}>
            Test Model
          </Button>
          <Button onClick={() => navigate('/dashboard/models/train')} size="lg">
            <Plus className="h-4 w-4 mr-2" />
            Train New Model
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Bulk Actions */}
      {selectedModels.length > 0 && viewMode === 'list' && (
        <Card className="bg-primary/5 border-primary/20">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-primary" />
                <span className="font-medium">
                  {selectedModels.length} model(s) selected
                </span>
              </div>
              <Button
                variant="destructive"
                size="sm"
                onClick={handleBulkDelete}
                disabled={deleting}
                className="gap-2"
              >
                {deleting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4" />
                )}
                Delete Selected
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Filters and View Toggle */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search models by name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <div className="flex gap-2">
              {/* TODO: Re-enable 'clustering' / 'unsupervised' filter in v2 after full validation */}
              {['all', 'classification', 'regression', ...(featureFlags.enable_unsupervised ? ['clustering'] : [])].map((type) => (
                <Button
                  key={type}
                  variant={filterTaskType === type ? 'default' : 'outline'}
                  onClick={() => setFilterTaskType(type)}
                  size="sm"
                >
                  {type === 'all' ? 'All' : type.charAt(0).toUpperCase() + type.slice(1)}
                </Button>
              ))}
            </div>
            <div className="flex gap-1 border rounded-md p-1">
              <Button
                variant={viewMode === 'cards' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('cards')}
                className="px-3"
              >
                <LayoutGrid className="h-4 w-4" />
              </Button>
              <Button
                variant={viewMode === 'list' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('list')}
                className="px-3"
              >
                <List className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats Cards */}
      {models.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Models
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{models.length}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Avg Generalization Score
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(models.reduce((sum, m) => sum + m.best_generalization_score, 0) / models.length).toFixed(3)}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Models Trained Today
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {models.filter(m => {
                  const modelDate = new Date(m.created_at);
                  const today = new Date();
                  return modelDate.toDateString() === today.toDateString();
                }).length}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Training Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {formatDuration(models.reduce((sum, m) => sum + m.training_duration, 0))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Models Display */}
      {filteredModels.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <Bot className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No models found</h3>
            <p className="text-muted-foreground mb-4">
              {searchQuery || filterTaskType !== 'all'
                ? 'Try adjusting your filters'
                : 'Get started by training your first AutoML model'}
            </p>
            {!searchQuery && filterTaskType === 'all' && (
              <Button onClick={() => navigate('/dashboard/models/train')}>
                <Plus className="h-4 w-4 mr-2" />
                Train Your First Model
              </Button>
            )}
          </CardContent>
        </Card>
      ) : viewMode === 'cards' ? (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {paginatedModels.map((model) => (
            <Card
              key={model.id}
              className="cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => navigate(`/dashboard/models/${model.id}`)}
            >
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-xl mb-2">{model.name}</CardTitle>
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge className={getTaskTypeBadge(model.task_type)}>
                        {model.task_type}
                      </Badge>
                      <Badge variant="outline" className="font-mono">
                        {model.best_model_name}
                      </Badge>
                    </div>
                  </div>
                  <Bot className="h-10 w-10 text-primary" />
                </div>
                {model.description && (
                  <CardDescription className="mt-2">{model.description}</CardDescription>
                )}
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Generalization Score - Highlighted */}
                <div className="bg-primary/10 dark:bg-primary/20 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-primary" />
                      <span className="text-sm font-medium">Generalization Score</span>
                    </div>
                    <div className="text-2xl font-bold text-primary">
                      {model.best_generalization_score.toFixed(4)}
                    </div>
                  </div>
                </div>

                {/* Score Details */}
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground mb-1">CV Score</div>
                    <div className="font-semibold">{model.best_cv_score.toFixed(4)}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground mb-1">Test Score</div>
                    <div className="font-semibold">{model.best_test_score.toFixed(4)}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground mb-1">Overfit Gap</div>
                    <div className={`font-semibold ${
                      model.best_overfit_gap > 0.10 
                        ? 'text-red-600 dark:text-red-400' 
                        : 'text-green-600 dark:text-green-400'
                    }`}>
                      {model.best_overfit_gap.toFixed(4)}
                    </div>
                  </div>
                </div>

                {/* Model Stats */}
                <div className="flex items-center justify-between text-sm text-muted-foreground pt-2 border-t">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1">
                      <CheckCircle className="h-3.5 w-3.5" />
                      {model.total_models_evaluated - model.models_rejected} valid
                    </div>
                    {model.models_rejected > 0 && (
                      <div className="flex items-center gap-1 text-red-600 dark:text-red-400">
                        <XCircle className="h-3.5 w-3.5" />
                        {model.models_rejected} rejected
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <Calendar className="h-3.5 w-3.5" />
                    {formatDate(model.created_at)}
                  </div>
                </div>

                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <BarChart3 className="h-3.5 w-3.5" />
                  Training: {formatDuration(model.training_duration)}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Pagination for Cards */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2 mt-6">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <div className="flex items-center gap-1">
              {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
                <Button
                  key={page}
                  variant={currentPage === page ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setCurrentPage(page)}
                  className="w-10"
                >
                  {page}
                </Button>
              ))}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}
      </>
      ) : (
        <>
          {/* List View */}
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">
                    <Checkbox
                      checked={selectedModels.length === paginatedModels.length && paginatedModels.length > 0}
                      onCheckedChange={handleSelectAll}
                      aria-label="Select all models"
                    />
                  </TableHead>
                  <TableHead>Model Name</TableHead>
                  <TableHead>Task Type</TableHead>
                  <TableHead>Best Model</TableHead>
                  <TableHead className="text-right">Gen. Score</TableHead>
                  <TableHead className="text-right">CV Score</TableHead>
                  <TableHead className="text-right">Test Score</TableHead>
                  <TableHead className="text-right">Overfit Gap</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {paginatedModels.map((model) => (
                  <TableRow
                    key={model.id}
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => navigate(`/dashboard/models/${model.id}`)}
                  >
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <Checkbox
                        checked={selectedModels.includes(model.id)}
                        onCheckedChange={(checked) => handleSelectModel(model.id, checked)}
                        aria-label={`Select ${model.name}`}
                      />
                    </TableCell>
                    <TableCell className="font-medium">{model.name}</TableCell>
                    <TableCell>
                      <Badge className={getTaskTypeBadge(model.task_type)}>
                        {model.task_type}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="font-mono text-xs">
                        {model.best_model_name}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right font-semibold text-primary">
                      {model.best_generalization_score.toFixed(4)}
                    </TableCell>
                    <TableCell className="text-right">
                      {model.best_cv_score.toFixed(4)}
                    </TableCell>
                    <TableCell className="text-right">
                      {model.best_test_score.toFixed(4)}
                    </TableCell>
                    <TableCell className="text-right">
                      <span className={
                        model.best_overfit_gap > 0.10 
                          ? 'text-red-600 dark:text-red-400 font-semibold' 
                          : 'text-green-600 dark:text-green-400 font-semibold'
                      }>
                        {model.best_overfit_gap.toFixed(4)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span className="text-sm">{model.total_models_evaluated - model.models_rejected}</span>
                        {model.models_rejected > 0 && (
                          <>
                            <XCircle className="h-4 w-4 text-red-500" />
                            <span className="text-sm text-red-600">{model.models_rejected}</span>
                          </>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(model.created_at)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>

          {/* Pagination for List */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <div className="flex items-center gap-1">
                {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
                  <Button
                    key={page}
                    variant={currentPage === page ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setCurrentPage(page)}
                    className="w-10"
                  >
                    {page}
                  </Button>
                ))}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
