/**
 * Test Model Page
 * Test trained models with custom input data
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Play,
  Loader2,
  Upload,
  Bot,
  CheckCircle,
  FileText,
  Download,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Textarea } from '../components/ui/textarea';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { Badge } from '../components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '../components/ui/dropdown-menu';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { automlAPI } from '../utils/api';

/**
 * Build categorical input groups using the Feature Encoding Reference
 * (encodingInfo.categorical_mappings) as the source of truth.
 *
 * Returns an array of group descriptors:
 *   { type: 'ohe',    catFeature, members: [colName,...], options: [label,...] }
 *   { type: 'label',  catFeature, options: [{value, label},...] }
 *
 * "ohe"   – the original feature was one-hot encoded into multiple columns
 *            (featureNames contains colName = `${catFeature}_${value}`).
 * "label" – the feature was label-encoded into a single numeric column.
 *
 * @param {string[]} featureNames       - actual column names the model expects
 * @param {object}   catMappings        - encodingInfo.categorical_mappings
 *                                        shape: { featureName: { encodedStr: originalLabel } }
 * @param {string[]} categoricalFeatures - original categorical column names from preprocessing metadata
 */
function buildCategoricalGroups(featureNames, catMappings, categoricalFeatures = []) {
  const groups = [];
  const featureSet = new Set(featureNames);

  // --- Pass 1: use categorical_mappings (backend-extracted encoder categories) ---
  const coveredByMappings = new Set();
  Object.entries(catMappings || {}).forEach(([catFeature, mapping]) => {
    // mapping: { "0": "<1H OCEAN", "1": "INLAND", ... }
    const labels = Object.entries(mapping)
      .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
      .map(([, label]) => label);

    // Check for OHE columns: at least one `{catFeature}_{label}` exists in featureNames
    const oheMembers = labels
      .map(label => `${catFeature}_${label}`)
      .filter(col => featureSet.has(col));

    if (oheMembers.length > 0) {
      groups.push({ type: 'ohe', catFeature, members: oheMembers, options: labels });
      oheMembers.forEach(m => coveredByMappings.add(m));
      coveredByMappings.add(catFeature);
    } else if (featureSet.has(catFeature)) {
      const options = Object.entries(mapping)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
        .map(([value, label]) => ({ value, label }));
      groups.push({ type: 'label', catFeature, options });
      coveredByMappings.add(catFeature);
    }
  });

  // --- Pass 2: infer OHE groups from feature_names pattern for columns not covered ---
  // For each original categorical feature, check if featureNames contains {catFeature}_{*} columns.
  // This handles the case where backend mappings weren't extracted (e.g. older saved models).
  categoricalFeatures.forEach(catFeature => {
    if (coveredByMappings.has(catFeature)) return; // already handled
    if (featureSet.has(catFeature)) return;         // single ordinal column, will render as input

    const prefix = `${catFeature}_`;
    const oheMembers = featureNames.filter(f => f.startsWith(prefix));
    if (oheMembers.length > 0) {
      const options = oheMembers.map(m => m.slice(prefix.length));
      groups.push({ type: 'ohe', catFeature, members: oheMembers, options });
      oheMembers.forEach(m => coveredByMappings.add(m));
      coveredByMappings.add(catFeature);
    }
  });

  // --- Pass 3: infer OHE groups from featureNames alone (no metadata) ---
  // Group `{base}_{suffix}` columns where {base} appears in 2+ feature names but not alone.
  // Only run if there are still uncovered OHE-looking columns.
  const uncovered = featureNames.filter(f => !coveredByMappings.has(f));
  const baseCount = {};
  uncovered.forEach(f => {
    const lastUnderscore = f.lastIndexOf('_');
    if (lastUnderscore > 0) {
      const base = f.slice(0, lastUnderscore);
      baseCount[base] = (baseCount[base] || 0) + 1;
    }
  });
  Object.entries(baseCount).forEach(([base, count]) => {
    if (count < 2) return; // need at least 2 columns to be OHE
    if (featureSet.has(base)) return; // base column exists → ordinal, not OHE
    if (coveredByMappings.has(base)) return;
    const prefix = `${base}_`;
    const members = uncovered.filter(f => f.startsWith(prefix));
    if (members.length >= 2) {
      const options = members.map(m => m.slice(prefix.length));
      groups.push({ type: 'ohe', catFeature: base, members, options });
      members.forEach(m => coveredByMappings.add(m));
    }
  });

  return groups;
}

export default function TestModel() {
  const navigate = useNavigate();
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [testing, setTesting] = useState(false);
  const [inputMode, setInputMode] = useState('manual'); // 'manual' or 'json'
  const [manualInputs, setManualInputs] = useState({});
  const [jsonInput, setJsonInput] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [encodingInfo, setEncodingInfo] = useState(null);
  const [loadingEncodingInfo, setLoadingEncodingInfo] = useState(false);
  const [preprocessingMetadata, setPreprocessingMetadata] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await automlAPI.listModels();
      setModels(response);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch models');
    } finally {
      setLoading(false);
    }
  };

  const handleModelSelect = async (modelId) => {
    const model = models.find(m => m.id === modelId);
    setSelectedModel(model);
    setResults(null);
    setError(null);
    setEncodingInfo(null);
    setPreprocessingMetadata(null);
    
    // Fetch encoding information
    if (model) {
      try {
        setLoadingEncodingInfo(true);
        const encodingData = await automlAPI.getEncodingInfo(model.id);
        console.log('🔍 Encoding data received:', encodingData);
        console.log('   Categorical mappings:', encodingData.categorical_mappings);
        console.log('   Number of features:', Object.keys(encodingData.categorical_mappings || {}).length);
        setEncodingInfo(encodingData);
      } catch (err) {
        console.error('❌ Failed to fetch encoding info:', err);
        // Don't show error to user, encoding info is optional
      } finally {
        setLoadingEncodingInfo(false);
      }
    }
    
    // Initialize manual inputs with sample values
    if (model && model.feature_names) {
      const initialInputs = {};
      
      // Check if model has preprocessing_metadata (for text features)
      const metadata = model.preprocessing_metadata || {};
      setPreprocessingMetadata(metadata);
      
      console.log('🔍 Model metadata:', metadata);
      console.log('   Feature names count:', model.feature_names?.length);
      
      // If text features exist, use original features
      const textFeatures = metadata.text_features || [];
      const hasTextFeatures = textFeatures.length > 0;
      
      console.log('   Text features:', textFeatures);
      console.log('   Has text features:', hasTextFeatures);
      
      let rawFeatures;
      if (hasTextFeatures) {
        // Use original features (numerical, categorical, text)
        rawFeatures = [
          ...(metadata.numerical_features || []),
          ...(metadata.categorical_features || []),
          ...textFeatures
        ];
        console.log('   Using original features:', rawFeatures);
      } else {
        // Filter out generated features (missing indicators, TF-IDF features)
        rawFeatures = model.feature_names.filter(f => 
          !f.startsWith('missingindicator_') && !f.startsWith('text_tfidf_')
        );
        console.log('   Using filtered feature names:', rawFeatures);
        console.log('   Filtered out:', model.feature_names.length - rawFeatures.length, 'features');
      }
      
      console.log('   Final raw features count:', rawFeatures.length);
      
      rawFeatures.forEach(feature => {
        // Check if this is a text feature
        if (textFeatures.includes(feature)) {
          initialInputs[feature] = 'Enter your text here...';
        }
        // Try to provide sensible default values based on feature name
        else if (feature.toLowerCase().includes('age')) {
          initialInputs[feature] = '30';
        } else if (feature.toLowerCase().includes('price') || feature.toLowerCase().includes('amount')) {
          initialInputs[feature] = '100.00';
        } else if (feature.toLowerCase().includes('rate') || feature.toLowerCase().includes('percentage')) {
          initialInputs[feature] = '0.5';
        } else if (feature.toLowerCase().includes('count') || feature.toLowerCase().includes('number')) {
          initialInputs[feature] = '5';
        } else {
          initialInputs[feature] = '0';
        }
      });

      // For categorical features (OHE or label-encoded), set sensible defaults
      // We don't have encodingInfo yet at this point (it's fetched async), so we
      // re-apply defaults after encodingInfo arrives via a useEffect (see below).
      // For now, keep raw defaults; the useEffect will fix them once encoding loads.
      setManualInputs(initialInputs);
      
      console.log('📝 Initialized manual inputs:', initialInputs);
      console.log('   Number of fields:', Object.keys(initialInputs).length);
      
      // Generate JSON preview
      setJsonInput(JSON.stringify(initialInputs, null, 2));
    }
  };

  // Once encodingInfo loads, fix categorical defaults in manualInputs
  useEffect(() => {
    if (!encodingInfo || !selectedModel) return;
    const catMappings = encodingInfo.categorical_mappings || {};
    const catFeaturesMeta = encodingInfo.categorical_features || [];

    const allFeatures = selectedModel.feature_names?.filter(f =>
      !f.startsWith('missingindicator_') && !f.startsWith('text_tfidf_')
    ) || [];

    const catGroups = buildCategoricalGroups(allFeatures, catMappings, catFeaturesMeta);
    if (catGroups.length === 0) return;

    setManualInputs(prev => {
      const updated = { ...prev };
      catGroups.forEach(group => {
        if (group.type === 'ohe') {
          // Default: first member = 1, rest = 0
          group.members.forEach((m, idx) => {
            updated[m] = idx === 0 ? '1' : '0';
          });
        } else {
          // Default: first encoded value
          updated[group.catFeature] = group.options[0]?.value ?? '0';
        }
      });
      setJsonInput(JSON.stringify(updated, null, 2));
      return updated;
    });
  }, [encodingInfo, selectedModel]);

  const handleManualInputChange = (feature, value) => {
    const updated = { ...manualInputs, [feature]: value };
    setManualInputs(updated);
    setJsonInput(JSON.stringify(updated, null, 2));
  };

  const handleTest = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    // Validate inputs
    const hasEmptyFields = Object.values(manualInputs).some(v => !v || v.trim() === '');
    if (hasEmptyFields) {
      setError('Please fill in all input fields');
      return;
    }

    try {
      setTesting(true);
      setError(null);

      // Convert string inputs to numbers where appropriate
      const parsedData = {};
      Object.entries(manualInputs).forEach(([key, value]) => {
        const numValue = parseFloat(value);
        parsedData[key] = isNaN(numValue) ? value : numValue;
      });

      // Make API call to prediction endpoint
      const result = await automlAPI.predict(selectedModel.id, parsedData);
      
      setResults({
        prediction: result.prediction,
        confidence: result.confidence ? (result.confidence * 100).toFixed(2) : null,
        probabilities: result.probabilities,
        processing_time: result.processing_time,
        input_features: result.input_features,
      });
      setTesting(false);

    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to test model');
      setTesting(false);
    }
  };

  const getTaskTypeBadge = (taskType) => {
    const colors = {
      classification: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
      regression: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
      clustering: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
    };
    return colors[taskType] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="space-y-6 w-full mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => navigate('/dashboard/models/automl')}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
              <Bot className="h-8 w-8 text-primary" />
              Test Model
            </h1>
            <p className="text-muted-foreground">
              Make predictions with your trained models
            </p>
          </div>
        </div>
        <Button onClick={() => navigate('/dashboard/models/train')}>
          Train New Model
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - Input */}
        <div className="space-y-6">
          {/* Model Selection */}
          <Card>
            <CardHeader>
              <CardTitle>Select Model</CardTitle>
              <CardDescription>
                Choose a trained model to test
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-primary" />
                </div>
              ) : models.length === 0 ? (
                <Alert>
                  <AlertDescription>
                    No trained models found. Train a model first.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-2">
                  <Label>Model</Label>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" className="w-full justify-between">
                        {selectedModel ? (
                          <span>{selectedModel.name}</span>
                        ) : (
                          <span className="text-muted-foreground">Select a model...</span>
                        )}
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="w-[500px] max-h-[400px] overflow-y-auto" align="start">
                      {models.map((model) => (
                        <DropdownMenuItem
                          key={model.id}
                          onClick={() => handleModelSelect(model.id)}
                          className="cursor-pointer p-4"
                        >
                          <div className="flex-1">
                            <div className="font-medium">{model.name}</div>
                            <div className="flex gap-2 mt-2">
                              <Badge className={getTaskTypeBadge(model.task_type)}>
                                {model.task_type}
                              </Badge>
                              <Badge variant="outline" className="font-mono text-xs">
                                {model.best_model_name}
                              </Badge>
                              <Badge variant="outline" className="text-xs">
                                Score: {model.best_generalization_score.toFixed(4)}
                              </Badge>
                            </div>
                          </div>
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              )}

              {selectedModel && (
                <div className="mt-4 p-4 border rounded-lg bg-muted/20">
                  <div className="text-sm space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Task Type:</span>
                      <Badge className={getTaskTypeBadge(selectedModel.task_type)}>
                        {selectedModel.task_type}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Best Model:</span>
                      <span className="font-mono text-xs">{selectedModel.best_model_name}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Gen. Score:</span>
                      <span className="font-semibold text-primary">{selectedModel.best_generalization_score.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Input Data */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Input Data</CardTitle>
                  <CardDescription>
                    Enter values for each feature
                  </CardDescription>
                </div>
                <div className="flex gap-1 border rounded-md p-1">
                  <Button
                    variant={inputMode === 'manual' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setInputMode('manual')}
                    className="text-xs"
                  >
                    Manual
                  </Button>
                  <Button
                    variant={inputMode === 'json' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setInputMode('json')}
                    className="text-xs"
                  >
                    JSON
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {!selectedModel ? (
                <div className="text-center py-8 text-muted-foreground">
                  <p>Select a model to enter input data</p>
                </div>
              ) : inputMode === 'manual' ? (
                <div className="space-y-3 max-h-[420px] overflow-y-auto pr-2">
                  {(() => {
                    const textFeatures = preprocessingMetadata?.text_features || [];
                    const hasTextFeatures = textFeatures.length > 0;

                    let featuresToShow;
                    if (hasTextFeatures) {
                      featuresToShow = [
                        ...(preprocessingMetadata?.numerical_features || []),
                        ...(preprocessingMetadata?.categorical_features || []),
                        ...textFeatures
                      ];
                    } else {
                      featuresToShow = selectedModel.feature_names?.filter(f =>
                        !f.startsWith('missingindicator_') && !f.startsWith('text_tfidf_')
                      ) || [];
                    }

                    // Use categorical_mappings + metadata + feature name inference for dropdown detection.
                    const catMappings = encodingInfo?.categorical_mappings || {};
                    const catFeaturesMeta = encodingInfo?.categorical_features || 
                      preprocessingMetadata?.categorical_features || [];
                    const catGroups = buildCategoricalGroups(featuresToShow, catMappings, catFeaturesMeta);

                    // Build a set of all raw column names already represented by a group
                    const catCovered = new Set();
                    catGroups.forEach(g => {
                      if (g.type === 'ohe') g.members.forEach(m => catCovered.add(m));
                      else catCovered.add(g.catFeature);
                    });

                    // Map first-appearance column → group (for insertion ordering)
                    const groupByFirstCol = new Map();
                    catGroups.forEach(g => {
                      const firstCol = g.type === 'ohe' ? g.members[0] : g.catFeature;
                      groupByFirstCol.set(firstCol, g);
                    });

                    // Walk featuresToShow to build ordered render rows
                    const rendered = new Set();
                    const rows = [];
                    featuresToShow.forEach(feature => {
                      if (rendered.has(feature)) return;
                      if (groupByFirstCol.has(feature)) {
                        const g = groupByFirstCol.get(feature);
                        if (g.type === 'ohe') g.members.forEach(m => rendered.add(m));
                        else rendered.add(g.catFeature);
                        rows.push({ type: 'cat', group: g });
                      } else if (!catCovered.has(feature)) {
                        rendered.add(feature);
                        rows.push({ type: 'single', feature });
                      }
                    });

                    return rows.map((row) => {
                      // ── Categorical dropdown (OHE or label-encoded) ──────────
                      if (row.type === 'cat') {
                        const { group } = row;
                        const { catFeature } = group;

                        if (group.type === 'ohe') {
                          // Active label = the member col currently set to '1'
                          const activeCol = group.members.find(m => String(manualInputs[m]) === '1') || '';
                          const activeLabel = activeCol ? activeCol.slice(catFeature.length + 1) : '';
                          return (
                            <div key={catFeature} className="space-y-1">
                              <Label className="text-sm flex items-center gap-2">
                                {catFeature}
                                <Badge variant="outline" className="text-xs bg-violet-50 text-violet-700 border-violet-200">
                                  Categorical
                                </Badge>
                              </Label>
                              <Select
                                value={activeLabel}
                                onValueChange={(val) => {
                                  const updated = { ...manualInputs };
                                  group.members.forEach(m => {
                                    updated[m] = m === `${catFeature}_${val}` ? '1' : '0';
                                  });
                                  setManualInputs(updated);
                                  setJsonInput(JSON.stringify(updated, null, 2));
                                }}
                              >
                                <SelectTrigger className="font-mono">
                                  <SelectValue placeholder={`Select ${catFeature}`} />
                                </SelectTrigger>
                                <SelectContent>
                                  {group.options.map(label => (
                                    <SelectItem key={label} value={label}>{label}</SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                          );
                        }

                        // Label-encoded single column
                        const currentEncoded = String(manualInputs[catFeature] ?? '');
                        const currentLabel = (group.options.find(o => o.value === currentEncoded) || group.options[0])?.label || '';
                        return (
                          <div key={catFeature} className="space-y-1">
                            <Label className="text-sm flex items-center gap-2">
                              {catFeature}
                              <Badge variant="outline" className="text-xs bg-violet-50 text-violet-700 border-violet-200">
                                Categorical
                              </Badge>
                            </Label>
                            <Select
                              value={currentEncoded}
                              onValueChange={(val) => {
                                const updated = { ...manualInputs, [catFeature]: val };
                                setManualInputs(updated);
                                setJsonInput(JSON.stringify(updated, null, 2));
                              }}
                            >
                              <SelectTrigger className="font-mono">
                                <SelectValue placeholder={`Select ${catFeature}`}>
                                  {currentLabel}
                                </SelectValue>
                              </SelectTrigger>
                              <SelectContent>
                                {group.options.map(({ value, label }) => (
                                  <SelectItem key={value} value={value}>{label}</SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                        );
                      }

                      // ── Regular numeric / text input ─────────────────────────
                      const { feature } = row;
                      const isTextFeature = textFeatures.includes(feature);
                      return (
                        <div key={feature} className="space-y-1">
                          <Label className="text-sm flex items-center gap-2">
                            {feature}
                            {isTextFeature && (
                              <Badge variant="outline" className="text-xs bg-emerald-50 text-emerald-700 border-emerald-200">
                                Text
                              </Badge>
                            )}
                          </Label>
                          {isTextFeature ? (
                            <Textarea
                              value={manualInputs[feature] || ''}
                              onChange={(e) => handleManualInputChange(feature, e.target.value)}
                              placeholder={`Enter text for ${feature}`}
                              rows={4}
                              className="font-mono text-sm"
                            />
                          ) : (
                            <Input
                              type="text"
                              value={manualInputs[feature] || ''}
                              onChange={(e) => handleManualInputChange(feature, e.target.value)}
                              placeholder={`Enter ${feature}`}
                              className="font-mono"
                            />
                          )}
                        </div>
                      );
                    });
                  })()}
                </div>
              ) : (
                <div className="space-y-2">
                  <Label>JSON Format</Label>
                  <Textarea
                    value={jsonInput}
                    onChange={(e) => {
                      setJsonInput(e.target.value);
                      try {
                        const parsed = JSON.parse(e.target.value);
                        setManualInputs(parsed);
                      } catch {}
                    }}
                    rows={12}
                    className="font-mono text-sm"
                  />
                </div>
              )}

              <Button 
                onClick={handleTest} 
                disabled={!selectedModel || testing}
                className="w-full"
                size="lg"
              >
                {testing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Testing...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Run Prediction
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

        </div>

        {/* Right Column - Results */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Prediction Results</CardTitle>
              <CardDescription>
                Model predictions and confidence scores
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!results ? (
                <div className="text-center py-12 text-muted-foreground">
                  <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No predictions yet</p>
                  <p className="text-sm mt-2">Select a model and provide input data to get started</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Prediction Result */}
                  <div className="bg-primary/10 rounded-lg p-6 text-center">
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <CheckCircle className="h-5 w-5 text-primary" />
                      <span className="text-sm text-muted-foreground">Prediction</span>
                    </div>
                    <div className="text-3xl font-bold text-primary">{results.prediction}</div>
                    {results.confidence && (
                      <div className="text-sm text-muted-foreground mt-2">
                        Confidence: {results.confidence}%
                      </div>
                    )}
                  </div>

                  {/* Class Probabilities */}
                  {results.probabilities && (
                    <div className="border rounded-lg p-4">
                      <h4 className="font-semibold mb-3 text-sm">Class Probabilities</h4>
                      <div className="space-y-2">
                        {Object.entries(results.probabilities)
                          .sort((a, b) => b[1] - a[1])
                          .map(([className, probability]) => (
                            <div key={className} className="flex items-center justify-between">
                              <span className="text-sm">{className}</span>
                              <div className="flex items-center gap-2 w-2/3">
                                <div className="flex-1 bg-muted rounded-full h-2">
                                  <div 
                                    className="bg-primary h-2 rounded-full" 
                                    style={{ width: `${(probability * 100).toFixed(1)}%` }}
                                  />
                                </div>
                                <span className="text-xs text-muted-foreground w-12 text-right">
                                  {(probability * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}

                  {/* Details */}
                  <div className="border rounded-lg">
                    <Table>
                      <TableBody>
                        <TableRow>
                          <TableCell className="font-medium">Processing Time</TableCell>
                          <TableCell className="text-right">{results.processing_time}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Input Features</TableCell>
                          <TableCell className="text-right">{results.input_features}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Model</TableCell>
                          <TableCell className="text-right font-mono text-sm">{selectedModel?.best_model_name}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Task Type</TableCell>
                          <TableCell className="text-right">
                            <Badge className={getTaskTypeBadge(selectedModel?.task_type)}>
                              {selectedModel?.task_type}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </div>

                  <Alert>
                    <AlertTitle>Note</AlertTitle>
                    <AlertDescription>
                      This is a single prediction. For batch predictions, use the model download feature and run predictions locally.
                    </AlertDescription>
                  </Alert>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Encoding Reference - below Prediction Results */}
          {encodingInfo && Object.keys(encodingInfo.categorical_mappings || {}).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Feature Encoding Reference</CardTitle>
                <CardDescription>
                  Categorical values and their encoded numbers
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2">
                  {Object.entries(encodingInfo.categorical_mappings).map(([feature, mapping]) => (
                    <div key={feature} className="border rounded-lg p-3">
                      <div className="font-medium text-sm mb-2 text-primary">{feature}</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {Object.entries(mapping)
                          .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
                          .map(([encodedValue, originalValue]) => (
                            <div key={encodedValue} className="flex items-center gap-2 py-1">
                              <Badge variant="outline" className="font-mono text-xs min-w-[30px] justify-center">
                                {encodedValue}
                              </Badge>
                              <span className="text-muted-foreground truncate">
                                {originalValue}
                              </span>
                            </div>
                          ))}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-3 text-xs text-muted-foreground">
                  💡 Use these encoded values when entering data above
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
