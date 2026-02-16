/**
 * Dataset upload page with EDA
 */
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { datasetsAPI } from '../utils/api';
import EDAProgressLog from '../components/ui/eda-progress-log';
import EDASummary from '../components/ui/eda-summary';
import EDACharts from '../components/ui/eda-charts';
import { Check, X, Search, Columns3 } from 'lucide-react';

const DatasetUpload = () => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
  });
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadPhase, setUploadPhase] = useState('idle'); // 'idle', 'uploading', 'selecting', 'analyzing', 'complete'
  const [edaSteps, setEdaSteps] = useState([]);
  const [edaResults, setEdaResults] = useState(null);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [activeResultsTab, setActiveResultsTab] = useState('summary');
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [columnSearch, setColumnSearch] = useState('');
  const [uploadedDatasetId, setUploadedDatasetId] = useState(null);
  const [uploadedDatasetInfo, setUploadedDatasetInfo] = useState(null); // Store initial dataset info
  const [savingColumns, setSavingColumns] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.name.endsWith('.csv')) {
      setFile(selectedFile);
      setError('');
    } else {
      setError('Please select a valid CSV file');
      setFile(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setUploadPhase('uploading');
    setUploadComplete(false);
    setEdaResults(null);

    try {
      // Step 1: Upload file (without EDA)
      const data = await datasetsAPI.upload(file, formData.name, formData.description);
      
      setUploadedDatasetId(data.id);
      setUploadedDatasetInfo(data);
      
      // Initialize all columns as selected by default
      if (data?.column_names) {
        setSelectedColumns(data.column_names);
      }
      
      // Step 2: Show column selector inline
      setUploadPhase('selecting');
      
    } catch (error) {
      console.error('Upload error:', error);
      setError(error.response?.data?.detail || error.message || 'Failed to upload dataset');
      setUploadPhase('idle');
    } finally {
      setLoading(false);
    }
  };

  const handleToggleColumn = (columnName) => {
    setSelectedColumns(prev => {
      if (prev.includes(columnName)) {
        return prev.filter(col => col !== columnName);
      } else {
        return [...prev, columnName];
      }
    });
  };

  const handleSelectAll = () => {
    if (uploadedDatasetInfo?.column_names) {
      setSelectedColumns(uploadedDatasetInfo.column_names);
    }
  };

  const handleDeselectAll = () => {
    setSelectedColumns([]);
  };

  const handleSaveColumnSelection = async () => {
    if (selectedColumns.length === 0) {
      alert('Please select at least one column');
      return;
    }

    setSavingColumns(true);
    setUploadPhase('analyzing');
    
    try {
      // Step 3: Run EDA on selected columns
      await runEDAWithProgress(selectedColumns);
    } catch (error) {
      console.error('Failed to run EDA:', error);
      setError('Failed to analyze dataset. Please try again.');
      setUploadPhase('idle');
    } finally {
      setSavingColumns(false);
    }
  };

  const handleUseAllColumns = async () => {
    // Save all columns as selected
    setSavingColumns(true);
    setUploadPhase('analyzing');
    
    try {
      // Use all columns from uploaded dataset
      const allColumns = uploadedDatasetInfo?.column_names || [];
      setSelectedColumns(allColumns);
      
      // Step 3: Run EDA on all columns
      await runEDAWithProgress(allColumns);
    } catch (error) {
      console.error('Failed to run EDA:', error);
      setError('Failed to analyze dataset. Please try again.');
      setUploadPhase('idle');
    } finally {
      setSavingColumns(false);
    }
  };

  const runEDAWithProgress = async (columns) => {
    // Define EDA steps
    const steps = [
      { step: 'Preparing analysis', status: 'running', elapsed: '0s' },
      { step: 'Loading dataset', status: 'pending' },
      { step: 'Analyzing dataset structure', status: 'pending' },
      { step: 'Checking missing values', status: 'pending' },
      { step: 'Detecting duplicate rows', status: 'pending' },
      { step: 'Analyzing distributions', status: 'pending' },
      { step: 'Detecting outliers', status: 'pending' },
      { step: 'Computing correlations', status: 'pending' },
      { step: 'Analyzing categorical features', status: 'pending' },
      { step: 'Assessing data quality', status: 'pending' },
      { step: 'Generating recommendations', status: 'pending' },
    ];

    setEdaSteps(steps);
    setLoading(true);

    const startTime = Date.now();
    let lastStepTime = startTime;
    let stepIntervalId;
    
    // Simulate progressive step completion
    stepIntervalId = setInterval(() => {
      const now = Date.now();
      const timeSinceLastStep = now - lastStepTime;
      
      if (timeSinceLastStep < 300) {
        return;
      }
      
      setEdaSteps(prev => {
        const currentRunningIndex = prev.findIndex(s => s.status === 'running');
        
        if (currentRunningIndex === -1 || currentRunningIndex >= prev.length - 1) {
          return prev;
        }
        
        lastStepTime = now;
        const elapsed = ((now - startTime) / 1000).toFixed(1) + 's';
        
        return prev.map((step, idx) => {
          if (idx === currentRunningIndex) {
            return { ...step, status: 'completed', elapsed };
          }
          if (idx === currentRunningIndex + 1) {
            return { ...step, status: 'running', elapsed: '0s' };
          }
          return step;
        });
      });
    }, 100);
    
    try {
      // Run EDA on backend
      const result = await datasetsAPI.runEDA(uploadedDatasetId, columns);
      
      // Wait for last step to complete
      await new Promise(resolve => {
        const checkLastStep = setInterval(() => {
          setEdaSteps(prev => {
            const lastStepIndex = prev.length - 1;
            const lastStep = prev[lastStepIndex];
            
            if (lastStep.status === 'running') {
              const now = Date.now();
              if (now - lastStepTime >= 300) {
                clearInterval(checkLastStep);
                clearInterval(stepIntervalId);
                const finalElapsed = ((now - startTime) / 1000).toFixed(1) + 's';
                resolve();
                return prev.map((step, idx) => 
                  idx === lastStepIndex 
                    ? { ...step, status: 'completed', elapsed: finalElapsed }
                    : step
                );
              }
            }
            return prev;
          });
        }, 100);
      });

      // Set EDA results
      if (result?.eda_results) {
        setEdaResults(result.eda_results);
      }
      
      setUploadComplete(true);
      setUploadPhase('complete');
      
    } catch (error) {
      clearInterval(stepIntervalId);
      setEdaSteps(prev => prev.map(step => 
        step.status === 'running' ? { ...step, status: 'error' } : step
      ));
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const filteredColumns = uploadedDatasetInfo?.column_names?.filter(col =>
    col.toLowerCase().includes(columnSearch.toLowerCase())
  ) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold">Upload Dataset</h1>
        <p className="mt-2 text-muted-foreground">
          Upload a CSV file to start analyzing and training models
        </p>
      </div>

      {/* Two column layout for upload/progress */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Upload Form - Hide when loading/complete/selecting */}
        {!loading && !uploadComplete && uploadPhase !== 'selecting' && (
          <div className="lg:col-span-2">
            <form onSubmit={handleSubmit} className="bg-card border rounded-2xl p-8 space-y-6">
            {error && (
              <div className="bg-destructive/10 border border-destructive/50 text-destructive px-4 py-3 rounded-xl text-sm">
                {error}
              </div>
            )}

            <div>
              <label htmlFor="name" className="block text-sm font-semibold mb-2">
                Dataset Name <span className="text-destructive">*</span>
              </label>
              <input
                type="text"
                name="name"
                id="name"
                required
                disabled={loading}
                value={formData.name}
                onChange={handleChange}
                className="w-full px-4 py-3 bg-background border rounded-lg focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50 disabled:cursor-not-allowed"
                placeholder="e.g., Customer Analysis Dataset"
              />
            </div>

            <div>
              <label htmlFor="description" className="block text-sm font-semibold mb-2">
                Description
              </label>
              <textarea
                name="description"
                id="description"
                rows={4}
                disabled={loading}
                value={formData.description}
                onChange={handleChange}
                className="w-full px-4 py-3 bg-background border rounded-lg focus:outline-none focus:ring-2 focus:ring-ring resize-none disabled:opacity-50 disabled:cursor-not-allowed"
                placeholder="Describe your dataset..."
              />
            </div>

            <div>
              <label className="block text-sm font-semibold mb-3">
                CSV File <span className="text-destructive">*</span>
              </label>
              <div className="bg-muted/50 border-2 border-dashed hover:border-primary transition-colors p-8 rounded-xl">
                <div className="text-center">
                  <div className="mx-auto w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center mb-4">
                    <svg className="w-8 h-8 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <div className="flex justify-center text-sm text-muted-foreground">
                    <label
                      htmlFor="file-upload"
                      className="relative cursor-pointer font-semibold text-primary hover:text-primary/80 transition-colors"
                    >
                      <span>Choose file</span>
                      <input
                        id="file-upload"
                        name="file-upload"
                        type="file"
                        accept=".csv"
                        disabled={loading}
                        className="sr-only"
                        onChange={handleFileChange}
                      />
                    </label>
                    <p className="pl-1">or drag and drop</p>
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">CSV files only</p>
                  {file && (
                    <div className="mt-4 flex items-center justify-center gap-2 text-sm font-medium bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20 rounded-xl py-2 px-4">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      {file.name}
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex justify-end space-x-3 pt-4">
              <button
                type="button"
                onClick={() => navigate('/dashboard/datasets')}
                disabled={loading}
                className="px-6 py-3 rounded-xl font-medium bg-secondary hover:bg-secondary/80 focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading || !file}
                className="inline-flex items-center justify-center px-6 py-3 rounded-xl font-medium bg-primary hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm text-primary-foreground dark:text-white"
              >
                {loading ? (
                  <span className="flex items-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    Upload & Analyze
                  </span>
                )}
              </button>
            </div>
          </form>
        </div>
        )}

        {/* Column Selection Section - Show after upload */}
        {uploadPhase === 'selecting' && uploadedDatasetInfo && (
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-card border rounded-2xl p-8 space-y-6"
            >
              {/* Header */}
              <div className="flex items-start gap-4">
                <div className="p-3 bg-primary/10 rounded-xl">
                  <Columns3 className="w-6 h-6 text-primary" />
                </div>
                <div className="flex-1">
                  <h2 className="text-2xl font-bold">Select Columns for Analysis</h2>
                  <p className="text-muted-foreground mt-1">
                    Choose which columns to include in the exploratory data analysis and training
                  </p>
                  <div className="mt-2 text-sm text-muted-foreground">
                    {selectedColumns.length} of {uploadedDatasetInfo?.column_names?.length || 0} columns selected
                  </div>
                </div>
              </div>

              {error && (
                <div className="bg-destructive/10 border border-destructive/50 text-destructive px-4 py-3 rounded-xl text-sm">
                  {error}
                </div>
              )}

              {/* Search and Actions */}
              <div className="space-y-4">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <input
                    type="text"
                    value={columnSearch}
                    onChange={(e) => setColumnSearch(e.target.value)}
                    placeholder="Search columns..."
                    className="w-full pl-10 pr-4 py-3 bg-background border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>

                {/* Toggle Select All Button */}
                <div className="flex justify-between items-center">
                  <button
                    onClick={() => {
                      if (selectedColumns.length === uploadedDatasetInfo?.column_names?.length) {
                        handleDeselectAll();
                      } else {
                        handleSelectAll();
                      }
                    }}
                    className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium bg-primary/10 text-primary rounded-lg hover:bg-primary/20 transition-colors"
                  >
                    {selectedColumns.length === uploadedDatasetInfo?.column_names?.length ? (
                      <>
                        <X className="w-4 h-4" />
                        Deselect All
                      </>
                    ) : (
                      <>
                        <Check className="w-4 h-4" />
                        Select All
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Column List */}
              <div className="border rounded-lg p-4 bg-muted/30 max-h-[500px] overflow-y-auto">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                  {filteredColumns.map((columnName) => {
                    const isSelected = selectedColumns.includes(columnName);
                    const columnType = uploadedDatasetInfo?.column_types?.[columnName] || 'unknown';
                    
                    return (
                      <motion.button
                        key={columnName}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => handleToggleColumn(columnName)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${
                          isSelected
                            ? 'border-primary bg-primary/5'
                            : 'border-border bg-card hover:border-primary/50'
                        }`}
                      >
                        <div className="flex items-start gap-2">
                          <div className={`mt-0.5 w-5 h-5 rounded flex items-center justify-center border-2 transition-colors ${
                            isSelected
                              ? 'bg-primary border-primary'
                              : 'border-muted-foreground'
                          }`}>
                            {isSelected && <Check className="w-3 h-3 text-primary-foreground" strokeWidth={3} />}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-sm truncate" title={columnName}>
                              {columnName}
                            </div>
                            <div className="text-xs text-muted-foreground mt-0.5">
                              {columnType}
                            </div>
                          </div>
                        </div>
                      </motion.button>
                    );
                  })}
                </div>

                {filteredColumns.length === 0 && (
                  <div className="text-center py-12 text-muted-foreground">
                    <Search className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No columns found matching "{columnSearch}"</p>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 justify-end pt-4 border-t">
                <button
                  onClick={async () => {
                    // Delete the uploaded dataset from backend before canceling
                    if (uploadedDatasetId) {
                      try {
                        await datasetsAPI.delete(uploadedDatasetId);
                      } catch (error) {
                        console.error('Failed to delete dataset:', error);
                      }
                    }
                    // Reset state
                    setUploadPhase('idle');
                    setUploadedDatasetId(null);
                    setUploadedDatasetInfo(null);
                    setSelectedColumns([]);
                    setColumnSearch('');
                    setFile(null);
                    setFormData({ name: '', description: '' });
                    setError('');
                  }}
                  disabled={savingColumns}
                  className="px-6 py-3 rounded-xl font-medium bg-secondary hover:bg-secondary/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveColumnSelection}
                  disabled={savingColumns || selectedColumns.length === 0}
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-medium bg-primary hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-primary-foreground dark:text-white shadow-sm"
                >
                  {savingColumns ? (
                    <>
                      <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Check className="w-5 h-5" />
                      Run Analysis
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          </div>
        )}

        {/* Progress Log - Left Column when complete */}
        {(loading || uploadComplete) && edaSteps.length > 0 && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-card border rounded-2xl p-6"
          >
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              {loading ? (
                <>
                  <svg className="w-5 h-5 text-primary animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Running Analysis
                </>
              ) : (
                <>
                  <svg className="w-5 h-5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Analysis Complete
                </>
              )}
            </h3>
            <EDAProgressLog steps={edaSteps} />
          </motion.div>
        )}

        {/* Results - Right Column when complete */}
        {uploadComplete && edaResults && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-card border rounded-2xl p-6"
          >
            {/* Results Tabs */}
            <div className="flex gap-4 mb-6 border-b">
              <button
                onClick={() => setActiveResultsTab('summary')}
                className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                  activeResultsTab === 'summary'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
              >
                EDA Summary
              </button>
              <button
                onClick={() => setActiveResultsTab('charts')}
                className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                  activeResultsTab === 'charts'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
              >
                Charts & Visualizations
              </button>
            </div>

            {/* Results Content */}
            {activeResultsTab === 'summary' && <EDASummary edaResults={edaResults} />}
            {activeResultsTab === 'charts' && <EDACharts edaResults={edaResults} />}
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="mt-6 flex justify-center"
            >
              <button
                onClick={() => navigate('/dashboard/datasets')}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-medium bg-primary hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all shadow-sm text-primary-foreground dark:text-white"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                View All Datasets
              </button>
            </motion.div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default DatasetUpload;
