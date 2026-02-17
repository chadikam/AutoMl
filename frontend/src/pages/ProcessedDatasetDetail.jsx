/**
 * Processed Dataset Detail page with preview and processing analysis
 */
import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { createPortal } from 'react-dom';
import { motion } from 'framer-motion';
import { datasetsAPI } from '../utils/api';
import { 
  CheckCircle, 
  TrendingUp, 
  AlertTriangle, 
  Database,
  Zap,
  XCircle,
  Plus,
  BarChart3,
  Info,
  GitBranch
} from 'lucide-react';

const ProcessedDatasetDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [dataset, setDataset] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [activeTab, setActiveTab] = useState('preview');
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadingTrain, setDownloadingTrain] = useState(false);
  const [downloadingTest, setDownloadingTest] = useState(false);
  const [expandedChange, setExpandedChange] = useState(null);
  const [showActionsMenu, setShowActionsMenu] = useState(false);

  useEffect(() => {
    fetchDataset();
    fetchPreview(); // Load preview on mount since it's the default tab
  }, [id]);

  const fetchDataset = async () => {
    try {
      const data = await datasetsAPI.get(id);
      console.log('📊 Dataset loaded:', data);
      console.log('📊 Changes:', data.preprocessing_summary?.changes);
      console.log('📊 Rare values removed:', data.preprocessing_summary?.changes?.rare_values_removed);
      console.log('📊 Columns removed structure:', data.preprocessing_summary?.columns_removed);
      setDataset(data);
    } catch (error) {
      console.error('Failed to fetch dataset:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPreview = async () => {
    if (preview) return; // Already loaded
    
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      // Always use preprocessed=true for this page
      const data = await datasetsAPI.preview(id, 10, true);
      setPreview(data);
    } catch (error) {
      console.error('Failed to fetch preview:', error);
      setPreviewError(error.response?.data?.detail || error.message || 'Failed to load preview');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleDelete = async () => {
    try {
      // Remove from processed datasets list in localStorage
      const savedProcessed = localStorage.getItem('processedDatasets');
      if (savedProcessed) {
        const processedIds = JSON.parse(savedProcessed);
        const updatedIds = processedIds.filter(datasetId => datasetId !== id);
        localStorage.setItem('processedDatasets', JSON.stringify(updatedIds));
      }
      
      // Navigate back to processed list
      navigate('/dashboard/processed');
    } catch (error) {
      console.error('Failed to remove from processed list:', error);
    } finally {
      setShowDeleteDialog(false);
    }
  };

  const handleDownload = async () => {
    setDownloading(true);
    try {
      await datasetsAPI.download(id, true); // true = download preprocessed version
    } catch (error) {
      console.error('Failed to download dataset:', error);
      alert('Failed to download dataset. Please try again.');
    } finally {
      setDownloading(false);
    }
  };

  const handleDownloadTrain = async () => {
    setDownloadingTrain(true);
    try {
      await datasetsAPI.downloadSplit(id, 'train');
    } catch (error) {
      console.error('Failed to download train split:', error);
      alert('Failed to download train dataset. Please try again.');
    } finally {
      setDownloadingTrain(false);
    }
  };

  const handleDownloadTest = async () => {
    setDownloadingTest(true);
    try {
      await datasetsAPI.downloadSplit(id, 'test');
    } catch (error) {
      console.error('Failed to download test split:', error);
      alert('Failed to download test dataset. Please try again.');
    } finally {
      setDownloadingTest(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="bg-card border rounded-2xl p-16 text-center">
        <h3 className="text-xl font-semibold mb-2">Dataset not found</h3>
        <p className="text-muted-foreground mb-6">
          The dataset you're looking for doesn't exist.
        </p>
        <Link
          to="/dashboard/processed"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-primary-foreground bg-primary hover:bg-primary/90 transition-colors"
        >
          Back to Processed Datasets
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-full overflow-hidden">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-4 flex-1 min-w-0">
          <button
            onClick={() => navigate('/dashboard/processed')}
            className="p-2 hover:bg-muted rounded-lg transition-colors shrink-0"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <div className="min-w-0 flex-1">
            <h1 className="text-4xl font-bold">{dataset.name}</h1>
            {dataset.description && (
              <p className="mt-2 text-muted-foreground line-clamp-2">{dataset.description}</p>
            )}
          </div>
        </div>
        <div className="relative">
          <button
            onClick={() => setShowActionsMenu(!showActionsMenu)}
            className="inline-flex items-center justify-center w-10 h-10 rounded-lg text-foreground hover:bg-muted transition-colors"
            title="Actions"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
            </svg>
          </button>
          
          {/* Dropdown Menu */}
          {showActionsMenu && (
            <>
              <div 
                className="fixed inset-0 z-10" 
                onClick={() => setShowActionsMenu(false)}
              ></div>
              <div className="absolute right-0 mt-2 w-56 bg-card border rounded-lg shadow-lg z-20 overflow-hidden">
                {dataset.preprocessing_summary?.has_train_test_split ? (
                  <>
                    <button
                      onClick={() => {
                        handleDownloadTrain();
                        setShowActionsMenu(false);
                      }}
                      disabled={downloadingTrain}
                      className="w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left"
                    >
                      <svg className="w-4 h-4 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      <div className="flex-1">
                        <div className="font-medium">Download Train Data</div>
                        <div className="text-xs text-muted-foreground">
                          {dataset.preprocessing_summary.train_size} rows
                        </div>
                      </div>
                      {downloadingTrain && (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                      )}
                    </button>
                    <button
                      onClick={() => {
                        handleDownloadTest();
                        setShowActionsMenu(false);
                      }}
                      disabled={downloadingTest}
                      className="w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left border-t"
                    >
                      <svg className="w-4 h-4 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      <div className="flex-1">
                        <div className="font-medium">Download Test Data</div>
                        <div className="text-xs text-muted-foreground">
                          {dataset.preprocessing_summary.test_size} rows
                        </div>
                      </div>
                      {downloadingTest && (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                      )}
                    </button>
                    <button
                      onClick={() => {
                        handleDownload();
                        setShowActionsMenu(false);
                      }}
                      disabled={downloading}
                      className="w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left border-t"
                    >
                      <svg className="w-4 h-4 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      <div className="flex-1">
                        <div className="font-medium">Download Full Dataset</div>
                        <div className="text-xs text-muted-foreground">
                          Complete preprocessed data
                        </div>
                      </div>
                      {downloading && (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                      )}
                    </button>
                  </>
                ) : (
                  <button
                    onClick={() => {
                      handleDownload();
                      setShowActionsMenu(false);
                    }}
                    disabled={downloading}
                    className="w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left"
                  >
                    <svg className="w-4 h-4 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    <div className="flex-1">
                      <div className="font-medium">Download Dataset</div>
                      <div className="text-xs text-muted-foreground">
                        Preprocessed data
                      </div>
                    </div>
                    {downloading && (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                    )}
                  </button>
                )}
                <button
                  onClick={() => {
                    setShowDeleteDialog(true);
                    setShowActionsMenu(false);
                  }}
                  className="w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-red-500/10 transition-colors text-left border-t"
                >
                  <svg className="w-4 h-4 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  <div className="flex-1">
                    <div className="font-medium text-red-600 dark:text-red-400">Delete Dataset</div>
                    <div className="text-xs text-muted-foreground">
                      Remove from list
                    </div>
                  </div>
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b">
        <div className="flex gap-4">
          <button
            onClick={() => {
              setActiveTab('preview');
              fetchPreview();
            }}
            className={`px-4 py-2 font-medium border-b-2 transition-colors ${
              activeTab === 'preview'
                ? 'border-primary text-primary'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            Data Preview
          </button>
          <button
            onClick={() => setActiveTab('processing')}
            className={`px-4 py-2 font-medium border-b-2 transition-colors ${
              activeTab === 'processing'
                ? 'border-primary text-primary'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            Processing Analysis
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="max-w-full overflow-hidden"
      >
        {activeTab === 'preview' && (
          <div className="space-y-4">
            {/* Preprocessed indicator */}
            {dataset.preprocessing_summary && (
              <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-xl p-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-green-500/10 rounded-lg">
                    <svg className="w-5 h-5 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-green-600 dark:text-green-400">Preprocessed Dataset Preview</h4>
                    <p className="text-sm text-muted-foreground">
                      Showing the processed version with {dataset.preprocessing_summary.changes?.categorical_columns_encoded || 0} encoded columns, 
                      {' '}{dataset.preprocessing_summary.changes?.numerical_columns_scaled || 0} scaled columns, 
                      and {dataset.preprocessing_summary.changes?.new_features_created || 0} engineered features.
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            <div className="bg-card border rounded-2xl overflow-hidden">
            {previewLoading ? (
              <div className="flex justify-center items-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
              </div>
            ) : previewError ? (
              <div className="p-16 text-center">
                <div className="mx-auto w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mb-4">
                  <svg className="w-8 h-8 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold mb-2">Preview Unavailable</h3>
                <p className="text-muted-foreground mb-4">{previewError}</p>
                <button
                  onClick={() => fetchPreview()}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Retry
                </button>
              </div>
            ) : preview ? (
              <div className="overflow-x-auto overflow-y-auto max-h-[600px]">
                {preview.has_split === true ? (
                  // Train/Test Split View
                  <div className="space-y-8 p-6">
                    {/* Train Data */}
                    <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
                      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
                        <div className="flex items-center gap-3">
                          <div className="px-3 py-1.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-lg text-sm font-semibold">
                            TRAIN DATA
                          </div>
                          <span className="text-sm text-gray-600 dark:text-gray-400">
                            {preview.train_size?.toLocaleString()} rows total • Showing 8 sample rows
                          </span>
                          {preview.column_notice && (
                            <span className="text-sm text-amber-600 dark:text-amber-400">
                              • Showing {preview.columns?.length || 0} / {preview.total_columns || 0} columns
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-border">
                          <thead className="bg-muted/50">
                            <tr>
                              {preview.columns && preview.columns.map((col, idx) => (
                                <th key={idx} className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider whitespace-nowrap">
                                  {col}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border">
                            {preview.train_data && preview.train_data.map((row, rowIdx) => (
                              <tr key={rowIdx} className="hover:bg-muted/30 transition-colors">
                                {preview.columns.map((col, cellIdx) => (
                                  <td key={cellIdx} className="px-6 py-4 text-sm whitespace-nowrap">
                                    {row[col] !== null && row[col] !== undefined ? row[col].toString() : <span className="text-muted-foreground italic">null</span>}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Test Data */}
                    <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
                      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
                        <div className="flex items-center gap-3">
                          <div className="px-3 py-1.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-lg text-sm font-semibold">
                            TEST DATA
                          </div>
                          <span className="text-sm text-gray-600 dark:text-gray-400">
                            {preview.test_size?.toLocaleString()} rows total • Showing 2 sample rows
                          </span>
                          {preview.column_notice && (
                            <span className="text-sm text-amber-600 dark:text-amber-400">
                              • Showing {preview.columns?.length || 0} / {preview.total_columns || 0} columns
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-border">
                          <thead className="bg-muted/50">
                            <tr>
                              {preview.columns && preview.columns.map((col, idx) => (
                                <th key={idx} className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider whitespace-nowrap">
                                  {col}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border">
                            {preview.test_data && preview.test_data.map((row, rowIdx) => (
                              <tr key={rowIdx} className="hover:bg-muted/30 transition-colors">
                                {preview.columns.map((col, cellIdx) => (
                                  <td key={cellIdx} className="px-6 py-4 text-sm whitespace-nowrap">
                                    {row[col] !== null && row[col] !== undefined ? row[col].toString() : <span className="text-muted-foreground italic">null</span>}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                ) : (
                  // Regular View (no split)
                  <div className="space-y-0">
                    <table className="min-w-full divide-y divide-border">
                      <thead className="bg-muted/50 sticky top-0 z-10">
                        <tr>
                          {preview.columns && preview.columns.map((col, idx) => (
                            <th
                              key={idx}
                              className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider whitespace-nowrap bg-muted/50"
                            >
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {preview.data && preview.data.length > 0 ? (
                          preview.data.map((row, rowIdx) => (
                            <tr key={rowIdx} className="hover:bg-muted/30 transition-colors">
                              {preview.columns.map((col, cellIdx) => (
                                <td key={cellIdx} className="px-6 py-4 text-sm whitespace-nowrap">
                                  {row[col] !== null && row[col] !== undefined ? row[col].toString() : <span className="text-muted-foreground italic">null</span>}
                                </td>
                              ))}
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan={preview.columns?.length || 1} className="px-6 py-8 text-center text-muted-foreground">
                              No data rows available
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                    {preview.column_notice && (
                      <div className="px-6 py-4 bg-amber-50 dark:bg-amber-900/20 border border-t-0 border-amber-200 dark:border-amber-800 text-sm text-amber-700 dark:text-amber-300">
                        <strong>Note:</strong> {preview.column_notice} (showing {preview.columns?.length || 0} of {preview.total_columns || 0} total columns)
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="p-16 text-center">
                <p className="text-muted-foreground">No preview available</p>
              </div>
            )}
            </div>
          </div>
        )}

        {activeTab === 'processing' && (
          <>
            {!dataset?.preprocessing_summary ? (
              <div className="bg-card border rounded-2xl p-16 text-center">
                <div className="mx-auto w-16 h-16 bg-blue-500/10 rounded-full flex items-center justify-center mb-4">
                  <svg className="w-8 h-8 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold mb-2">No Processing Analysis Available</h3>
                <p className="text-muted-foreground">
                  This dataset hasn't been preprocessed yet. Run preprocessing to see the analysis here.
                </p>
              </div>
            ) : (
          <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-2xl font-bold flex items-center gap-2">
                  <Zap className="w-6 h-6 text-green-600 dark:text-green-400" />
                  Preprocessing Complete
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Dataset has been preprocessed and is ready for machine learning
                </p>
              </div>
              {dataset.preprocessing_summary.final_quality && (
                <QualityBadge 
                  score={dataset.preprocessing_summary.final_quality.quality_score} 
                  assessment={dataset.preprocessing_summary.final_quality.assessment} 
                />
              )}
            </div>

            {/* Overview Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Original Shape */}
              <StatCard
                icon={<Database className="w-5 h-5" />}
                label="Original Shape"
                value={`${dataset.preprocessing_summary.original?.rows?.toLocaleString() || 0} × ${dataset.preprocessing_summary.original?.columns || 0}`}
                subtitle={`${dataset.preprocessing_summary.original?.rows?.toLocaleString() || 0} rows, ${dataset.preprocessing_summary.original?.columns || 0} cols`}
                color="blue"
              />

              {/* Processed Shape */}
              <StatCard
                icon={<CheckCircle className="w-5 h-5" />}
                label="Processed Shape"
                value={`${dataset.preprocessing_summary.processed?.rows?.toLocaleString() || 0} × ${dataset.preprocessing_summary.processed?.columns || 0}`}
                subtitle={`${dataset.preprocessing_summary.processed?.rows?.toLocaleString() || 0} rows, ${dataset.preprocessing_summary.processed?.columns || 0} cols`}
                color="green"
              />
              
              {/* Class Balance - Only show if class imbalance was handled */}
              {dataset.preprocessing_summary.class_imbalance_handled && (
                <ClickableStatCard
                  icon={<BarChart3 className="w-5 h-5" />}
                  label="Class Balance"
                  value={dataset.preprocessing_summary.imbalance_strategy || 'Balanced'}
                  subtitle="Click to see distribution"
                  color="purple"
                  onClick={() => setExpandedChange(expandedChange === 'class_balance' ? null : 'class_balance')}
                  hasDetails={true}
                  isExpanded={expandedChange === 'class_balance'}
                />
              )}

              {/* Initial Quality - shift right if no class balance */}
              {!dataset.preprocessing_summary.class_imbalance_handled && (
                <StatCard
                  icon={<TrendingUp className="w-5 h-5" />}
                  label="Initial Quality"
                  value={`${dataset.preprocessing_summary.initial_quality?.quality_score?.toFixed(1) || 0}%`}
                  subtitle={dataset.preprocessing_summary.initial_quality?.assessment || 'Unknown'}
                  color="orange"
                />
              )}

              {/* Final Quality */}
              <StatCard
                icon={<CheckCircle className="w-5 h-5" />}
                label="Final Quality"
                value={`${dataset.preprocessing_summary.final_quality?.quality_score?.toFixed(1) || 0}%`}
                subtitle={dataset.preprocessing_summary.final_quality?.assessment || 'Unknown'}
                color="green"
              />
            </div>
            
            {/* Class Balance Details Expansion */}
            {expandedChange === 'class_balance' && dataset.preprocessing_summary.class_imbalance_handled && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                {renderClassBalanceDetails(dataset.preprocessing_summary)}
              </motion.div>
            )}

            {/* Changes Made Grid */}
            {dataset.preprocessing_summary.changes && Object.keys(dataset.preprocessing_summary.changes).length > 0 && (
              <div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(dataset.preprocessing_summary.changes)
                    .filter(([key, value]) => {
                      // Special case: show outliers_handled if there are any outlier_details (even if count is 0)
                      if (key === 'outliers_handled') {
                        const outlierDetails = dataset.preprocessing_summary.outlier_details || [];
                        return outlierDetails.length > 0;
                      }
                      // Special case: show multicollinearity_handled if there are any pairs detected
                      if (key === 'multicollinearity_handled') {
                        const multicollinearity = dataset.preprocessing_summary.multicollinearity || {};
                        return (multicollinearity.pairs_detected || 0) > 0;
                      }
                      return value > 0;
                    })
                    .map(([key, value]) => {
                      // Determine icon and color based on change type
                      let icon = <BarChart3 className="w-5 h-5" />;
                      let color = 'blue';
                      
                      if (key.includes('removed') || key.includes('duplicates')) {
                        icon = <XCircle className="w-5 h-5" />;
                        color = 'red';
                      } else if (key.includes('multicollinearity')) {
                        icon = <GitBranch className="w-5 h-5" />;
                        color = 'indigo';
                      } else if (key.includes('filled') || key.includes('handled')) {
                        icon = <AlertTriangle className="w-5 h-5" />;
                        color = 'yellow';
                      } else if (key.includes('encoded') || key.includes('scaled') || key.includes('created')) {
                        icon = <Plus className="w-5 h-5" />;
                        color = 'green';
                      } else if (key.includes('corrected')) {
                        icon = <TrendingUp className="w-5 h-5" />;
                        color = 'purple';
                      }

                      // Special handling for outliers: show total count of columns with outliers
                      let displayValue = value;
                      if (key === 'outliers_handled') {
                        const outlierDetails = dataset.preprocessing_summary.outlier_details || [];
                        displayValue = outlierDetails.length;
                        color = 'orange'; // Use orange for outliers
                      }
                      
                      // Special handling for multicollinearity: show pairs detected
                      if (key === 'multicollinearity_handled') {
                        const multicollinearity = dataset.preprocessing_summary.multicollinearity || {};
                        displayValue = multicollinearity.pairs_detected || 0;
                      }

                      const hasDetails = getChangeDetails(key, dataset.preprocessing_summary);
                      const isExpanded = expandedChange === key;

                      return (
                        <div key={key}>
                          <ClickableStatCard
                            icon={icon}
                            label={key.replace(/_/g, ' ')}
                            value={displayValue.toLocaleString()}
                            subtitle={getChangeSubtitle(key)}
                            color={color}
                            hasDetails={hasDetails}
                            onClick={() => setExpandedChange(isExpanded ? null : key)}
                            isExpanded={isExpanded}
                          />
                        </div>
                      );
                    })}
                </div>

                {/* Expanded Details */}
                {expandedChange && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-4"
                  >
                    {renderChangeDetails(expandedChange, dataset.preprocessing_summary)}
                  </motion.div>
                )}
              </div>
            )}

            {/* Quality Improvement */}
            {dataset.preprocessing_summary.initial_quality && dataset.preprocessing_summary.final_quality && (
              <div className="bg-card border rounded-xl p-6">
                <h4 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
                  Quality Improvement
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <QualityMetric
                    label="Completeness"
                    before={dataset.preprocessing_summary.initial_quality.completeness}
                    after={dataset.preprocessing_summary.final_quality.completeness}
                  />
                  <QualityMetric
                    label="Uniqueness"
                    before={dataset.preprocessing_summary.initial_quality.uniqueness}
                    after={dataset.preprocessing_summary.final_quality.uniqueness}
                  />
                  <QualityMetric
                    label="Overall Score"
                    before={dataset.preprocessing_summary.initial_quality.quality_score}
                    after={dataset.preprocessing_summary.final_quality.quality_score}
                  />
                </div>
              </div>
            )}

            {/* Processing Steps */}
            {dataset.preprocessing_summary.steps && dataset.preprocessing_summary.steps.length > 0 && (
              <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-200 dark:border-gray-800">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-gray-900 dark:text-white" />
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Processing Steps
                      </h4>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {dataset.preprocessing_summary.steps.length} steps completed
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-1 max-h-[500px] overflow-y-auto">
                  {dataset.preprocessing_summary.steps.map((step, index) => {
                    const isWarning = step.includes('⚠️') || (step.toLowerCase().includes('imbalance') && step.includes('detected'));
                    const isLastStep = index === dataset.preprocessing_summary.steps.length - 1;
                    
                    return (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className={`flex items-start gap-4 p-4 rounded-lg border transition-all duration-200 hover:border-gray-400 dark:hover:border-gray-600 ${
                          isWarning 
                            ? 'bg-yellow-50 dark:bg-yellow-900/10 border-yellow-200 dark:border-yellow-800/30' 
                            : isLastStep
                            ? 'bg-gray-50 dark:bg-gray-800/50 border-gray-300 dark:border-gray-700'
                            : 'bg-gray-50/50 dark:bg-gray-800/30 border-gray-200 dark:border-gray-800'
                        }`}
                      >
                        <div className={`flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center font-semibold text-sm ${
                          isWarning
                            ? 'bg-yellow-200 dark:bg-yellow-800/50 text-yellow-900 dark:text-yellow-100'
                            : isLastStep
                            ? 'bg-gray-900 dark:bg-white text-white dark:text-gray-900'
                            : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                        }`}>
                          {index + 1}
                        </div>
                        <span className={`text-sm flex-1 leading-relaxed ${
                          isWarning 
                            ? 'text-yellow-900 dark:text-yellow-100 font-medium' 
                            : 'text-gray-700 dark:text-gray-300'
                        }`}>
                          {step}
                        </span>
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
            )}
          </>
        )}
      </motion.div>

      {/* Delete Confirmation Dialog */}
      {showDeleteDialog && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center" style={{ zIndex: 99999, marginTop: 0 }} onClick={() => setShowDeleteDialog(false)}>
          <div className="bg-card border rounded-2xl p-6 max-w-md w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start gap-4">
              <div className="p-3 bg-red-500/10 rounded-lg shrink-0">
                <svg className="w-6 h-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold mb-2">Remove from Processed List</h3>
                <p className="text-muted-foreground mb-6">
                  Are you sure you want to remove "{dataset?.name}" from the processed datasets list? 
                  This will not delete the actual dataset or its preprocessing results.
                </p>
                <div className="flex gap-3 justify-end">
                  <button
                    onClick={() => setShowDeleteDialog(false)}
                    className="px-4 py-2 rounded-lg font-medium bg-muted hover:bg-muted/80 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleDelete}
                    className="px-4 py-2 rounded-lg font-medium text-white bg-red-500 hover:bg-red-600 transition-colors"
                  >
                    Remove
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
};

// Helper Components
const ClickableStatCard = ({ icon, label, value, subtitle, color = "blue", hasDetails, onClick, isExpanded }) => {
  const colorClasses = {
    blue: 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20',
    green: 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20',
    yellow: 'bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/20',
    red: 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20',
    purple: 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20',
    orange: 'bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/20',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      onClick={hasDetails ? onClick : undefined}
      className={`bg-card border rounded-xl p-4 transition-all ${
        hasDetails 
          ? 'cursor-pointer hover:bg-muted/50 hover:shadow-md hover:scale-[1.02]' 
          : 'hover:bg-muted/50'
      } ${isExpanded ? 'ring-2 ring-primary' : ''}`}
    >
      <div className="flex items-center gap-3 mb-3">
        <div className={`p-2 rounded-lg border ${colorClasses[color]}`}>
          {icon}
        </div>
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide flex-1">
          {label}
        </span>
        {hasDetails && (
          <svg 
            className={`w-4 h-4 text-muted-foreground transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        )}
      </div>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-xs text-muted-foreground mt-1">{subtitle}</div>
      {hasDetails && !isExpanded && (
        <div className="text-xs text-primary mt-2">Click to see details</div>
      )}
    </motion.div>
  );
};

const StatCard = ({ icon, label, value, subtitle, color = "blue" }) => {
  const colorClasses = {
    blue: 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20',
    green: 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20',
    yellow: 'bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/20',
    red: 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20',
    purple: 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20',
    orange: 'bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/20',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-card border rounded-xl p-4 hover:bg-muted/50 transition-colors"
    >
      <div className="flex items-center gap-3 mb-3">
        <div className={`p-2 rounded-lg border ${colorClasses[color]}`}>
          {icon}
        </div>
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          {label}
        </span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-xs text-muted-foreground mt-1">{subtitle}</div>
    </motion.div>
  );
};

const QualityMetric = ({ label, before, after }) => {
  const improvement = after - before;
  const getColor = (val) => {
    if (val >= 90) return 'text-green-600 dark:text-green-400';
    if (val >= 70) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <div className="p-3 bg-muted/50 rounded-lg">
      <span className="text-sm text-muted-foreground">{label}</span>
      <div className="flex items-center justify-between mt-2">
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">{before.toFixed(1)}%</span>
          <svg className="w-4 h-4 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
          <span className={`text-lg font-bold ${getColor(after)}`}>
            {after.toFixed(1)}%
          </span>
        </div>
        {improvement !== 0 && (
          <span className={`text-xs font-semibold ${improvement > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
            {improvement > 0 ? '+' : ''}{improvement.toFixed(1)}
          </span>
        )}
      </div>
    </div>
  );
};

const QualityBadge = ({ score, assessment }) => {
  const getConfig = () => {
    const assessmentLower = assessment.toLowerCase();
    if (assessmentLower.includes('excellent')) {
      return { color: 'bg-green-500/20 text-green-700 dark:text-green-300 border-green-500/30', icon: CheckCircle };
    }
    if (assessmentLower.includes('good')) {
      return { color: 'bg-blue-500/20 text-blue-700 dark:text-blue-300 border-blue-500/30', icon: CheckCircle };
    }
    if (assessmentLower.includes('fair')) {
      return { color: 'bg-yellow-500/20 text-yellow-700 dark:text-yellow-300 border-yellow-500/30', icon: AlertTriangle };
    }
    return { color: 'bg-red-500/20 text-red-700 dark:text-red-300 border-red-500/30', icon: AlertTriangle };
  };

  const config = getConfig();
  const Icon = config.icon;

  return (
    <div className={`flex items-center gap-2 px-4 py-2 rounded-lg border font-semibold ${config.color}`}>
      <Icon className="w-4 h-4" />
      <div className="flex flex-col">
        <span className="text-xs font-medium uppercase tracking-wide">
          {assessment}
        </span>
        <span className="text-xs opacity-90">{score.toFixed(0)}/100</span>
      </div>
    </div>
  );
};

const getChangeSubtitle = (key) => {
  const subtitles = {
    'rows_removed': 'Data cleaning',
    'columns_removed': 'Feature reduction',
    'duplicates_removed': 'Deduplication',
    'outliers_handled': 'Outlier treatment',
    'multicollinearity_handled': 'Correlation reduction',
    'skewness_corrected': 'Distribution normalization',
    'missing_values_filled': 'Imputation',
    'constant_columns_removed': 'Constant features',
    'datetime_columns_engineered': 'Temporal features',
    'categorical_columns_encoded': 'Encoding',
    'numerical_columns_scaled': 'Normalization',
    'new_features_created': 'Feature engineering'
  };
  return subtitles[key] || 'Processing';
};

const getChangeDetails = (key, summary) => {
  // Return true if this change type has column details to show
  if (key === 'duplicates_removed') {
    const duplicateCount = summary.changes?.duplicates_removed || 0;
    return duplicateCount > 0;
  }
  
  if (key === 'rows_removed') {
    const rareValuesRemoved = summary.changes?.rare_values_removed || [];
    return rareValuesRemoved.length > 0;
  }
  
  if (key === 'columns_removed') {
    // The backend stores these as separate fields in preprocessing_summary, not nested
    const removedCols = [
      ...(summary.id_columns_removed || []),
      ...(summary.constant_columns_removed || []),
      ...(summary.high_missing_columns_removed || [])
    ];
    
    console.log('🔍 getChangeDetails - columns_removed:', {
      id_columns_removed: summary.id_columns_removed,
      constant_columns_removed: summary.constant_columns_removed,
      high_missing_columns_removed: summary.high_missing_columns_removed,
      removedCols,
      removedCols_length: removedCols.length,
      hasDetails: removedCols.length > 0
    });
    
    return removedCols.length > 0;
  }
  
  if (key === 'new_features_created') {
    const newCols = summary.processed?.column_names?.filter(col => 
      !summary.original?.column_names?.includes(col)
    ) || [];
    return newCols.length > 0;
  }
  
  if (key === 'categorical_columns_encoded') {
    // Always return true to show encoding details (even if 0 columns)
    return true;
  }
  
  if (key === 'numerical_columns_scaled') {
    // Use the scaled_columns list from backend if available
    const scaledCols = summary.scaled_columns || 
      summary.processed?.column_names?.filter(col => 
        summary.original?.column_names?.includes(col) &&
        !col.includes('_')
      ) || [];
    return scaledCols.length > 0;
  }
  
  if (key === 'outliers_handled') {
    const outlierDetails = summary.outlier_details || [];
    return outlierDetails.length > 0;
  }
  
  if (key === 'skewness_corrected') {
    const skewnessDetails = summary.skewness_details || [];
    return skewnessDetails.length > 0;
  }
  
  if (key === 'missing_values_filled') {
    const imputationDetails = summary.imputation_details || [];
    return imputationDetails.length > 0;
  }
  
  if (key === 'class_imbalance_handled') {
    return summary.class_imbalance_handled === true;
  }
  
  if (key === 'multicollinearity_handled') {
    const multicollinearity = summary.multicollinearity || {};
    return (multicollinearity.pairs_detected || 0) > 0;
  }
  
  return false;
};

const renderClassBalanceDetails = (summary) => {
  const originalDist = summary.original_class_distribution || {};
  const resampledDist = summary.resampled_class_distribution || {};
  const strategy = summary.imbalance_strategy || 'Unknown';
  
  if (Object.keys(originalDist).length === 0) {
    return null;
  }
  
  // Calculate totals and percentages
  const originalTotal = Object.values(originalDist).reduce((sum, count) => sum + count, 0);
  const resampledTotal = Object.values(resampledDist).reduce((sum, count) => sum + count, 0);
  
  const originalPercentages = {};
  const resampledPercentages = {};
  
  Object.keys(originalDist).forEach(className => {
    originalPercentages[className] = (originalDist[className] / originalTotal * 100).toFixed(1);
    if (resampledDist[className]) {
      resampledPercentages[className] = (resampledDist[className] / resampledTotal * 100).toFixed(1);
    }
  });
  
  // Calculate imbalance ratio
  const counts = Object.values(originalDist);
  const maxCount = Math.max(...counts);
  const minCount = Math.min(...counts);
  const imbalanceRatio = (maxCount / minCount).toFixed(2);
  
  const newCounts = Object.values(resampledDist);
  const newMaxCount = Math.max(...newCounts);
  const newMinCount = Math.min(...newCounts);
  const newImbalanceRatio = newCounts.length > 0 ? (newMaxCount / newMinCount).toFixed(2) : '-';
  
  return (
    <div className="bg-purple-50 dark:bg-purple-950/30 border border-purple-200 dark:border-purple-800/50 rounded-xl p-6">
      <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-purple-900 dark:text-purple-100">
        <BarChart3 className="w-5 h-5 text-purple-600 dark:text-purple-400" />
        Class Distribution Before & After Resampling
      </h4>
      
      {/* Strategy Badge */}
      <div className="mb-4 p-4 bg-purple-100/50 dark:bg-purple-900/20 rounded-lg border border-purple-200/50 dark:border-purple-800/30">
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-purple-900 dark:text-purple-100">
            Resampling Strategy
          </span>
          <span className="px-3 py-1 bg-purple-200 dark:bg-purple-800 rounded-full text-sm font-semibold text-purple-900 dark:text-purple-100">
            {strategy}
          </span>
        </div>
      </div>
      
      {/* Before and After Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Original Distribution */}
        <div className="p-4 bg-white/50 dark:bg-purple-900/20 rounded-lg border border-purple-100 dark:border-purple-800/30">
          <div className="flex items-center justify-between mb-3">
            <h5 className="text-sm font-semibold text-purple-900 dark:text-purple-100">
              Original Distribution
            </h5>
            <span className="text-xs px-2 py-1 bg-red-200 dark:bg-red-800 rounded text-red-900 dark:text-red-100">
              Imbalanced ({imbalanceRatio}:1)
            </span>
          </div>
          <div className="space-y-3">
            {Object.entries(originalDist).map(([className, count]) => (
              <div key={className} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono text-purple-900 dark:text-purple-100">Class {className}</span>
                  <span className="font-semibold text-purple-700 dark:text-purple-300">
                    {count.toLocaleString()} ({originalPercentages[className]}%)
                  </span>
                </div>
                <div className="w-full bg-purple-200 dark:bg-purple-800/30 rounded-full h-3">
                  <div
                    className="bg-red-500 dark:bg-red-600 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${originalPercentages[className]}%` }}
                  />
                </div>
              </div>
            ))}
            <div className="pt-2 mt-2 border-t border-purple-200 dark:border-purple-800">
              <div className="flex items-center justify-between text-xs font-semibold">
                <span className="text-purple-900 dark:text-purple-100">Total Samples</span>
                <span className="text-purple-700 dark:text-purple-300">{originalTotal.toLocaleString()}</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Resampled Distribution */}
        {Object.keys(resampledDist).length > 0 && (
          <div className="p-4 bg-white/50 dark:bg-purple-900/20 rounded-lg border border-purple-100 dark:border-purple-800/30">
            <div className="flex items-center justify-between mb-3">
              <h5 className="text-sm font-semibold text-purple-900 dark:text-purple-100">
                After Resampling
              </h5>
              <span className="text-xs px-2 py-1 bg-green-200 dark:bg-green-800 rounded text-green-900 dark:text-green-100">
                Balanced ({newImbalanceRatio}:1)
              </span>
            </div>
            <div className="space-y-3">
              {Object.entries(resampledDist).map(([className, count]) => (
                <div key={className} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-mono text-purple-900 dark:text-purple-100">Class {className}</span>
                    <span className="font-semibold text-purple-700 dark:text-purple-300">
                      {count.toLocaleString()} ({resampledPercentages[className]}%)
                    </span>
                  </div>
                  <div className="w-full bg-purple-200 dark:bg-purple-800/30 rounded-full h-3">
                    <div
                      className="bg-green-500 dark:bg-green-600 h-3 rounded-full transition-all duration-500"
                      style={{ width: `${resampledPercentages[className]}%` }}
                    />
                  </div>
                </div>
              ))}
              <div className="pt-2 mt-2 border-t border-purple-200 dark:border-purple-800">
                <div className="flex items-center justify-between text-xs font-semibold">
                  <span className="text-purple-900 dark:text-purple-100">Total Samples</span>
                  <span className="text-purple-700 dark:text-purple-300">{resampledTotal.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Summary */}
      <div className="mt-4 p-3 bg-green-50 dark:bg-green-950/30 rounded-lg border border-green-200 dark:border-green-800/50">
        <div className="flex items-center gap-2 text-sm text-green-900 dark:text-green-100">
          <svg className="w-4 h-4 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>
            Added <span className="font-semibold">{(resampledTotal - originalTotal).toLocaleString()}</span> samples 
            using {strategy} to balance class distribution
          </span>
        </div>
        
        {/* Original Split Info */}
        {summary.original_train_size > 0 && (
          <div className="mt-2 pt-2 border-t border-green-200 dark:border-green-800/50 text-xs text-green-800 dark:text-green-200">
            <span className="font-semibold">Original 80/20 split:</span>{' '}
            {summary.original_train_size.toLocaleString()} train + {summary.original_test_size.toLocaleString()} test = {(summary.original_train_size + summary.original_test_size).toLocaleString()} total
            {' '}→ After resampling: {resampledTotal.toLocaleString()} train + {summary.original_test_size.toLocaleString()} test = {(resampledTotal + summary.original_test_size).toLocaleString()} total
          </div>
        )}
      </div>
    </div>
  );
};

const renderChangeDetails = (key, summary) => {
  if (key === 'duplicates_removed') {
    const duplicateCount = summary.changes?.duplicates_removed || 0;
    const duplicateDetails = summary.changes?.duplicate_removal_details || [];
    
    return (
      <div className="bg-yellow-50 dark:bg-yellow-950/30 border border-yellow-200 dark:border-yellow-800/50 rounded-xl p-6">
        <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-yellow-900 dark:text-yellow-100">
          <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
          Duplicate Rows Removed ({duplicateCount})
        </h4>
        
        <div className="p-4 bg-yellow-100/50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200/50 dark:border-yellow-800/30 mb-4">
          <p className="text-sm text-yellow-900 dark:text-yellow-100 mb-3">
            <span className="font-semibold">{duplicateCount}</span> exact duplicate rows were identified and removed during preprocessing.
          </p>
          <div className="text-xs text-yellow-800 dark:text-yellow-200 space-y-1">
            <p>• Duplicates are rows that have identical values across all columns</p>
            <p>• Only the first occurrence of each duplicate set is kept</p>
            <p>• This improves data quality and prevents model bias</p>
          </div>
        </div>
        
        {/* Show detailed duplicate groups */}
        {duplicateDetails.length > 0 && (
          <div className="space-y-3">
            <h5 className="text-sm font-semibold text-yellow-900 dark:text-yellow-100">
              Duplicate Patterns Found ({duplicateDetails.length} unique patterns)
            </h5>
            
            <div className="max-h-[400px] overflow-y-auto space-y-3">
              {duplicateDetails.map((detail, idx) => (
                <div key={idx} className="bg-white dark:bg-gray-800 border border-yellow-200 dark:border-yellow-800/50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-yellow-700 dark:text-yellow-300 bg-yellow-100 dark:bg-yellow-900/40 px-2 py-1 rounded">
                        Pattern #{idx + 1}
                      </span>
                      <span className="text-xs text-yellow-600 dark:text-yellow-400">
                        Occurred {detail.occurrence_count} times
                      </span>
                    </div>
                  </div>
                  
                  {/* Row indices */}
                  <div className="mb-3">
                    <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                      Row Indices:
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {detail.row_indices.map((idx, i) => (
                        <span key={i} className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                          {idx}
                        </span>
                      ))}
                      {detail.total_indices > detail.row_indices.length && (
                        <span className="text-xs text-gray-500 dark:text-gray-400 px-2 py-1">
                          +{detail.total_indices - detail.row_indices.length} more
                        </span>
                      )}
                    </div>
                  </div>
                  
                  {/* Sample data */}
                  <div>
                    <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">
                      Duplicate Row Data:
                    </p>
                    <div className="bg-gray-50 dark:bg-gray-900/50 rounded border border-gray-200 dark:border-gray-700 overflow-hidden">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="bg-gray-100 dark:bg-gray-800">
                            <th className="px-3 py-2 text-left font-medium text-gray-600 dark:text-gray-400">Column</th>
                            <th className="px-3 py-2 text-left font-medium text-gray-600 dark:text-gray-400">Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(detail.sample_data).map(([key, value], i) => (
                            <tr key={i} className="border-t border-gray-200 dark:border-gray-700">
                              <td className="px-3 py-2 font-mono text-yellow-700 dark:text-yellow-300">{key}</td>
                              <td className="px-3 py-2 text-gray-900 dark:text-gray-100">
                                {value === null ? (
                                  <span className="text-gray-400 italic">null</span>
                                ) : (
                                  <span className="font-mono">{String(value)}</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            {duplicateDetails.length >= 20 && (
              <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-2">
                Showing top 20 duplicate patterns. Additional patterns may exist.
              </p>
            )}
          </div>
        )}
      </div>
    );
  }
  
  if (key === 'rows_removed') {
    const rareValuesRemoved = summary.changes?.rare_values_removed || [];
    
    if (rareValuesRemoved.length > 0) {
      const totalRowsRemoved = rareValuesRemoved.reduce((sum, item) => sum + item.rows_removed, 0);
      
      return (
        <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/50 rounded-xl p-6">
          <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-amber-900 dark:text-amber-100">
            <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400" />
            Rare Values Cleaned
          </h4>
          <div className="mb-4 p-4 bg-amber-100/50 dark:bg-amber-900/20 rounded-lg border border-amber-200/50 dark:border-amber-800/30">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-amber-900 dark:text-amber-100">
                Total Rows Removed
              </span>
              <span className="text-2xl font-bold text-amber-700 dark:text-amber-400">
                {totalRowsRemoved.toLocaleString()}
              </span>
            </div>
            <p className="text-xs text-amber-700 dark:text-amber-300 mt-2">
              Removed rare values from {rareValuesRemoved.length} column{rareValuesRemoved.length > 1 ? 's' : ''}
            </p>
          </div>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {rareValuesRemoved.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="p-4 bg-white/50 dark:bg-amber-900/20 rounded-lg border border-amber-100 dark:border-amber-800/30"
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-mono font-semibold text-amber-900 dark:text-amber-100">
                    {item.column}
                  </span>
                  <span className="text-xs font-semibold px-2 py-1 bg-amber-200 dark:bg-amber-800 rounded text-amber-900 dark:text-amber-100">
                    {item.rows_removed} rows
                  </span>
                </div>
                <div className="mb-2">
                  <div className="text-xs text-amber-700 dark:text-amber-300 mb-2">
                    {item.values_removed.length} value{item.values_removed.length > 1 ? 's' : ''} removed:
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {item.values_removed.map((val, vidx) => (
                      <span
                        key={vidx}
                        className="inline-block px-3 py-1 bg-amber-200 dark:bg-amber-800 text-amber-900 dark:text-amber-100 rounded-md text-sm font-mono"
                      >
                        {val}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      );
    }
  }
  
  if (key === 'columns_removed') {
    // The backend stores these as separate fields in preprocessing_summary, not nested
    const multicollinearityDropped = summary.multicollinearity?.features_dropped || [];
    
    const removedCols = [
      ...(summary.id_columns_removed || []),
      ...(summary.constant_columns_removed || []),
      ...(summary.high_missing_columns_removed || []),
      ...multicollinearityDropped
    ];
    
    console.log('🔍 Columns removed debug:', {
      id_columns_removed: summary.id_columns_removed,
      constant_columns_removed: summary.constant_columns_removed,
      high_missing_columns_removed: summary.high_missing_columns_removed,
      multicollinearity_dropped: multicollinearityDropped,
      removedCols,
      removedCols_length: removedCols.length,
      constant_column_details: summary.constant_column_details
    });
    
    return (
      <div className="bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800/50 rounded-xl p-6">
        <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-red-900 dark:text-red-100">
          <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
          Removed Columns ({removedCols.length})
        </h4>
        
        {/* Breakdown by reason */}
        <div className="mb-4 p-3 bg-red-100/50 dark:bg-red-900/20 rounded-lg border border-red-200/50 dark:border-red-800/30">
          <div className="text-xs text-red-700 dark:text-red-300 space-y-1">
            {(summary.id_columns_removed || []).length > 0 && (
              <div>ID columns: {(summary.id_columns_removed || []).length}</div>
            )}
            {(summary.constant_columns_removed || []).length > 0 && (
              <div>
                Constant/low-variance columns: {(summary.constant_columns_removed || []).length}
              </div>
            )}
            {(summary.high_missing_columns_removed || []).length > 0 && (
              <div>High missing columns: {(summary.high_missing_columns_removed || []).length}</div>
            )}
            {multicollinearityDropped.length > 0 && (
              <div>Multicollinearity: {multicollinearityDropped.length}</div>
            )}
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-64 overflow-y-auto">
          {removedCols.map((col, index) => {
            // Check if this column has detailed info (constant/low-variance)
            const details = summary.constant_column_details?.[col];
            const isConstantCol = summary.constant_columns_removed?.includes(col);
            
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.02 }}
                className="p-2 bg-white/50 dark:bg-red-900/20 rounded-lg border border-red-100 dark:border-red-800/30"
              >
                <div className="text-sm font-mono text-red-900 dark:text-red-100 font-semibold">{col}</div>
                {details ? (
                  <div className="text-xs text-red-700 dark:text-red-300 mt-1 space-y-0.5">
                    {details.type === 'high_missing' ? (
                      <div>Missing: {details.missing_count.toLocaleString()} / {details.total_count.toLocaleString()} rows ({details.percentage.toFixed(1)}%)</div>
                    ) : (
                      <>
                        <div>Value: '{details.value}' ({details.percentage.toFixed(1)}%)</div>
                        {details.variance !== null && details.variance !== undefined && (
                          <div className="text-red-600 dark:text-red-400">
                            Variance: {details.variance < 0.01 ? details.variance.toExponential(2) : details.variance.toFixed(4)}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                ) : isConstantCol && (
                  <div className="text-xs text-red-600 dark:text-red-400 mt-1 italic">
                    Re-run preprocessing for details
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>
      </div>
    );
  }
  
  if (key === 'new_features_created') {
    const newCols = summary.processed?.column_names?.filter(col => 
      !summary.original?.column_names?.includes(col)
    ) || [];
    
    return (
      <div className="bg-green-50 dark:bg-green-950/30 border border-green-200 dark:border-green-800/50 rounded-xl p-6">
        <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-green-900 dark:text-green-100">
          <Plus className="w-5 h-5 text-green-600 dark:text-green-400" />
          New Columns ({newCols.length})
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-64 overflow-y-auto">
          {newCols.map((col, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.02 }}
              className="p-2 bg-white/50 dark:bg-green-900/20 rounded-lg border border-green-100 dark:border-green-800/30"
            >
              <span className="text-sm font-mono text-green-900 dark:text-green-100">{col}</span>
            </motion.div>
          ))}
        </div>
      </div>
    );
  }
  
  if (key === 'categorical_columns_encoded') {
    // Use encoding details from backend
    const encodingDetails = summary.encoding_details || [];
    
    // If no encoding details, fall back to old detection method
    let encodedCols = [];
    let encodingMap = {};
    
    if (encodingDetails.length > 0) {
      // Use backend encoding details
      encodedCols = encodingDetails.map(detail => detail.column);
      
      // For one-hot encoded columns, find the new columns
      encodingDetails.forEach(detail => {
        if (detail.creates_new_columns) {
          const encodedVersions = summary.processed?.column_names?.filter(col => 
            col.startsWith(detail.column + '_')
          ) || [];
          encodingMap[detail.column] = {
            newColumns: encodedVersions,
            strategy: detail.strategy
          };
        } else {
          // Ordinal encoding - same column name
          encodingMap[detail.column] = {
            newColumns: [detail.column],
            strategy: detail.strategy
          };
        }
      });
    } else {
      // Fallback: Find original categorical columns that were one-hot encoded
      encodedCols = summary.original?.column_names?.filter(col => {
        return !summary.processed?.column_names?.includes(col) &&
               summary.processed?.column_names?.some(newCol => newCol.startsWith(col + '_'));
      }) || [];
      
      encodedCols.forEach(originalCol => {
        const encodedVersions = summary.processed?.column_names?.filter(col => 
          col.startsWith(originalCol + '_')
        ) || [];
        encodingMap[originalCol] = {
          newColumns: encodedVersions,
          strategy: 'One-Hot Encoding'
        };
      });
    }
    
    return (
      <div className="bg-purple-50 dark:bg-purple-950/30 border border-purple-200 dark:border-purple-800/50 rounded-xl p-6">
        <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-purple-900 dark:text-purple-100">
          <BarChart3 className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          Encoded Categorical Columns ({encodedCols.length})
        </h4>
        {encodedCols.length === 0 ? (
          <div className="p-4 bg-purple-100/50 dark:bg-purple-900/20 rounded-lg border border-purple-200/50 dark:border-purple-800/30">
            <p className="text-sm text-purple-900 dark:text-purple-100">
              ✓ No categorical columns were encoded. All features are numerical.
            </p>
          </div>
        ) : (
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {encodedCols.map((originalCol, index) => {
              const encoding = encodingMap[originalCol];
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="p-4 bg-white/50 dark:bg-purple-900/20 rounded-lg border border-purple-100 dark:border-purple-800/30"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-mono font-semibold text-purple-900 dark:text-purple-100">
                      {originalCol}
                    </span>
                    <svg className="w-4 h-4 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                    <span className="text-xs px-2 py-1 bg-purple-200 dark:bg-purple-800 rounded text-purple-900 dark:text-purple-100 font-medium">
                      {encoding.strategy}
                    </span>
                  </div>
                  {encoding.newColumns.length > 0 && (
                    <>
                      <div className="text-xs text-purple-600 dark:text-purple-400 mb-2">
                        {encoding.newColumns.length === 1 && encoding.newColumns[0] === originalCol
                          ? 'Encoded in-place (same column)'
                          : `${encoding.newColumns.length} encoded features`
                        }
                      </div>
                      {encoding.newColumns.length > 1 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 ml-6">
                          {encoding.newColumns.map((col, idx) => (
                            <div key={idx} className="text-xs font-mono text-purple-700 dark:text-purple-300 p-1 bg-purple-100/50 dark:bg-purple-800/20 rounded">
                              {col}
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    );
  }
  
  if (key === 'numerical_columns_scaled') {
    // Use the scaled_columns list from backend if available
    const scaledCols = summary.scaled_columns || 
      summary.processed?.column_names?.filter(col => 
        summary.original?.column_names?.includes(col) &&
        !col.includes('_')
      ) || [];
    
    return (
      <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800/50 rounded-xl p-6">
        <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-blue-900 dark:text-blue-100">
          <TrendingUp className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          Scaled Numerical Columns ({scaledCols.length})
        </h4>
        <div className="mb-3 p-3 bg-blue-100/50 dark:bg-blue-900/20 rounded-lg border border-blue-200/50 dark:border-blue-800/30">
          <p className="text-sm text-blue-900 dark:text-blue-100">
            <span className="font-semibold">StandardScaler applied:</span> Each column normalized to mean=0 and std=1
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-64 overflow-y-auto">
          {scaledCols.map((col, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.02 }}
              className="p-2 bg-white/50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-800/30"
            >
              <span className="text-sm font-mono text-blue-900 dark:text-blue-100">{col}</span>
            </motion.div>
          ))}
        </div>
      </div>
    );
  }
  
  if (key === 'outliers_handled') {
    const outlierDetails = summary.outlier_details || [];
    const outlierRowsRemoved = summary.outlier_rows_removed || 0;
    const totalOutliers = summary.changes?.outliers_handled || 0;
    
    // Separate capped and removed outliers
    const cappedOutliers = outlierDetails.filter(d => d.action === 'capped');
    const removedOutliers = outlierDetails.filter(d => d.action === 'removed');
    
    return (
      <div className="bg-orange-50 dark:bg-orange-950/30 border border-orange-200 dark:border-orange-800/50 rounded-xl p-6">
        <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-orange-900 dark:text-orange-100">
          <AlertTriangle className="w-5 h-5 text-orange-600 dark:text-orange-400" />
          Outlier Treatment
        </h4>
        
        {/* Summary Stats */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          {cappedOutliers.length > 0 && (
            <div className="p-4 bg-orange-100/50 dark:bg-orange-900/20 rounded-lg border border-orange-200/50 dark:border-orange-800/30">
              <div className="text-2xl font-bold text-orange-700 dark:text-orange-400">
                {cappedOutliers.length}
              </div>
              <div className="text-xs text-orange-600 dark:text-orange-300 mt-1">
                Columns with Capped Outliers
              </div>
            </div>
          )}
          {outlierRowsRemoved > 0 && (
            <div className="p-4 bg-red-100/50 dark:bg-red-900/20 rounded-lg border border-red-200/50 dark:border-red-800/30">
              <div className="text-2xl font-bold text-red-700 dark:text-red-400">
                {outlierRowsRemoved}
              </div>
              <div className="text-xs text-red-600 dark:text-red-300 mt-1">
                Rows Removed
              </div>
            </div>
          )}
        </div>
        
        {/* Capped Outliers Section */}
        {cappedOutliers.length > 0 && (
          <div className="mb-4">
            <h5 className="text-sm font-semibold mb-3 text-orange-900 dark:text-orange-100 flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Outliers Capped (Winsorization)
            </h5>
            <div className="mb-3 p-3 bg-orange-100/50 dark:bg-orange-900/20 rounded-lg border border-orange-200/50 dark:border-orange-800/30">
              <p className="text-sm text-orange-900 dark:text-orange-100">
                <span className="font-semibold">IQR Method:</span> Values beyond 2.5×IQR from quartiles were clipped to bounds
              </p>
            </div>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {cappedOutliers.map((detail, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.03 }}
                  className="p-4 bg-white/50 dark:bg-orange-900/20 rounded-lg border border-orange-100 dark:border-orange-800/30"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-mono font-semibold text-orange-900 dark:text-orange-100">
                      {detail.column}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold px-2 py-1 bg-orange-200 dark:bg-orange-800 rounded text-orange-900 dark:text-orange-100">
                        {detail.count} outliers
                      </span>
                      <span className="text-xs font-semibold px-2 py-1 bg-orange-300 dark:bg-orange-700 rounded text-orange-900 dark:text-orange-100">
                        {detail.percentage.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="p-2 bg-orange-100/50 dark:bg-orange-800/20 rounded">
                      <div className="text-orange-600 dark:text-orange-400 font-medium">Lower Bound</div>
                      <div className="font-mono text-orange-900 dark:text-orange-100">{detail.lower_bound.toFixed(2)}</div>
                    </div>
                    <div className="p-2 bg-orange-100/50 dark:bg-orange-800/20 rounded">
                      <div className="text-orange-600 dark:text-orange-400 font-medium">Upper Bound</div>
                      <div className="font-mono text-orange-900 dark:text-orange-100">{detail.upper_bound.toFixed(2)}</div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
        
        {/* Removed Outliers Section */}
        {removedOutliers.length > 0 && (
          <div>
            <h5 className="text-sm font-semibold mb-3 text-red-900 dark:text-red-100 flex items-center gap-2">
              <XCircle className="w-4 h-4" />
              Outlier Rows Removed ({outlierRowsRemoved} rows)
            </h5>
            <div className="mb-3 p-3 bg-red-100/50 dark:bg-red-900/20 rounded-lg border border-red-200/50 dark:border-red-800/30">
              <p className="text-sm text-red-900 dark:text-red-100">
                <span className="font-semibold">Warning:</span> Entire rows were removed when outliers were detected in these columns
              </p>
            </div>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {removedOutliers.map((detail, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.03 }}
                  className="p-4 bg-white/50 dark:bg-red-900/20 rounded-lg border border-red-100 dark:border-red-800/30"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-mono font-semibold text-red-900 dark:text-red-100">
                      {detail.column}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold px-2 py-1 bg-red-200 dark:bg-red-800 rounded text-red-900 dark:text-red-100">
                        {detail.count} outliers
                      </span>
                      <span className="text-xs font-semibold px-2 py-1 bg-red-300 dark:bg-red-700 rounded text-red-900 dark:text-red-100">
                        {detail.percentage.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="p-2 bg-red-100/50 dark:bg-red-800/20 rounded">
                      <div className="text-red-600 dark:text-red-400 font-medium">Lower Bound</div>
                      <div className="font-mono text-red-900 dark:text-red-100">{detail.lower_bound.toFixed(2)}</div>
                    </div>
                    <div className="p-2 bg-red-100/50 dark:bg-red-800/20 rounded">
                      <div className="text-red-600 dark:text-red-400 font-medium">Upper Bound</div>
                      <div className="font-mono text-red-900 dark:text-red-100">{detail.upper_bound.toFixed(2)}</div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
        
        {outlierDetails.length === 0 && (
          <div className="p-4 bg-gray-100/50 dark:bg-gray-900/20 rounded-lg border border-gray-200 dark:border-gray-700 text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              No outlier handling was applied to this dataset
            </p>
          </div>
        )}
      </div>
    );
  }
  
  if (key === 'skewness_corrected') {
    const skewnessDetails = summary.skewness_details || [];
    const transformedFeatures = summary.skewed_features_transformed || [];
    
    // Separate transformed and detected-only features
    const transformed = skewnessDetails.filter(d => d.transformed);
    const detectedOnly = skewnessDetails.filter(d => !d.transformed);
    
    return (
      <div className="bg-purple-50 dark:bg-purple-950/30 border border-purple-200 dark:border-purple-800/50 rounded-xl p-6">
        <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-purple-900 dark:text-purple-100">
          <TrendingUp className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          Skewness Correction
        </h4>
        
        {/* Summary Stats */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="p-4 bg-purple-100/50 dark:bg-purple-900/20 rounded-lg border border-purple-200/50 dark:border-purple-800/30">
            <div className="text-2xl font-bold text-purple-700 dark:text-purple-400">
              {transformed.length}
            </div>
            <div className="text-xs text-purple-600 dark:text-purple-300 mt-1">
              Features Transformed
            </div>
          </div>
          <div className="p-4 bg-gray-100/50 dark:bg-gray-900/20 rounded-lg border border-gray-200/50 dark:border-gray-800/30">
            <div className="text-2xl font-bold text-gray-700 dark:text-gray-400">
              {detectedOnly.length}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-300 mt-1">
              Skewed (No Transform)
            </div>
          </div>
        </div>
        
        {/* Explanation */}
        <div className="mb-4 p-3 bg-purple-100/50 dark:bg-purple-900/20 rounded-lg border border-purple-200/50 dark:border-purple-800/30">
          <p className="text-sm text-purple-900 dark:text-purple-100 mb-2">
            <span className="font-semibold">Skewness Threshold:</span> |skewness| &gt; 0.75
          </p>
          <p className="text-xs text-purple-700 dark:text-purple-300">
            Features with high skewness were transformed using Yeo-Johnson power transformation to normalize their distribution. This improves model performance, especially for linear and distance-based algorithms.
          </p>
        </div>
        
        {/* Transformed Features */}
        {transformed.length > 0 && (
          <div className="mb-4">
            <h5 className="text-sm font-semibold mb-3 text-purple-900 dark:text-purple-100 flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Transformed Features ({transformed.length})
            </h5>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {transformed.map((detail, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.02 }}
                  className="p-4 bg-white/50 dark:bg-purple-900/20 rounded-lg border border-purple-100 dark:border-purple-800/30"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-mono font-semibold text-purple-900 dark:text-purple-100">
                      {detail.column}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold px-2 py-1 bg-purple-200 dark:bg-purple-800 rounded text-purple-900 dark:text-purple-100">
                        Skew: {detail.skewness.toFixed(2)}
                      </span>
                      {detail.method && (
                        <span className="text-xs font-semibold px-2 py-1 bg-purple-300 dark:bg-purple-700 rounded text-purple-900 dark:text-purple-100">
                          {detail.method}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-xs text-purple-700 dark:text-purple-300">
                    {Math.abs(detail.skewness) > 2 ? (
                      <span>⚠️ Highly skewed - transformation significantly improves distribution</span>
                    ) : (
                      <span>Moderately skewed - transformation helps normalize distribution</span>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
        
        {/* Detected but not transformed */}
        {detectedOnly.length > 0 && (
          <div>
            <h5 className="text-sm font-semibold mb-3 text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <Info className="w-4 h-4" />
              Skewed Features (Not Transformed)
            </h5>
            <div className="mb-3 p-3 bg-gray-100/50 dark:bg-gray-900/20 rounded-lg border border-gray-200/50 dark:border-gray-800/30">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                These features are skewed but weren't transformed because your model type (tree-based) is robust to skewness.
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-64 overflow-y-auto">
              {detectedOnly.map((detail, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.02 }}
                  className="p-3 bg-white/50 dark:bg-gray-900/20 rounded-lg border border-gray-100 dark:border-gray-800/30"
                >
                  <div className="font-mono font-semibold text-sm text-gray-900 dark:text-gray-100 mb-1">
                    {detail.column}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    Skew: {detail.skewness.toFixed(2)}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
        
        {skewnessDetails.length === 0 && (
          <div className="p-4 bg-gray-100/50 dark:bg-gray-900/20 rounded-lg border border-gray-200 dark:border-gray-700 text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              No skewed features detected or transformation not needed
            </p>
          </div>
        )}
      </div>
    );
  }
  
  if (key === 'multicollinearity_handled') {
    const multicollinearity = summary.multicollinearity || {};
    const pairsDetected = multicollinearity.pairs_detected || 0;
    const featuresDropped = multicollinearity.features_dropped || [];
    const featuresCombined = multicollinearity.features_combined || [];
    const modelFamily = multicollinearity.model_family || 'unknown';
    
    if (pairsDetected === 0) {
      return null;
    }
    
    return (
      <div className="bg-indigo-50 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-800/50 rounded-xl p-6">
        <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-indigo-900 dark:text-indigo-100">
          <GitBranch className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
          Multicollinearity Reduction
        </h4>
        
        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="p-4 bg-indigo-100/50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200/50 dark:border-indigo-800/30">
            <div className="text-2xl font-bold text-indigo-700 dark:text-indigo-400">
              {pairsDetected}
            </div>
            <div className="text-xs text-indigo-600 dark:text-indigo-300 mt-1">
              Correlated Pairs (r&gt;0.9)
            </div>
          </div>
          <div className="p-4 bg-red-100/50 dark:bg-red-900/20 rounded-lg border border-red-200/50 dark:border-red-800/30">
            <div className="text-2xl font-bold text-red-700 dark:text-red-400">
              {featuresDropped.length}
            </div>
            <div className="text-xs text-red-600 dark:text-red-300 mt-1">
              Features Dropped
            </div>
          </div>
          <div className="p-4 bg-green-100/50 dark:bg-green-900/20 rounded-lg border border-green-200/50 dark:border-green-800/30">
            <div className="text-2xl font-bold text-green-700 dark:text-green-400">
              {featuresCombined.length}
            </div>
            <div className="text-xs text-green-600 dark:text-green-300 mt-1">
              Features Combined
            </div>
          </div>
        </div>
        
        {/* Strategy Explanation */}
        <div className="mb-4 p-3 bg-indigo-100/50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200/50 dark:border-indigo-800/30">
          <p className="text-sm text-indigo-900 dark:text-indigo-100 mb-1">
            <span className="font-semibold">Strategy for {modelFamily} models:</span>
          </p>
          <p className="text-xs text-indigo-700 dark:text-indigo-300">
            {modelFamily === 'linear' && 'Severe impact - Features combined or dropped to prevent multicollinearity issues'}
            {modelFamily === 'distance_based' && 'Medium impact - Correlated features removed to avoid distance bias'}
            {modelFamily === 'tree_based' && 'Moderate impact - Features kept (trees naturally robust to correlation)'}
            {!['linear', 'distance_based', 'tree_based'].includes(modelFamily) && 'Features analyzed and handled appropriately'}
          </p>
        </div>
        
        {/* Combined Features */}
        {featuresCombined.length > 0 && (
          <div className="mb-4">
            <h5 className="text-sm font-semibold mb-3 text-green-900 dark:text-green-100 flex items-center gap-2">
              <Plus className="w-4 h-4" />
              Combined Features ({featuresCombined.length})
            </h5>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {featuresCombined.map((combined, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.03 }}
                  className="p-4 bg-white/50 dark:bg-green-900/20 rounded-lg border border-green-100 dark:border-green-800/30"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-mono font-semibold text-green-900 dark:text-green-100">
                      {combined.new_feature}
                    </span>
                    <span className="text-xs font-semibold px-2 py-1 bg-green-200 dark:bg-green-800 rounded text-green-900 dark:text-green-100">
                      {combined.method}
                    </span>
                  </div>
                  <div className="text-xs text-green-700 dark:text-green-300 mb-2">
                    <span className="font-medium">From:</span> {combined.original_features.join(' + ')}
                  </div>
                  <div className="text-xs text-green-600 dark:text-green-400 italic">
                    {combined.reason}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
        
        {/* Dropped Features */}
        {featuresDropped.length > 0 && (
          <div>
            <h5 className="text-sm font-semibold mb-3 text-red-900 dark:text-red-100 flex items-center gap-2">
              <XCircle className="w-4 h-4" />
              Dropped Features ({featuresDropped.length})
            </h5>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-48 overflow-y-auto">
              {featuresDropped.map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.02 }}
                  className="p-2 bg-white/50 dark:bg-red-900/20 rounded-lg border border-red-100 dark:border-red-800/30"
                >
                  <span className="text-sm font-mono text-red-900 dark:text-red-100">{feature}</span>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }
  
  if (key === 'missing_values_filled') {
    const imputationDetails = summary.imputation_details || [];
    
    if (imputationDetails.length === 0) {
      return null;
    }
    
    const totalMissing = imputationDetails.reduce((sum, item) => sum + item.missing_count, 0);
    
    return (
      <div className="bg-teal-50 dark:bg-teal-950/30 border border-teal-200 dark:border-teal-800/50 rounded-xl p-6">
        <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-teal-900 dark:text-teal-100">
          <BarChart3 className="w-5 h-5 text-teal-600 dark:text-teal-400" />
          Missing Values Imputed ({imputationDetails.length} columns)
        </h4>
        <div className="mb-4 p-4 bg-teal-100/50 dark:bg-teal-900/20 rounded-lg border border-teal-200/50 dark:border-teal-800/30">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-teal-900 dark:text-teal-100">
              Total Values Imputed
            </span>
            <span className="text-2xl font-bold text-teal-700 dark:text-teal-400">
              {totalMissing.toLocaleString()}
            </span>
          </div>
          <p className="text-xs text-teal-700 dark:text-teal-300 mt-2">
            Missing values filled across {imputationDetails.length} column{imputationDetails.length > 1 ? 's' : ''}
          </p>
        </div>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {imputationDetails.map((detail, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.03 }}
              className="p-4 bg-white/50 dark:bg-teal-900/20 rounded-lg border border-teal-100 dark:border-teal-800/30"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-mono font-semibold text-teal-900 dark:text-teal-100">
                  {detail.column}
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold px-2 py-1 bg-teal-200 dark:bg-teal-800 rounded text-teal-900 dark:text-teal-100">
                    {detail.type}
                  </span>
                  <span className="text-xs font-semibold px-2 py-1 bg-teal-300 dark:bg-teal-700 rounded text-teal-900 dark:text-teal-100">
                    {detail.missing_count} filled
                  </span>
                </div>
              </div>
              <div className="p-2 bg-teal-100/50 dark:bg-teal-800/20 rounded">
                <div className="text-xs text-teal-600 dark:text-teal-400 font-medium">Strategy</div>
                <div className="text-sm font-mono text-teal-900 dark:text-teal-100">{detail.strategy}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    );
  }
  
  if (key === 'class_imbalance_handled') {
    const originalDist = summary.original_class_distribution || {};
    const resampledDist = summary.resampled_class_distribution || {};
    const strategy = summary.imbalance_strategy || 'Unknown';
    
    if (Object.keys(originalDist).length === 0) {
      return null;
    }
    
    // Calculate totals and percentages
    const originalTotal = Object.values(originalDist).reduce((sum, count) => sum + count, 0);
    const resampledTotal = Object.values(resampledDist).reduce((sum, count) => sum + count, 0);
    
    const originalPercentages = {};
    const resampledPercentages = {};
    
    Object.keys(originalDist).forEach(className => {
      originalPercentages[className] = (originalDist[className] / originalTotal * 100).toFixed(1);
      if (resampledDist[className]) {
        resampledPercentages[className] = (resampledDist[className] / resampledTotal * 100).toFixed(1);
      }
    });
    
    // Calculate imbalance ratio
    const counts = Object.values(originalDist);
    const maxCount = Math.max(...counts);
    const minCount = Math.min(...counts);
    const imbalanceRatio = (maxCount / minCount).toFixed(2);
    
    const newCounts = Object.values(resampledDist);
    const newMaxCount = Math.max(...newCounts);
    const newMinCount = Math.min(...newCounts);
    const newImbalanceRatio = newCounts.length > 0 ? (newMaxCount / newMinCount).toFixed(2) : '-';
    
    return (
      <div className="bg-purple-50 dark:bg-purple-950/30 border border-purple-200 dark:border-purple-800/50 rounded-xl p-6">
        <h4 className="text-lg font-semibold mb-4 flex items-center gap-2 text-purple-900 dark:text-purple-100">
          <BarChart3 className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          Class Imbalance Handled
        </h4>
        
        {/* Strategy Badge */}
        <div className="mb-4 p-4 bg-purple-100/50 dark:bg-purple-900/20 rounded-lg border border-purple-200/50 dark:border-purple-800/30">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-purple-900 dark:text-purple-100">
              Resampling Strategy
            </span>
            <span className="px-3 py-1 bg-purple-200 dark:bg-purple-800 rounded-full text-sm font-semibold text-purple-900 dark:text-purple-100">
              {strategy}
            </span>
          </div>
        </div>
        
        {/* Before and After Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
          {/* Original Distribution */}
          <div className="p-4 bg-white/50 dark:bg-purple-900/20 rounded-lg border border-purple-100 dark:border-purple-800/30">
            <div className="flex items-center justify-between mb-3">
              <h5 className="text-sm font-semibold text-purple-900 dark:text-purple-100">
                Original Distribution
              </h5>
              <span className="text-xs px-2 py-1 bg-red-200 dark:bg-red-800 rounded text-red-900 dark:text-red-100">
                Imbalanced ({imbalanceRatio}:1)
              </span>
            </div>
            <div className="space-y-2">
              {Object.entries(originalDist).map(([className, count]) => (
                <div key={className} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-mono text-purple-900 dark:text-purple-100">Class {className}</span>
                    <span className="font-semibold text-purple-700 dark:text-purple-300">
                      {count.toLocaleString()} ({originalPercentages[className]}%)
                    </span>
                  </div>
                  <div className="w-full bg-purple-200 dark:bg-purple-800/30 rounded-full h-2">
                    <div
                      className="bg-purple-600 dark:bg-purple-400 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${originalPercentages[className]}%` }}
                    />
                  </div>
                </div>
              ))}
              <div className="pt-2 mt-2 border-t border-purple-200 dark:border-purple-800">
                <div className="flex items-center justify-between text-xs font-semibold">
                  <span className="text-purple-900 dark:text-purple-100">Total Samples</span>
                  <span className="text-purple-700 dark:text-purple-300">{originalTotal.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Resampled Distribution */}
          {Object.keys(resampledDist).length > 0 && (
            <div className="p-4 bg-white/50 dark:bg-purple-900/20 rounded-lg border border-purple-100 dark:border-purple-800/30">
              <div className="flex items-center justify-between mb-3">
                <h5 className="text-sm font-semibold text-purple-900 dark:text-purple-100">
                  After Resampling
                </h5>
                <span className="text-xs px-2 py-1 bg-green-200 dark:bg-green-800 rounded text-green-900 dark:text-green-100">
                  Balanced ({newImbalanceRatio}:1)
                </span>
              </div>
              <div className="space-y-2">
                {Object.entries(resampledDist).map(([className, count]) => (
                  <div key={className} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-mono text-purple-900 dark:text-purple-100">Class {className}</span>
                      <span className="font-semibold text-purple-700 dark:text-purple-300">
                        {count.toLocaleString()} ({resampledPercentages[className]}%)
                      </span>
                    </div>
                    <div className="w-full bg-purple-200 dark:bg-purple-800/30 rounded-full h-2">
                      <div
                        className="bg-green-600 dark:bg-green-400 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${resampledPercentages[className]}%` }}
                      />
                    </div>
                  </div>
                ))}
                <div className="pt-2 mt-2 border-t border-purple-200 dark:border-purple-800">
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span className="text-purple-900 dark:text-purple-100">Total Samples</span>
                    <span className="text-purple-700 dark:text-purple-300">{resampledTotal.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Summary */}
        <div className="p-3 bg-purple-100/50 dark:bg-purple-900/20 rounded-lg border border-purple-200/50 dark:border-purple-800/30">
          <div className="flex items-center gap-2 text-sm text-purple-900 dark:text-purple-100">
            <svg className="w-4 h-4 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>
              Added <span className="font-semibold">{(resampledTotal - originalTotal).toLocaleString()}</span> samples 
              to balance class distribution and improve model accuracy
            </span>
          </div>
        </div>
      </div>
    );
  }
  
  return null;
};

export default ProcessedDatasetDetail;
