/**
 * Dataset Detail page with EDA visualization
 */
import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { datasetsAPI } from '../utils/api';
import EDASummary from '../components/ui/eda-summary';
import EDACharts from '../components/ui/eda-charts';

const DatasetDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [dataset, setDataset] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [activeTab, setActiveTab] = useState('preview');
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showActionsMenu, setShowActionsMenu] = useState(false);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    fetchDataset();
  }, [id]);

  useEffect(() => {
    if (dataset) {
      fetchPreview(); // Load preview when dataset is ready
    }
  }, [dataset]);

  const fetchDataset = async () => {
    try {
      const data = await datasetsAPI.get(id);
      setDataset(data);
      // Reset preview to refetch with correct preprocessed flag
      setPreview(null);
    } catch (error) {
      console.error('Failed to fetch dataset:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPreview = async () => {
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      // Always use original data (preprocessed=false) for regular dataset detail page
      const data = await datasetsAPI.preview(id, 10, false);
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
      await datasetsAPI.delete(id);
      navigate('/dashboard/datasets');
    } catch (error) {
      console.error('Failed to delete dataset:', error);
    } finally {
      setShowDeleteDialog(false);
    }
  };

  const handleDownload = async () => {
    setDownloading(true);
    try {
      await datasetsAPI.download(id, false); // false = download original version
    } catch (error) {
      console.error('Failed to download dataset:', error);
      alert('Failed to download dataset. Please try again.');
    } finally {
      setDownloading(false);
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
          to="/dashboard/datasets"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-primary-foreground bg-primary hover:bg-primary/90 transition-colors"
        >
          Back to Datasets
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
            onClick={() => navigate('/dashboard/datasets')}
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
                <button
                  onClick={() => {
                    handleDownload();
                    setShowActionsMenu(false);
                  }}
                  disabled={downloading}
                  className="w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left"
                >
                  <svg className="w-4 h-4 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  <div className="flex-1">
                    <div className="font-medium">Download Dataset</div>
                    <div className="text-xs text-muted-foreground">
                      {dataset.selected_columns ? 'Filtered columns' : 'Original data'}
                    </div>
                  </div>
                  {downloading && (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                  )}
                </button>
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
                      Permanently remove
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
          {dataset.eda_results && (
            <>
              <button
                onClick={() => setActiveTab('eda')}
                className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                  activeTab === 'eda'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
              >
                EDA Analysis
              </button>
              <button
                onClick={() => setActiveTab('charts')}
                className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                  activeTab === 'charts'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
              >
                Charts & Visualizations
              </button>
            </>
          )}
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
              <div className="overflow-x-auto overflow-y-auto max-h-[600px] space-y-0">
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
            ) : (
              <div className="p-16 text-center">
                <p className="text-muted-foreground">No preview available</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'eda' && dataset.eda_results && (
          <div className="bg-card border rounded-2xl p-6">
            <EDASummary edaResults={dataset.eda_results} />
          </div>
        )}

        {activeTab === 'charts' && dataset.eda_results && (
          <div className="bg-card border rounded-2xl p-6">
            <EDACharts edaResults={dataset.eda_results} />
          </div>
        )}
      </motion.div>

      {/* Delete Confirmation Dialog */}
      {showDeleteDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowDeleteDialog(false)}>
          <div className="bg-card border rounded-2xl p-6 max-w-md w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start gap-4">
              <div className="p-3 bg-red-500/10 rounded-lg shrink-0">
                <svg className="w-6 h-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold mb-2">Delete Dataset</h3>
                <p className="text-muted-foreground mb-6">
                  Are you sure you want to delete "{dataset?.name}"? This action cannot be undone.
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
                    Delete
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DatasetDetail;