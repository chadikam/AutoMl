/**
 * Processed Datasets page - manage and preprocess datasets
 */
import React, { useEffect, useState, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { createPortal } from 'react-dom';
import { datasetsAPI } from '../utils/api';

const ProcessedDatasets = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [allDatasets, setAllDatasets] = useState([]); // All available datasets for selection
  const [processedDatasets, setProcessedDatasets] = useState([]); // Only processed datasets for display
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedDatasets, setSelectedDatasets] = useState([]);
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState('desc');
  const [showPreprocessModal, setShowPreprocessModal] = useState(false);
  const [modalSearchQuery, setModalSearchQuery] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [openMenuId, setOpenMenuId] = useState(null); // Track which dataset menu is open
  const [deleteDialogDataset, setDeleteDialogDataset] = useState(null);
  const [showBulkDeleteDialog, setShowBulkDeleteDialog] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isBulkDownloading, setIsBulkDownloading] = useState(false);
  const [downloadingId, setDownloadingId] = useState(null);
  const [taskTypeFilter, setTaskTypeFilter] = useState('all');
  const itemsPerPage = 10;

  useEffect(() => {
    fetchDatasets();
  }, []);

  // Refetch when navigation state indicates a refresh is needed
  useEffect(() => {
    if (location.state?.refresh) {
      fetchDatasets();
      // Clear the refresh flag from state
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  // Refetch when window regains focus (useful if user processes in different context)
  useEffect(() => {
    const handleFocus = () => {
      fetchDatasets();
    };

    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, []);

  const fetchDatasets = async () => {
    try {
      // Fetch processed datasets for display
      const processedData = await datasetsAPI.listProcessed();
      setProcessedDatasets(processedData);
      
      // Fetch ALL datasets (original + processed) for preprocessing selection
      const allData = await datasetsAPI.list();
      setAllDatasets(allData);
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    setLoading(true);
    fetchDatasets();
  };

  const handleSelectAll = () => {
    if (selectedDatasets.length === paginatedDatasets.length) {
      setSelectedDatasets([]);
    } else {
      setSelectedDatasets(paginatedDatasets.map(d => d.id));
    }
  };

  const handleSelectDataset = (datasetId) => {
    setSelectedDatasets(prev => {
      if (prev.includes(datasetId)) {
        return prev.filter(id => id !== datasetId);
      } else {
        return [...prev, datasetId];
      }
    });
  };

  const handlePreprocessClick = (dataset) => {
    // Navigate to preprocessing page
    navigate(`/dashboard/processed/${dataset.id}/preprocess`);
  };

  const handleDeleteProcessed = async (datasetId) => {
    setIsDeleting(true);
    try {
      // Delete preprocessing data from backend
      await datasetsAPI.deletePreprocessing(datasetId);
      
      // Remove from local state
      const updatedProcessed = processedDatasets.filter(d => d.id !== datasetId);
      setProcessedDatasets(updatedProcessed);
      setAllDatasets(updatedProcessed);
      
      setDeleteDialogDataset(null);
    } catch (error) {
      console.error('Failed to delete preprocessing data:', error);
      alert('Failed to delete preprocessing data. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDownload = async (datasetId) => {
    setDownloadingId(datasetId);
    try {
      await datasetsAPI.download(datasetId, true);
    } catch (error) {
      console.error('Failed to download dataset:', error);
      alert('Failed to download dataset. Please try again.');
    } finally {
      setDownloadingId(null);
    }
  };

  const handleDownloadTrain = async (datasetId) => {
    setDownloadingId(datasetId);
    try {
      await datasetsAPI.downloadSplit(datasetId, 'train');
    } catch (error) {
      console.error('Failed to download train dataset:', error);
      alert('Failed to download train dataset. Please try again.');
    } finally {
      setDownloadingId(null);
      setOpenMenuId(null);
    }
  };

  const handleDownloadTest = async (datasetId) => {
    setDownloadingId(datasetId);
    try {
      await datasetsAPI.downloadSplit(datasetId, 'test');
    } catch (error) {
      console.error('Failed to download test dataset:', error);
      alert('Failed to download test dataset. Please try again.');
    } finally {
      setDownloadingId(null);
      setOpenMenuId(null);
    }
  };

  const handleBulkDelete = async () => {
    setIsDeleting(true);
    try {
      // Delete preprocessing data for all selected datasets
      await Promise.all(
        selectedDatasets.map(datasetId => datasetsAPI.deletePreprocessing(datasetId))
      );
      
      // Remove from local state
      const updatedProcessed = processedDatasets.filter(d => !selectedDatasets.includes(d.id));
      setProcessedDatasets(updatedProcessed);
      setAllDatasets(updatedProcessed);
      
      setSelectedDatasets([]);
    } catch (error) {
      console.error('Failed to delete preprocessing data:', error);
      alert('Failed to delete some datasets. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleBulkDownload = async () => {
    if (selectedDatasets.length === 0) {
      alert('Please select at least one dataset to download');
      return;
    }

    setIsBulkDownloading(true);
    let successCount = 0;
    let failCount = 0;
    const failedDatasets = [];
    
    try {
      console.log('Starting bulk download for datasets:', selectedDatasets);
      
      // Download each selected dataset sequentially
      for (const datasetId of selectedDatasets) {
        try {
          // Find the dataset to check if it has been preprocessed
          const dataset = processedDatasets.find(d => d.id === datasetId);
          console.log(`Attempting to download dataset:`, dataset);
          
          if (!dataset) {
            console.warn(`Dataset ${datasetId} not found in processed list`);
            failCount++;
            failedDatasets.push(datasetId);
            continue;
          }
          
          // Check if dataset has preprocessed file
          if (!dataset.preprocessed_file_path && !dataset.preprocessing_summary) {
            console.warn(`Dataset "${dataset.name}" hasn't been preprocessed yet`);
            failCount++;
            failedDatasets.push(dataset.name);
            continue;
          }
          
          console.log(`Downloading dataset ID: ${datasetId} (${dataset.name})`);
          const result = await datasetsAPI.download(datasetId, true);
          console.log(`Download result for ${datasetId}:`, result);
          successCount++;
          // Small delay between downloads to prevent overwhelming the browser
          await new Promise(resolve => setTimeout(resolve, 800));
        } catch (err) {
          console.error(`Failed to download dataset ${datasetId}:`, err);
          console.error('Error details:', err.response?.data || err.message);
          const dataset = processedDatasets.find(d => d.id === datasetId);
          failedDatasets.push(dataset?.name || datasetId);
          failCount++;
        }
      }
      
      console.log(`Bulk download complete. Success: ${successCount}, Failed: ${failCount}`);
      
      if (failCount === 0) {
        alert(`✅ Successfully downloaded ${successCount} dataset(s)`);
      } else if (successCount > 0) {
        alert(`⚠️ Downloaded ${successCount} dataset(s).\n${failCount} failed: ${failedDatasets.join(', ')}`);
      } else {
        alert(`❌ Failed to download all datasets.\n\nFailed datasets:\n${failedDatasets.join('\n')}\n\nNote: Make sure datasets have been preprocessed before downloading.`);
      }
    } catch (error) {
      console.error('Bulk download error:', error);
      alert('Failed to download datasets. Please check the console for details.');
    } finally {
      setIsBulkDownloading(false);
    }
  };

  // Filter datasets for modal search
  const filteredModalDatasets = allDatasets.filter(dataset =>
    dataset.name.toLowerCase().includes(modalSearchQuery.toLowerCase()) ||
    (dataset.description && dataset.description.toLowerCase().includes(modalSearchQuery.toLowerCase()))
  );

  // Filter processed datasets based on search query
  const filteredDatasets = processedDatasets.filter(dataset => {
    const matchesSearch = dataset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (dataset.description && dataset.description.toLowerCase().includes(searchQuery.toLowerCase()));
    
    // Apply task type filter
    if (taskTypeFilter === 'all') {
      return matchesSearch;
    }
    
    const taskType = dataset.preprocessing_summary?.task_type?.toLowerCase();
    return matchesSearch && taskType === taskTypeFilter;
  });

  // Sort datasets
  const sortedDatasets = [...filteredDatasets].sort((a, b) => {
    let aValue, bValue;
    
    switch (sortBy) {
      case 'name':
        aValue = a.name.toLowerCase();
        bValue = b.name.toLowerCase();
        break;
      case 'rows':
        aValue = a.rows;
        bValue = b.rows;
        break;
      case 'columns':
        aValue = a.columns;
        bValue = b.columns;
        break;
      case 'file_size':
        aValue = a.file_size || 0;
        bValue = b.file_size || 0;
        break;
      case 'created_at':
      default:
        aValue = new Date(a.created_at);
        bValue = new Date(b.created_at);
        break;
    }
    
    if (sortOrder === 'asc') {
      return aValue > bValue ? 1 : -1;
    } else {
      return aValue < bValue ? 1 : -1;
    }
  });

  // Calculate pagination
  const totalPages = Math.ceil(sortedDatasets.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedDatasets = sortedDatasets.slice(startIndex, endIndex);

  // Reset to page 1 when search changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, sortBy, sortOrder, taskTypeFilter]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold">Processed Datasets</h1>
          <p className="mt-2 text-muted-foreground">
            Preprocess and manage your datasets
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleRefresh}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium bg-secondary hover:bg-secondary/80 border-2 border-secondary shadow-sm hover:shadow-md transition-all"
            title="Refresh dataset list"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>
          <button
            onClick={() => setShowPreprocessModal(true)}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium bg-primary hover:bg-primary/90 border-2 border-primary shadow-sm hover:shadow-md transition-all text-primary-foreground dark:text-white"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Preprocess Data
          </button>
        </div>
      </div>

      {processedDatasets.length === 0 ? (
        <div className="bg-card border rounded-2xl p-16 text-center">
          <div className="mx-auto w-20 h-20 bg-primary/10 rounded-full flex items-center justify-center mb-6">
            <svg className="w-10 h-10 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold mb-2">No processed datasets yet</h3>
          <p className="text-muted-foreground mb-6">
            Click "Preprocess Data" to select a dataset and start preprocessing.
          </p>
          <button
            onClick={() => setShowPreprocessModal(true)}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-primary hover:bg-primary/90 transition-colors shadow-sm text-primary-foreground dark:text-white"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Preprocess Data
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Search and Sort Controls */}
          <div className="bg-card border rounded-lg p-4">
            <div className="flex flex-col sm:flex-row gap-4">
              {/* Search Box */}
              <div className="relative flex-1">
                <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  placeholder="Search datasets by name or description..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-background border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              {/* Sort Controls */}
              <div className="flex gap-2">
                {/* Task Type Filter */}
                <select
                  value={taskTypeFilter}
                  onChange={(e) => setTaskTypeFilter(e.target.value)}
                  className="px-3 py-2 bg-background border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm"
                >
                  <option value="all">All Types</option>
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                  <option value="clustering">Unsupervised</option>
                </select>

                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="px-3 py-2 bg-background border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm"
                >
                  <option value="created_at">Date Created</option>
                  <option value="name">Name</option>
                  <option value="rows">Rows</option>
                  <option value="columns">Columns</option>
                  <option value="file_size">Size</option>
                </select>

                <button
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                  className="px-3 py-2 bg-background border rounded-lg hover:bg-muted/50 transition-colors"
                  title={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    {sortOrder === 'asc' ? (
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12" />
                    ) : (
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h9m5-4v12m0 0l-4-4m4 4l4-4" />
                    )}
                  </svg>
                </button>
              </div>
            </div>
          </div>

          {/* Results Counter */}
          <div className="flex items-center justify-between text-sm px-1">
            <span className="text-muted-foreground">
              Showing {sortedDatasets.length === 0 ? 0 : startIndex + 1} to {Math.min(endIndex, sortedDatasets.length)} of {sortedDatasets.length} datasets
            </span>
            
            {/* Bulk Actions */}
            {selectedDatasets.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground font-medium">
                  {selectedDatasets.length} selected
                </span>
                <button
                  onClick={handleBulkDownload}
                  disabled={isBulkDownloading || isDeleting}
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-white text-sm"
                >
                  {isBulkDownloading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                      Downloading...
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      Download All
                    </>
                  )}
                </button>
                <button
                  onClick={() => setShowBulkDeleteDialog(true)}
                  disabled={isDeleting || isBulkDownloading}
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-white text-sm"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  Remove All
                </button>
              </div>
            )}
          </div>

          {/* Table */}
          {sortedDatasets.length === 0 ? (
            <div className="bg-card border rounded-2xl p-12 text-center">
              <svg className="mx-auto w-12 h-12 text-muted-foreground mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <h3 className="text-lg font-semibold mb-2">No datasets found</h3>
              <p className="text-muted-foreground">
                Try adjusting your search query
              </p>
            </div>
          ) : (
            <div className="bg-card border rounded-2xl overflow-hidden">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-border">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="py-4 pl-6 pr-3 text-left">
                        <input
                          type="checkbox"
                          checked={selectedDatasets.length === paginatedDatasets.length && paginatedDatasets.length > 0}
                          onChange={handleSelectAll}
                          className="w-4 h-4 rounded border-gray-300 text-primary focus:ring-primary cursor-pointer"
                        />
                      </th>
                      <th className="px-3 py-4 text-left text-xs font-semibold uppercase tracking-wider">
                        Name
                      </th>
                      <th className="px-3 py-4 text-left text-xs font-semibold uppercase tracking-wider">
                        Rows
                      </th>
                      <th className="px-3 py-4 text-left text-xs font-semibold uppercase tracking-wider">
                        Columns
                      </th>
                      <th className="px-3 py-4 text-left text-xs font-semibold uppercase tracking-wider">
                        Size
                      </th>
                      <th className="px-3 py-4 text-left text-xs font-semibold uppercase tracking-wider">
                        Created
                      </th>
                      <th className="relative py-4 pl-3 pr-6">
                        <span className="sr-only">Actions</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {paginatedDatasets.map((dataset) => (
                      <tr 
                        key={dataset.id} 
                        className="hover:bg-muted/50 transition-colors"
                      >
                        <td className="whitespace-nowrap py-2 pl-6 pr-3" onClick={(e) => e.stopPropagation()}>
                          <input
                            type="checkbox"
                            checked={selectedDatasets.includes(dataset.id)}
                            onChange={() => handleSelectDataset(dataset.id)}
                            className="w-4 h-4 rounded border-gray-300 text-primary focus:ring-primary cursor-pointer"
                          />
                        </td>
                        <td 
                          className="whitespace-nowrap py-2 px-3 cursor-pointer"
                          onClick={() => navigate(`/dashboard/processed/${dataset.id}`)}
                        >
                          <div className="flex items-center gap-3">
                            <div className="p-1.5 bg-primary/10 rounded-lg">
                              <svg className="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                              </svg>
                            </div>
                            <div>
                              <span className="text-sm font-semibold truncate max-w-xs block">
                                {dataset.name.length > 50 ? dataset.name.substring(0, 50) + '...' : dataset.name}
                              </span>
                              <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
                                {dataset.preprocessing_summary?.task_type && (
                                  <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${
                                    dataset.preprocessing_summary.task_type === 'classification'
                                      ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20'
                                      : dataset.preprocessing_summary.task_type === 'regression'
                                      ? 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20'
                                      : 'bg-gray-500/10 text-gray-600 dark:text-gray-400 border-gray-500/20'
                                  }`}>
                                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      {dataset.preprocessing_summary.task_type === 'classification' ? (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                                      ) : dataset.preprocessing_summary.task_type === 'regression' ? (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                                      ) : (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                                      )}
                                    </svg>
                                    {dataset.preprocessing_summary.task_type.charAt(0).toUpperCase() + dataset.preprocessing_summary.task_type.slice(1)}
                                  </span>
                                )}
                                {dataset.preprocessing_summary?.has_train_test_split && (
                                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-orange-500/10 text-orange-600 dark:text-orange-400 border border-orange-500/20">
                                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                                    </svg>
                                    Train/Test Split
                                  </span>
                                )}
                                {dataset.preprocessing_summary?.text_features?.columns?.length > 0 && (
                                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    TF-IDF
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-sm text-muted-foreground">
                          <span className="font-medium">{dataset.rows.toLocaleString()}</span>
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-sm text-muted-foreground">
                          <span className="font-medium">{dataset.columns}</span>
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-sm text-muted-foreground">
                          <span className="font-medium">
                            {dataset.file_size ? (dataset.file_size / 1024).toFixed(1) + ' KB' : 'N/A'}
                          </span>
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-sm text-muted-foreground">
                          {new Date(dataset.created_at).toLocaleDateString()}
                        </td>
                        <td className="relative whitespace-nowrap py-2 pl-3 pr-6 text-right">
                          <div className="flex items-center gap-2 justify-end">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handlePreprocessClick(dataset);
                              }}
                              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-white bg-primary hover:bg-primary/90 transition-colors"
                              title="View preprocessing details"
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                              </svg>
                              <span className="text-white">Analyze</span>
                            </button>
                            
                            {/* 3-dot menu */}
                            <div className="relative">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setOpenMenuId(openMenuId === dataset.id ? null : dataset.id);
                                }}
                                data-menu-id={dataset.id}
                                className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-muted/50 transition-colors"
                                title="More options"
                              >
                                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                  <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
                                </svg>
                              </button>
                              
                              {/* Dropdown menu */}
                              {openMenuId === dataset.id && createPortal(
                                <>
                                  {/* Backdrop to close menu when clicking outside */}
                                  <div 
                                    className="fixed inset-0 z-[9998]" 
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setOpenMenuId(null);
                                    }}
                                  />
                                  <div 
                                    className="fixed w-56 bg-card border rounded-lg shadow-lg z-[9999] py-1"
                                    style={{
                                      top: `${document.querySelector(`[data-menu-id="${dataset.id}"]`)?.getBoundingClientRect().bottom + 8}px`,
                                      right: `${window.innerWidth - document.querySelector(`[data-menu-id="${dataset.id}"]`)?.getBoundingClientRect().right}px`
                                    }}
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    {dataset.preprocessing_summary?.has_train_test_split && (
                                      <>
                                        <button
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            handleDownloadTrain(dataset.id);
                                          }}
                                          disabled={downloadingId === dataset.id}
                                          className="w-full text-left px-4 py-2 text-sm hover:bg-muted/50 transition-colors flex items-center gap-2 disabled:opacity-50"
                                        >
                                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                          </svg>
                                          Download Train Dataset
                                        </button>
                                        <button
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            handleDownloadTest(dataset.id);
                                          }}
                                          disabled={downloadingId === dataset.id}
                                          className="w-full text-left px-4 py-2 text-sm hover:bg-muted/50 transition-colors flex items-center gap-2 disabled:opacity-50"
                                        >
                                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                          </svg>
                                          Download Test Dataset
                                        </button>
                                        <div className="border-t my-1" />
                                      </>
                                    )}
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleDownload(dataset.id);
                                        setOpenMenuId(null);
                                      }}
                                      disabled={downloadingId === dataset.id}
                                      className="w-full text-left px-4 py-2 text-sm hover:bg-muted/50 transition-colors flex items-center gap-2 disabled:opacity-50"
                                    >
                                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                      </svg>
                                      Download Full Dataset
                                    </button>
                                    <div className="border-t my-1" />
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setDeleteDialogDataset(dataset);
                                        setOpenMenuId(null);
                                      }}
                                      className="w-full text-left px-4 py-2 text-sm hover:bg-red-50 dark:hover:bg-red-950/30 text-red-600 dark:text-red-400 transition-colors flex items-center gap-2"
                                    >
                                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                      </svg>
                                      Delete Dataset
                                    </button>
                                  </div>
                                </>,
                                document.body
                              )}
                            </div>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Pagination Controls */}
          {sortedDatasets.length > 0 && totalPages > 1 && (
            <div className="flex items-center justify-between px-1">
              <button
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-card border hover:bg-muted/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Previous
              </button>

              <div className="flex items-center gap-2">
                {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                  <button
                    key={page}
                    onClick={() => setCurrentPage(page)}
                    className={`px-3 py-1 rounded-lg font-medium transition-colors ${
                      currentPage === page
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-card border hover:bg-muted/50'
                    }`}
                  >
                    {page}
                  </button>
                ))}
              </div>

              <button
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-card border hover:bg-muted/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Preprocess Modal - Choose Dataset */}
      {showPreprocessModal && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center" style={{ zIndex: 99999 }} onClick={() => {
          setShowPreprocessModal(false);
          setModalSearchQuery('');
          setIsDropdownOpen(false);
        }}>
          <div className="bg-card border rounded-2xl p-6 max-w-md w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start gap-4 mb-6">
              <div className="p-3 bg-primary/10 rounded-lg shrink-0">
                <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold mb-2">Choose Dataset to Preprocess</h3>
                <p className="text-muted-foreground text-sm">
                  Select a dataset from the dropdown to start preprocessing
                </p>
              </div>
              <button
                onClick={() => {
                  setShowPreprocessModal(false);
                  setModalSearchQuery('');
                  setIsDropdownOpen(false);
                }}
                className="p-2 hover:bg-muted rounded-lg transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Dropdown with Search */}
            {allDatasets.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <svg className="mx-auto w-12 h-12 text-muted-foreground mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="font-medium">No datasets available</p>
                <p className="text-sm mt-1">Upload a dataset first to start preprocessing</p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Search Input with Dropdown */}
                <div className="relative">
                  <label className="block text-sm font-medium mb-2">Select Dataset</label>
                  <div className="relative">
                    <input
                      type="text"
                      value={modalSearchQuery}
                      onChange={(e) => {
                        setModalSearchQuery(e.target.value);
                        setIsDropdownOpen(true);
                      }}
                      onFocus={() => setIsDropdownOpen(true)}
                      placeholder="Search datasets..."
                      className="w-full px-4 py-2.5 pr-10 bg-background border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                    <svg className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>

                  {/* Dropdown List */}
                  {isDropdownOpen && (
                    <div className="absolute z-10 w-full mt-2 bg-background border rounded-lg shadow-lg max-h-64 overflow-y-auto">
                      {filteredModalDatasets.length === 0 ? (
                        <div className="px-4 py-8 text-center text-muted-foreground text-sm">
                          <svg className="mx-auto w-10 h-10 text-muted-foreground mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                          </svg>
                          <p>No datasets found</p>
                          <p className="text-xs mt-1">Try a different search term</p>
                        </div>
                      ) : (
                        filteredModalDatasets.map((dataset) => {
                          const isAlreadyProcessed = processedDatasets.find(d => d.id === dataset.id);
                          return (
                            <button
                              key={dataset.id}
                              onClick={() => handlePreprocessClick(dataset)}
                              className="w-full text-left px-4 py-3 hover:bg-muted/50 transition-colors border-b last:border-b-0 focus:outline-none focus:bg-muted/50"
                            >
                              <div className="flex items-center gap-3">
                                <div className="p-1.5 bg-primary/10 rounded-lg shrink-0">
                                  <svg className="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                  </svg>
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2">
                                    <span className="font-medium truncate">{dataset.name}</span>
                                    {isAlreadyProcessed && (
                                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20 shrink-0">
                                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                        </svg>
                                        Processed
                                      </span>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
                                    <span>{dataset.rows.toLocaleString()} rows</span>
                                    <span>•</span>
                                    <span>{dataset.columns} cols</span>
                                    <span>•</span>
                                    <span>{dataset.file_size ? (dataset.file_size / 1024).toFixed(1) + ' KB' : 'N/A'}</span>
                                  </div>
                                </div>
                              </div>
                            </button>
                          );
                        })
                      )}
                    </div>
                  )}
                </div>

                {/* Helper Text */}
                <div className="flex items-start gap-2 p-3 bg-muted/50 rounded-lg">
                  <svg className="w-5 h-5 text-primary shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div className="text-sm text-muted-foreground">
                    <p className="font-medium text-foreground">Tip:</p>
                    <p className="mt-0.5">Type to search through your datasets. Datasets already processed will be marked with a badge.</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>,
        document.body
      )}

      {/* Delete Confirmation Dialog */}
      {deleteDialogDataset && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center" style={{ zIndex: 99999 }} onClick={() => setDeleteDialogDataset(null)}>
          <div className="bg-card border rounded-2xl p-6 max-w-md w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start gap-4">
              <div className="p-3 bg-red-500/10 rounded-lg shrink-0">
                <svg className="w-6 h-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold mb-2">Remove from Processed List</h3>
                <p className="text-muted-foreground mb-6">
                  Are you sure you want to remove "{deleteDialogDataset.name}" from the processed datasets list? 
                  This will not delete the actual dataset, only remove it from this list.
                </p>
                <div className="flex gap-3 justify-end">
                  <button
                    onClick={() => setDeleteDialogDataset(null)}
                    disabled={isDeleting}
                    className="px-4 py-2 rounded-lg font-medium bg-muted hover:bg-muted/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => handleDeleteProcessed(deleteDialogDataset.id)}
                    disabled={isDeleting}
                    className="px-4 py-2 rounded-lg font-medium text-white bg-red-500 hover:bg-red-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isDeleting ? 'Removing...' : 'Remove'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* Bulk Delete Confirmation Dialog */}
      {showBulkDeleteDialog && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center" style={{ zIndex: 99999 }} onClick={() => setShowBulkDeleteDialog(false)}>
          <div className="bg-card border rounded-2xl p-6 max-w-md w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start gap-4">
              <div className="p-3 bg-red-500/10 rounded-lg shrink-0">
                <svg className="w-6 h-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold mb-2">Remove from Processed List</h3>
                <p className="text-muted-foreground mb-4">
                  Are you sure you want to remove <strong>{selectedDatasets.length} dataset{selectedDatasets.length > 1 ? 's' : ''}</strong> from the processed datasets list?
                </p>
                <p className="text-sm text-muted-foreground mb-6">
                  This will not delete the actual datasets, only remove them from this list.
                </p>
                <div className="flex gap-3 justify-end">
                  <button
                    onClick={() => setShowBulkDeleteDialog(false)}
                    disabled={isDeleting}
                    className="px-4 py-2 rounded-lg font-medium bg-muted hover:bg-muted/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => {
                      handleBulkDelete();
                      setShowBulkDeleteDialog(false);
                    }}
                    disabled={isDeleting}
                    className="px-4 py-2 rounded-lg font-medium text-white bg-red-500 hover:bg-red-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isDeleting ? 'Removing...' : `Remove ${selectedDatasets.length} Dataset${selectedDatasets.length > 1 ? 's' : ''}`}
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

export default ProcessedDatasets;
