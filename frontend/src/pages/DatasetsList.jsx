/**
 * Datasets list page
 */
import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { createPortal } from 'react-dom';
import { datasetsAPI } from '../utils/api';

const DatasetsList = () => {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [deleteDialogDataset, setDeleteDialogDataset] = useState(null);
  const [cascadeDeleteDialog, setCascadeDeleteDialog] = useState(null); // For cascade warning
  const [editDialogDataset, setEditDialogDataset] = useState(null);
  const [editFormData, setEditFormData] = useState({ name: '', description: '' });
  const [selectedDatasets, setSelectedDatasets] = useState([]);
  const [bulkDeleteDialog, setBulkDeleteDialog] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState('desc');
  const [openMenuId, setOpenMenuId] = useState(null);
  const [downloadingId, setDownloadingId] = useState(null);
  const itemsPerPage = 10;

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      const data = await datasetsAPI.list();
      setDatasets(data);
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (datasetId, cascade = false) => {
    setIsDeleting(true);
    let hasError = false;
    let errorStatus = null;
    try {
      await datasetsAPI.delete(datasetId, cascade);
      fetchDatasets();
      setDeleteDialogDataset(null);
    } catch (error) {
      hasError = true;
      errorStatus = error.response?.status;
      console.error('Failed to delete dataset:', error);
      
      // Check if error is due to preprocessing data existing
      if (error.response?.status === 409) {
        // Show cascade delete warning
        const dataset = datasets.find(d => d.id === datasetId);
        setCascadeDeleteDialog(dataset);
        setDeleteDialogDataset(null);
      } else {
        alert('Failed to delete dataset. Please try again.');
        setDeleteDialogDataset(null);
      }
    } finally {
      setIsDeleting(false);
    }
  };

  const handleCascadeDelete = async (datasetId) => {
    setIsDeleting(true);
    try {
      await datasetsAPI.delete(datasetId, true); // cascade = true
      fetchDatasets();
      setCascadeDeleteDialog(null);
    } catch (error) {
      console.error('Failed to cascade delete dataset:', error);
      alert('Failed to delete dataset. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleEdit = (dataset) => {
    setEditDialogDataset(dataset);
    setEditFormData({
      name: dataset.name,
      description: dataset.description || ''
    });
  };

  const handleUpdate = async (e) => {
    e.preventDefault();
    setIsDeleting(true);
    try {
      await datasetsAPI.update(
        editDialogDataset.id,
        editFormData.name,
        editFormData.description
      );
      fetchDatasets();
      setEditDialogDataset(null);
    } catch (error) {
      console.error('Failed to update dataset:', error);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDownload = async (datasetId) => {
    setDownloadingId(datasetId);
    try {
      await datasetsAPI.download(datasetId, false); // false = original file
    } catch (error) {
      console.error('Failed to download dataset:', error);
      alert('Failed to download dataset. Please try again.');
    } finally {
      setDownloadingId(null);
    }
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

  const handleBulkDelete = async () => {
    setIsDeleting(true);
    try {
      await datasetsAPI.bulkDelete(selectedDatasets);
      setSelectedDatasets([]);
      fetchDatasets();
    } catch (error) {
      console.error('Failed to bulk delete datasets:', error);
    } finally {
      setIsDeleting(false);
      setBulkDeleteDialog(false);
    }
  };

  // Filter datasets based on search query
  const filteredDatasets = datasets.filter(dataset =>
    dataset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (dataset.description && dataset.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

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
      case 'columns_selected':
        // Sort by whether columns are selected (true values first)
        aValue = a.selected_columns && a.selected_columns.length > 0 && a.selected_columns.length < (a.column_names?.length || 0) ? 1 : 0;
        bValue = b.selected_columns && b.selected_columns.length > 0 && b.selected_columns.length < (b.column_names?.length || 0) ? 1 : 0;
        break;
      case 'eda_available':
        // Sort by whether EDA is available (true values first)
        aValue = a.eda_results ? 1 : 0;
        bValue = b.eda_results ? 1 : 0;
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
  }, [searchQuery, sortBy, sortOrder]);

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
          <h1 className="text-4xl font-bold">Datasets</h1>
          <p className="mt-2 text-muted-foreground">
            Manage and explore your uploaded datasets
          </p>
        </div>
        <Link
          to="/dashboard/datasets/upload"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium bg-primary hover:bg-primary/90 border-2 border-primary shadow-sm hover:shadow-md transition-all text-primary-foreground dark:text-white"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Upload Dataset
        </Link>
      </div>

      {datasets.length === 0 ? (
        <div className="bg-card border rounded-2xl p-16 text-center">
          <div className="mx-auto w-20 h-20 bg-primary/10 rounded-full flex items-center justify-center mb-6">
            <svg className="w-10 h-10 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold mb-2">No datasets yet</h3>
          <p className="text-muted-foreground mb-6">
            Get started by uploading your first dataset.
          </p>
          <Link
            to="/dashboard/datasets/upload"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-primary hover:bg-primary/90 transition-colors shadow-sm text-primary-foreground dark:text-white"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 1 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            Upload Dataset
          </Link>
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
                  <option value="columns_selected">Columns Selected</option>
                  <option value="eda_available">EDA Available</option>
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
          <div className="flex items-center justify-between text-sm text-muted-foreground px-1">
            <span>
              Showing {sortedDatasets.length === 0 ? 0 : startIndex + 1} to {Math.min(endIndex, sortedDatasets.length)} of {sortedDatasets.length} datasets
            </span>
            {selectedDatasets.length > 0 && (
              <button
                onClick={() => setBulkDeleteDialog(true)}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-white bg-red-500 hover:bg-red-600 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Delete {selectedDatasets.length} Selected
              </button>
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
                      onClick={() => navigate(`/dashboard/datasets/${dataset.id}`)}
                    >
                      <div className="flex items-center gap-3">
                        <div className="p-1.5 bg-primary/10 rounded-lg">
                          <svg className="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </div>
                        <div>
                          <span className="text-sm font-semibold truncate max-w-xs block">{dataset.name.length > 50 ? dataset.name.substring(0, 50) + '...' : dataset.name}</span>
                          <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
                            {dataset.eda_results && (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
                                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                                EDA Available
                              </span>
                            )}
                            {dataset.selected_columns && dataset.selected_columns.length > 0 && dataset.selected_columns.length < (dataset.column_names?.length || 0) && (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20">
                                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                                </svg>
                                Modified Dataset
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
                        {/* Analyze/Re-run EDA Button */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/dashboard/datasets/upload?dataset=${dataset.id}`);
                          }}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-white bg-primary hover:bg-primary/90 transition-colors"
                          title={dataset.eda_results ? "Re-analyze dataset" : "Analyze dataset"}
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                          <span className="text-white">Analyze</span>
                        </button>
                        
                        {/* 3-dot menu */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setOpenMenuId(openMenuId === dataset.id ? null : dataset.id);
                          }}
                          data-menu-id={dataset.id}
                          className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-muted transition-colors"
                          title="Actions"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
                          </svg>
                        </button>
                        
                        {/* Dropdown Menu */}
                        {openMenuId === dataset.id && createPortal(
                          <>
                            <div 
                              className="fixed inset-0 z-[9998]" 
                              onClick={(e) => {
                                e.stopPropagation();
                                setOpenMenuId(null);
                              }}
                            ></div>
                            <div 
                              className="fixed w-48 bg-card border rounded-lg shadow-lg z-[9999] overflow-hidden"
                              style={{
                                top: `${document.querySelector(`[data-menu-id="${dataset.id}"]`)?.getBoundingClientRect().bottom + 8}px`,
                                right: `${window.innerWidth - document.querySelector(`[data-menu-id="${dataset.id}"]`)?.getBoundingClientRect().right}px`
                              }}
                              onClick={(e) => e.stopPropagation()}
                            >
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDownload(dataset.id);
                                  setOpenMenuId(null);
                                }}
                                disabled={downloadingId === dataset.id}
                                className="w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left"
                              >
                                <svg className="w-4 h-4 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                </svg>
                                <div className="flex-1">
                                  <div className="font-medium">Download</div>
                                </div>
                                {downloadingId === dataset.id && (
                                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                                )}
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleEdit(dataset);
                                  setOpenMenuId(null);
                                }}
                                className="w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-muted transition-colors text-left border-t"
                              >
                                <svg className="w-4 h-4 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                </svg>
                                <div className="flex-1">
                                  <div className="font-medium">Edit</div>
                                </div>
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setDeleteDialogDataset(dataset);
                                  setOpenMenuId(null);
                                }}
                                className="w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-red-500/10 transition-colors text-left border-t"
                              >
                                <svg className="w-4 h-4 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                                <div className="flex-1">
                                  <div className="font-medium text-red-600 dark:text-red-400">Delete</div>
                                </div>
                              </button>
                            </div>
                          </>,
                          document.body
                        )}
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
                <h3 className="text-lg font-semibold mb-2">Delete Dataset</h3>
                <p className="text-muted-foreground mb-6">
                  Are you sure you want to delete "{deleteDialogDataset.name}"? This action cannot be undone.
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
                    onClick={() => handleDelete(deleteDialogDataset.id)}
                    disabled={isDeleting}
                    className="px-4 py-2 rounded-lg font-medium text-white bg-red-500 hover:bg-red-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isDeleting ? 'Deleting...' : 'Delete'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* Cascade Delete Warning Dialog */}
      {cascadeDeleteDialog && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center" style={{ zIndex: 99999 }} onClick={() => setCascadeDeleteDialog(null)}>
          <div className="bg-card border rounded-2xl p-6 max-w-md w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start gap-4">
              <div className="p-3 bg-orange-500/10 rounded-lg shrink-0">
                <svg className="w-6 h-6 text-orange-600 dark:text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold mb-2">Delete Dataset & Processed Version</h3>
                <p className="text-muted-foreground mb-4">
                  This dataset has a processed version. Deleting it will also delete:
                </p>
                <div className="bg-muted/50 rounded-lg p-3 mb-4">
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li className="flex items-center gap-2">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Original dataset
                    </li>
                    <li className="flex items-center gap-2">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Processed dataset
                    </li>
                    <li className="flex items-center gap-2">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Preprocessing pipeline
                    </li>
                  </ul>
                </div>
                <p className="text-sm text-muted-foreground mb-6">
                  This action cannot be undone. Are you sure?
                </p>
                <div className="flex gap-3 justify-end">
                  <button
                    onClick={() => setCascadeDeleteDialog(null)}
                    disabled={isDeleting}
                    className="px-4 py-2 rounded-lg font-medium bg-muted hover:bg-muted/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => handleCascadeDelete(cascadeDeleteDialog.id)}
                    disabled={isDeleting}
                    className="px-4 py-2 rounded-lg font-medium text-white bg-red-500 hover:bg-red-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isDeleting ? 'Deleting...' : 'Delete Both'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* Edit Dataset Dialog */}
      {editDialogDataset && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center" style={{ zIndex: 99999 }} onClick={() => setEditDialogDataset(null)}>
          <div className="bg-card border rounded-2xl p-6 max-w-md w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start gap-4">
              <div className="p-3 bg-blue-500/10 rounded-lg shrink-0">
                <svg className="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold mb-4">Edit Dataset</h3>
                <form onSubmit={handleUpdate} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-1.5">Dataset Name</label>
                    <input
                      type="text"
                      value={editFormData.name}
                      onChange={(e) => setEditFormData({ ...editFormData, name: e.target.value })}
                      required
                      className="w-full px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
                      placeholder="Enter dataset name"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1.5">Description (Optional)</label>
                    <textarea
                      value={editFormData.description}
                      onChange={(e) => setEditFormData({ ...editFormData, description: e.target.value })}
                      rows={3}
                      className="w-full px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                      placeholder="Enter description"
                    />
                  </div>
                  <div className="flex gap-3 justify-end pt-2">
                    <button
                      type="button"
                      onClick={() => setEditDialogDataset(null)}
                      className="px-4 py-2 rounded-lg font-medium bg-muted hover:bg-muted/80 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={isDeleting}
                      className="px-4 py-2 rounded-lg font-medium text-white bg-blue-500 hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isDeleting ? 'Saving...' : 'Save Changes'}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* Bulk Delete Confirmation Dialog */}
      {bulkDeleteDialog && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center" style={{ zIndex: 99999 }} onClick={() => setBulkDeleteDialog(false)}>
          <div className="bg-card border rounded-2xl p-6 max-w-md w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start gap-4">
              <div className="p-3 bg-red-500/10 rounded-lg shrink-0">
                <svg className="w-6 h-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold mb-2">Delete Multiple Datasets</h3>
                <p className="text-muted-foreground mb-6">
                  Are you sure you want to delete {selectedDatasets.length} dataset{selectedDatasets.length > 1 ? 's' : ''}? This action cannot be undone.
                </p>
                <div className="flex gap-3 justify-end">
                  <button
                    onClick={() => setBulkDeleteDialog(false)}
                    className="px-4 py-2 rounded-lg font-medium bg-muted hover:bg-muted/80 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleBulkDelete}
                    disabled={isDeleting}
                    className="px-4 py-2 rounded-lg font-medium text-white bg-red-500 hover:bg-red-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isDeleting ? 'Deleting...' : `Delete ${selectedDatasets.length} Dataset${selectedDatasets.length > 1 ? 's' : ''}`}
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

export default DatasetsList;
