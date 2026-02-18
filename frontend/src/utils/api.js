/**
 * API client for AutoML Framework
 * Handles all HTTP requests to the backend API
 */
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor to handle errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.status, error.response?.data);
    return Promise.reject(error);
  }
);

// Datasets API
export const datasetsAPI = {
  upload: async (file, name, description) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    if (description) {
      formData.append('description', description);
    }

    const response = await apiClient.post('/api/datasets/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  list: async () => {
    const response = await apiClient.get('/api/datasets/');
    return response.data;
  },

  listProcessed: async () => {
    // Add cache-busting parameter to ensure fresh data
    const timestamp = Date.now();
    const response = await apiClient.get(`/api/datasets/processed/list?_t=${timestamp}`);
    return response.data;
  },

  get: async (datasetId) => {
    const response = await apiClient.get(`/api/datasets/${datasetId}`);
    return response.data;
  },

  preview: async (datasetId, rows = 10, usePreprocessed = true) => {
    const response = await apiClient.get(`/api/datasets/${datasetId}/preview?rows=${rows}&use_preprocessed=${usePreprocessed}`);
    return response.data;
  },

  update: async (datasetId, name, description) => {
    const formData = new FormData();
    formData.append('name', name);
    if (description !== null && description !== undefined) {
      formData.append('description', description);
    }

    const response = await apiClient.put(`/api/datasets/${datasetId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  updateSelectedColumns: async (datasetId, columns) => {
    const response = await apiClient.put(`/api/datasets/${datasetId}/selected_columns`, {
      selected_columns: columns,
    });
    return response.data;
  },

  runEDA: async (datasetId, selectedColumns = null) => {
    const response = await apiClient.post(`/api/datasets/${datasetId}/run-eda`, {
      selected_columns: selectedColumns,
    });
    return response.data;
  },

  getEDA: async (datasetId) => {
    const response = await apiClient.get(`/api/datasets/${datasetId}/eda`);
    return response.data;
  },

  delete: async (datasetId, cascade = false) => {
    const response = await apiClient.delete(`/api/datasets/${datasetId}?cascade=${cascade}`);
    return response.data;
  },

  deletePreprocessing: async (datasetId) => {
    const response = await apiClient.delete(`/api/datasets/${datasetId}/preprocessing`);
    return response.data;
  },

  bulkDelete: async (datasetIds) => {
    const deletePromises = datasetIds.map(id => apiClient.delete(`/api/datasets/${id}`));
    await Promise.all(deletePromises);
    return { success: true };
  },

  preprocess: async (datasetId, targetColumn = null, rareValueSelections = {}, outlierPreferences = {}) => {
    const params = targetColumn ? { target_column: targetColumn } : {};
    
    // Build request body
    const requestBody = {};
    
    if (Object.keys(rareValueSelections).length > 0) {
      requestBody.remove_rare_values = rareValueSelections;
    }
    
    if (Object.keys(outlierPreferences).length > 0) {
      requestBody.outlier_preferences = outlierPreferences;
    }
    
    // Send request
    const response = await apiClient.post(
      `/api/datasets/${datasetId}/preprocess`, 
      requestBody, 
      { params }
    );
    return response.data;
  },

  download: async (datasetId, usePreprocessed = true) => {
    const response = await apiClient.get(
      `/api/datasets/${datasetId}/download?use_preprocessed=${usePreprocessed}`,
      {
        responseType: 'blob', // Important for file downloads
      }
    );
    
    // Get filename from Content-Disposition header or use default
    const contentDisposition = response.headers['content-disposition'];
    console.log('📥 Content-Disposition header:', contentDisposition);
    console.log('📥 All headers:', response.headers);
    let filename = 'dataset.csv';
    if (contentDisposition) {
      // Try to match filename*=UTF-8''encoded_name first (RFC 5987)
      const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/);
      console.log('📥 UTF-8 match:', utf8Match);
      if (utf8Match) {
        filename = decodeURIComponent(utf8Match[1]);
      } else {
        // Fallback to regular filename="name" or filename=name
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        console.log('📥 Filename match:', filenameMatch);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
    }
    console.log('📥 Final filename:', filename);
    
    // Create download link and trigger download
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    
    return { success: true, filename };
  },

  downloadSplit: async (datasetId, splitType = 'train') => {
    const response = await apiClient.get(
      `/api/datasets/${datasetId}/download/split?split_type=${splitType}`,
      {
        responseType: 'blob',
      }
    );
    
    // Get filename from Content-Disposition header
    const contentDisposition = response.headers['content-disposition'];
    let filename = `${splitType}.csv`;
    if (contentDisposition) {
      // Try to match filename*=UTF-8''encoded_name first (RFC 5987)
      const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/);
      if (utf8Match) {
        filename = decodeURIComponent(utf8Match[1]);
      } else {
        // Fallback to regular filename="name" or filename=name
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
    }
    
    // Create download link and trigger download
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    
    return { success: true, filename };
  },
};

// Models API
export const modelsAPI = {
  train: async (modelData) => {
    const response = await apiClient.post('/api/models/train', modelData);
    return response.data;
  },

  list: async (datasetId = null) => {
    const url = datasetId ? `/api/models/?dataset_id=${datasetId}` : '/api/models/';
    const response = await apiClient.get(url);
    return response.data;
  },

  get: async (modelId) => {
    const response = await apiClient.get(`/api/models/${modelId}`);
    return response.data;
  },

  delete: async (modelId) => {
    const response = await apiClient.delete(`/api/models/${modelId}`);
    return response.data;
  },

  compare: async (modelIds) => {
    const params = new URLSearchParams();
    modelIds.forEach(id => params.append('model_ids', id));
    const response = await apiClient.get(`/api/models/compare?${params.toString()}`);
    return response.data;
  },
};

// AutoML API
export const automlAPI = {
  startTraining: async (trainingConfig) => {
    const response = await apiClient.post('/api/automl/train/start', trainingConfig);
    return response.data;
  },

  train: async (trainingConfig, trainingId, options = {}) => {
    const url = trainingId ? `/api/automl/train?training_id=${trainingId}` : '/api/automl/train';
    const response = await apiClient.post(url, trainingConfig, options);
    return response.data;
  },

  cancelTraining: async (trainingId) => {
    // Try new endpoint first, fall back to legacy
    try {
      const response = await apiClient.post('/api/automl/cancel-training');
      return response.data;
    } catch {
      const response = await apiClient.post(`/api/automl/cancel/${trainingId}`);
      return response.data;
    }
  },

  skipModel: async () => {
    const response = await apiClient.post('/api/automl/skip-model');
    return response.data;
  },

  getTrainingStatus: async () => {
    const response = await apiClient.get('/api/automl/training-status');
    return response.data;
  },

  listModels: async () => {
    const response = await apiClient.get('/api/automl/models');
    return response.data;
  },

  getModel: async (modelId) => {
    const response = await apiClient.get(`/api/automl/models/${modelId}`);
    return response.data;
  },

  deleteModel: async (modelId) => {
    const response = await apiClient.delete(`/api/automl/models/${modelId}`);
    return response.data;
  },

  downloadModel: async (modelId) => {
    const response = await apiClient.get(`/api/automl/models/${modelId}/download`, {
      responseType: 'blob',
    });
    return response.data;
  },

  exportModel: async (modelId) => {
    const response = await apiClient.get(`/api/automl/models/${modelId}/export`, {
      responseType: 'blob',
    });
    return response.data;
  },

  predict: async (modelId, inputData) => {
    const response = await apiClient.post(`/api/automl/models/${modelId}/predict`, inputData);
    return response.data;
  },

  getEncodingInfo: async (modelId) => {
    const response = await apiClient.get(`/api/automl/models/${modelId}/encoding-info`);
    return response.data;
  },

  getSystemInfo: async () => {
    const response = await apiClient.get('/api/automl/system-info');
    return response.data;
  },
};

export default apiClient;
