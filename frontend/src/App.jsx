/**
 * Main App component with routing
 */
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import Dashboard from './pages/Dashboard';
import DashboardHome from './pages/DashboardHome';
import DatasetsList from './pages/DatasetsList';
import DatasetUpload from './pages/DatasetUpload';
import DatasetDetail from './pages/DatasetDetail';
import ProcessedDatasets from './pages/ProcessedDatasets';
import ProcessedDatasetDetail from './pages/ProcessedDatasetDetail';
import PreprocessDataset from './pages/PreprocessDataset';
import Documentation from './pages/Documentation';
import AutoMLModels from './pages/AutoMLModels';
import TrainAutoMLModel from './pages/TrainAutoMLModel';
import AutoMLModelDetail from './pages/AutoMLModelDetail';
import TestModel from './pages/TestModel';

function App() {
  return (
    <Router>
      <ThemeProvider>
        <Routes>
          {/* Dashboard Routes */}
          <Route path="/dashboard" element={<Dashboard />}>
            <Route index element={<DashboardHome />} />
            <Route path="datasets" element={<DatasetsList />} />
            <Route path="datasets/upload" element={<DatasetUpload />} />
            <Route path="datasets/:id" element={<DatasetDetail />} />
            <Route path="processed" element={<ProcessedDatasets />} />
            <Route path="processed/:id" element={<ProcessedDatasetDetail />} />
            <Route path="processed/:id/preprocess" element={<PreprocessDataset />} />
            <Route path="models" element={<Navigate to="/dashboard/models/automl" replace />} />
            <Route path="models/automl" element={<AutoMLModels />} />
            <Route path="models/train" element={<TrainAutoMLModel />} />
            <Route path="models/test" element={<TestModel />} />
            <Route path="models/:id" element={<AutoMLModelDetail />} />
            <Route path="docs" element={<Documentation />} />
          </Route>

          {/* Default Route */}
          <Route path="/" element={<Navigate to="/dashboard" />} />
          <Route path="*" element={<Navigate to="/dashboard" />} />
        </Routes>
      </ThemeProvider>
    </Router>
  );
}

export default App;
