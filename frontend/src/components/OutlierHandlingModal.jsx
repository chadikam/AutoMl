import React, { useState, useEffect } from 'react';
import { X, AlertTriangle, Info, TrendingUp, Target, Database, CheckCircle2, XCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const OutlierHandlingModal = ({ isOpen, onClose, outlierData, onApply, taskType }) => {
  const [selectedColumns, setSelectedColumns] = useState({});
  const [globalAction, setGlobalAction] = useState('keep'); // 'keep', 'cap', or 'remove'

  useEffect(() => {
    if (isOpen && outlierData) {
      // Initialize all columns with default action based on task type and severity
      const initial = {};
      Object.entries(outlierData.outliers_by_column || {}).forEach(([column, data]) => {
        if (data.count > 0) {
          // Default recommendation based on task type and outlier percentage
          let defaultAction = 'keep';
          
          if (taskType === 'regression') {
            // Numerical target: outliers bias model, recommend handling
            defaultAction = data.percentage > 10 ? 'cap' : 'keep';
          } else if (taskType === 'classification') {
            // Categorical target: be careful not to remove minority examples
            defaultAction = data.percentage > 15 ? 'cap' : 'keep';
          } else {
            // Unsupervised: outliers dominate distances, suggest robust handling
            defaultAction = data.percentage > 5 ? 'cap' : 'keep';
          }
          
          initial[column] = defaultAction;
        }
      });
      setSelectedColumns(initial);
    }
  }, [isOpen, outlierData, taskType]);

  if (!isOpen || !outlierData) return null;

  const columnsWithOutliers = Object.entries(outlierData.outliers_by_column || {})
    .filter(([_, data]) => data.count > 0)
    .sort((a, b) => b[1].percentage - a[1].percentage);

  const handleColumnToggle = (column, action) => {
    setSelectedColumns(prev => ({
      ...prev,
      [column]: action
    }));
  };

  const handleApplyToAll = (action) => {
    setGlobalAction(action);
    const updated = {};
    columnsWithOutliers.forEach(([column]) => {
      updated[column] = action;
    });
    setSelectedColumns(updated);
  };

  const handleApply = () => {
    onApply(selectedColumns);
    onClose();
  };

  const getActionCounts = () => {
    const counts = { keep: 0, cap: 0, remove: 0 };
    Object.values(selectedColumns).forEach(action => {
      counts[action] = (counts[action] || 0) + 1;
    });
    return counts;
  };

  const actionCounts = getActionCounts();

  const getTaskTypeWarning = () => {
    if (taskType === 'regression') {
      return {
        icon: Target,
        color: 'red',
        title: 'Regression Task - Outliers Bias Model',
        message: 'In regression, outliers can significantly bias your model predictions. Consider capping or removing extreme values, but consult domain experts first.'
      };
    } else if (taskType === 'classification') {
      return {
        icon: Target,
        color: 'yellow',
        title: 'Classification Task - Handle with Care',
        message: 'Be careful not to remove outliers that might be important minority class examples. Capping is often safer than removal for classification tasks.'
      };
    } else {
      return {
        icon: Database,
        color: 'blue',
        title: 'Unsupervised Learning - Outliers Dominate Distances',
        message: 'In clustering and other unsupervised methods, outliers can dominate distance calculations. Consider using robust methods or capping extreme values.'
      };
    }
  };

  const taskWarning = getTaskTypeWarning();

  return (
    <AnimatePresence>
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-hidden"
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-orange-500 to-red-500 p-6 text-white">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
                  <TrendingUp className="w-6 h-6" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold">Outlier Handling</h2>
                  <p className="text-orange-100 text-sm mt-1">
                    {columnsWithOutliers.length} column{columnsWithOutliers.length !== 1 ? 's' : ''} with outliers detected
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/20 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Critical Warning Banner */}
          <div className="p-4 bg-red-50 dark:bg-red-950/30 border-b-2 border-red-200 dark:border-red-800">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="font-semibold text-red-900 dark:text-red-100 mb-1">
                  ⚠️ Warning: Not All Outliers Are Bad!
                </h3>
                <p className="text-sm text-red-800 dark:text-red-200 leading-relaxed">
                  Outliers can represent important patterns, rare events, or critical data points. Removing or modifying outliers without domain knowledge can lead to loss of valuable information. <span className="font-semibold">Always consult a domain expert</span> before making decisions about outlier handling.
                </p>
              </div>
            </div>
          </div>

          {/* Task-Specific Warning */}
          <div className={`p-4 bg-${taskWarning.color}-50 dark:bg-${taskWarning.color}-950/20 border-b border-${taskWarning.color}-200 dark:border-${taskWarning.color}-800`}>
            <div className="flex items-start gap-3">
              <taskWarning.icon className={`w-5 h-5 text-${taskWarning.color}-600 dark:text-${taskWarning.color}-400 flex-shrink-0 mt-0.5`} />
              <div className="flex-1">
                <h4 className={`font-semibold text-${taskWarning.color}-900 dark:text-${taskWarning.color}-100 text-sm mb-1`}>
                  {taskWarning.title}
                </h4>
                <p className={`text-xs text-${taskWarning.color}-800 dark:text-${taskWarning.color}-200`}>
                  {taskWarning.message}
                </p>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-400px)]">
            {/* Global Actions */}
            <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-xl border border-gray-200 dark:border-gray-700">
              <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                <Info className="w-4 h-4" />
                Quick Actions - Apply to All Columns
              </h3>
              <div className="flex gap-3">
                <button
                  onClick={() => handleApplyToAll('keep')}
                  className="flex-1 px-4 py-3 bg-green-100 dark:bg-green-900/30 hover:bg-green-200 dark:hover:bg-green-900/50 text-green-900 dark:text-green-100 rounded-lg border-2 border-green-300 dark:border-green-700 transition-colors font-medium text-sm"
                >
                  <CheckCircle2 className="w-4 h-4 inline mr-2" />
                  Keep All Outliers
                </button>
                <button
                  onClick={() => handleApplyToAll('cap')}
                  className="flex-1 px-4 py-3 bg-orange-100 dark:bg-orange-900/30 hover:bg-orange-200 dark:hover:bg-orange-900/50 text-orange-900 dark:text-orange-100 rounded-lg border-2 border-orange-300 dark:border-orange-700 transition-colors font-medium text-sm"
                >
                  <TrendingUp className="w-4 h-4 inline mr-2" />
                  Cap All (Recommended)
                </button>
                <button
                  onClick={() => handleApplyToAll('remove')}
                  className="flex-1 px-4 py-3 bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-900/50 text-red-900 dark:text-red-100 rounded-lg border-2 border-red-300 dark:border-red-700 transition-colors font-medium text-sm"
                >
                  <XCircle className="w-4 h-4 inline mr-2" />
                  Remove All
                </button>
              </div>
            </div>

            {/* Column-by-Column Selection */}
            <div className="space-y-3">
              <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Per-Column Outlier Handling
              </h3>
              {columnsWithOutliers.map(([column, data]) => (
                <div
                  key={column}
                  className="p-4 bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-orange-300 dark:hover:border-orange-700 transition-colors"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                        {column}
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          data.percentage > 15
                            ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200'
                            : data.percentage > 10
                            ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-200'
                            : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200'
                        }`}>
                          {data.percentage.toFixed(1)}% outliers
                        </span>
                      </h4>
                      <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        <span className="font-medium">{data.count}</span> outliers detected
                        {' • '}Range: [{data.min.toFixed(2)} - {data.max.toFixed(2)}]
                        {' • '}IQR bounds: [{data.lower_bound.toFixed(2)} - {data.upper_bound.toFixed(2)}]
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => handleColumnToggle(column, 'keep')}
                      className={`flex-1 px-3 py-2 rounded-lg border-2 transition-all text-sm font-medium ${
                        selectedColumns[column] === 'keep'
                          ? 'bg-green-100 dark:bg-green-900/30 border-green-500 dark:border-green-600 text-green-900 dark:text-green-100 shadow-sm'
                          : 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-green-300 dark:hover:border-green-700'
                      }`}
                    >
                      Keep
                    </button>
                    <button
                      onClick={() => handleColumnToggle(column, 'cap')}
                      className={`flex-1 px-3 py-2 rounded-lg border-2 transition-all text-sm font-medium ${
                        selectedColumns[column] === 'cap'
                          ? 'bg-orange-100 dark:bg-orange-900/30 border-orange-500 dark:border-orange-600 text-orange-900 dark:text-orange-100 shadow-sm'
                          : 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-orange-300 dark:hover:border-orange-700'
                      }`}
                    >
                      Cap (Winsorize)
                    </button>
                    <button
                      onClick={() => handleColumnToggle(column, 'remove')}
                      className={`flex-1 px-3 py-2 rounded-lg border-2 transition-all text-sm font-medium ${
                        selectedColumns[column] === 'remove'
                          ? 'bg-red-100 dark:bg-red-900/30 border-red-500 dark:border-red-600 text-red-900 dark:text-red-100 shadow-sm'
                          : 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-red-300 dark:hover:border-red-700'
                      }`}
                    >
                      Remove Rows
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Footer */}
          <div className="p-6 bg-gray-50 dark:bg-gray-900/50 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm text-gray-600 dark:text-gray-400">
                <span className="font-semibold text-gray-900 dark:text-gray-100">Summary:</span>
                {' '}
                <span className="text-green-600 dark:text-green-400 font-medium">{actionCounts.keep} Keep</span>
                {' • '}
                <span className="text-orange-600 dark:text-orange-400 font-medium">{actionCounts.cap} Cap</span>
                {' • '}
                <span className="text-red-600 dark:text-red-400 font-medium">{actionCounts.remove} Remove</span>
              </div>
            </div>
            
            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="flex-1 px-6 py-3 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 rounded-xl font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleApply}
                className="flex-1 px-6 py-3 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white rounded-xl font-medium transition-all shadow-lg hover:shadow-xl"
              >
                Apply Outlier Handling
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};

export default OutlierHandlingModal;
