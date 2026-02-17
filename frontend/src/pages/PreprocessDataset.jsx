import { useState, useEffect, useRef, forwardRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { datasetsAPI } from '../utils/api';
import { ArrowLeft, CheckCircle, AlertTriangle, Info, Loader2, Check, Database, Copy, Activity, Award, Zap, TrendingUp, Search, X, ChevronDown, Target, Brain, List, Code, Settings, ChevronLeft, ChevronRight, Trash2, Sparkles, XCircle } from 'lucide-react';

// Step Item Component (similar to EDA) - wrapped with forwardRef for AnimatePresence
const PreprocessingStepItem = forwardRef(({ step, index, icon: Icon }, ref) => {
  const getStatusIcon = () => {
    if (step.status === 'completed') {
      return (
        <div className="w-5 h-5 rounded-full bg-green-500/20 flex items-center justify-center shrink-0">
          <Check className="w-3 h-3 text-green-400" strokeWidth={3} />
        </div>
      );
    }
    
    if (step.status === 'running') {
      return (
        <div className="w-5 h-5 rounded-full bg-blue-500/20 flex items-center justify-center shrink-0">
          <Loader2 className="w-3 h-3 text-blue-400 animate-spin" strokeWidth={3} />
        </div>
      );
    }
    
    if (step.status === 'error') {
      return (
        <div className="w-5 h-5 rounded-full bg-red-500/20 flex items-center justify-center shrink-0">
          <svg className="w-3 h-3 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
      );
    }
    
    return (
      <div className="w-5 h-5 rounded-full bg-muted/50 flex items-center justify-center shrink-0">
        <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground" />
      </div>
    );
  };

  const getStatusBadge = () => {
    const statusConfig = {
      completed: {
        icon: <Check className="w-3 h-3" strokeWidth={3} />,
        className: 'bg-green-500/10 text-green-400 border-green-500/20',
      },
      running: {
        icon: <Loader2 className="w-3 h-3 animate-spin" strokeWidth={3} />,
        className: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
      },
      error: {
        icon: <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>,
        className: 'bg-red-500/10 text-red-400 border-red-500/20',
      },
      pending: {
        icon: null,
        className: 'bg-muted/50 text-muted-foreground border-border',
        label: '○',
      },
    };

    const config = statusConfig[step.status] || statusConfig.pending;

    return (
      <div className={`flex items-center justify-center w-6 h-6 rounded-md border font-mono text-xs ${config.className}`}>
        {config.icon || config.label}
      </div>
    );
  };

  const iconColorClass = 
    step.status === 'completed' 
      ? 'text-green-400' 
      : step.status === 'running' 
      ? 'text-blue-400' 
      : step.status === 'error'
      ? 'text-red-400'
      : 'text-muted-foreground/50';

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.2, delay: index * 0.05 }}
      className="flex items-center justify-between gap-4 py-2 px-4 rounded-lg border bg-card backdrop-blur-sm hover:bg-muted/50 transition-colors"
    >
      {/* Left: Step name with icon */}
      <div className="flex items-center gap-3 flex-1 min-w-0">
        {getStatusIcon()}
        {Icon && (
          <div className={`w-5 h-5 flex items-center justify-center shrink-0 transition-colors ${iconColorClass}`}>
            <Icon className="w-4 h-4" strokeWidth={2} />
          </div>
        )}
        <span className="text-sm truncate font-medium">
          {step.step}
        </span>
      </div>

      {/* Right: Status and time */}
      <div className="flex items-center gap-3 shrink-0">
        {(step.status === 'running' || step.status === 'completed') && step.elapsed && (
          <span className="text-xs text-muted-foreground font-mono">
            {step.elapsed}
          </span>
        )}
        {getStatusBadge()}
      </div>
    </motion.div>
  );
});

PreprocessingStepItem.displayName = 'PreprocessingStepItem';

export default function PreprocessDataset() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [dataset, setDataset] = useState(null);
  const [loading, setLoading] = useState(true);
  const [preprocessing, setPreprocessing] = useState(false);
  const [results, setResults] = useState(null);
  const [preprocessingSteps, setPreprocessingSteps] = useState([]);
  const [showTargetSelection, setShowTargetSelection] = useState(true);
  const [targetColumn, setTargetColumn] = useState('');
  const [columns, setColumns] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef(null);
  const logsEndRef = useRef(null);
  const rareValueSelectionsRef = useRef({}); // NEW: Use ref to accumulate selections
  const detectedTaskTypeRef = useRef('unsupervised'); // Use ref for immediate updates
  
  // Phase tracking for inline workflow
  const [preprocessingPhase, setPreprocessingPhase] = useState('idle'); // 'idle', 'target', 'rare-values', 'outliers', 'processing', 'complete'
  
  // Rare value review state
  const [rareValueColumns, setRareValueColumns] = useState([]);
  const [currentRareValueIndex, setCurrentRareValueIndex] = useState(0);
  const [rareValueSelections, setRareValueSelections] = useState({}); // { columnName: [value1, value2, ...] }
  const [selectedRareValues, setSelectedRareValues] = useState({}); // { rareValue: boolean }
  const [isRareValuesExpanded, setIsRareValuesExpanded] = useState(false);
  const [isColumnsRemovedExpanded, setIsColumnsRemovedExpanded] = useState(false);
  
  // Outlier handling state
  const [outlierData, setOutlierData] = useState(null);
  const [outlierPreferences, setOutlierPreferences] = useState({}); // { columnName: 'keep'|'cap'|'remove' }
  const [taskType, setTaskType] = useState('unsupervised'); // Will be detected from target column
  const [detectedTaskType, setDetectedTaskType] = useState('unsupervised'); // Store detected task type

  // Define preprocessing steps with icons
  const stepIcons = {
    'Starting adaptive preprocessing': Zap,
    'Detecting task type (classification/regression/unsupervised)': Target,
    'Auto-selecting optimal model family': Brain,
    'Analyzing columns (ID, constant, high-missing)': Database,
    'Categorizing features (numerical, categorical)': List,
    'Choosing encoding strategy': Code,
    'Choosing scaling strategy': Activity,
    'Building preprocessing pipeline': Settings,
    'Applying transformations': Zap,
    'Calculating quality metrics': Award,
    'Preprocessing complete': CheckCircle,
    // Legacy steps (for backward compatibility)
    'Starting automatic preprocessing': Zap,
    'Analyzing dataset structure': Database,
    'Detecting duplicate rows': Copy,
    'Removing duplicate rows': Copy,
    'Detecting constant columns': Activity,
    'Removing constant columns': Activity,
    'Checking missing values': AlertTriangle,
    'Removing high-missing columns': AlertTriangle,
    'Detecting outliers': Activity,
    'Clipping outliers': Activity,
    'Filling missing values': Info,
    'Assessing data quality': Award,
  };

  useEffect(() => {
    fetchDataset();
  }, [id]);

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [preprocessingSteps]);

  // Click outside handler for dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Initialize outlier preferences with smart defaults when outliers phase starts
  useEffect(() => {
    if (preprocessingPhase === 'outliers' && outlierData && Object.keys(outlierPreferences).length === 0) {
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
      setOutlierPreferences(initial);
    }
  }, [preprocessingPhase, outlierData, taskType, outlierPreferences]);

  const fetchDataset = async () => {
    try {
      setLoading(true);
      const data = await datasetsAPI.get(id);
      setDataset(data);
      
      // Get column names from the dataset - use selected_columns if available
      if (data.selected_columns && data.selected_columns.length > 0) {
        setColumns(data.selected_columns);
      } else if (data.column_names && data.column_names.length > 0) {
        setColumns(data.column_names);
      } else if (data.columns && data.columns.length > 0) {
        setColumns(data.columns);
      }
      
      // Don't automatically start preprocessing - wait for target selection
    } catch (error) {
      console.error('Error fetching dataset:', error);
    } finally {
      setLoading(false);
    }
  };

  const checkForRareValues = async () => {
    try {
      // Get EDA results which contain rare value detections
      const edaResults = await datasetsAPI.getEDA(id);
      
      if (edaResults?.categorical_analysis?.categorical_features) {
        const columnsWithRareValues = [];
        
        Object.entries(edaResults.categorical_analysis.categorical_features).forEach(([columnName, analysis]) => {
          if (analysis.rare_values && analysis.rare_values.length > 0) {
            columnsWithRareValues.push({
              name: columnName,
              rare_values: analysis.rare_values,
              total_count: dataset.row_count || 0
            });
          }
        });
        
        if (columnsWithRareValues.length > 0) {
          // Show rare value review section
          setRareValueColumns(columnsWithRareValues);
          setCurrentRareValueIndex(0);
          setRareValueSelections({});
          rareValueSelectionsRef.current = {}; // Reset ref
          
          // Initialize selections for first column (all checked by default)
          const firstColumn = columnsWithRareValues[0];
          const initialSelections = {};
          firstColumn.rare_values.forEach(rv => {
            initialSelections[rv.rare_value] = true;
          });
          setSelectedRareValues(initialSelections);
          
          setPreprocessingPhase('rare-values');
          return true; // Has rare values
        }
      }
      
      return false; // No rare values
    } catch (error) {
      console.error('Error checking for rare values:', error);
      return false;
    }
  };
  
  const checkForOutliers = async (detectedTask = 'unsupervised') => {
    try {
      console.log('🎯 checkForOutliers called with detectedTask:', detectedTask);
      
      // Get EDA results which contain outlier detections
      const edaResults = await datasetsAPI.getEDA(id);
      
      if (edaResults?.outliers?.outliers_by_column) {
        // Check if there are any columns with outliers
        const hasOutliers = Object.values(edaResults.outliers.outliers_by_column).some(
          data => data.count > 0
        );
        
        if (hasOutliers) {
          setOutlierData(edaResults.outliers);
          console.log('🎯 Setting taskType to:', detectedTask);
          setTaskType(detectedTask); // Set task type right before showing section
          setPreprocessingPhase('outliers');
          return true;
        }
      }
      
      return false; // No outliers
    } catch (error) {
      console.error('Error checking for outliers:', error);
      return false;
    }
  };

  const handleStartPreprocessing = async () => {
    setShowTargetSelection(false);
    
    console.log('🎯🎯🎯 handleStartPreprocessing called with targetColumn:', targetColumn);
    
    // Detect task type based on target column selection FIRST
    let taskTypeDetected = 'unsupervised';
    if (targetColumn && dataset) {
      console.log('🎯 Fetching EDA for target column:', targetColumn);
      // Check if target is numeric or categorical
      const edaResults = await datasetsAPI.getEDA(id);
      
      // Column types structure: {numerical_columns: [], categorical_columns: []}
      const columnTypes = edaResults?.basic_info?.column_types;
      const isNumerical = columnTypes?.numerical_columns?.includes(targetColumn);
      const isCategorical = columnTypes?.categorical_columns?.includes(targetColumn);
      
      console.log('🎯 Is numerical?', isNumerical);
      console.log('🎯 Is categorical?', isCategorical);
      
      if (isNumerical) {
        taskTypeDetected = 'regression';
      } else if (isCategorical) {
        taskTypeDetected = 'classification';
      }
    }
    
    // Store detected task type in BOTH state and ref (ref updates immediately)
    setDetectedTaskType(taskTypeDetected);
    detectedTaskTypeRef.current = taskTypeDetected;
    console.log('🎯🎯🎯 Task type detected and stored:', taskTypeDetected, '- Ref value:', detectedTaskTypeRef.current);
    
    // Check for rare values first
    const hasRareValues = await checkForRareValues();
    
    // If no rare values, check for outliers (pass detected task type)
    if (!hasRareValues) {
      const hasOutliers = await checkForOutliers(taskTypeDetected);
      
      // If no outliers either, start preprocessing immediately
      if (!hasOutliers) {
        setTaskType(taskTypeDetected); // Set for preprocessing
        startPreprocessing(dataset);
      }
    }
    // Otherwise, modals will be shown and user will trigger next step from there
  };

  const startPreprocessing = async (datasetData, outlierPrefs = null) => {
    try {
      setPreprocessing(true);
      
      // Use provided outlier preferences or fall back to state
      const outlierPreferencesToUse = outlierPrefs !== null ? outlierPrefs : outlierPreferences;
      console.log('🎯 startPreprocessing using outlier preferences:', outlierPreferencesToUse);
      
      // Initialize preprocessing steps
      const steps = [
        { step: 'Starting adaptive preprocessing', status: 'pending', elapsed: null },
        { step: 'Detecting task type (classification/regression/unsupervised)', status: 'pending', elapsed: null },
        { step: 'Auto-selecting optimal model family', status: 'pending', elapsed: null },
        { step: 'Analyzing columns (ID, constant, high-missing)', status: 'pending', elapsed: null },
        { step: 'Categorizing features (numerical, categorical)', status: 'pending', elapsed: null },
        { step: 'Choosing encoding strategy', status: 'pending', elapsed: null },
        { step: 'Choosing scaling strategy', status: 'pending', elapsed: null },
        { step: 'Building preprocessing pipeline', status: 'pending', elapsed: null },
        { step: 'Applying transformations', status: 'pending', elapsed: null },
        { step: 'Calculating quality metrics', status: 'pending', elapsed: null },
        { step: 'Preprocessing complete', status: 'pending', elapsed: null },
      ];
      setPreprocessingSteps(steps);
      
      const startTime = Date.now();
      
      // Simulate step progression
      for (let i = 0; i < steps.length - 1; i++) {
        await new Promise(resolve => setTimeout(resolve, 300));
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1) + 's';
        setPreprocessingSteps(prev => 
          prev.map((s, idx) => 
            idx === i 
              ? { ...s, status: 'running', elapsed }
              : idx < i 
              ? { ...s, status: 'completed', elapsed: s.elapsed || elapsed }
              : s
          )
        );
      }
      
      // Mark all as running before API call
      setPreprocessingSteps(prev => 
        prev.map(s => ({ ...s, status: s.status === 'pending' ? 'running' : s.status }))
      );
      
      // Use the ref if it has values, otherwise use state
      const selectionsToSend = Object.keys(rareValueSelectionsRef.current).length > 0 
        ? rareValueSelectionsRef.current 
        : rareValueSelections;
      
      console.log('🔍 Rare value selections being sent:', selectionsToSend);
      console.log('🔍 Number of columns with selections:', Object.keys(selectionsToSend).length);
      console.log('🔍 Outlier preferences being sent:', outlierPreferencesToUse);
      
      const response = await datasetsAPI.preprocess(id, targetColumn || null, selectionsToSend, outlierPreferencesToUse);
      
      // Mark all steps as completed
      const finalElapsed = ((Date.now() - startTime) / 1000).toFixed(1) + 's';
      setPreprocessingSteps(prev => 
        prev.map(s => ({ ...s, status: 'completed', elapsed: s.elapsed || finalElapsed }))
      );
      
      setResults(response);
      
      // Add to processed datasets list in localStorage
      const savedProcessed = localStorage.getItem('processedDatasets');
      let processedIds = savedProcessed ? JSON.parse(savedProcessed) : [];
      
      if (!processedIds.includes(id)) {
        processedIds.push(id);
        localStorage.setItem('processedDatasets', JSON.stringify(processedIds));
      }
    } catch (error) {
      console.error('Error preprocessing dataset:', error);
      console.error('Error details:', error.response?.data);
      setPreprocessingSteps(prev => 
        prev.map(s => s.status === 'running' || s.status === 'pending' ? { ...s, status: 'error' } : s)
      );
      
      // Show error in results
      setResults({
        error: true,
        message: error.response?.data?.detail || error.message || 'Unknown error occurred'
      });
    } finally {
      setPreprocessing(false);
    }
  };

  const handleBack = () => {
    navigate('/dashboard/processed', { state: { refresh: true } });
  };

  const handleRareValueDecision = () => {
    const currentColumn = rareValueColumns[currentRareValueIndex];
    
    // Get selected rare values for removal
    const selectedForRemoval = Object.entries(selectedRareValues)
      .filter(([value, isSelected]) => isSelected)
      .map(([value]) => value);
    
    console.log('🔍 Column:', currentColumn.name);
    console.log('🔍 Selected for removal:', selectedForRemoval);
    
    // Record the selected values for this column using ref (immediate update)
    if (selectedForRemoval.length > 0) {
      rareValueSelectionsRef.current[currentColumn.name] = selectedForRemoval;
      console.log('🔍 Ref updated:', rareValueSelectionsRef.current);
      
      // Also update state for consistency
      setRareValueSelections(prev => {
        const updated = {
          ...prev,
          [currentColumn.name]: selectedForRemoval
        };
        console.log('🔍 State updated:', updated);
        return updated;
      });
    }
    
    // Move to next column or finish
    if (currentRareValueIndex < rareValueColumns.length - 1) {
      const nextIndex = currentRareValueIndex + 1;
      setCurrentRareValueIndex(nextIndex);
      
      // Initialize selections for next column (all checked by default)
      const nextColumn = rareValueColumns[nextIndex];
      const nextSelections = {};
      nextColumn.rare_values.forEach(rv => {
        nextSelections[rv.rare_value] = true;
      });
      setSelectedRareValues(nextSelections);
    } else {
      // All columns reviewed, move to outliers phase
      console.log('🔍 Final selections from ref:', rareValueSelectionsRef.current);
      setRareValueSelections(rareValueSelectionsRef.current); // Set final state from ref
      
      // Check for outliers before starting preprocessing (use ref for immediate value)
      const taskTypeToUse = detectedTaskTypeRef.current;
      console.log('🎯 About to check for outliers. Using task type from ref:', taskTypeToUse);
      checkForOutliers(taskTypeToUse).then(hasOutliers => {
        console.log('🎯 Outliers check result:', hasOutliers);
        if (!hasOutliers) {
          setTaskType(taskTypeToUse); // Set for preprocessing
          startPreprocessing(dataset);
        }
      });
    }
  };

  const handleSkipColumn = () => {
    // Skip this column (don't remove any values)
    // Move to next column or finish
    if (currentRareValueIndex < rareValueColumns.length - 1) {
      const nextIndex = currentRareValueIndex + 1;
      setCurrentRareValueIndex(nextIndex);
      
      // Initialize selections for next column
      const nextColumn = rareValueColumns[nextIndex];
      const nextSelections = {};
      nextColumn.rare_values.forEach(rv => {
        nextSelections[rv.rare_value] = true;
      });
      setSelectedRareValues(nextSelections);
    } else {
      // All columns reviewed, check for outliers
      const taskTypeToUse = detectedTaskTypeRef.current;
      checkForOutliers(taskTypeToUse).then(hasOutliers => {
        if (!hasOutliers) {
          setTaskType(taskTypeToUse); // Set for preprocessing
          startPreprocessing(dataset);
        }
      });
    }
  };

  const toggleRareValue = (rareValue) => {
    setSelectedRareValues(prev => ({
      ...prev,
      [rareValue]: !prev[rareValue]
    }));
  };

  const selectAllRareValues = () => {
    const currentColumn = rareValueColumns[currentRareValueIndex];
    const allSelected = {};
    currentColumn.rare_values.forEach(rv => {
      allSelected[rv.rare_value] = true;
    });
    setSelectedRareValues(allSelected);
  };

  const deselectAllRareValues = () => {
    const currentColumn = rareValueColumns[currentRareValueIndex];
    const noneSelected = {};
    currentColumn.rare_values.forEach(rv => {
      noneSelected[rv.rare_value] = false;
    });
    setSelectedRareValues(noneSelected);
  };

  const calculateRemainingRows = (currentColumn) => {
    if (!currentColumn || !currentColumn.rare_values) {
      return 0;
    }
    
    // Get total from first rare value entry (they all have the same total_rows)
    const totalRows = currentColumn.rare_values[0]?.total_rows || 0;
    
    // Only count selected rare values
    const selectedCount = currentColumn.rare_values.reduce((sum, rv) => {
      return sum + (selectedRareValues[rv.rare_value] ? rv.count : 0);
    }, 0);
    
    return totalRows - selectedCount;
  };

  const checkTargetColumnImpact = () => {
    // Check if removing selected values will cause issues with target column
    const currentColumn = rareValueColumns[currentRareValueIndex];
    
    if (!targetColumn || currentColumn.name !== targetColumn) {
      return null; // No issue if not the target column
    }
    
    // Count remaining samples per class
    const selectedValues = Object.entries(selectedRareValues)
      .filter(([_, isSelected]) => isSelected)
      .map(([value]) => value);
    
    if (selectedValues.length === 0) {
      return null; // Nothing being removed
    }
    
    // Check if any of the selected values are the target column
    // This would leave only 0 samples for that class
    const willHaveEmptyClasses = selectedValues.length > 0;
    
    return willHaveEmptyClasses ? 
      "Warning: Removing these values from the target column may cause issues with train/test split. Each class needs at least 2 samples." : 
      null;
  };
  
  // Outlier handling functions
  const handleOutlierPreferences = (preferences) => {
    console.log('🔍 Outlier preferences received:', preferences);
    setOutlierPreferences(preferences);
    
    // Move to processing phase
    setPreprocessingPhase('processing');
    
    // Start preprocessing with the preferences (use parameter, not state)
    startPreprocessingWithOutliers(dataset, preferences);
  };
  
  const handleSkipOutliers = () => {
    // User wants to keep all outliers - proceed without changes
    const keepAllPrefs = {};
    const columns = Object.keys(outlierData?.outliers_by_column || {});
    columns.forEach(col => {
      keepAllPrefs[col] = 'keep';
    });
    handleOutlierPreferences(keepAllPrefs);
  };
  
  const startPreprocessingWithOutliers = async (datasetData, outlierPrefs) => {
    console.log('🎯 Starting preprocessing with outlier preferences:', outlierPrefs);
    
    // Update state for display purposes
    setOutlierPreferences(outlierPrefs);
    
    // Call the regular startPreprocessing
    await startPreprocessing(datasetData, outlierPrefs);
  };


  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="mb-6">
          <button
            onClick={handleBack}
            className="flex items-center text-muted-foreground hover:text-foreground mb-4 transition-colors"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Back to Processed Datasets
          </button>
          <div>
            <h1 className="text-3xl font-bold text-foreground">
              Preprocessing: {dataset?.name || 'Dataset'}
            </h1>
            <p className="text-muted-foreground mt-2">
              Automatic preprocessing and data quality improvements
            </p>
          </div>
        </div>

        {/* Workflow Timeline Stepper */}
        <div className="bg-card border rounded-2xl p-6">
          <div className="flex items-center justify-between">
            {/* Step 1: Target Selection */}
            <div className="flex items-center gap-3 flex-1">
              <button
                onClick={() => {
                  if (!preprocessing && !results) {
                    setPreprocessingPhase('idle');
                    setShowTargetSelection(true);
                  }
                }}
                disabled={preprocessing || results}
                className={`flex items-center gap-3 transition-all ${preprocessing || results ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:scale-105'}`}
              >
                <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold transition-all ${
                  preprocessingPhase === 'idle' || preprocessingPhase === 'target' || showTargetSelection
                    ? 'bg-primary text-primary-foreground shadow-lg'
                    : 'bg-muted text-muted-foreground'
                }`}>
                  1
                </div>
                <div className="text-left">
                  <div className={`font-semibold text-sm ${
                    preprocessingPhase === 'idle' || preprocessingPhase === 'target' || showTargetSelection
                      ? 'text-foreground'
                      : 'text-muted-foreground'
                  }`}>
                    Target Column
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {targetColumn || 'Not selected'}
                  </div>
                </div>
              </button>
            </div>
              
            {/* Connector Line */}
            <div className={`h-0.5 mx-2 transition-colors ${
              preprocessingPhase === 'rare-values' || preprocessingPhase === 'outliers' || preprocessingPhase === 'processing' || results || (rareValueColumns.length === 0 && outlierData && Object.keys(outlierData?.outliers_by_column || {}).filter(col => outlierData.outliers_by_column[col].count > 0).length > 0) || (rareValueColumns.length === 0 && outlierData && Object.keys(outlierData?.outliers_by_column || {}).filter(col => outlierData.outliers_by_column[col].count > 0).length === 0)
                ? 'bg-primary'
                : 'bg-muted'
            }`} />

            {/* Step 2: Rare Values (conditional) */}
            {rareValueColumns.length > 0 && (
              <>
                <div className="flex items-center gap-3 flex-1">
                  <button
                    onClick={() => {
                      if (!preprocessing && !results && preprocessingPhase !== 'idle' && preprocessingPhase !== 'target') {
                        setPreprocessingPhase('rare-values');
                        setShowTargetSelection(false);
                      }
                    }}
                    disabled={preprocessing || results || preprocessingPhase === 'idle' || preprocessingPhase === 'target'}
                    className={`flex items-center gap-3 transition-all ${
                      preprocessing || results || preprocessingPhase === 'idle' || preprocessingPhase === 'target'
                        ? 'cursor-not-allowed opacity-50'
                        : 'cursor-pointer hover:scale-105'
                    }`}
                  >
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold transition-all ${
                      preprocessingPhase === 'rare-values'
                        ? 'bg-primary text-primary-foreground shadow-lg'
                        : (preprocessingPhase === 'outliers' || preprocessingPhase === 'processing' || results)
                        ? 'bg-primary/50 text-primary-foreground'
                        : 'bg-muted text-muted-foreground'
                    }`}>
                      2
                    </div>
                    <div className="text-left">
                      <div className={`font-semibold text-sm ${
                        preprocessingPhase === 'rare-values'
                          ? 'text-foreground'
                          : 'text-muted-foreground'
                      }`}>
                        Rare Values
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {rareValueColumns.length} column{rareValueColumns.length !== 1 ? 's' : ''}
                      </div>
                    </div>
                  </button>
                  
                  {/* Connector Line */}
                  <div className={`flex-1 h-0.5 mx-2 transition-colors ${
                    preprocessingPhase === 'outliers' || preprocessingPhase === 'processing' || results
                      ? 'bg-primary'
                      : 'bg-muted'
                  }`} />
                </div>
              </>
            )}

            {/* Step 3: Outliers (conditional) */}
            {outlierData && Object.keys(outlierData?.outliers_by_column || {}).filter(col => outlierData.outliers_by_column[col].count > 0).length > 0 && (
              <>
                <div className="flex items-center gap-3 flex-1">
                  <button
                    onClick={() => {
                      if (!preprocessing && !results && preprocessingPhase !== 'idle' && preprocessingPhase !== 'target' && preprocessingPhase !== 'rare-values') {
                        setPreprocessingPhase('outliers');
                        setShowTargetSelection(false);
                      }
                    }}
                    disabled={preprocessing || results || preprocessingPhase === 'idle' || preprocessingPhase === 'target' || preprocessingPhase === 'rare-values'}
                    className={`flex items-center gap-3 transition-all ${
                      preprocessing || results || preprocessingPhase === 'idle' || preprocessingPhase === 'target' || preprocessingPhase === 'rare-values'
                        ? 'cursor-not-allowed opacity-50'
                        : 'cursor-pointer hover:scale-105'
                    }`}
                  >
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold transition-all ${
                      preprocessingPhase === 'outliers'
                        ? 'bg-primary text-primary-foreground shadow-lg'
                        : (preprocessingPhase === 'processing' || results)
                        ? 'bg-primary/50 text-primary-foreground'
                        : 'bg-muted text-muted-foreground'
                    }`}>
                      {rareValueColumns.length > 0 ? '3' : '2'}
                    </div>
                    <div className="text-left">
                      <div className={`font-semibold text-sm ${
                        preprocessingPhase === 'outliers'
                          ? 'text-foreground'
                          : 'text-muted-foreground'
                      }`}>
                        Outliers
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {Object.keys(outlierData?.outliers_by_column || {}).filter(col => outlierData.outliers_by_column[col].count > 0).length} column{Object.keys(outlierData?.outliers_by_column || {}).filter(col => outlierData.outliers_by_column[col].count > 0).length !== 1 ? 's' : ''}
                      </div>
                    </div>
                  </button>
                  
                  {/* Connector Line */}
                  <div className={`flex-1 h-0.5 mx-2 transition-colors ${
                    preprocessingPhase === 'processing' || results
                      ? 'bg-primary'
                      : 'bg-muted'
                  }`} />
                </div>
              </>
            )}

            {/* Final Step: Processing */}
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold transition-all ${
                preprocessingPhase === 'processing' || results
                  ? 'bg-primary text-primary-foreground shadow-lg'
                  : 'bg-muted text-muted-foreground'
              } ${preprocessing ? 'animate-pulse' : ''}`}>
                {rareValueColumns.length > 0 && outlierData && Object.keys(outlierData?.outliers_by_column || {}).filter(col => outlierData.outliers_by_column[col].count > 0).length > 0
                  ? '4'
                  : rareValueColumns.length > 0 || (outlierData && Object.keys(outlierData?.outliers_by_column || {}).filter(col => outlierData.outliers_by_column[col].count > 0).length > 0)
                  ? '3'
                  : '2'}
              </div>
              <div className="text-left">
                <div className={`font-semibold text-sm ${
                  preprocessingPhase === 'processing' || results
                    ? 'text-foreground'
                    : 'text-muted-foreground'
                }`}>
                  Processing
                </div>
                <div className="text-xs text-muted-foreground">
                  {preprocessing ? 'Running...' : results ? 'Complete' : 'Ready'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Target Column Selection Panel */}
        {showTargetSelection && !preprocessing && !results && preprocessingPhase !== 'rare-values' && preprocessingPhase !== 'outliers' && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-card border rounded-2xl p-6"
          >
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-500" />
              Select Target Column (Optional)
            </h3>
            
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                If you're planning to use this dataset for supervised learning (classification or regression), 
                select the target column. This will ensure it's encoded correctly (label encoding) instead of 
                being one-hot encoded.
              </p>
              
              {columns.length > 0 && (
                <div className="text-xs text-muted-foreground">
                  {columns.length} columns available: {columns.slice(0, 5).join(', ')}
                  {columns.length > 5 && `, ... and ${columns.length - 5} more`}
                </div>
              )}
              
              <div className="space-y-2">
                <label className="text-sm font-medium">Target Column</label>
                <div className="relative" ref={dropdownRef}>
                  {/* Search Input */}
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <input
                      type="text"
                      value={targetColumn || searchTerm}
                      onChange={(e) => {
                        setSearchTerm(e.target.value);
                        setTargetColumn('');
                        setShowDropdown(true);
                      }}
                      onFocus={() => setShowDropdown(true)}
                      placeholder="Search or select target column..."
                      className="w-full pl-10 pr-10 py-2 bg-background border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                    {targetColumn && (
                      <button
                        onClick={() => {
                          setTargetColumn('');
                          setSearchTerm('');
                        }}
                        className="absolute right-10 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}
                    <ChevronDown 
                      className={`absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground transition-transform ${showDropdown ? 'rotate-180' : ''}`}
                    />
                  </div>

                  {/* Dropdown */}
                  {showDropdown && (
                    <div className="absolute z-10 w-full mt-1 bg-card border rounded-lg shadow-lg max-h-60 overflow-y-auto">
                      {/* None option */}
                      <button
                        onClick={() => {
                          setTargetColumn('');
                          setSearchTerm('');
                          setShowDropdown(false);
                        }}
                        className="w-full px-4 py-2 text-left hover:bg-muted/50 transition-colors text-sm"
                      >
                        <span className="text-muted-foreground italic">None (unsupervised learning)</span>
                      </button>
                      
                      {/* Filtered columns */}
                      {columns
                        .filter(col => col.toLowerCase().includes(searchTerm.toLowerCase()))
                        .map((col) => (
                          <button
                            key={col}
                            onClick={() => {
                              setTargetColumn(col);
                              setSearchTerm('');
                              setShowDropdown(false);
                            }}
                            className={`w-full px-4 py-2 text-left hover:bg-muted/50 transition-colors text-sm ${
                              targetColumn === col ? 'bg-primary/10 text-primary font-medium' : ''
                            }`}
                          >
                            {col}
                          </button>
                        ))}
                      
                      {/* No results */}
                      {columns.filter(col => col.toLowerCase().includes(searchTerm.toLowerCase())).length === 0 && searchTerm && (
                        <div className="px-4 py-3 text-sm text-muted-foreground text-center">
                          No columns found matching "{searchTerm}"
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                {/* Selected column display */}
                {targetColumn && (
                  <div className="text-sm text-muted-foreground">
                    Selected: <span className="font-medium text-foreground">{targetColumn}</span>
                  </div>
                )}
              </div>
              
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                <div className="flex gap-3">
                  <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
                  <div className="text-sm text-blue-400">
                    <p className="font-medium mb-1">Encoding Strategy:</p>
                    <ul className="space-y-1 text-xs opacity-90">
                      <li>• <strong>Target column:</strong> Label encoded (0, 1, 2, ...)</li>
                      <li>• <strong>Feature columns:</strong> One-hot encoded (if &lt;10 categories)</li>
                      <li>• <strong>One-hot encoded columns:</strong> Kept as binary (0 or 1)</li>
                    </ul>
                  </div>
                </div>
              </div>
              
              <button
                onClick={handleStartPreprocessing}
                className="w-full bg-primary text-primary-foreground px-6 py-3 rounded-lg font-medium hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
              >
                <Zap className="w-5 h-5" />
                Start Preprocessing
              </button>
            </div>
          </motion.div>
        )}

        {/* Rare Values Review Section - INLINE */}
        {preprocessingPhase === 'rare-values' && rareValueColumns.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-card border rounded-2xl p-8 space-y-6"
          >
            {/* Header with Navigation */}
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-500/10 rounded-lg">
                  <Sparkles className="w-6 h-6 text-purple-500" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold">Rare Values Review</h2>
                  <p className="text-muted-foreground text-sm mt-1">
                    Column {currentRareValueIndex + 1} of {rareValueColumns.length}: <span className="font-semibold">{rareValueColumns[currentRareValueIndex]?.name}</span>
                  </p>
                </div>
              </div>
              
              {/* Column Navigation */}
              {rareValueColumns.length > 1 && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      if (currentRareValueIndex > 0) {
                        setCurrentRareValueIndex(currentRareValueIndex - 1);
                        const prevColumn = rareValueColumns[currentRareValueIndex - 1];
                        const prevSelections = {};
                        prevColumn.rare_values.forEach(rv => {
                          prevSelections[rv.rare_value] = rareValueSelectionsRef.current[prevColumn.name]?.includes(rv.rare_value) || false;
                        });
                        setSelectedRareValues(prevSelections);
                      }
                    }}
                    disabled={currentRareValueIndex === 0}
                    className="p-2 border rounded-lg hover:bg-muted transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Previous column"
                  >
                    <ChevronLeft className="w-5 h-5" />
                  </button>
                  
                  <button
                    onClick={() => {
                      if (currentRareValueIndex < rareValueColumns.length - 1) {
                        setCurrentRareValueIndex(currentRareValueIndex + 1);
                        const nextColumn = rareValueColumns[currentRareValueIndex + 1];
                        const nextSelections = {};
                        nextColumn.rare_values.forEach(rv => {
                          nextSelections[rv.rare_value] = rareValueSelectionsRef.current[nextColumn.name]?.includes(rv.rare_value) || false;
                        });
                        setSelectedRareValues(nextSelections);
                      }
                    }}
                    disabled={currentRareValueIndex === rareValueColumns.length - 1}
                    className="p-2 border rounded-lg hover:bg-muted transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Next column"
                  >
                    <ChevronRight className="w-5 h-5" />
                  </button>
                </div>
              )}
            </div>

            {/* Warning Banner */}
            <div className="p-4 bg-yellow-50 dark:bg-yellow-950/30 border border-yellow-200 dark:border-yellow-800 rounded-lg">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-yellow-900 dark:text-yellow-100 mb-1 text-sm">
                    Rare Values Detected
                  </h3>
                  <p className="text-xs text-yellow-800 dark:text-yellow-200">
                    These values occur very infrequently in your dataset and may not have enough samples for proper model training.
                  </p>
                </div>
              </div>
            </div>

            {/* Current Column Info - Compact */}
            {rareValueColumns[currentRareValueIndex] && (
              <>
                <div className="flex items-center justify-between p-4 bg-muted/30 rounded-lg border">
                  <div className="flex items-center gap-6">
                    <div>
                      <div className="text-xs text-muted-foreground">Rare Values</div>
                      <div className="text-lg font-bold">{rareValueColumns[currentRareValueIndex].rare_values.length}</div>
                    </div>
                    <div className="w-px h-8 bg-border" />
                    <div>
                      <div className="text-xs text-muted-foreground">Rows After Removal</div>
                      <div className="text-lg font-bold">{calculateRemainingRows(rareValueColumns[currentRareValueIndex]).toLocaleString()}</div>
                    </div>
                  </div>
                  
                  {/* Toggle Select/Deselect All */}
                  <button
                    onClick={() => {
                      const allSelected = rareValueColumns[currentRareValueIndex]?.rare_values.every(
                        rv => selectedRareValues[rv.rare_value]
                      );
                      if (allSelected) {
                        deselectAllRareValues();
                      } else {
                        selectAllRareValues();
                      }
                    }}
                    className="px-4 py-2 bg-background hover:bg-muted border rounded-lg transition-colors font-medium text-sm flex items-center gap-2"
                  >
                    {rareValueColumns[currentRareValueIndex]?.rare_values.every(
                      rv => selectedRareValues[rv.rare_value]
                    ) ? (
                      <>
                        <XCircle className="w-4 h-4" />
                        Deselect All
                      </>
                    ) : (
                      <>
                        <CheckCircle className="w-4 h-4" />
                        Select All
                      </>
                    )}
                  </button>
                </div>

                {/* Rare Values List - Two Columns */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-96 overflow-y-auto">
                  {rareValueColumns[currentRareValueIndex].rare_values.map((rv) => (
                    <div
                      key={rv.rare_value}
                      onClick={() => toggleRareValue(rv.rare_value)}
                      className="p-3 bg-background rounded-lg border hover:border-purple-300 dark:hover:border-purple-700 cursor-pointer transition-colors"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          <input
                            type="checkbox"
                            checked={selectedRareValues[rv.rare_value] || false}
                            onChange={() => {}}
                            className="w-4 h-4 text-purple-600 rounded focus:ring-purple-500 flex-shrink-0"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="font-medium truncate">
                              {String(rv.rare_value)}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {rv.count} samples ({rv.percentage.toFixed(2)}%)
                            </div>
                          </div>
                        </div>
                        <div className={`px-2 py-1 rounded text-xs font-medium flex-shrink-0 ${
                          rv.percentage < 1
                            ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200'
                            : rv.percentage < 2
                            ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-200'
                            : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200'
                        }`}>
                          {rv.percentage < 1 ? 'Very Rare' : rv.percentage < 2 ? 'Rare' : 'Low Freq'}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Target Column Warning */}
                {checkTargetColumnImpact() && (
                  <div className="p-3 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-lg">
                    <div className="flex items-start gap-3">
                      <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                      <p className="text-xs text-red-800 dark:text-red-200">
                        {checkTargetColumnImpact()}
                      </p>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Action Buttons */}
            <div className="grid grid-cols-3 gap-3 pt-4 border-t">
              {/* Skip All Rare Values - Quick Action */}
              <button
                type="button"
                onClick={async (e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  // Skip all columns and proceed
                  setRareValueSelections({});
                  rareValueSelectionsRef.current = {};
                  const taskTypeToUse = detectedTaskTypeRef.current;
                  try {
                    const hasOutliers = await checkForOutliers(taskTypeToUse);
                    if (!hasOutliers) {
                      setTaskType(taskTypeToUse);
                      startPreprocessing(dataset);
                    }
                  } catch (error) {
                    console.error('Error checking outliers:', error);
                    setTaskType(taskTypeToUse);
                    startPreprocessing(dataset);
                  }
                }}
                className="px-4 py-3 bg-background hover:bg-muted border rounded-lg transition-colors font-medium text-sm"
              >
                Skip All
              </button>

              {/* Skip This Column */}
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  handleSkipColumn();
                }}
                className="px-4 py-3 bg-background hover:bg-muted border rounded-lg transition-colors font-medium text-sm"
              >
                Skip Column
              </button>

              {/* Confirm/Next */}
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  handleRareValueDecision();
                }}
                disabled={Object.values(selectedRareValues).every(v => !v)}
                className="px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              >
                {currentRareValueIndex < rareValueColumns.length - 1 ? 'Next Column' : 'Confirm'}
              </button>
            </div>
          </motion.div>
        )}

        {/* Outliers Review Section - INLINE */}
        {preprocessingPhase === 'outliers' && outlierData && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-card border rounded-2xl p-8 space-y-6"
          >
            {/* Header */}
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-orange-500/10 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-orange-500" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold">Outlier Handling</h2>
                  <p className="text-muted-foreground text-sm mt-1">
                    {Object.keys(outlierData?.outliers_by_column || {}).filter(col => outlierData.outliers_by_column[col].count > 0).length} column{Object.keys(outlierData?.outliers_by_column || {}).filter(col => outlierData.outliers_by_column[col].count > 0).length !== 1 ? 's' : ''} with outliers detected
                  </p>
                </div>
              </div>
            </div>

            {/* Critical Warning Banner */}
            <div className="p-4 bg-red-50 dark:bg-red-950/30 border-2 border-red-200 dark:border-red-800 rounded-xl">
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
            {taskType && (
              <div className={`p-4 rounded-xl border-2 ${
                taskType === 'regression' 
                  ? 'bg-red-50 dark:bg-red-950/20 border-red-200 dark:border-red-800'
                  : taskType === 'classification'
                  ? 'bg-yellow-50 dark:bg-yellow-950/20 border-yellow-200 dark:border-yellow-800'
                  : 'bg-blue-50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800'
              }`}>
                <div className="flex items-start gap-3">
                  {taskType === 'regression' && <Target className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />}
                  {taskType === 'classification' && <Target className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />}
                  {taskType === 'unsupervised' && <Database className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />}
                  <div className="flex-1">
                    <h4 className={`font-semibold text-sm mb-1 ${
                      taskType === 'regression' ? 'text-red-900 dark:text-red-100'
                      : taskType === 'classification' ? 'text-yellow-900 dark:text-yellow-100'
                      : 'text-blue-900 dark:text-blue-100'
                    }`}>
                      {taskType === 'regression' && 'Regression Task - Outliers Bias Model'}
                      {taskType === 'classification' && 'Classification Task - Handle with Care'}
                      {taskType === 'unsupervised' && 'Unsupervised Learning - Outliers Dominate Distances'}
                    </h4>
                    <p className={`text-xs ${
                      taskType === 'regression' ? 'text-red-800 dark:text-red-200'
                      : taskType === 'classification' ? 'text-yellow-800 dark:text-yellow-200'
                      : 'text-blue-800 dark:text-blue-200'
                    }`}>
                      {taskType === 'regression' && 'In regression, outliers can significantly bias your model predictions. Consider capping or removing extreme values, but consult domain experts first.'}
                      {taskType === 'classification' && 'Be careful not to remove outliers that might be important minority class examples. Capping is often safer than removal for classification tasks.'}
                      {taskType === 'unsupervised' && 'In clustering and other unsupervised methods, outliers can dominate distance calculations. Consider using robust methods or capping extreme values.'}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Global Actions */}
            <div className="p-4 bg-muted/50 rounded-xl border">
              <h3 className="font-semibold text-sm mb-3 flex items-center gap-2">
                <Info className="w-4 h-4" />
                Quick Actions - Apply to All Columns
              </h3>
              <div className="flex gap-3">
                <button
                  onClick={() => {
                    const keepAllPrefs = {};
                    Object.keys(outlierData?.outliers_by_column || {}).forEach(col => {
                      if (outlierData.outliers_by_column[col].count > 0) {
                        keepAllPrefs[col] = 'keep';
                      }
                    });
                    setOutlierPreferences(keepAllPrefs);
                  }}
                  className="flex-1 px-4 py-3 bg-green-100 dark:bg-green-900/30 hover:bg-green-200 dark:hover:bg-green-900/50 text-green-900 dark:text-green-100 rounded-lg border-2 border-green-300 dark:border-green-700 transition-colors font-medium text-sm flex items-center justify-center gap-2"
                >
                  <CheckCircle className="w-4 h-4" />
                  Keep All
                </button>
                <button
                  onClick={() => {
                    const capAllPrefs = {};
                    Object.keys(outlierData?.outliers_by_column || {}).forEach(col => {
                      if (outlierData.outliers_by_column[col].count > 0) {
                        capAllPrefs[col] = 'cap';
                      }
                    });
                    setOutlierPreferences(capAllPrefs);
                  }}
                  className="flex-1 px-4 py-3 bg-orange-100 dark:bg-orange-900/30 hover:bg-orange-200 dark:hover:bg-orange-900/50 text-orange-900 dark:text-orange-100 rounded-lg border-2 border-orange-300 dark:border-orange-700 transition-colors font-medium text-sm flex items-center justify-center gap-2"
                >
                  <TrendingUp className="w-4 h-4" />
                  Cap All (Recommended)
                </button>
                <button
                  onClick={() => {
                    const removeAllPrefs = {};
                    Object.keys(outlierData?.outliers_by_column || {}).forEach(col => {
                      if (outlierData.outliers_by_column[col].count > 0) {
                        removeAllPrefs[col] = 'remove';
                      }
                    });
                    setOutlierPreferences(removeAllPrefs);
                  }}
                  className="flex-1 px-4 py-3 bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-900/50 text-red-900 dark:text-red-100 rounded-lg border-2 border-red-300 dark:border-red-700 transition-colors font-medium text-sm flex items-center justify-center gap-2"
                >
                  <XCircle className="w-4 h-4" />
                  Remove All
                </button>
              </div>
            </div>

            {/* Column-by-Column Selection */}
            <div className="space-y-3">
              <h3 className="font-semibold mb-4">Per-Column Outlier Handling</h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {Object.entries(outlierData?.outliers_by_column || {})
                  .filter(([_, data]) => data.count > 0)
                  .sort((a, b) => b[1].percentage - a[1].percentage)
                  .map(([column, data]) => (
                    <div
                      key={column}
                      className="p-4 bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-orange-300 dark:hover:border-orange-700 transition-colors"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <h4 className="font-semibold flex items-center gap-2 flex-wrap">
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
                          <div className="text-sm text-muted-foreground mt-1">
                            <span className="font-medium">{data.count}</span> outliers detected
                            {' • '}Range: [{data.min.toFixed(2)} - {data.max.toFixed(2)}]
                            {' • '}IQR bounds: [{data.lower_bound.toFixed(2)} - {data.upper_bound.toFixed(2)}]
                          </div>
                        </div>
                      </div>

                      <div className="flex gap-2">
                        <button
                          onClick={() => setOutlierPreferences(prev => ({...prev, [column]: 'keep'}))}
                          className={`flex-1 px-3 py-2 rounded-lg border-2 transition-all text-sm font-medium ${
                            outlierPreferences[column] === 'keep'
                              ? 'bg-green-100 dark:bg-green-900/30 border-green-500 dark:border-green-600 text-green-900 dark:text-green-100 shadow-sm'
                              : 'bg-muted/50 border-gray-200 dark:border-gray-700 hover:border-green-300 dark:hover:border-green-700'
                          }`}
                        >
                          Keep
                        </button>
                        <button
                          onClick={() => setOutlierPreferences(prev => ({...prev, [column]: 'cap'}))}
                          className={`flex-1 px-3 py-2 rounded-lg border-2 transition-all text-sm font-medium ${
                            outlierPreferences[column] === 'cap'
                              ? 'bg-orange-100 dark:bg-orange-900/30 border-orange-500 dark:border-orange-600 text-orange-900 dark:text-orange-100 shadow-sm'
                              : 'bg-muted/50 border-gray-200 dark:border-gray-700 hover:border-orange-300 dark:hover:border-orange-700'
                          }`}
                        >
                          Cap (Winsorize)
                        </button>
                        <button
                          onClick={() => setOutlierPreferences(prev => ({...prev, [column]: 'remove'}))}
                          className={`flex-1 px-3 py-2 rounded-lg border-2 transition-all text-sm font-medium ${
                            outlierPreferences[column] === 'remove'
                              ? 'bg-red-100 dark:bg-red-900/30 border-red-500 dark:border-red-600 text-red-900 dark:text-red-100 shadow-sm'
                              : 'bg-muted/50 border-gray-200 dark:border-gray-700 hover:border-red-300 dark:hover:border-red-700'
                          }`}
                        >
                          Remove Rows
                        </button>
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3 pt-4 border-t">
              <button
                onClick={() => handleOutlierPreferences(outlierPreferences)}
                className="px-6 py-2.5 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg font-medium transition-colors text-sm"
              >
                Apply Outlier Handling
              </button>
              <button
                onClick={handleSkipOutliers}
                className="px-6 py-2.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 rounded-lg font-medium transition-colors text-sm"
              >
                Skip (Keep All Outliers)
              </button>
            </div>
          </motion.div>
        )}

        {/* Two Column Layout - Like EDA */}
        {!showTargetSelection && preprocessingPhase !== 'rare-values' && preprocessingPhase !== 'outliers' && (
        <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
          {/* Left Column - Processing Log */}
          <div className="bg-card border rounded-2xl overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-6 pb-4">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                {preprocessing ? (
                  <>
                    <Loader2 className="w-5 h-5 text-primary animate-spin" />
                    Running Preprocessing
                  </>
                ) : results ? (
                  <>
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    Preprocessing Complete
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5 text-blue-500" />
                    Preprocessing Log
                  </>
                )}
              </h3>
            </div>
            
            {/* Progress Steps */}
            <div className="space-y-2 px-6 pb-6">
              <AnimatePresence mode="popLayout">
                {preprocessingSteps.map((step, index) => (
                  <PreprocessingStepItem key={step.step} step={step} index={index} icon={stepIcons[step.step]} />
                ))}
              </AnimatePresence>
              <div ref={logsEndRef} />
            </div>
          </div>

          {/* Right Column - Summary */}
          <div className="bg-card border rounded-2xl p-6">
            <h2 className="text-xl font-semibold mb-4">Preprocessing Summary</h2>
            
            {results ? (
              results.error ? (
                <div className="flex flex-col items-center justify-center h-64 text-center">
                  <AlertTriangle className="w-16 h-16 text-red-500 mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Preprocessing Failed</h3>
                  <p className="text-muted-foreground max-w-md">{results.message}</p>
                  <button
                    onClick={handleBack}
                    className="mt-6 bg-primary text-primary-foreground px-4 py-2 rounded-lg hover:bg-primary/90 transition-colors"
                  >
                    Return to Processed Datasets
                  </button>
                </div>
              ) : (
              <div className="space-y-6">
                {/* Dataset Changes */}
                <div>
                  <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2">
                    <Database className="w-4 h-4" />
                    Dataset Changes
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">Original Size</p>
                      <p className="text-2xl font-bold text-foreground">
                        {results.original?.rows || 0} × {results.original?.columns || 0}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {results.original?.rows?.toLocaleString() || 0} rows, {results.original?.columns || 0} cols
                      </p>
                    </div>
                    <div className="bg-green-500/10 border border-green-500/20 p-4 rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">Processed Size</p>
                      <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {results.processed?.rows || 0} × {results.processed?.columns || 0}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {results.processed?.rows?.toLocaleString() || 0} rows, {results.processed?.columns || 0} cols
                      </p>
                    </div>
                  </div>

                  {/* Train/Test Split Info */}
                  {results.preprocessing_summary?.has_train_test_split && (
                    <div className="mt-4 grid grid-cols-2 gap-4">
                      <div className="bg-green-500/10 border border-green-500/20 p-4 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Train Set</p>
                        <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {results.preprocessing_summary.train_size}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {((results.preprocessing_summary.train_size / (results.preprocessing_summary.train_size + results.preprocessing_summary.test_size)) * 100).toFixed(0)}% of data
                        </p>
                      </div>
                      <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Test Set</p>
                        <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                          {results.preprocessing_summary.test_size}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {((results.preprocessing_summary.test_size / (results.preprocessing_summary.train_size + results.preprocessing_summary.test_size)) * 100).toFixed(0)}% of data
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Changes Made */}
                {results.changes && Object.values(results.changes).some(v => v > 0) && (
                  <div>
                    <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2">
                      <CheckCircle className="w-4 h-4" />
                      Changes Applied
                    </h3>
                    <div className="space-y-2">
                      {results.changes.duplicates_removed > 0 && (
                        <div className="flex justify-between items-center bg-yellow-500/10 border border-yellow-500/20 p-3 rounded-lg">
                          <span className="text-sm flex items-center gap-2">
                            <Copy className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                            Duplicates Removed
                          </span>
                          <span className="font-semibold text-yellow-600 dark:text-yellow-400">
                            {results.changes.duplicates_removed.toLocaleString()}
                          </span>
                        </div>
                      )}
                      {results.changes.outliers_handled > 0 && (
                        <div className="flex justify-between items-center bg-purple-500/10 border border-purple-500/20 p-3 rounded-lg">
                          <span className="text-sm flex items-center gap-2">
                            <Activity className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                            Outliers Clipped
                          </span>
                          <span className="font-semibold text-purple-600 dark:text-purple-400">
                            {results.changes.outliers_handled.toLocaleString()}
                          </span>
                        </div>
                      )}
                      {results.changes.missing_values_filled > 0 && (
                        <div className="flex justify-between items-center bg-blue-500/10 border border-blue-500/20 p-3 rounded-lg">
                          <span className="text-sm flex items-center gap-2">
                            <Info className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                            Missing Values Filled
                          </span>
                          <span className="font-semibold text-blue-600 dark:text-blue-400">
                            {results.changes.missing_values_filled.toLocaleString()}
                          </span>
                        </div>
                      )}
                      {results.changes.constant_columns_removed > 0 && (
                        <div className="flex justify-between items-center bg-red-500/10 border border-red-500/20 p-3 rounded-lg">
                          <span className="text-sm flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400" />
                            Constant Columns Removed
                          </span>
                          <span className="font-semibold text-red-600 dark:text-red-400">
                            {results.changes.constant_columns_removed}
                          </span>
                        </div>
                      )}
                      {(results.changes.columns_removed - (results.changes.constant_columns_removed || 0)) > 0 && (
                        <div className="flex justify-between items-center bg-orange-500/10 border border-orange-500/20 p-3 rounded-lg">
                          <span className="text-sm flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                            High-Missing Columns Removed
                          </span>
                          <span className="font-semibold text-orange-600 dark:text-orange-400">
                            {results.changes.columns_removed - (results.changes.constant_columns_removed || 0)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Rare Values Removed */}
                {results.changes?.rare_values_removed && results.changes.rare_values_removed.length > 0 && (
                  <div className="relative">
                    <button
                      onClick={() => setIsRareValuesExpanded(!isRareValuesExpanded)}
                      className={`w-full flex items-center justify-between p-3 bg-amber-500/10 border border-amber-500/20 hover:bg-amber-500/20 transition-colors ${
                        isRareValuesExpanded ? 'rounded-t-lg' : 'rounded-lg'
                      }`}
                    >
                      <h3 className="font-semibold text-foreground flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" />
                        Rare Values Cleaned
                      </h3>
                      <ChevronDown className={`w-5 h-5 text-muted-foreground transition-transform ${isRareValuesExpanded ? 'rotate-180' : ''}`} />
                    </button>
                    
                    {isRareValuesExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        style={{ overflow: 'hidden' }}
                      >
                        <div className="bg-amber-500/10 border border-amber-500/20 border-t-0 p-4 rounded-b-lg space-y-3">
                            <div className="flex justify-between items-center pb-3 border-b border-amber-500/20">
                              <span className="text-sm font-semibold text-amber-700 dark:text-amber-400">
                                Total Rows Removed
                              </span>
                              <span className="text-2xl font-bold text-amber-700 dark:text-amber-400">
                                {results.changes.rare_values_removed.reduce((sum, item) => sum + item.rows_removed, 0).toLocaleString()}
                              </span>
                            </div>
                            <div className="space-y-2">
                              <p className="text-xs text-muted-foreground mb-2">
                                Removed rare values from {results.changes.rare_values_removed.length} column{results.changes.rare_values_removed.length > 1 ? 's' : ''}:
                              </p>
                              {results.changes.rare_values_removed.map((item, idx) => (
                                <div key={idx} className="bg-background/50 border border-amber-500/10 p-3 rounded-md">
                                  <div className="flex justify-between items-start mb-2">
                                    <span className="text-sm font-medium text-foreground">{item.column}</span>
                                    <span className="text-xs font-semibold text-amber-700 dark:text-amber-400">
                                      {item.rows_removed.toLocaleString()} rows
                                    </span>
                                  </div>
                                  <div className="text-xs text-muted-foreground">
                                    <span className="font-medium">{item.values_removed.length} value{item.values_removed.length > 1 ? 's' : ''} removed:</span>
                                    <div className="mt-1 flex flex-wrap gap-1">
                                      {item.values_removed.map((val, vidx) => (
                                        <span key={vidx} className="bg-amber-500/20 text-amber-800 dark:text-amber-300 px-2 py-0.5 rounded">
                                          {val}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </motion.div>
                      )}
                  </div>
                )}

                {/* Columns Removed Details */}
                {results.changes?.constant_column_details && Object.keys(results.changes.constant_column_details).length > 0 && (
                  <div className="relative">
                    <button
                      onClick={() => setIsColumnsRemovedExpanded(!isColumnsRemovedExpanded)}
                      className={`w-full flex items-center justify-between p-3 bg-red-500/10 border border-red-500/20 hover:bg-red-500/20 transition-colors ${
                        isColumnsRemovedExpanded ? 'rounded-t-lg' : 'rounded-lg'
                      }`}
                    >
                      <h3 className="font-semibold text-foreground flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" />
                        Columns Removed Details
                      </h3>
                      <ChevronDown className={`w-5 h-5 text-muted-foreground transition-transform ${isColumnsRemovedExpanded ? 'rotate-180' : ''}`} />
                    </button>
                    
                    {isColumnsRemovedExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        style={{ overflow: 'hidden' }}
                      >
                        <div className="bg-red-500/10 border border-red-500/20 border-t-0 p-4 rounded-b-lg space-y-3">
                            <div className="space-y-2">
                              {Object.entries(results.changes.constant_column_details)
                                .sort((a, b) => {
                                  // Sort by type: constant, low_variance, high_missing
                                  const typeOrder = { constant: 0, low_variance: 1, high_missing: 2 };
                                  return typeOrder[a[1].type] - typeOrder[b[1].type];
                                })
                                .map(([colName, details], idx) => (
                                  <div key={idx} className="bg-background/50 border border-red-500/10 p-3 rounded-md">
                                    <div className="flex justify-between items-start mb-2">
                                      <div className="flex items-center gap-2">
                                        <span className="text-sm font-medium text-foreground">{colName}</span>
                                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                                          details.type === 'constant' 
                                            ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200'
                                            : details.type === 'low_variance'
                                            ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-200'
                                            : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200'
                                        }`}>
                                          {details.type === 'constant' ? 'Constant' : details.type === 'low_variance' ? 'Low Variance' : 'High Missing'}
                                        </span>
                                      </div>
                                    </div>
                                    
                                    <div className="text-xs text-muted-foreground space-y-1">
                                      {details.type === 'high_missing' ? (
                                        <>
                                          <div>
                                            <span className="font-medium">Missing:</span> {details.missing_count?.toLocaleString() || 0} / {details.total_count?.toLocaleString() || 0} rows ({details.percentage?.toFixed(1) || 0}%)
                                          </div>
                                        </>
                                      ) : (
                                        <>
                                          <div>
                                            <span className="font-medium">Value:</span> '{details.value}' ({details.percentage?.toFixed(1) || 0}%)
                                          </div>
                                          {details.variance !== null && details.variance !== undefined && (
                                            <div>
                                              <span className="font-medium">Variance:</span> {details.variance?.toFixed(4) || 0}
                                            </div>
                                          )}
                                        </>
                                      )}
                                    </div>
                                  </div>
                                ))}
                            </div>
                          </div>
                        </motion.div>
                      )}
                  </div>
                )}

                {/* Data Quality Improvement */}
                {results.initial_quality && results.final_quality && (
                  <div>
                    <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2">
                      <Award className="w-4 h-4" />
                      Data Quality
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-muted/50 border p-4 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Initial Quality</p>
                        <p className="text-3xl font-bold text-foreground">
                          {results.initial_quality.quality_score || 0}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {results.initial_quality.assessment || 'Unknown'}
                        </p>
                      </div>
                      <div className="bg-green-500/10 border border-green-500/20 p-4 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Final Quality</p>
                        <p className="text-3xl font-bold text-green-600 dark:text-green-400">
                          {results.final_quality.quality_score || 0}
                        </p>
                        <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                          {results.final_quality.assessment || 'Unknown'}
                        </p>
                      </div>
                    </div>
                    <div className="mt-3 bg-primary/10 border border-primary/20 p-3 rounded-lg">
                      <p className="text-sm font-semibold text-primary flex items-center gap-2">
                        <TrendingUp className="w-4 h-4" />
                        Quality Improvement: +{
                          ((results.final_quality.quality_score || 0) - 
                           (results.initial_quality.quality_score || 0)).toFixed(1)
                        } points
                      </p>
                    </div>
                  </div>
                )}

                {/* Preprocessing Steps */}
                {results.steps && results.steps.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-foreground mb-4">Steps Completed</h3>
                    <div className="space-y-2.5">
                      {results.steps.map((step, index) => {
                        // Determine icon based on step content
                        let IconComponent = CheckCircle;
                        if (step.includes('Task Type')) IconComponent = Target;
                        else if (step.includes('Model Family')) IconComponent = Brain;
                        else if (step.includes('Removed') || step.includes('columns')) IconComponent = Trash2;
                        else if (step.includes('Features')) IconComponent = Database;
                        else if (step.includes('encoding') || step.includes('Encoding')) IconComponent = Code;
                        else if (step.includes('imbalance')) IconComponent = AlertTriangle;
                        else if (step.includes('scaling') || step.includes('Scaling')) IconComponent = Activity;
                        else if (step.includes('Quality Score')) IconComponent = Award;

                        return (
                          <div key={index} className="flex items-start gap-3 group">
                            {/* Minimal black icon */}
                            <div className="w-5 h-5 flex items-center justify-center mt-0.5 flex-shrink-0 text-foreground/80 group-hover:text-foreground transition-colors">
                              <IconComponent className="w-4 h-4" strokeWidth={2} />
                            </div>
                            {/* Step text */}
                            <span className="text-sm text-foreground leading-relaxed group-hover:text-foreground/80 transition-colors">{step}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Action Button */}
                <div className="pt-4 border-t">
                  <button
                    onClick={handleBack}
                    className="w-full bg-primary text-primary-foreground px-4 py-3 rounded-lg hover:bg-primary/90 transition-colors font-medium"
                  >
                    Return to Processed Datasets
                  </button>
                </div>
              </div>
              )
            ) : (
              <div className="flex items-center justify-center h-64">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
              </div>
            )}
          </div>
        </div>
        )}
      </div>
    </div>
  );
}
