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
        {preprocessingPhase === 'rare-values' && rareValueColumns.length > 0 && (() => {
          const currentColumn = rareValueColumns[currentRareValueIndex];
          const totalRows = currentColumn?.rare_values[0]?.total_rows || 0;
          const selectedCount = currentColumn?.rare_values.reduce((sum, rv) => sum + (selectedRareValues[rv.rare_value] ? rv.count : 0), 0) || 0;
          const remainingRows = totalRows - selectedCount;
          const impactPct = totalRows > 0 ? ((selectedCount / totalRows) * 100).toFixed(2) : '0.00';
          const allSelected = currentColumn?.rare_values.every(rv => selectedRareValues[rv.rare_value]);

          // Compute dynamic step number: Target = step 1, Rare Values = step 2
          const hasOutlierStep = outlierData && Object.keys(outlierData?.outliers_by_column || {}).filter(col => outlierData.outliers_by_column[col].count > 0).length > 0;
          const totalSteps = 2 + (hasOutlierStep ? 1 : 0) + 1; // target + rare + outliers? + processing

          return (
          <div className="bg-card border rounded-2xl p-8 space-y-6">
            {/* ── Header ─────────────────────────────────────────────── */}
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                  <span>Step 2 of {totalSteps}</span>
                  <span>·</span>
                  <span>Rare Categories</span>
                </div>
                <h2 className="text-xl font-bold">
                  {currentColumn?.name}
                  <span className="ml-2 text-sm font-normal text-muted-foreground">
                    Column {currentRareValueIndex + 1} of {rareValueColumns.length}
                  </span>
                </h2>
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
                    className="p-1.5 border rounded-md hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Previous column"
                  >
                    <ChevronLeft className="w-4 h-4" />
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
                    className="p-1.5 border rounded-md hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Next column"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              )}
            </div>

            {/* ── Compact Info Line ──────────────────────────────────── */}
            <p className="text-sm text-muted-foreground">
              <Info className="w-3.5 h-3.5 inline -mt-0.5 mr-1" />
              {currentColumn?.rare_values.length} rare categor{currentColumn?.rare_values.length === 1 ? 'y' : 'ies'} detected below the <span className="font-medium text-foreground">&lt;2%</span> rarity threshold. Removing all would affect <span className="font-medium text-foreground">{totalRows > 0 ? ((currentColumn?.rare_values.reduce((s, rv) => s + rv.count, 0) / totalRows) * 100).toFixed(1) : 0}%</span> of rows.
            </p>

            {/* ── Impact Summary ─────────────────────────────────────── */}
            {currentColumn && (
              <>
                <div className="grid grid-cols-4 gap-4 text-center py-3 px-4 bg-muted/30 rounded-lg border">
                  <div>
                    <div className="text-xs text-muted-foreground">Original Rows</div>
                    <div className="text-lg font-semibold">{totalRows.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Rows Removed</div>
                    <div className={`text-lg font-semibold ${selectedCount > 0 ? 'text-red-600 dark:text-red-400' : ''}`}>{selectedCount.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Impact</div>
                    <div className={`text-lg font-semibold ${parseFloat(impactPct) > 5 ? 'text-red-600 dark:text-red-400' : parseFloat(impactPct) > 0 ? 'text-amber-600 dark:text-amber-400' : ''}`}>{impactPct}%</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Remaining</div>
                    <div className="text-lg font-semibold">{remainingRows.toLocaleString()}</div>
                  </div>
                </div>

                {/* ── Table Header with Select/Deselect All ──────────── */}
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{currentColumn.rare_values.length} rare categor{currentColumn.rare_values.length === 1 ? 'y' : 'ies'}</span>
                  <button
                    onClick={() => { allSelected ? deselectAllRareValues() : selectAllRareValues(); }}
                    className="px-3 py-1.5 bg-background hover:bg-muted border rounded-md font-medium text-xs flex items-center gap-1.5"
                  >
                    {allSelected ? <XCircle className="w-3.5 h-3.5" /> : <CheckCircle className="w-3.5 h-3.5" />}
                    {allSelected ? 'Deselect All' : 'Select All for Removal'}
                  </button>
                </div>

                {/* ── Structured Table ────────────────────────────────── */}
                <div className="border rounded-lg overflow-hidden">
                  {/* Table Header */}
                  <div className="grid grid-cols-[1fr_100px_100px_60px] gap-2 px-4 py-2.5 bg-muted/50 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
                    <div>Category</div>
                    <div className="text-right">Count</div>
                    <div className="text-right">% of Dataset</div>
                    <div className="text-center">Remove</div>
                  </div>

                  {/* Table Body */}
                  <div className="max-h-80 overflow-y-auto divide-y">
                    {currentColumn.rare_values.map((rv) => (
                      <div
                        key={rv.rare_value}
                        onClick={() => toggleRareValue(rv.rare_value)}
                        className={`grid grid-cols-[1fr_100px_100px_60px] gap-2 px-4 py-2.5 cursor-pointer hover:bg-muted/40 items-center ${
                          selectedRareValues[rv.rare_value] ? 'bg-purple-50/50 dark:bg-purple-950/20' : ''
                        }`}
                      >
                        <div className="font-medium text-sm truncate" title={String(rv.rare_value)}>
                          {String(rv.rare_value)}
                        </div>
                        <div className="text-sm text-right tabular-nums text-muted-foreground">
                          {rv.count.toLocaleString()}
                        </div>
                        <div className="text-sm text-right tabular-nums text-muted-foreground">
                          {rv.percentage.toFixed(2)}%
                        </div>
                        <div className="flex justify-center">
                          <input
                            type="checkbox"
                            checked={selectedRareValues[rv.rare_value] || false}
                            onChange={() => {}}
                            className="w-4 h-4 text-purple-600 rounded focus:ring-purple-500"
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Target Column Warning */}
                {checkTargetColumnImpact() && (
                  <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-lg">
                    <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                    <p className="text-xs text-red-800 dark:text-red-200">
                      {checkTargetColumnImpact()}
                    </p>
                  </div>
                )}
              </>
            )}

            {/* ── Action Buttons ─────────────────────────────────────── */}
            <div className="flex items-center justify-between pt-4 border-t">
              <button
                type="button"
                onClick={async (e) => {
                  e.preventDefault();
                  e.stopPropagation();
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
                className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted border rounded-lg"
              >
                Keep All Categories
              </button>

              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    handleSkipColumn();
                  }}
                  className="px-4 py-2 bg-background hover:bg-muted border rounded-lg font-medium text-sm"
                >
                  Skip This Column
                </button>

                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    handleRareValueDecision();
                  }}
                  className="px-5 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Apply Changes & Continue
                </button>
              </div>
            </div>
          </div>
          );
        })()}

        {/* Outliers Review Section - INLINE */}
        {preprocessingPhase === 'outliers' && outlierData && (() => {
          const columnsWithOutliers = Object.entries(outlierData?.outliers_by_column || {})
            .filter(([_, d]) => d.count > 0)
            .sort((a, b) => b[1].percentage - a[1].percentage);

          const totalDatasetRows = outlierData?.total_rows || columnsWithOutliers[0]?.[1]?.total_rows || 0;
          const totalOutlierRows = columnsWithOutliers.reduce((sum, [_, d]) => sum + d.count, 0);
          const highestPct = columnsWithOutliers.length > 0 ? columnsWithOutliers[0][1].percentage : 0;
          const affectedCols = columnsWithOutliers.length;

          // Live impact estimate based on current preferences
          const rowsToRemove = columnsWithOutliers
            .filter(([col]) => outlierPreferences[col] === 'remove')
            .reduce((sum, [_, d]) => sum + d.count, 0);
          const valuesToCap = columnsWithOutliers
            .filter(([col]) => outlierPreferences[col] === 'cap')
            .reduce((sum, [_, d]) => sum + d.count, 0);

          // Dynamic step number: target=1, rare-values=2 (if present), outliers=2 or 3
          const hasRareStep = rareValueColumns.length > 0;
          const outlierStepNum = hasRareStep ? 3 : 2;
          const totalSteps = 1 + (hasRareStep ? 1 : 0) + 1 + 1; // target + rare? + outliers + processing

          const applyToAll = (strategy) => {
            const prefs = {};
            columnsWithOutliers.forEach(([col]) => { prefs[col] = strategy; });
            setOutlierPreferences(prefs);
          };

          // Determine which global strategy is fully active
          const allKeep = columnsWithOutliers.every(([col]) => outlierPreferences[col] === 'keep');
          const allCap = columnsWithOutliers.every(([col]) => outlierPreferences[col] === 'cap');
          const allRemove = columnsWithOutliers.every(([col]) => outlierPreferences[col] === 'remove');

          return (
          <div className="bg-card border rounded-2xl p-8 space-y-6">
            {/* ── Header ─────────────────────────────────────────────── */}
            <div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                <span>Step {outlierStepNum} of {totalSteps}</span>
                <span>·</span>
                <span>Outlier Handling</span>
              </div>
              <h2 className="text-xl font-bold">Outlier Handling</h2>
            </div>

            {/* ── Compact Info Line ──────────────────────────────────── */}
            <p className="text-sm text-muted-foreground">
              <Info className="w-3.5 h-3.5 inline -mt-0.5 mr-1" />
              {affectedCols} column{affectedCols !== 1 ? 's' : ''} contain outliers identified via the IQR method. The default strategy is <span className="font-medium text-foreground">capping</span> (Winsorization to IQR bounds). Override per column as needed.
            </p>

            {/* ── Global Impact Summary ──────────────────────────────── */}
            <div className="grid grid-cols-5 gap-4 text-center py-3 px-4 bg-muted/30 rounded-lg border">
              <div>
                <div className="text-xs text-muted-foreground">Total Rows</div>
                <div className="text-lg font-semibold">{totalDatasetRows.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Rows w/ Outliers</div>
                <div className="text-lg font-semibold">{totalOutlierRows.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Impact</div>
                <div className={`text-lg font-semibold ${totalDatasetRows > 0 && (totalOutlierRows / totalDatasetRows * 100) > 10 ? 'text-red-600 dark:text-red-400' : ''}`}>
                  {totalDatasetRows > 0 ? (totalOutlierRows / totalDatasetRows * 100).toFixed(1) : 0}%
                </div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Affected Cols</div>
                <div className="text-lg font-semibold">{affectedCols}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Highest Col %</div>
                <div className="text-lg font-semibold">{highestPct.toFixed(1)}%</div>
              </div>
            </div>

            {/* ── Default Strategy Selector ──────────────────────────── */}
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Default Strategy</span>
              <div className="flex items-center gap-1 border rounded-lg p-0.5">
                <button
                  onClick={() => applyToAll('keep')}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium ${
                    allKeep ? 'bg-foreground text-background' : 'hover:bg-muted'
                  }`}
                >
                  Keep All
                </button>
                <button
                  onClick={() => applyToAll('cap')}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium ${
                    allCap ? 'bg-foreground text-background' : 'hover:bg-muted'
                  }`}
                >
                  Cap All
                </button>
                <button
                  onClick={() => applyToAll('remove')}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium ${
                    allRemove ? 'bg-foreground text-background' : 'hover:bg-muted'
                  }`}
                >
                  Remove Rows
                </button>
              </div>
            </div>

            {/* ── Per-Column Table ────────────────────────────────────── */}
            <div className="border rounded-lg overflow-hidden">
              {/* Table Header */}
              <div className="grid grid-cols-[1fr_80px_70px_140px_130px] gap-2 px-4 py-2.5 bg-muted/50 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
                <div>Column</div>
                <div className="text-right">Outliers</div>
                <div className="text-right">%</div>
                <div className="text-right">Value Range</div>
                <div className="text-center">Strategy</div>
              </div>

              {/* Table Body */}
              <div className="max-h-[420px] overflow-y-auto divide-y">
                {columnsWithOutliers.map(([column, data]) => (
                  <div key={column} className="grid grid-cols-[1fr_80px_70px_140px_130px] gap-2 px-4 py-2.5 items-center hover:bg-muted/30">
                    <div className="min-w-0">
                      <div className="font-medium text-sm truncate" title={column}>{column}</div>
                      <div className="text-[11px] text-muted-foreground">
                        IQR [{data.lower_bound.toFixed(1)}, {data.upper_bound.toFixed(1)}]
                      </div>
                    </div>
                    <div className="text-sm text-right tabular-nums text-muted-foreground">
                      {data.count.toLocaleString()}
                    </div>
                    <div className={`text-sm text-right tabular-nums ${
                      data.percentage > 15 ? 'text-red-600 dark:text-red-400 font-medium' : data.percentage > 10 ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground'
                    }`}>
                      {data.percentage.toFixed(1)}%
                    </div>
                    <div className="text-xs text-right tabular-nums text-muted-foreground">
                      {data.min.toFixed(2)} – {data.max.toFixed(2)}
                    </div>
                    <div className="flex justify-center">
                      <select
                        value={outlierPreferences[column] || 'keep'}
                        onChange={(e) => setOutlierPreferences(prev => ({...prev, [column]: e.target.value}))}
                        className="px-2 py-1 text-xs border rounded-md bg-background focus:ring-1 focus:ring-primary focus:outline-none w-full max-w-[120px]"
                      >
                        <option value="keep">Keep</option>
                        <option value="cap">Cap (IQR)</option>
                        <option value="remove">Remove Rows</option>
                      </select>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Live Impact Estimate ────────────────────────────────── */}
            {(rowsToRemove > 0 || valuesToCap > 0) && (
              <div className="flex items-center gap-6 px-4 py-3 bg-muted/30 rounded-lg border text-sm">
                <span className="text-muted-foreground font-medium">Estimated impact:</span>
                {valuesToCap > 0 && (
                  <span>
                    <span className="font-semibold text-amber-600 dark:text-amber-400">{valuesToCap.toLocaleString()}</span>
                    <span className="text-muted-foreground ml-1">values capped</span>
                  </span>
                )}
                {rowsToRemove > 0 && (
                  <span>
                    <span className="font-semibold text-red-600 dark:text-red-400">{rowsToRemove.toLocaleString()}</span>
                    <span className="text-muted-foreground ml-1">rows removed</span>
                    <span className="text-muted-foreground ml-1">({totalDatasetRows > 0 ? (rowsToRemove / totalDatasetRows * 100).toFixed(1) : 0}%)</span>
                  </span>
                )}
              </div>
            )}

            {/* ── Action Buttons ─────────────────────────────────────── */}
            <div className="flex items-center justify-between pt-4 border-t">
              <button
                onClick={handleSkipOutliers}
                className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted border rounded-lg"
              >
                Skip This Step
              </button>
              <button
                onClick={() => handleOutlierPreferences(outlierPreferences)}
                className="px-5 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg font-medium text-sm"
              >
                Apply Strategy & Continue
              </button>
            </div>
          </div>
          );
        })()}

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
