/**
 * EDA Charts Component - Visual analytics for exploratory data analysis
 */
import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  PieChart as RechartsPieChart,
  Pie,
  ComposedChart,
  Area,
  ReferenceLine,
  ReferenceArea
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './card';
import { AlertCircle, BarChart3, TrendingUp, PieChart, Activity, Filter, Box, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const COLORS = ['#8b5cf6', '#6366f1', '#3b82f6', '#0ea5e9', '#06b6d4', '#14b8a6', '#10b981', '#84cc16', '#eab308', '#f59e0b'];
const HEATMAP_COLORS = ['#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe', '#ffffff', '#fecaca', '#fca5a5', '#f87171', '#ef4444'];

// Utility function to check if column is an ID column
const isIdColumn = (columnName) => {
  if (!columnName) return false;
  const lowerName = columnName.toLowerCase();
  return lowerName === 'id' || lowerName.endsWith('_id') || lowerName.startsWith('id_');
};

// Helper function to generate normal distribution curve points
const generateBellCurve = (mean, std, min, max, points = 100) => {
  const data = [];
  const range = max - min;
  const step = range / points;
  
  for (let i = 0; i <= points; i++) {
    const x = min + (step * i);
    // Normal distribution formula
    const exponent = -Math.pow(x - mean, 2) / (2 * Math.pow(std, 2));
    const y = (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
    data.push({ x: parseFloat(x.toFixed(2)), y: parseFloat(y.toFixed(6)) });
  }
  
  return data;
};

// Outlier Bell Curve Visualization Component
const OutlierBellCurveVisualization = ({ outliersData }) => {
  const [selectedColumn, setSelectedColumn] = useState(null);
  
  // Filter columns that have outliers
  const columnsWithOutliers = Object.entries(outliersData || {})
    .filter(([_, data]) => data.count > 0)
    .map(([name, data]) => ({ name, ...data }))
    .sort((a, b) => b.count - a.count);
  
  if (columnsWithOutliers.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No outliers detected in any numerical columns
      </div>
    );
  }
  
  // Set initial column if not set
  if (!selectedColumn && columnsWithOutliers.length > 0) {
    setSelectedColumn(columnsWithOutliers[0].name);
  }
  
  const currentData = columnsWithOutliers.find(col => col.name === selectedColumn);
  
  if (!currentData) return null;
  
  // Check if we have the required data for bell curve visualization
  const hasFullData = currentData.mean !== undefined && 
                       currentData.std !== undefined && 
                       currentData.min !== undefined && 
                       currentData.max !== undefined;
  
  // If we don't have full data, show simple outlier count
  if (!hasFullData) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium">Select Column:</label>
          <select
            value={selectedColumn}
            onChange={(e) => setSelectedColumn(e.target.value)}
            className="px-3 py-2 border rounded-lg bg-background text-sm"
          >
            {columnsWithOutliers.map(col => (
              <option key={col.name} value={col.name}>
                {col.name} ({col.count} outliers - {col.percentage?.toFixed(1) || '0.0'}%)
              </option>
            ))}
          </select>
        </div>
        
        <div className="p-4 bg-muted/30 rounded-lg space-y-2">
          <div className="text-sm font-medium">{currentData.name}</div>
          <div className="text-2xl font-bold text-destructive">{currentData.count}</div>
          <div className="text-xs text-muted-foreground">
            {currentData.percentage?.toFixed(1) || '0.0'}% of total values
          </div>
          {currentData.lower_bound !== undefined && currentData.upper_bound !== undefined && (
            <div className="pt-2 border-t mt-2 space-y-1">
              <div className="text-xs">
                <span className="text-muted-foreground">Lower Bound (IQR): </span>
                <span className="font-semibold">{currentData.lower_bound.toFixed(2)}</span>
              </div>
              <div className="text-xs">
                <span className="text-muted-foreground">Upper Bound (IQR): </span>
                <span className="font-semibold">{currentData.upper_bound.toFixed(2)}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }
  
  // Generate bell curve data
  const bellCurveData = generateBellCurve(
    currentData.mean,
    currentData.std,
    currentData.min,
    currentData.max,
    150
  );
  
  // Create histogram bins for inliers and outliers
  const createHistogram = (values, bins = 20, color) => {
    if (!values || values.length === 0) return [];
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binWidth = (max - min) / bins;
    
    const histogram = Array(bins).fill(0).map((_, i) => ({
      x: min + (binWidth * i) + (binWidth / 2),
      count: 0,
      color
    }));
    
    values.forEach(val => {
      const binIndex = Math.min(Math.floor((val - min) / binWidth), bins - 1);
      if (binIndex >= 0 && binIndex < bins) {
        histogram[binIndex].count++;
      }
    });
    
    return histogram;
  };
  
  const inlierHistogram = createHistogram(currentData.inlier_sample || [], 30, COLORS[2]);
  const outlierHistogram = createHistogram(currentData.outlier_values || [], 30, '#ef4444');
  
  // Combine histograms
  const allHistogramData = [...inlierHistogram, ...outlierHistogram];
  const combinedHistogram = allHistogramData.length > 0 
    ? allHistogramData.sort((a, b) => a.x - b.x) 
    : [];
  
  // Normalize histogram for overlay
  const maxCount = combinedHistogram.length > 0 ? Math.max(...combinedHistogram.map(d => d.count)) : 1;
  const maxBellY = Math.max(...bellCurveData.map(d => d.y));
  const scale = maxBellY > 0 ? (maxCount / maxBellY * 0.8) : 1;
  
  const normalizedBellCurve = bellCurveData.map(point => ({
    ...point,
    y: point.y * scale
  }));
  
  return (
    <div className="space-y-4">
      {/* Column Selector */}
      <div className="flex items-center gap-3">
        <label className="text-sm font-medium">Select Column:</label>
        <select
          value={selectedColumn}
          onChange={(e) => setSelectedColumn(e.target.value)}
          className="px-3 py-2 border rounded-lg bg-background text-sm"
        >
          {columnsWithOutliers.map(col => (
            <option key={col.name} value={col.name}>
              {col.name} ({col.count} outliers - {col.percentage.toFixed(1)}%)
            </option>
          ))}
        </select>
      </div>
      
      {/* Statistics Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4 bg-muted/30 rounded-lg">
        <div>
          <div className="text-xs text-muted-foreground">Mean</div>
          <div className="font-semibold">{currentData.mean.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Std Dev</div>
          <div className="font-semibold">{currentData.std.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Lower Bound (IQR)</div>
          <div className="font-semibold text-orange-500">{currentData.lower_bound?.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Upper Bound (IQR)</div>
          <div className="font-semibold text-orange-500">{currentData.upper_bound?.toFixed(2)}</div>
        </div>
      </div>
      
      {/* Bell Curve with Outliers */}
      <div className="space-y-2">
        <div className="text-sm font-medium">Distribution with Outlier Boundaries</div>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={normalizedBellCurve}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis 
              dataKey="x" 
              type="number"
              domain={['dataMin', 'dataMax']}
              tick={{ fill: 'currentColor' }} 
              className="text-xs fill-foreground"
              label={{ value: 'Value', position: 'insideBottom', offset: -5 }}
              tickFormatter={(value) => typeof value === 'number' ? value.toFixed(2) : value}
            />
            <YAxis 
              tick={{ fill: 'currentColor' }} 
              className="text-xs fill-foreground"
              label={{ value: 'Density / Count', angle: -90, position: 'insideLeft' }}
              tickFormatter={(value) => typeof value === 'number' ? value.toFixed(2) : value}
            />
            <Tooltip 
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
                      <div className="text-sm font-semibold">Value: {payload[0].payload.x.toFixed(2)}</div>
                      {payload.map((entry, index) => (
                        <div key={index} className="text-xs" style={{ color: entry.color }}>
                          {entry.name}: {entry.value.toFixed(2)}
                        </div>
                      ))}
                    </div>
                  );
                }
                return null;
              }}
            />
            
            {/* Lower bound reference line */}
            {currentData.lower_bound !== undefined && (
              <ReferenceLine 
                x={currentData.lower_bound} 
                stroke="#f97316" 
                strokeWidth={3}
                strokeDasharray="5 5"
                label={{ value: `Lower: ${currentData.lower_bound.toFixed(2)}`, position: 'top', fill: '#f97316', fontSize: 11 }}
              />
            )}
            
            {/* Upper bound reference line */}
            {currentData.upper_bound !== undefined && (
              <ReferenceLine 
                x={currentData.upper_bound} 
                stroke="#f97316" 
                strokeWidth={3}
                strokeDasharray="5 5"
                label={{ value: `Upper: ${currentData.upper_bound.toFixed(2)}`, position: 'top', fill: '#f97316', fontSize: 11 }}
              />
            )}
            
            {/* Outlier regions */}
            {currentData.lower_bound !== undefined && currentData.min !== undefined && (
              <ReferenceArea 
                x1={currentData.min} 
                x2={currentData.lower_bound} 
                fill="#ef4444" 
                fillOpacity={0.15}
                label={{ value: 'Low Outliers', position: 'center', fill: '#ef4444', fontSize: 10 }}
              />
            )}
            {currentData.upper_bound !== undefined && currentData.max !== undefined && (
              <ReferenceArea 
                x1={currentData.upper_bound} 
                x2={currentData.max} 
                fill="#ef4444" 
                fillOpacity={0.15}
                label={{ value: 'High Outliers', position: 'center', fill: '#ef4444', fontSize: 10 }}
              />
            )}
            
            {/* Normal distribution curve */}
            <Area 
              type="monotone" 
              dataKey="y" 
              fill={COLORS[2]} 
              fillOpacity={0.3}
              stroke={COLORS[2]} 
              strokeWidth={2}
              name="Normal Distribution"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      {/* Data Distribution Histogram */}
      <div className="space-y-2">
        <div className="text-sm font-medium">Actual Data Distribution</div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={combinedHistogram}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis 
              dataKey="x" 
              tick={{ fill: 'currentColor' }} 
              className="text-xs fill-foreground"
              label={{ value: 'Value', position: 'insideBottom', offset: -5 }}
              tickFormatter={(value) => typeof value === 'number' ? value.toFixed(2) : value}
            />
            <YAxis 
              tick={{ fill: 'currentColor' }} 
              className="text-xs fill-foreground"
              label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip 
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const isOutlier = payload[0].payload.color === '#ef4444';
                  return (
                    <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
                      <div className="text-sm font-semibold">Value: {payload[0].payload.x.toFixed(2)}</div>
                      <div className="text-xs">Count: {payload[0].value}</div>
                      <div className="text-xs font-semibold" style={{ color: isOutlier ? '#ef4444' : COLORS[2] }}>
                        {isOutlier ? '⚠️ Outlier' : '✓ Normal'}
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {combinedHistogram.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
            <ReferenceLine 
              x={currentData.lower_bound} 
              stroke="#f97316" 
              strokeWidth={2}
              strokeDasharray="5 5"
            />
            <ReferenceLine 
              x={currentData.upper_bound} 
              stroke="#f97316" 
              strokeWidth={2}
              strokeDasharray="5 5"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: COLORS[2] }}></div>
          <span>Normal Data</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-red-500"></div>
          <span>Outliers</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-12 h-0.5 bg-orange-500" style={{ borderTop: '2px dashed #f97316' }}></div>
          <span>IQR Bounds</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-12 h-2 rounded" style={{ backgroundColor: COLORS[2], opacity: 0.3 }}></div>
          <span>Bell Curve</span>
        </div>
      </div>
    </div>
  );
};

const EDACharts = ({ edaResults, className }) => {
  const [showFilter, setShowFilter] = useState(false);
  const [selectedCharts, setSelectedCharts] = useState('all');
  const [selectedHistogramColumn, setSelectedHistogramColumn] = useState(0);
  const [selectedCategoryColumn, setSelectedCategoryColumn] = useState(0);

  if (!edaResults) return null;

  const { basic_info, missing_values, duplicates, data_quality, distributions, correlations, categorical_analysis, outliers } = edaResults;

  // Prepare data for missing values chart
  const missingValuesData = Object.entries(missing_values?.columns_with_missing || {})
    .map(([column, data]) => ({
      name: column.length > 20 ? column.substring(0, 17) + '...' : column,
      fullName: column,
      percentage: data.percentage,
      count: data.count
    }))
    .sort((a, b) => b.percentage - a.percentage)
    .slice(0, 10); // Top 10

  // Prepare data for column types distribution
  const columnTypesData = basic_info?.column_types ? [
    { name: 'Numerical', value: basic_info.column_types.numerical_count, color: COLORS[0] },
    { name: 'Categorical', value: basic_info.column_types.categorical_count, color: COLORS[1] },
    { name: 'DateTime', value: basic_info.column_types.datetime_count, color: COLORS[2] }
  ].filter(item => item.value > 0) : [];

  // Prepare data for distribution statistics (mean, median, std) - EXCLUDE ID COLUMNS
  const distributionStatsData = distributions?.distributions ? 
    Object.entries(distributions.distributions)
      .filter(([column]) => !isIdColumn(column)) // Exclude ID columns
      .map(([column, stats]) => ({
        name: column.length > 15 ? column.substring(0, 12) + '...' : column,
        fullName: column,
        mean: stats.mean,
        median: stats.median,
        std: stats.std
      }))
      .filter(item => item.mean !== null && item.median !== null)
    : [];

  // Prepare data for outliers - EXCLUDE ID COLUMNS
  const outliersData = outliers?.outliers_by_column ?
    Object.entries(outliers.outliers_by_column)
      .filter(([column]) => !isIdColumn(column)) // Exclude ID columns
      .map(([column, data]) => ({
        name: column.length > 20 ? column.substring(0, 17) + '...' : column,
        fullName: column,
        count: data.count,
        percentage: data.percentage
      }))
      .filter(item => item.count > 0)
      .sort((a, b) => b.count - a.count)
      .slice(0, 10)
    : [];

  // Prepare data for categorical cardinality
  const categoricalCardinalityData = categorical_analysis?.categorical_features ?
    Object.entries(categorical_analysis.categorical_features)
      .map(([column, data]) => ({
        name: column.length > 20 ? column.substring(0, 17) + '...' : column,
        fullName: column,
        uniqueCount: data.unique_count,
        cardinality: data.cardinality
      }))
      .sort((a, b) => b.uniqueCount - a.uniqueCount)
      .slice(0, 10)
    : [];

  // Prepare data for data quality radar chart
  const qualityData = data_quality ? [
    {
      metric: 'Quality Score',
      value: data_quality.quality_score || 0
    },
    {
      metric: 'Completeness',
      value: data_quality.completeness || 0
    },
    {
      metric: 'Uniqueness',
      value: data_quality.uniqueness || 0
    }
  ] : [];

  // Prepare correlation heatmap data (top correlations) - EXCLUDE ID COLUMNS
  const correlationData = correlations?.high_correlations ?
    correlations.high_correlations
      .filter(item => !isIdColumn(item.feature1) && !isIdColumn(item.feature2)) // Exclude ID columns
      .slice(0, 10)
      .map(item => ({
        pair: `${item.feature1.substring(0, 10)}... × ${item.feature2.substring(0, 10)}...`,
        fullPair: `${item.feature1} × ${item.feature2}`,
        correlation: Math.abs(item.correlation),
        originalCorrelation: item.correlation
      }))
      .sort((a, b) => b.correlation - a.correlation)
    : [];

  // Prepare skewness data - EXCLUDE ID COLUMNS
  const skewnessData = distributions?.distributions ?
    Object.entries(distributions.distributions)
      .filter(([column]) => !isIdColumn(column)) // Exclude ID columns
      .map(([column, stats]) => ({
        name: column.length > 15 ? column.substring(0, 12) + '...' : column,
        fullName: column,
        skewness: Math.abs(stats.skewness || 0),
        originalSkewness: stats.skewness
      }))
      .filter(item => item.skewness > 0.5) // Only show significantly skewed
      .sort((a, b) => b.skewness - a.skewness)
      .slice(0, 10)
    : [];

  // NEW: Prepare full correlation matrix for heatmap - EXCLUDE ID COLUMNS
  const correlationMatrix = correlations?.correlation_matrix || {};
  const nonIdFeatures = Object.keys(correlationMatrix).filter(col => !isIdColumn(col));
  const corrMatrixData = nonIdFeatures.map(feature1 => {
    const row = { feature: feature1.length > 12 ? feature1.substring(0, 10) + '...' : feature1, fullFeature: feature1 };
    nonIdFeatures.forEach(feature2 => {
      row[feature2] = correlationMatrix[feature1]?.[feature2] || 0;
    });
    return row;
  });

  // NEW: Prepare histogram data for numerical distributions - EXCLUDE ID COLUMNS
  const histogramData = distributions?.distributions ? 
    Object.entries(distributions.distributions)
      .filter(([column]) => !isIdColumn(column)) // Exclude ID columns
      .map(([column, stats]) => ({
        column,
        shortName: column.length > 15 ? column.substring(0, 12) + '...' : column,
        data: [
          { range: 'Min-Q1', value: stats.q25 - stats.min, label: `${stats.min?.toFixed(1)} - ${stats.q25?.toFixed(1)}` },
          { range: 'Q1-Q2', value: stats.q50 - stats.q25, label: `${stats.q25?.toFixed(1)} - ${stats.q50?.toFixed(1)}` },
          { range: 'Q2-Q3', value: stats.q75 - stats.q50, label: `${stats.q50?.toFixed(1)} - ${stats.q75?.toFixed(1)}` },
          { range: 'Q3-Max', value: stats.max - stats.q75, label: `${stats.q75?.toFixed(1)} - ${stats.max?.toFixed(1)}` }
        ].filter(item => item.value > 0)
      }))
    : [];

  // NEW: Prepare box plot data - EXCLUDE ID COLUMNS
  const boxPlotData = distributions?.distributions ?
    Object.entries(distributions.distributions)
      .filter(([column]) => !isIdColumn(column)) // Exclude ID columns
      .map(([column, stats]) => ({
        name: column.length > 12 ? column.substring(0, 10) + '...' : column,
        fullName: column,
        min: stats.min,
        q1: stats.q25,
        median: stats.q50,
        q3: stats.q75,
        max: stats.max,
        mean: stats.mean
      }))
      .filter(item => item.min !== null && item.max !== null)
    : [];

  // NEW: Prepare pie chart data for categorical balance - ALL columns
  const categoryBalanceData = categorical_analysis?.categorical_features ?
    Object.entries(categorical_analysis.categorical_features)
      .map(([column, data]) => ({
        column,
        shortName: column.length > 20 ? column.substring(0, 17) + '...' : column,
        values: Object.entries(data.top_5_values || {}).map(([cat, count]) => ({
          name: cat.length > 15 ? cat.substring(0, 12) + '...' : cat,
          fullName: cat,
          value: count
        }))
      }))
    : [];

  // Count excluded ID columns
  const allNumericalColumns = distributions?.distributions ? Object.keys(distributions.distributions) : [];
  const excludedIdColumns = allNumericalColumns.filter(col => isIdColumn(col));
  const hasExcludedIds = excludedIdColumns.length > 0;

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-700 rounded-lg p-3 shadow-lg">
          <p className="font-medium text-sm text-foreground">{payload[0].payload.fullName || label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-xs text-foreground" style={{ color: entry.color }}>
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className={className}>
      {/* Info Notification about ID columns */}
      {hasExcludedIds && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 bg-blue-500/10 border border-blue-500/20 rounded-lg p-4 flex items-start gap-3"
        >
          <Info className="h-5 w-5 text-blue-500 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-1">
              ID Columns Excluded from Charts
            </p>
            <p className="text-xs text-blue-600/80 dark:text-blue-400/80">
              {excludedIdColumns.length} ID column{excludedIdColumns.length > 1 ? 's' : ''} ({excludedIdColumns.join(', ')}) 
              {excludedIdColumns.length > 1 ? ' have' : ' has'} been automatically excluded from visualizations to improve chart clarity.
            </p>
          </div>
        </motion.div>
      )}

      {/* Filter and Export Controls */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4 bg-card border rounded-xl p-4">
        <div className="flex items-center gap-3">
          <Filter className="h-5 w-5 text-muted-foreground" />
          <select
            value={selectedCharts}
            onChange={(e) => setSelectedCharts(e.target.value)}
            className="px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="all">All Charts</option>
            <option value="overview">Overview Charts</option>
            <option value="quality">Quality & Missing Data</option>
            <option value="distributions">Distributions & Stats</option>
            <option value="correlations">Correlations</option>
            <option value="advanced">Advanced Charts</option>
          </select>
        </div>
        <div className="text-sm text-muted-foreground">
          {selectedCharts === 'all' ? 'Showing all visualizations' : `Filtered: ${selectedCharts}`}
        </div>
      </div>

      {/* CHART GRID WITH ANIMATED TRANSITIONS */}
      <AnimatePresence mode="wait">
        <motion.div 
          key={selectedCharts}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
        
        {/* Dataset Overview - Column Types */}
        {(selectedCharts === 'all' || selectedCharts === 'overview') && columnTypesData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChart className="h-5 w-5" />
                Column Type Distribution
              </CardTitle>
              <CardDescription>Breakdown of column data types</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={columnTypesData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="name" tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <YAxis tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                    {columnTypesData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Data Quality Radar */}
        {(selectedCharts === 'all' || selectedCharts === 'overview' || selectedCharts === 'quality') && qualityData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Data Quality Metrics
              </CardTitle>
              <CardDescription>Overall quality assessment (0-100)</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={qualityData}>
                  <PolarGrid stroke="currentColor" strokeOpacity={0.15} />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <Radar name="Score" dataKey="value" stroke={COLORS[0]} fill={COLORS[0]} fillOpacity={0.3} isAnimationActive={false} dot={false} />
                  <Tooltip content={<CustomTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Missing Values */}
        {(selectedCharts === 'all' || selectedCharts === 'quality') && missingValuesData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5" />
                Missing Values by Column
              </CardTitle>
              <CardDescription>Top 10 columns with missing data</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={missingValuesData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis type="number" tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" unit="%" />
                  <YAxis type="category" dataKey="name" width={120} tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="percentage" fill={COLORS[3]} radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Outliers Detection with Bell Curve */}
        {(selectedCharts === 'all' || selectedCharts === 'quality') && outliersData.length > 0 && (
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Outlier Distribution Analysis
              </CardTitle>
              <CardDescription>
                Interactive bell curve showing normal distribution, IQR bounds, and outliers for each column
              </CardDescription>
            </CardHeader>
            <CardContent>
              <OutlierBellCurveVisualization 
                outliersData={edaResults.outliers?.outliers_by_column || {}} 
              />
            </CardContent>
          </Card>
        )}


        {/* Distribution Statistics */}
        {(selectedCharts === 'all' || selectedCharts === 'distributions') && distributionStatsData.length > 0 && (
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Distribution Statistics (Mean, Median, Std Dev)
              </CardTitle>
              <CardDescription>Central tendency and spread for numerical columns</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={Math.max(400, distributionStatsData.length * 40)}>
                <BarChart data={distributionStatsData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="name" tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <YAxis tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="mean" fill={COLORS[0]} radius={[8, 8, 0, 0]} />
                  <Bar dataKey="median" fill={COLORS[1]} radius={[8, 8, 0, 0]} />
                  <Bar dataKey="std" fill={COLORS[2]} radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Categorical Cardinality */}
        {(selectedCharts === 'all' || selectedCharts === 'overview') && categoricalCardinalityData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Categorical Cardinality
              </CardTitle>
              <CardDescription>Unique value counts for categorical features</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={categoricalCardinalityData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="name" tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <YAxis tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="uniqueCount" radius={[8, 8, 0, 0]}>
                    {categoricalCardinalityData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.cardinality === 'high' ? COLORS[8] : entry.cardinality === 'medium' ? COLORS[4] : COLORS[6]} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              
              {/* Color Legend for Cardinality */}
              <div className="flex flex-wrap items-center justify-center gap-4 mt-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: COLORS[6] }}></div>
                  <span>Low Cardinality (&lt; 10)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: COLORS[4] }}></div>
                  <span>Medium Cardinality (10-50)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: COLORS[8] }}></div>
                  <span>High Cardinality (&gt; 50)</span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* High Correlations */}
        {(selectedCharts === 'all' || selectedCharts === 'correlations') && correlationData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                High Correlations
              </CardTitle>
              <CardDescription>Feature pairs with |correlation| &gt; 0.7</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={correlationData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis type="number" domain={[0, 1]} tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <YAxis type="category" dataKey="pair" width={150} tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="correlation" fill={COLORS[5]} radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Skewness */}
        {(selectedCharts === 'all' || selectedCharts === 'distributions') && skewnessData.length > 0 && (
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Distribution Skewness
              </CardTitle>
              <CardDescription>Columns with significantly skewed distributions (|skew| &gt; 0.5)</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={skewnessData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="name" tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <YAxis tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="skewness" radius={[8, 8, 0, 0]}>
                    {skewnessData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.originalSkewness > 0 ? COLORS[8] : COLORS[3]} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              
              {/* Color Legend for Skewness */}
              <div className="flex flex-wrap items-center justify-center gap-4 mt-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: COLORS[8] }}></div>
                  <span>Right-Skewed (Positive)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: COLORS[3] }}></div>
                  <span>Left-Skewed (Negative)</span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* NEW: Correlation Heatmap */}
        {(selectedCharts === 'all' || selectedCharts === 'correlations' || selectedCharts === 'advanced') && corrMatrixData.length > 0 && (
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Correlation Matrix Heatmap
              </CardTitle>
              <CardDescription>Full correlation matrix for numerical features</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2 text-xs font-medium text-left text-foreground">Feature</th>
                      {nonIdFeatures.map((feat, idx) => (
                        <th key={idx} className="p-2 text-xs font-medium text-center text-foreground" title={feat}>
                          {feat.length > 8 ? feat.substring(0, 6) + '..' : feat}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {corrMatrixData.map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        <td className="p-2 text-xs font-medium text-foreground" title={row.fullFeature}>{row.feature}</td>
                        {nonIdFeatures.map((feat, colIdx) => {
                          const value = row[feat] || 0;
                          const colorIdx = Math.floor((value + 1) / 2 * (HEATMAP_COLORS.length - 1));
                          return (
                            <td
                              key={colIdx}
                              className="p-2 text-center text-xs font-medium text-black dark:text-black"
                              style={{ backgroundColor: HEATMAP_COLORS[colorIdx] }}
                              title={`${row.fullFeature} × ${feat}: ${value.toFixed(3)}`}
                            >
                              {value.toFixed(2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {/* NEW: Box Plots for Distributions */}
        {(selectedCharts === 'all' || selectedCharts === 'distributions' || selectedCharts === 'advanced') && boxPlotData.length > 0 && (
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Box className="h-5 w-5" />
                Distribution Box Plots
              </CardTitle>
              <CardDescription>Five-number summary for numerical columns</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={Math.max(450, boxPlotData.length * 60)}>
                <ComposedChart data={boxPlotData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="name" tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <YAxis tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <Tooltip content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-700 rounded-lg p-3 shadow-lg">
                          <p className="font-medium text-sm mb-2 text-foreground">{data.fullName}</p>
                          <p className="text-xs text-foreground">Min: {data.min?.toFixed(2)}</p>
                          <p className="text-xs text-foreground">Q1: {data.q1?.toFixed(2)}</p>
                          <p className="text-xs text-foreground">Median: {data.median?.toFixed(2)}</p>
                          <p className="text-xs text-foreground">Mean: {data.mean?.toFixed(2)}</p>
                          <p className="text-xs text-foreground">Q3: {data.q3?.toFixed(2)}</p>
                          <p className="text-xs text-foreground">Max: {data.max?.toFixed(2)}</p>
                        </div>
                      );
                    }
                    return null;
                  }} />
                  <Bar dataKey="q1" stackId="box" fill={COLORS[0]} />
                  <Bar dataKey="median" stackId="box" fill={COLORS[1]} />
                  <Bar dataKey="q3" stackId="box" fill={COLORS[2]} />
                  <Line type="monotone" dataKey="mean" stroke={COLORS[8]} strokeWidth={2} dot={{ r: 4 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* NEW: Histograms for Distributions */}
        {(selectedCharts === 'all' || selectedCharts === 'distributions' || selectedCharts === 'advanced') && histogramData.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Value Distribution by Quartiles
                  </CardTitle>
                  <CardDescription>Distribution breakdown for selected column</CardDescription>
                </div>
                <select
                  value={selectedHistogramColumn}
                  onChange={(e) => setSelectedHistogramColumn(Number(e.target.value))}
                  className="px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  {histogramData.map((item, idx) => (
                    <option key={idx} value={idx}>{item.column}</option>
                  ))}
                </select>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={histogramData[selectedHistogramColumn]?.data || []}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="range" tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <YAxis tick={{ fill: 'currentColor' }} className="text-xs fill-foreground" />
                  <Tooltip content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-700 rounded-lg p-3 shadow-lg">
                          <p className="font-medium text-sm text-foreground">{payload[0].payload.label}</p>
                          <p className="text-xs text-foreground">Range: {payload[0].value?.toFixed(2)}</p>
                        </div>
                      );
                    }
                    return null;
                  }} />
                  <Bar dataKey="value" fill={COLORS[selectedHistogramColumn % COLORS.length]} radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* NEW: Category Balance Pie Chart */}
        {(selectedCharts === 'all' || selectedCharts === 'overview' || selectedCharts === 'advanced') && categoryBalanceData.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <PieChart className="h-5 w-5" />
                  Category Balance
                </CardTitle>
                {categoryBalanceData.length > 1 && (
                  <select
                    value={selectedCategoryColumn}
                    onChange={(e) => setSelectedCategoryColumn(parseInt(e.target.value))}
                    className="px-3 py-1.5 text-sm border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
                  >
                    {categoryBalanceData.map((item, idx) => (
                      <option key={idx} value={idx}>
                        {item.column}
                      </option>
                    ))}
                  </select>
                )}
              </div>
              <CardDescription>Distribution of top 5 categories</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <RechartsPieChart>
                  <Pie
                    data={categoryBalanceData[selectedCategoryColumn]?.values || []}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    labelLine={true}
                  >
                    {(categoryBalanceData[selectedCategoryColumn]?.values || []).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-700 rounded-lg p-3 shadow-lg">
                          <p className="font-medium text-sm text-foreground">{payload[0].payload.fullName}</p>
                          <p className="text-xs text-foreground">Count: {payload[0].value}</p>
                        </div>
                      );
                    }
                    return null;
                  }} />
                  <Legend />
                </RechartsPieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default EDACharts;
