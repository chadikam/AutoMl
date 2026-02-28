/**
 * EDA Summary Component - Display comprehensive EDA results
 */
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Database, 
  AlertCircle, 
  TrendingUp, 
  Copy, 
  AlertTriangle,
  CheckCircle,
  Info,
  BarChart3,
  Hash,
  Percent,
  Calendar,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { cn } from '../../lib/utils';

const EDASummary = ({ edaResults, className }) => {
  if (!edaResults) return null;

  const { basic_info, missing_values, duplicates, data_quality, recommendations } = edaResults;

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            Dataset Analysis Complete
          </h3>
          <p className="text-sm text-muted-foreground mt-1">
            Comprehensive exploratory data analysis performed successfully
          </p>
        </div>
        {data_quality && (
          <QualityBadge score={data_quality.quality_score} assessment={data_quality.assessment} />
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Dataset Shape */}
        <StatCard
          icon={<Database className="w-5 h-5" />}
          label="Dataset Shape"
          value={`${basic_info?.shape?.rows?.toLocaleString() || 0} × ${basic_info?.shape?.columns || 0}`}
          subtitle={`${basic_info?.shape?.rows?.toLocaleString() || 0} rows, ${basic_info?.shape?.columns || 0} columns`}
          color="blue"
        />

        {/* Numerical Columns */}
        <StatCard
          icon={<Hash className="w-5 h-5" />}
          label="Numerical Columns"
          value={`${basic_info?.column_types?.numerical_count || 0}`}
          subtitle="Numeric features"
          color="green"
        />

        {/* Categorical Columns */}
        <StatCard
          icon={<Percent className="w-5 h-5" />}
          label="Categorical Columns"
          value={`${basic_info?.column_types?.categorical_count || 0}`}
          subtitle="Categorical features"
          color="purple"
        />

        {/* DateTime Columns */}
        <StatCard
          icon={<Calendar className="w-5 h-5" />}
          label="DateTime Columns"
          value={`${basic_info?.column_types?.datetime_count || 0}`}
          subtitle="Temporal features"
          color="orange"
        />
      </div>

      {/* Second Row of Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Missing Values - Expandable Card */}
        <MissingValuesCard missingValues={missing_values} />

        {/* Duplicates */}
        <StatCard
          icon={<Copy className="w-5 h-5" />}
          label="Duplicate Rows"
          value={`${duplicates?.total_duplicates?.toLocaleString() || 0}`}
          subtitle={`${duplicates?.duplicate_percentage?.toFixed(2) || 0}% duplicates`}
          color={duplicates?.total_duplicates > 0 ? "yellow" : "green"}
        />
      </div>

      {/* Data Quality Metrics */}
      {data_quality && (
        <div className="bg-card border rounded-xl p-6">
          <h4 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
            Data Quality Assessment
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <QualityMetric
              label="Completeness"
              value={data_quality.completeness}
              unit="%"
            />
            <QualityMetric
              label="Uniqueness"
              value={data_quality.uniqueness}
              unit="%"
            />
            <QualityMetric
              label="Overall Score"
              value={data_quality.quality_score}
              unit="/100"
            />
          </div>
        </div>
      )}

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <DataQualityTabs recommendations={recommendations} edaResults={edaResults} />
      )}
    </div>
  );
};

const MissingValuesCard = ({ missingValues }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  if (!missingValues) return null;
  
  const hasDetails = missingValues?.columns_with_missing && Object.keys(missingValues.columns_with_missing).length > 0;
  const totalMissing = missingValues?.total_missing || 0;
  const columnsWithMissing = missingValues?.columns_missing_count || 0;
  const missingPercentage = missingValues?.missing_percentage || 0;
  
  const color = totalMissing > 0 ? "yellow" : "green";
  const colorClasses = {
    yellow: 'bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/20',
    green: 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20',
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-card border rounded-xl overflow-hidden"
    >
      {/* Header - Clickable */}
      <div 
        className={cn(
          "p-4 transition-colors",
          hasDetails && totalMissing > 0 ? "cursor-pointer hover:bg-muted/50" : ""
        )}
        onClick={() => hasDetails && totalMissing > 0 && setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={cn('p-2 rounded-lg border', colorClasses[color])}>
              <AlertCircle className="w-5 h-5" />
            </div>
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Missing Values
            </span>
          </div>
          {hasDetails && totalMissing > 0 && (
            <motion.div
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <ChevronDown className="w-5 h-5 text-muted-foreground" />
            </motion.div>
          )}
        </div>
        
        <div className="flex items-baseline gap-3">
          <div className="text-2xl font-bold">{totalMissing.toLocaleString()}</div>
          {columnsWithMissing > 0 && (
            <div className="text-sm text-muted-foreground">
              in {columnsWithMissing} column{columnsWithMissing > 1 ? 's' : ''}
            </div>
          )}
        </div>
        <div className="text-xs text-muted-foreground mt-1">
          {missingPercentage.toFixed(2)}% of all cells
        </div>
      </div>
      
      {/* Expandable Details */}
      <AnimatePresence>
        {isExpanded && hasDetails && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="border-t overflow-hidden"
          >
            <div className="p-4 bg-muted/30 space-y-2 max-h-64 overflow-y-auto">
              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                Columns with Missing Values
              </div>
              {Object.entries(missingValues.columns_with_missing)
                .sort((a, b) => b[1].percentage - a[1].percentage)
                .map(([col, data]) => (
                  <div
                    key={col}
                    className="flex items-center justify-between p-3 bg-background rounded-lg border"
                  >
                    <span className="text-sm font-medium truncate flex-1 mr-4">{col}</span>
                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-xs text-muted-foreground">{data.count.toLocaleString()} missing</span>
                      <span className={cn(
                        "text-xs font-mono font-semibold px-2 py-1 rounded",
                        data.percentage >= 50 ? "bg-red-100 dark:bg-red-950 text-red-700 dark:text-red-300" :
                        data.percentage >= 20 ? "bg-yellow-100 dark:bg-yellow-950 text-yellow-700 dark:text-yellow-300" :
                        "bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300"
                      )}>
                        {data.percentage.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
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
        <div className={cn('p-2 rounded-lg border', colorClasses[color])}>
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

const QualityMetric = ({ label, value, unit }) => {
  const getColor = (val) => {
    if (val >= 90) return 'text-green-600 dark:text-green-400';
    if (val >= 70) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className={cn('text-lg font-bold', getColor(value))}>
        {value.toFixed(1)}{unit}
      </span>
    </div>
  );
};

const QualityBadge = ({ score, assessment }) => {
  const badgeConfig = {
    excellent: { color: 'bg-green-500/20 text-green-700 dark:text-green-300 border-green-500/30', icon: CheckCircle },
    good: { color: 'bg-blue-500/20 text-blue-700 dark:text-blue-300 border-blue-500/30', icon: CheckCircle },
    fair: { color: 'bg-yellow-500/20 text-yellow-700 dark:text-yellow-300 border-yellow-500/30', icon: AlertCircle },
    poor: { color: 'bg-red-500/20 text-red-700 dark:text-red-300 border-red-500/30', icon: AlertTriangle },
  };

  const config = badgeConfig[assessment] || badgeConfig.fair;
  const Icon = config.icon;

  return (
    <div className={cn('flex items-center gap-2 px-4 py-2 rounded-lg border font-semibold', config.color)}>
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

const DataQualityTabs = ({ recommendations, edaResults }) => {
  const [activeTab, setActiveTab] = useState('all');
  
  // Parse recommendations into categories
  const rareValuesRecs = recommendations.filter(rec => rec.includes('rare values') || rec.includes('Data Quality:'));
  const multicollinearityRecs = recommendations.filter(rec => rec.includes('multicollinearity') || rec.includes('correlated features'));
  const outlierRecs = recommendations.filter(rec => rec.includes('outlier'));
  const otherRecs = recommendations.filter(rec => 
    !rec.includes('rare values') && 
    !rec.includes('Data Quality:') && 
    !rec.includes('multicollinearity') && 
    !rec.includes('correlated features') && 
    !rec.includes('outlier')
  );
  
  // Get outlier count from EDA results
  const outlierCount = edaResults?.outliers?.total_outliers || 0;
  const outlierColumns = edaResults?.outliers?.columns_with_outliers || 0;
  
  const tabs = [
    { id: 'all', label: 'All Issues', count: recommendations.length, icon: Info },
    { id: 'rare', label: 'Rare Values', count: rareValuesRecs.length, icon: AlertTriangle },
    { id: 'correlation', label: 'Multicollinearity', count: multicollinearityRecs.length, icon: TrendingUp },
    { id: 'outliers', label: 'Outliers', count: outlierColumns, icon: AlertCircle },
  ];
  
  const renderContent = () => {
    let content = [];
    
    if (activeTab === 'all') {
      content = recommendations;
    } else if (activeTab === 'rare') {
      content = rareValuesRecs;
    } else if (activeTab === 'correlation') {
      content = multicollinearityRecs;
    } else if (activeTab === 'outliers') {
      // Show outlier summary from EDA results
      if (outlierCount > 0 && edaResults?.outliers?.outliers_by_column) {
        const outlierDetails = Object.entries(edaResults.outliers.outliers_by_column)
          .filter(([_, data]) => data.count > 0)
          .map(([col, data]) => {
            const boundsInfo = (data.lower_bound !== undefined && data.upper_bound !== undefined)
              ? ` [IQR bounds: ${data.lower_bound.toFixed(2)} - ${data.upper_bound.toFixed(2)}]`
              : '';
            return `⚠️ ${col}: ${data.count.toLocaleString()} outliers (${data.percentage.toFixed(1)}% of column)${boundsInfo}`;
          });
        content = outlierDetails.length > 0 ? outlierDetails : outlierRecs;
      } else {
        content = outlierRecs;
      }
    }
    
    if (content.length === 0) {
      return (
        <div className="text-center py-8 text-muted-foreground">
          <CheckCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>No issues found in this category</p>
        </div>
      );
    }
    
    return (
      <ul className="space-y-3">
        {content.map((rec, index) => (
          <motion.li
            key={index}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="flex items-start gap-3 text-sm bg-white dark:bg-gray-900/50 p-3 rounded-lg border"
          >
            <span className="text-muted-foreground mt-0.5">
              {rec.includes('⚠️') ? <AlertTriangle className="w-4 h-4" /> : '•'}
            </span>
            <span className="flex-1">{rec.replace('⚠️ ', '')}</span>
          </motion.li>
        ))}
      </ul>
    );
  };
  
  return (
    <div className="bg-card border rounded-xl overflow-hidden">
      {/* Header */}
      <div className="bg-blue-50 dark:bg-blue-950/30 border-b border-blue-200 dark:border-blue-800/50 p-4">
        <h4 className="text-lg font-semibold flex items-center gap-2 text-blue-900 dark:text-blue-100">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          Data Quality Issues & Recommendations
        </h4>
        <p className="text-xs text-blue-700 dark:text-blue-300 mt-1">
          Review potential data quality issues detected during analysis
        </p>
      </div>
      
      {/* Tabs */}
      <div className="flex items-center gap-2 p-2 bg-muted/30 border-b overflow-x-auto">
        {tabs.map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap',
                isActive
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'hover:bg-muted text-muted-foreground'
              )}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
              {tab.count > 0 && (
                <span className={cn(
                  'px-2 py-0.5 rounded-full text-xs font-bold',
                  isActive
                    ? 'bg-primary-foreground/20 text-primary-foreground'
                    : 'bg-primary/10 text-primary'
                )}>
                  {tab.count}
                </span>
              )}
            </button>
          );
        })}
      </div>
      
      {/* Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export default EDASummary;
