/**
 * EDA Progress Log Component - Vercel-style visual log table with chart icons
 */
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, Loader2, Database, AlertCircle, Copy, TrendingUp, Activity, BarChart3, PieChart, Award } from 'lucide-react';
import { cn } from '../../lib/utils';

// Map steps to their corresponding chart icons
const stepChartIcons = {
  'Loading dataset': Database,
  'Analyzing dataset structure': BarChart3,
  'Checking missing values': AlertCircle,
  'Detecting duplicate rows': Copy,
  'Analyzing distributions': TrendingUp,
  'Detecting outliers': Activity,
  'Computing correlations': Activity,
  'Analyzing categorical features': PieChart,
  'Assessing data quality': Award,
  'Generating recommendations': Award,
};

const EDAProgressLog = ({ steps, className }) => {
  return (
    <div className={cn("w-full space-y-2", className)}>
      <AnimatePresence mode="popLayout">
        {steps.map((step, index) => (
          <motion.div
            key={step.step}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, delay: index * 0.05 }}
            className="flex items-center justify-between gap-4 py-2 px-4 rounded-lg border bg-card backdrop-blur-sm hover:bg-muted/50 transition-colors"
          >
            {/* Left: Step name with chart icon */}
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <StepIcon status={step.status} />
              <ChartIcon stepName={step.step} status={step.status} />
              <span className="text-sm truncate font-medium">
                {step.step}
              </span>
            </div>

            {/* Right: Status and time */}
            <div className="flex items-center gap-3 shrink-0">
              {step.status === 'running' && (
                <span className="text-xs text-muted-foreground font-mono">
                  {step.elapsed || '0s'}
                </span>
              )}
              {step.status === 'completed' && step.elapsed && (
                <span className="text-xs text-muted-foreground font-mono">
                  {step.elapsed}
                </span>
              )}
              <StatusBadge status={step.status} />
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};


const ChartIcon = ({ stepName, status }) => {
  const IconComponent = stepChartIcons[stepName];
  
  if (!IconComponent) {
    return null;
  }

  const iconColorClass = 
    status === 'completed' 
      ? 'text-green-400' 
      : status === 'running' 
      ? 'text-blue-400' 
      : status === 'error'
      ? 'text-red-400'
      : 'text-muted-foreground/50';

  return (
    <div className={cn(
      "w-5 h-5 flex items-center justify-center shrink-0 transition-colors",
      iconColorClass
    )}>
      <IconComponent className="w-4 h-4" strokeWidth={2} />
    </div>
  );
};

const StepIcon = ({ status }) => {
  if (status === 'completed') {
    return (
      <div className="w-5 h-5 rounded-full bg-green-500/20 flex items-center justify-center shrink-0">
        <Check className="w-3 h-3 text-green-400" strokeWidth={3} />
      </div>
    );
  }
  
  if (status === 'running') {
    return (
      <div className="w-5 h-5 rounded-full bg-blue-500/20 flex items-center justify-center shrink-0">
        <Loader2 className="w-3 h-3 text-blue-400 animate-spin" strokeWidth={3} />
      </div>
    );
  }
  
  if (status === 'error') {
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

const StatusBadge = ({ status }) => {
  const statusConfig = {
    completed: {
      icon: <Check className="w-3 h-3" strokeWidth={3} />,
      className: 'bg-green-500/10 text-green-400 border-green-500/20',
      label: '✓',
    },
    running: {
      icon: <Loader2 className="w-3 h-3 animate-spin" strokeWidth={3} />,
      className: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
      label: '...',
    },
    error: {
      icon: (
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      ),
      className: 'bg-red-500/10 text-red-400 border-red-500/20',
      label: '✕',
    },
    pending: {
      icon: null,
      className: 'bg-muted/50 text-muted-foreground border-border',
      label: '○',
    },
  };

  const config = statusConfig[status] || statusConfig.pending;

  return (
    <div
      className={cn(
        'flex items-center justify-center w-6 h-6 rounded-md border font-mono text-xs',
        config.className
      )}
    >
      {config.icon || config.label}
    </div>
  );
};

export default EDAProgressLog;
