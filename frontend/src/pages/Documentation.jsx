/**
 * Documentation — Technical reference for the AutoML Framework.
 */
import React, { useState } from 'react';
import { motion } from 'framer-motion';

// --- Visual Components ---

const PipelineDiagram = () => {
  const stages = [
    { label: 'Upload', sub: 'CSV ingestion' },
    { label: 'EDA', sub: 'Statistical analysis' },
    { label: 'Preprocess', sub: 'Clean & transform' },
    { label: 'Train', sub: 'Optuna HPO' },
    { label: 'Select', sub: 'Gen. scoring' },
    { label: 'Results', sub: 'Metrics & plots' },
  ];
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 overflow-x-auto">
      <div className="flex items-center gap-1 min-w-[540px] py-1">
        {stages.map((stage, i) => (
          <React.Fragment key={stage.label}>
            <div className="flex-1 border border-border rounded-lg px-3 py-3 text-center bg-muted/30">
              <div className="text-sm font-medium text-foreground">{stage.label}</div>
              <div className="text-[11px] text-muted-foreground mt-0.5">{stage.sub}</div>
            </div>
            {i < stages.length - 1 && (
              <svg width="16" height="16" viewBox="0 0 16 16" className="shrink-0 text-muted-foreground/50">
                <path d="M5 3L11 8L5 13" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
          </React.Fragment>
        ))}
      </div>
    </motion.div>
  );
};

const StrategyMatrix = () => {
  const families = [
    { name: 'Linear', models: 'Ridge, Lasso, Logistic Reg.', scaling: 'Standard', encoding: 'OneHot', outliers: 'IQR cap' },
    { name: 'Tree-based', models: 'RF, XGB, LGBM, GBM', scaling: 'None', encoding: 'Ordinal', outliers: 'None' },
    { name: 'Distance', models: 'KNN, SVM, K-Means', scaling: 'Robust', encoding: 'OneHot', outliers: 'IQR cap' },
    { name: 'Neural Net', models: 'MLP', scaling: 'MinMax', encoding: 'OneHot', outliers: 'IQR cap' },
  ];
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6">
      <div className="border border-border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-muted/40 border-b border-border">
              <th className="text-left px-3 py-2 font-medium text-muted-foreground text-xs uppercase tracking-wide">Family</th>
              <th className="text-left px-3 py-2 font-medium text-muted-foreground text-xs uppercase tracking-wide">Scaling</th>
              <th className="text-left px-3 py-2 font-medium text-muted-foreground text-xs uppercase tracking-wide">Encoding</th>
              <th className="text-left px-3 py-2 font-medium text-muted-foreground text-xs uppercase tracking-wide">Outliers</th>
            </tr>
          </thead>
          <tbody>
            {families.map((f, i) => (
              <tr key={f.name} className={i < families.length - 1 ? 'border-b border-border' : ''}>
                <td className="px-3 py-2.5">
                  <div className="font-medium text-foreground text-sm">{f.name}</div>
                  <div className="text-[11px] text-muted-foreground">{f.models}</div>
                </td>
                <td className="px-3 py-2.5"><code className="text-xs bg-muted px-1.5 py-0.5 rounded">{f.scaling}</code></td>
                <td className="px-3 py-2.5"><code className="text-xs bg-muted px-1.5 py-0.5 rounded">{f.encoding}</code></td>
                <td className="px-3 py-2.5"><code className={`text-xs px-1.5 py-0.5 rounded ${f.outliers === 'None' ? 'bg-muted text-muted-foreground' : 'bg-muted'}`}>{f.outliers}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
};

const GeneralizationVisual = () => {
  const models = [
    { name: 'Model A', gen: 0.71, status: '' },
    { name: 'Model B', gen: 0.80, status: 'selected' },
    { name: 'Model C', gen: null, status: 'rejected' },
  ];
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-3">
      <div className="border border-border rounded-lg p-4 bg-muted/20">
        <div className="font-mono text-[13px] leading-relaxed text-muted-foreground">
          <div>overfit_gap = |train_score − cv_score|</div>
          <div>penalty = overfit_gap × penalty_factor</div>
          <div className="text-foreground font-medium">gen_score = cv_score − penalty</div>
        </div>
      </div>
      <div className="border border-border rounded-lg p-4 space-y-2.5">
        <div className="text-xs text-muted-foreground uppercase tracking-wide font-medium mb-3">Example</div>
        {models.map((m) => (
          <div key={m.name} className="flex items-center gap-3">
            <span className="w-14 text-xs font-mono text-muted-foreground shrink-0">{m.name}</span>
            <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
              {m.gen !== null && (
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${m.gen * 100}%` }}
                  transition={{ duration: 0.6, delay: 0.15 }}
                  className={`h-full rounded-full ${m.status === 'selected' ? 'bg-foreground/70' : 'bg-foreground/30'}`}
                />
              )}
            </div>
            <span className={`text-xs w-14 text-right font-mono shrink-0 ${m.status === 'rejected' ? 'text-destructive/60' : m.status === 'selected' ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>
              {m.gen !== null ? m.gen.toFixed(2) : 'rejected'}
            </span>
          </div>
        ))}
      </div>
    </motion.div>
  );
};

const ArchitectureTree = () => {
  const TreeNode = ({ name, desc, children = [], isLast = false, depth = 0 }) => (
    <div>
      <div className="flex items-start gap-2">
        <div className="mt-1 shrink-0">
          {children.length > 0 ? (
            <svg width="16" height="16" viewBox="0 0 16 16" className="text-muted-foreground/50">
              <path d="M6 3V13M6 3H2M6 3H10" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" className="text-muted-foreground/30">
              <circle cx="8" cy="8" r="2" fill="currentColor" />
            </svg>
          )}
        </div>
        <div className="flex-1">
          <div className="font-mono text-sm text-foreground">{name}</div>
          {desc && <div className="text-xs text-muted-foreground">{desc}</div>}
        </div>
      </div>
      {children.length > 0 && (
        <div className="ml-6 mt-1 border-l border-border/40 pl-3 space-y-2">
          {children.map((child, i) => (
            <TreeNode key={child.name} {...child} isLast={i === children.length - 1} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 border border-border rounded-lg p-4 bg-muted/10 font-mono text-sm">
      <TreeNode
        name="app/"
        children={[
          {
            name: 'routes/',
            children: [
              { name: 'datasets.py', desc: 'upload, EDA, preprocessing endpoints' },
              { name: 'automl.py', desc: 'training and model management' },
              { name: 'models.py', desc: 'retrieval and deletion' },
            ],
          },
          {
            name: 'services/',
            children: [
              { name: 'automl_engine.py', desc: 'Optuna HPO, scoring, stability' },
              { name: 'adaptive_preprocessing.py', desc: 'context-aware pipelines' },
              { name: 'eda_service.py', desc: 'analysis and typo detection' },
              { name: 'automl_plots.py', desc: 'visualization generation' },
            ],
          },
          { name: 'storage.py', desc: 'JSON file storage engine' },
          { name: 'config.py', desc: 'application settings' },
          { name: 'models/', desc: 'schemas.py, etc.' },
          { name: 'utils/', desc: 'utility functions' },
        ]}
      />
    </motion.div>
  );
};

const PreprocessingPipeline = () => {
  const steps = [
    { num: 1, title: 'Load dataset', desc: 'Reads CSV with encoding fallback chain (UTF-8, Latin-1, ISO-8859-1, CP1252, UTF-16). Falls back to chardet detection.' },
    { num: 2, title: 'Remove duplicates', desc: 'Drops exact duplicate rows.' },
    { num: 3, title: 'Remove constant columns', desc: 'Drops columns with one or fewer unique values.' },
    { num: 4, title: 'Flag high-missing columns', desc: 'Columns with >50% missing values are flagged for removal.' },
    { num: 5, title: 'Detect ID columns', desc: 'Pattern matching (col == "id", ends with "_id", starts with "id_"). Sequential integer columns with >95% uniqueness also detected.' },
    { num: 6, title: 'Clip outliers', desc: 'Per numerical column, values outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR] are clipped to the bounds.' },
    { num: 7, title: 'Impute missing values', desc: 'Numerical columns use median; categorical columns use mode.' },
    { num: 8, title: 'Encode categoricals', desc: 'Target column: label encoding. Features with <10 unique values: one-hot encoding (drop_first=True). Features with ≥10 unique values: label encoding.' },
    { num: 9, title: 'Scale numerical features', desc: 'StandardScaler (mean=0, std=1) applied to original numerical features only. One-hot columns and the target column are excluded.' },
    { num: 10, title: 'Convert types', desc: 'Numpy types converted to native Python types for JSON serialization.' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-3">
      {steps.map((step, idx) => (
        <motion.div
          key={step.num}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: idx * 0.05 }}
          className="relative"
        >
          <div className="flex gap-4">
            {/* Step number circle */}
            <div className="shrink-0 relative pt-1">
              <div className="w-8 h-8 rounded-full bg-muted/40 border border-border flex items-center justify-center">
                <span className="text-xs font-bold text-foreground">{step.num}</span>
              </div>
              {idx < steps.length - 1 && (
                <div className="absolute top-8 left-3.5 w-0.5 h-8 bg-border/40" />
              )}
            </div>
            {/* Step content */}
            <div className="flex-1 pb-2">
              <div className="text-sm font-medium text-foreground">{step.title}</div>
              <div className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{step.desc}</div>
            </div>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

const AdaptiveStrategyRules = () => {
  const strategies = {
    'Scaling': [
      { condition: 'Tree-based models', strategy: 'None' },
      { condition: 'Neural networks', strategy: 'MinMaxScaler' },
      { condition: 'Distance-based + outliers', strategy: 'RobustScaler' },
      { condition: 'All other models', strategy: 'StandardScaler' },
    ],
    'Encoding': [
      { condition: 'Tree-based models', strategy: 'OrdinalEncoder' },
      { condition: 'High-cardinality (>50 unique)', strategy: 'OrdinalEncoder' },
      { condition: 'All other cases', strategy: 'OneHotEncoder' },
    ],
    'Imputation': [
      { condition: 'High missing (>30%) + outliers', strategy: 'Median' },
      { condition: 'High missing (>30%) no outliers', strategy: 'Mean' },
      { condition: 'Low missing + outliers', strategy: 'Median' },
      { condition: 'Low missing no outliers', strategy: 'Mean' },
      { condition: 'Categorical columns', strategy: 'Most frequent' },
    ],
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-5">
      {Object.entries(strategies).map(([strategy, rules]) => (
        <div key={strategy} className="border border-border rounded-lg p-4 bg-muted/5">
          <div className="text-sm font-bold text-foreground mb-3 uppercase tracking-wide">{strategy}</div>
          <div className="grid grid-cols-2 gap-2">
            {rules.map((rule, idx) => (
              <motion.div
                key={`${strategy}-${idx}`}
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: idx * 0.04 }}
                className="border border-border rounded px-3 py-2.5 flex flex-col items-start justify-between"
              >
                <span className="text-xs text-muted-foreground mb-2">{rule.condition}</span>
                <code className="text-xs font-mono bg-background/40 text-foreground px-1.5 py-0.5 rounded">
                  {rule.strategy}
                </code>
              </motion.div>
            ))}
          </div>
        </div>
      ))}
    </motion.div>
  );
};

const PenaltyTiers = () => {
  const tiers = [
    { gap: 'Gap > 0.20', penalty: 'Rejected', reason: 'generalization_score = −1.0', status: 'rejected' },
    { gap: 'Gap 0.10 − 0.20', penalty: '3.0', reason: 'High penalty', status: 'warning' },
    { gap: 'Gap < 0.10', penalty: '2.0', reason: 'Low penalty', status: 'ok' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2">
      {tiers.map((tier, idx) => (
        <motion.div
          key={idx}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.06 }}
          className="border border-border rounded-lg px-4 py-3 flex items-center justify-between bg-muted/5"
        >
          <div className="flex-1">
            <div className="text-xs font-mono text-foreground font-medium">{tier.gap}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{tier.reason}</div>
          </div>
          <code className={`text-xs font-mono px-2 py-1 rounded border border-border/50 shrink-0 ml-4 ${
            tier.status === 'rejected' ? 'text-destructive/80 bg-destructive/5' :
            tier.status === 'warning' ? 'text-amber-700 dark:text-amber-300 bg-amber-500/5' :
            'text-green-700 dark:text-green-300 bg-green-500/5'
          }`}>
            {tier.penalty}
          </code>
        </motion.div>
      ))}
      <div className="text-xs text-muted-foreground italic border-t border-border/40 mt-3 pt-3">
        The model with the highest generalization score among non-rejected models is selected.
      </div>
    </motion.div>
  );
};

const StabilityDetection = () => {
  const checks = [
    { num: 1, title: 'Convergence', desc: 'Checks if iterative models hit max_iter without converging.' },
    { num: 2, title: 'NaN coefficients', desc: 'Checks linear model coefficients for NaN values.' },
    { num: 3, title: 'Large coefficients', desc: 'Flags coefficients exceeding 1e6.' },
    { num: 4, title: 'NaN/Inf feature importances', desc: 'Checks tree model feature importances.' },
    { num: 5, title: 'NaN/Inf predictions', desc: 'Validates model output on test data.' },
    { num: 6, title: 'Single-class prediction', desc: 'Flags classifiers that predict only one class.' },
    { num: 7, title: 'Zero-variance predictions', desc: 'Flags regressors with prediction std < 1e-10.' },
    { num: 8, title: 'Extreme overfitting', desc: 'Flags models with train/cv gap > 0.5.' },
    { num: 9, title: 'Invalid CV score', desc: 'Flags NaN or Inf cross-validation scores.' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-3">
      {checks.map((check, idx) => (
        <motion.div
          key={check.num}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: idx * 0.05 }}
          className="relative"
        >
          <div className="flex gap-4">
            <div className="shrink-0 relative pt-1">
              <div className="w-8 h-8 rounded-full bg-muted/40 border border-border flex items-center justify-center">
                <span className="text-xs font-bold text-foreground">{check.num}</span>
              </div>
              {idx < checks.length - 1 && (
                <div className="absolute top-8 left-3.5 w-0.5 h-8 bg-border/40" />
              )}
            </div>
            <div className="flex-1 pb-2">
              <div className="text-sm font-medium text-foreground">{check.title}</div>
              <div className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{check.desc}</div>
            </div>
          </div>
        </motion.div>
      ))}
      <div className="text-xs text-muted-foreground italic border-t border-border/40 mt-3 pt-3">
        During Optuna trials, unstable models return a score of −9999, steering the search away from unstable hyperparameter regions. An unstable final model is not automatically rejected but is flagged in the results.
      </div>
    </motion.div>
  );
};

const ModelComparison = () => {
  const models = [
    { name: 'A', train: 0.95, cv: 0.87, gap: 0.08, penalty: 0.16, gen: 0.71, status: null },
    { name: 'B', train: 0.89, cv: 0.86, gap: 0.03, penalty: 0.06, gen: 0.80, status: 'selected' },
    { name: 'C', train: 0.98, cv: 0.75, gap: 0.23, penalty: null, gen: null, status: 'rejected' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2.5">
      {models.map((model, idx) => (
        <motion.div
          key={model.name}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.08 }}
          className={`border rounded-lg p-3.5 ${
            model.status === 'selected' ? 'border-foreground/60 bg-foreground/5' :
            model.status === 'rejected' ? 'border-destructive/40 bg-destructive/5' :
            'border-border bg-muted/5'
          }`}
        >
          <div className="flex items-start justify-between mb-2.5">
            <div className="flex items-center gap-2">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                model.status === 'selected' ? 'bg-foreground text-background' :
                model.status === 'rejected' ? 'bg-destructive/80 text-background' :
                'bg-muted-foreground/40 text-foreground'
              }`}>
                {model.name}
              </div>
              <span className="text-sm font-semibold text-foreground">Model {model.name}</span>
            </div>
            {model.status && (
              <span className={`text-xs font-mono px-2 py-1 rounded ${
                model.status === 'selected' ? 'bg-foreground/10 text-foreground font-medium' :
                'bg-destructive/10 text-destructive/80'
              }`}>
                {model.status}
              </span>
            )}
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="border-t border-border/40 pt-2">
              <div className="text-muted-foreground text-[10px] uppercase tracking-wide mb-0.5">Train</div>
              <div className="font-mono text-foreground font-medium">{model.train.toFixed(2)}</div>
            </div>
            <div className="border-t border-border/40 pt-2">
              <div className="text-muted-foreground text-[10px] uppercase tracking-wide mb-0.5">CV</div>
              <div className="font-mono text-foreground font-medium">{model.cv.toFixed(2)}</div>
            </div>
            <div className="border-t border-border/40 pt-2">
              <div className="text-muted-foreground text-[10px] uppercase tracking-wide mb-0.5">Gap</div>
              <div className="font-mono text-foreground font-medium">{model.gap.toFixed(2)}</div>
            </div>
            {model.status !== 'rejected' && (
              <>
                <div className="border-t border-border/40 pt-2">
                  <div className="text-muted-foreground text-[10px] uppercase tracking-wide mb-0.5">Penalty</div>
                  <div className="font-mono text-foreground font-medium">{model.penalty.toFixed(2)}</div>
                </div>
                <div className="border-t border-border/40 pt-2">
                  <div className="text-muted-foreground text-[10px] uppercase tracking-wide mb-0.5">Gen Score</div>
                  <div className={`font-mono font-medium ${
                    model.status === 'selected' ? 'text-foreground' : 'text-muted-foreground'
                  }`}>{model.gen.toFixed(2)}</div>
                </div>
              </>
            )}
            {model.status === 'rejected' && (
              <div className="col-span-2 border-t border-border/40 pt-2">
                <div className="text-muted-foreground text-[10px] uppercase tracking-wide mb-0.5">Status</div>
                <div className="text-destructive/80 font-mono text-sm">Rejected (gap &gt; 0.20)</div>
              </div>
            )}
          </div>
        </motion.div>
      ))}
      <div className="text-xs text-muted-foreground italic border-t border-border/40 mt-4 pt-4">
        Model B is selected despite lower raw scores because it generalizes better.
      </div>
    </motion.div>
  );
};

const EnvironmentVariables = () => {
  const vars = [
    { name: 'DATA_DIR', desc: 'directory for JSON storage files', default: 'data' },
    { name: 'UPLOAD_DIR', desc: 'directory for uploaded CSV files', default: 'uploads' },
    { name: 'MODELS_DIR', desc: 'directory for saved model artifacts', default: 'trained_models' },
    { name: 'MAX_UPLOAD_SIZE', desc: 'maximum file upload size in bytes', default: '52428800 (50 MB)' },
    { name: 'HOST', desc: 'server bind address', default: '0.0.0.0' },
    { name: 'PORT', desc: 'server port', default: '8000' },
    { name: 'CORS_ORIGINS', desc: 'comma-separated allowed origins', default: 'http://localhost:3000,...' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2">
      {vars.map((v, idx) => (
        <motion.div
          key={v.name}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.04 }}
          className="border border-border rounded-lg p-3 flex items-start justify-between bg-muted/5"
        >
          <div className="flex-1">
            <div className="text-xs font-mono text-foreground font-bold">{v.name}</div>
            <div className="text-xs text-muted-foreground mt-1">{v.desc}</div>
          </div>
          <code className="text-xs font-mono bg-background/40 text-muted-foreground px-2 py-1 rounded border border-border/50 shrink-0 ml-3 whitespace-nowrap">
            {v.default}
          </code>
        </motion.div>
      ))}
    </motion.div>
  );
};

const TrainingDefaults = () => {
  const defaults = [
    { key: 'n_trials', value: '75' },
    { key: 'cv_folds', value: '5' },
    { key: 'test_size', value: '0.2' },
    { key: 'penalty_factor', value: '2.0' },
    { key: 'overfit_threshold_reject', value: '0.20' },
    { key: 'overfit_threshold_high', value: '0.10' },
    { key: 'max_gpu_usage', value: '0.8' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6">
      <div className="grid grid-cols-2 gap-2">
        {defaults.map((item, idx) => (
          <motion.div
            key={item.key}
            initial={{ opacity: 0, x: -4 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.2, delay: idx * 0.04 }}
            className="border border-border rounded-lg p-3 bg-muted/5"
          >
            <div className="text-xs font-mono text-muted-foreground mb-1">{item.key}</div>
            <div className="text-sm font-mono text-foreground font-bold">{item.value}</div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};

const PreprocessingDefaults = () => {
  const defaults = [
    { key: 'OUTLIER_IQR_MULTIPLIER', value: '1.5', desc: 'clipping bounds multiplier' },
    { key: 'HIGH_MISSING_THRESHOLD', value: '0.5', desc: '>50%  missing → remove column' },
    { key: 'LOW_CARDINALITY_THRESHOLD', value: '10', desc: '<10 unique → one-hot encoding' },
    { key: 'TYPO_SIMILARITY_THRESHOLD', value: '0.80', desc: 'similarity cutoff for typo detection' },
    { key: 'TYPO_MAX_UNIQUE_VALUES', value: '100', desc: 'columns with more skip typo check' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2">
      {defaults.map((item, idx) => (
        <motion.div
          key={item.key}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.04 }}
          className="border border-border rounded-lg p-3 flex items-start justify-between bg-muted/5"
        >
          <div className="flex-1">
            <div className="text-xs font-mono text-foreground font-bold">{item.key}</div>
            <div className="text-xs text-muted-foreground mt-1">{item.desc}</div>
          </div>
          <code className="text-xs font-mono bg-background/40 text-muted-foreground px-2 py-1 rounded border border-border/50 shrink-0 ml-3">
            {item.value}
          </code>
        </motion.div>
      ))}
    </motion.div>
  );
};

const APIEndpoints = () => {
  const endpoints = [
    { method: 'POST', path: '/api/datasets/upload', desc: 'upload a CSV' },
    { method: 'GET', path: '/api/datasets/', desc: 'list datasets' },
    { method: 'GET', path: '/api/datasets/{id}/eda', desc: 'run EDA' },
    { method: 'POST', path: '/api/datasets/{id}/preprocess', desc: 'preprocess a dataset' },
    { method: 'POST', path: '/api/automl/start/{id}', desc: 'start AutoML training' },
    { method: 'GET', path: '/api/automl/status/{id}', desc: 'check training status' },
    { method: 'GET', path: '/api/models/', desc: 'list trained models' },
    { method: 'DELETE', path: '/api/models/{id}', desc: 'delete a model' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2">
      {endpoints.map((endpoint, idx) => (
        <motion.div
          key={`${endpoint.method}${endpoint.path}`}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.04 }}
          className="border border-border rounded-lg p-3 flex items-start justify-between bg-muted/5"
        >
          <div className="flex-1 flex items-center gap-3">
            <span className={`text-xs font-mono font-bold px-1.5 py-0.5 rounded ${
              endpoint.method === 'GET' ? 'bg-blue-500/10 text-blue-700 dark:text-blue-300' :
              endpoint.method === 'POST' ? 'bg-green-500/10 text-green-700 dark:text-green-300' :
              endpoint.method === 'DELETE' ? 'bg-red-500/10 text-red-700 dark:text-red-300' :
              'bg-amber-500/10 text-amber-700 dark:text-amber-300'
            }`}>
              {endpoint.method}
            </span>
            <div className="flex-1">
              <div className="text-xs font-mono text-foreground">{endpoint.path}</div>
              <div className="text-xs text-muted-foreground mt-0.5">{endpoint.desc}</div>
            </div>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

const HyperparameterConfig = () => {
  const config = [
    { key: 'Trials', value: '75', range: '(10-200)', desc: 'per model' },
    { key: 'CV Folds', value: '5', range: 'stratified', desc: 'for classification' },
    { key: 'Train/Test', value: '80/20', range: 'split', desc: 'before preprocessing' },
    { key: 'Seed', value: '42', range: 'fixed', desc: 'for reproducibility' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 grid grid-cols-2 gap-3">
      {config.map((item, idx) => (
        <motion.div
          key={item.key}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.06 }}
          className="border border-border rounded-lg p-3 bg-muted/5"
        >
          <div className="text-xs font-mono text-muted-foreground mb-1 flex items-center gap-2">
            {item.key}
            {item.range && <span className="text-[10px] bg-muted/40 px-1.5 py-0.5 rounded">{item.range}</span>}
          </div>
          <div className="text-sm font-mono text-foreground font-bold">{item.value}</div>
          <div className="text-xs text-muted-foreground mt-1.5 text-[11px]">{item.desc}</div>
        </motion.div>
      ))}
    </motion.div>
  );
};

const SearchSpacesExamples = () => {
  const spaces = [
    {
      model: 'Random Forest',
      params: [
        'n_estimators: [50, 500]',
        'max_depth: [3, 30]',
        'min_samples_split: [2, 20]',
        'min_samples_leaf: [1, 10]',
        'max_features: {sqrt, log2, None}',
      ],
    },
    {
      model: 'XGBoost',
      params: [
        'n_estimators: [50, 500]',
        'learning_rate: [0.001, 0.3] (log)',
        'max_depth: [3, 10]',
        'min_child_weight: [1, 10]',
        'subsample: [0.5, 1.0]',
        'colsample_bytree: [0.5, 1.0]',
        'gamma: [0.0, 1.0]',
      ],
    },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-3">
      {spaces.map((space, spaceIdx) => (
        <div key={space.model} className="border border-border rounded-lg p-4 bg-muted/5">
          <div className="text-sm font-bold text-foreground mb-3">{space.model}</div>
          <div className="space-y-1.5">
            {space.params.map((param, paramIdx) => (
              <motion.div
                key={`${space.model}-${paramIdx}`}
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: (spaceIdx * 0.1) + (paramIdx * 0.04) }}
                className="text-xs font-mono text-muted-foreground pl-2 border-l border-border/50"
              >
                {param}
              </motion.div>
            ))}
          </div>
        </div>
      ))}
    </motion.div>
  );
};

const TrainingFlowSteps = () => {
  const steps = [
    { num: 1, title: 'Run optimization', desc: 'For each selected model, run Optuna optimization (N trials).' },
    { num: 2, title: 'Retrain best model', desc: 'Retrain with the best hyperparameters found.' },
    { num: 3, title: 'Compute scores', desc: 'Calculate train, cross-validation, and test scores.' },
    { num: 4, title: 'Generalization gap', desc: 'Calculate overfitting gap and generalization score.' },
    { num: 5, title: 'Stability checks', desc: 'Run 9 stability checks on the final model.' },
    { num: 6, title: 'Select best model', desc: 'Filter rejected models, then select the best by generalization score.' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-3">
      {steps.map((step, idx) => (
        <motion.div
          key={step.num}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: idx * 0.05 }}
          className="relative"
        >
          <div className="flex gap-4">
            <div className="shrink-0 relative pt-1">
              <div className="w-8 h-8 rounded-full bg-muted/40 border border-border flex items-center justify-center">
                <span className="text-xs font-bold text-foreground">{step.num}</span>
              </div>
              {idx < steps.length - 1 && (
                <div className="absolute top-8 left-3.5 w-0.5 h-8 bg-border/40" />
              )}
            </div>
            <div className="flex-1 pb-2">
              <div className="text-sm font-medium text-foreground">{step.title}</div>
              <div className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{step.desc}</div>
            </div>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

const GPUResourceConfig = () => {
  const info = [
    { label: 'GPU Memory Allocation', value: 'configurable', detail: 'max_gpu_usage (default 0.8)' },
    { label: 'XGBoost & LGBM', value: 'auto-scaled', detail: 'histogram bin count scales with GPU' },
    { label: 'Memory Monitoring', value: 'psutil', detail: 'tracked continuously' },
    { label: 'Garbage Collection', value: 'triggers at 80%', detail: 'usage, automatic cleanup' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2">
      {info.map((item, idx) => (
        <motion.div
          key={item.label}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.06 }}
          className="border border-border rounded-lg px-4 py-3 flex items-center justify-between bg-muted/5"
        >
          <div className="flex-1">
            <div className="text-xs font-mono text-foreground font-medium">{item.label}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{item.detail}</div>
          </div>
          <code className="text-xs font-mono bg-background/40 text-muted-foreground px-2 py-1 rounded border border-border/50 shrink-0 ml-4">
            {item.value}
          </code>
        </motion.div>
      ))}
    </motion.div>
  );
};

const ClassificationMetrics = () => {
  const metrics = [
    { name: 'Accuracy', desc: 'proportion of correct predictions' },
    { name: 'Precision', desc: 'proportion of true positives among predicted positives' },
    { name: 'Recall', desc: 'proportion of true positives among actual positives' },
    { name: 'F1-Score', desc: 'harmonic mean of precision and recall' },
    { name: 'Confusion matrix', desc: 'per-class true/false positive/negative counts' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2">
      {metrics.map((m, idx) => (
        <motion.div
          key={m.name}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.04 }}
          className="border border-border rounded-lg p-3 flex items-start justify-between bg-muted/5"
        >
          <div className="flex-1">
            <div className="text-xs font-mono text-foreground font-bold">{m.name}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{m.desc}</div>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

const RegressionMetrics = () => {
  const metrics = [
    { name: 'R-squared', desc: 'proportion of variance explained (0 to 1)' },
    { name: 'RMSE', desc: 'root mean squared error' },
    { name: 'MAE', desc: 'mean absolute error' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2">
      {metrics.map((m, idx) => (
        <motion.div
          key={m.name}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.04 }}
          className="border border-border rounded-lg p-3 flex items-start justify-between bg-muted/5"
        >
          <div className="flex-1">
            <div className="text-xs font-mono text-foreground font-bold">{m.name}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{m.desc}</div>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

const GeneratedPlots = () => {
  const plots = [
    { type: 'Confusion matrix', desc: 'classification', detail: 'per-class predictions' },
    { type: 'ROC curve', desc: 'binary classification', detail: 'TPR vs FPR' },
    { type: 'Residual plot', desc: 'regression', detail: 'prediction errors' },
    { type: 'Feature importance', desc: 'all models', detail: 'bar chart ranking' },
    { type: 'Actual vs. predicted', desc: 'regression', detail: 'scatter plot' },
    { type: 'Learning curve', desc: 'all models', detail: 'score vs sample size' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 grid grid-cols-2 gap-2">
      {plots.map((plot, idx) => (
        <motion.div
          key={plot.type}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.04 }}
          className="border border-border rounded-lg p-3 bg-muted/5"
        >
          <div className="text-xs font-mono text-foreground font-bold mb-1">{plot.type}</div>
          <div className="text-xs text-muted-foreground">{plot.desc}</div>
          <div className="text-[10px] text-muted-foreground/60 mt-1.5">{plot.detail}</div>
        </motion.div>
      ))}
    </motion.div>
  );
};

const DataQualityScore = () => {
  const deductions = [
    { category: 'Columns removed', max: '-20', example: '5 cols removed → -20' },
    { category: 'Remaining missing cells', max: '-30', example: '50 missing cells → -30' },
    { category: 'Final score', max: '[0, 100]', example: 'clamped to valid range' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-3">
      <div className="border border-border rounded-lg p-4 bg-muted/5">
        <div className="text-sm font-bold text-foreground mb-3">Quality Score Formula</div>
        <div className="font-mono text-xs text-muted-foreground space-y-1.5">
          <div>base_score = 100</div>
          <div>score −= min(20, columns_removed)</div>
          <div>score −= min(30, remaining_missing_cells)</div>
          <div className="text-foreground font-medium pt-1">final_score = clamp(score, 0, 100)</div>
        </div>
      </div>
      <div className="space-y-2">
        {deductions.map((item, idx) => (
          <motion.div
            key={item.category}
            initial={{ opacity: 0, x: -4 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.2, delay: idx * 0.06 }}
            className="border border-border rounded-lg p-3 flex items-start justify-between bg-muted/5"
          >
            <div className="flex-1">
              <div className="text-xs font-mono text-foreground font-medium">{item.category}</div>
              <div className="text-xs text-muted-foreground mt-0.5">{item.example}</div>
            </div>
            <code className="text-xs font-mono bg-background/40 text-muted-foreground px-2 py-1 rounded border border-border/50 shrink-0 ml-3">
              {item.max}
            </code>
          </motion.div>
        ))}
      </div>
      <div className="text-xs text-muted-foreground italic border-t border-border/40 mt-4 pt-4">
        This score is informational and does not affect model training.
      </div>
    </motion.div>
  );
};

const TextColumnDetection = () => {
  const conditions = [
    {
      title: 'Condition 1: Content-Rich',
      criteria: [
        'Average string length > 30',
        'Unique count > 50',
        'Unique ratio > 0.5',
      ],
      logic: 'All conditions required',
    },
    {
      title: 'Condition 2: Long Text with Structure',
      criteria: [
        'Average string length > 50',
        'Whitespace (>50% contain spaces) OR Punctuation (>30% contain . , ! ? ; :)',
      ],
      logic: 'All conditions required',
    },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-3">
      {conditions.map((cond, condIdx) => (
        <div key={cond.title} className="border border-border rounded-lg p-4 bg-muted/5">
          <div className="text-sm font-bold text-foreground mb-3">{cond.title}</div>
          <div className="space-y-2">
            {cond.criteria.map((criterion, idx) => (
              <motion.div
                key={`${cond.title}-${idx}`}
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: (condIdx * 0.1) + (idx * 0.05) }}
                className="flex items-start gap-2 text-xs text-muted-foreground pl-2 border-l border-border/50"
              >
                <span className="mt-0.5 shrink-0">•</span>
                <span>{criterion}</span>
              </motion.div>
            ))}
          </div>
          <div className="text-xs text-muted-foreground/70 italic mt-2.5 pt-2.5 border-t border-border/40">
            {cond.logic}
          </div>
        </div>
      ))}
      <div className="text-xs text-muted-foreground italic pt-2">
        Detected text columns are excluded from categorical encoding and processed separately (OR logic between conditions).
      </div>
    </motion.div>
  );
};

const TFIDFVectorization = () => {
  const params = [
    { key: 'max_features', value: '5000', desc: 'maximum vocabulary size' },
    { key: 'min_df', value: '2', desc: 'document frequency minimum' },
    { key: 'max_df', value: '0.95', desc: 'ignore words in >95% docs' },
    { key: 'ngram_range', value: '(1, 2)', desc: 'unigrams and bigrams' },
    { key: 'stop_words', value: 'english', desc: 'language' },
    { key: 'strip_accents', value: 'unicode', desc: 'accent removal' },
    { key: 'lowercase', value: 'True', desc: 'case normalization' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2">
      {params.map((param, idx) => (
        <motion.div
          key={param.key}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.04 }}
          className="border border-border rounded-lg p-3 flex items-start justify-between bg-muted/5"
        >
          <div className="flex-1">
            <div className="text-xs font-mono text-foreground font-bold">{param.key}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{param.desc}</div>
          </div>
          <code className="text-xs font-mono bg-background/40 text-muted-foreground px-2 py-1 rounded border border-border/50 shrink-0 ml-3">
            {param.value}
          </code>
        </motion.div>
      ))}
      <div className="text-xs text-muted-foreground italic border-t border-border/40 mt-3 pt-3">
        Each text column is vectorized independently. Feature names follow the pattern {'{column_name}_tfidf_{i}'}.
      </div>
    </motion.div>
  );
};

const PipelineIntegrationNotes = () => {
  const notes = [
    {
      label: 'Feature Combination',
      value: 'ColumnTransformer',
      desc: 'text + numerical + categorical features',
    },
    {
      label: 'Sparse Handling',
      value: '0.3',
      desc: 'sparse_threshold (memory efficient)',
    },
    {
      label: 'Compatibility',
      value: 'sklearn',
      desc: 'BaseEstimator & TransformerMixin',
    },
    {
      label: 'Error Handling',
      value: 'warning',
      desc: 'skipped columns logged (no silent failures)',
    },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2">
      {notes.map((note, idx) => (
        <motion.div
          key={note.label}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.06 }}
          className="border border-border rounded-lg px-4 py-3 flex items-center justify-between bg-muted/5"
        >
          <div className="flex-1">
            <div className="text-xs font-mono text-foreground font-medium">{note.label}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{note.desc}</div>
          </div>
          <code className="text-xs font-mono bg-background/40 text-muted-foreground px-2 py-1 rounded border border-border/50 shrink-0 ml-4">
            {note.value}
          </code>
        </motion.div>
      ))}
    </motion.div>
  );
};

const WorkflowSteps = () => {
  const steps = [
    { step: 1, title: 'Upload CSV', desc: 'Import a dataset' },
    { step: 2, title: 'Run EDA', desc: 'Inspect columns, distributions, missing values, outliers' },
    { step: 3, title: 'Preprocess', desc: 'Imputation, encoding, scaling, outlier treatment' },
    { step: 4, title: 'Train Models', desc: 'Optuna-based hyperparameter optimization' },
    { step: 5, title: 'Compare Results', desc: 'Ranked by generalization score' },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-2">
      {steps.map((item, idx) => (
        <motion.div
          key={item.step}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.05 }}
          className="border border-border rounded-lg p-3 bg-muted/5 flex items-start gap-4"
        >
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-background/60 border border-border/50 shrink-0 text-xs font-bold text-foreground/80">
            {item.step}
          </div>
          <div className="flex-1">
            <div className="text-sm font-semibold text-foreground">{item.title}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{item.desc}</div>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

const SupportedTasks = () => {
  const tasks = [
    {
      task: 'Classification',
      count: 9,
      algorithms: 'Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, KNN, Decision Tree, AdaBoost',
    },
    {
      task: 'Regression',
      count: 11,
      algorithms: 'Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVR, KNN, AdaBoost',
    },
    {
      task: 'Clustering',
      count: 3,
      algorithms: 'KMeans, DBSCAN, Agglomerative',
    },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 grid grid-cols-1 gap-3">
      {tasks.map((item, idx) => (
        <motion.div
          key={item.task}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.06 }}
          className="border border-border rounded-lg p-4 bg-muted/5"
        >
          <div className="flex items-baseline gap-2 mb-2">
            <h3 className="text-sm font-bold text-foreground">{item.task}</h3>
            <span className="text-xs font-mono bg-background/40 text-muted-foreground px-2 py-1 rounded border border-border/50">
              {item.count} algorithms
            </span>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">{item.algorithms}</p>
        </motion.div>
      ))}
    </motion.div>
  );
};

const KeyDesignDecisions = () => {
  const decisions = [
    {
      principle: 'Data Splitting',
      desc: 'Train (80%) and test (20%) split before transformations. All preprocessing fit on training set only.',
    },
    {
      principle: 'Ranking Metric',
      desc: 'Models ranked by generalization score, not raw accuracy. Penalizes overfitting.',
    },
    {
      principle: 'Adaptive Strategies',
      desc: 'Preprocessing strategies adapt to target model family when adaptive mode is enabled.',
    },
    {
      principle: 'Storage',
      desc: 'JSON file-based storage. No external database required.',
    },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="my-6 space-y-3">
      {decisions.map((item, idx) => (
        <motion.div
          key={item.principle}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2, delay: idx * 0.06 }}
          className="border border-border rounded-lg p-4 bg-muted/5"
        >
          <div className="text-sm font-bold text-foreground mb-2">{item.principle}</div>
          <div className="text-xs text-muted-foreground leading-relaxed">{item.desc}</div>
        </motion.div>
      ))}
    </motion.div>
  );
};

const Documentation = () => {
  const [activeDoc, setActiveDoc] = useState('overview');

  const docs = [
    {
      id: 'overview',
      title: 'Overview',
      category: 'General',
      content: `# Overview

AutoML Framework is a local, open-source tool for automated machine learning. It provides a web interface for uploading CSV datasets, running exploratory data analysis, preprocessing data, and training models with hyperparameter optimization.

## Workflow

[VISUAL:workflow-steps]

## Supported Tasks

[VISUAL:supported-tasks]

## Key Design Decisions

[VISUAL:key-design-decisions]`
    },
    {
      id: 'architecture',
      title: 'Architecture',
      category: 'General',
      content: `# Architecture

## System Flow

[VISUAL:pipeline-diagram]

## Components

[VISUAL:architecture-tree]

**Frontend (React + Vite)**
- Pages: Dashboard, Datasets, Processed Datasets, AutoML, Models, Documentation
- UI: TailwindCSS, shadcn/ui components, Recharts for visualization
- API client: Axios-based client in utils/api.js

## Storage

All metadata (datasets, models, experiments) is stored as JSON files in the data/ directory. Uploaded CSVs are stored in uploads/. Trained model artifacts (.joblib) are stored in trained_models/.`
    },
    {
      id: 'preprocessing',
      title: 'Data Preprocessing',
      category: 'Pipeline',
      content: `# Data Preprocessing

## Base Pipeline

The default preprocessing pipeline runs the following steps sequentially:

[VISUAL:preprocessing-pipeline]

## Adaptive Preprocessing

When adaptive mode is enabled (use_adaptive=true), preprocessing strategies are selected based on EDA results and the target model family.

### Task Type Detection

- If no target column: unsupervised
- If target is numerical and unique_ratio > 5%: regression
- Otherwise: classification

### Model-Aware Strategy Selection

[VISUAL:strategy-matrix]

[VISUAL:adaptive-strategy-rules]

### Custom Overrides

Users can supply per-column overrides for imputation and encoding strategies, force-drop or force-keep specific columns, adjust the missing value threshold, or disable ID detection.

### Decision Log

Every preprocessing decision is logged with a timestamp, category, description, reason, and impact level (info, warning, or critical). This log is returned in the preprocessing response.`
    },
    {
      id: 'training',
      title: 'Model Training',
      category: 'Pipeline',
      content: `# Model Training

## Hyperparameter Optimization

Training uses Optuna with TPESampler and MedianPruner. Default configuration:

[VISUAL:hyperparameter-config]

Each trial samples hyperparameters from model-specific search spaces, fits the model, and computes a cross-validation score. MedianPruner terminates unpromising trials early.

## Search Spaces (examples)

[VISUAL:search-spaces-examples]

## Training Flow

[VISUAL:training-flow-steps]

## GPU and Resource Configuration

[VISUAL:gpu-resource-config]

Memory is monitored via psutil; garbage collection triggers when usage exceeds 80%.

## Timeout

Training can be limited by a per-model timeout. If an Optuna study exceeds the allotted time, the best result found so far is used.`
    },
    {
      id: 'selection',
      title: 'Model Selection',
      category: 'Pipeline',
      content: `# Model Selection and Stability

## Generalization Scoring

Models are not ranked by raw accuracy. Instead, a generalization score penalizes overfitting:

[VISUAL:generalization-visual]

### Penalty Tiers

[VISUAL:penalty-tiers]

### Example

[VISUAL:model-comparison]

## Stability Detection

Nine checks are performed on each trained model:

[VISUAL:stability-detection]`
    },
    {
      id: 'evaluation',
      title: 'Evaluation',
      category: 'Pipeline',
      content: `# Evaluation

## Metrics

### Classification

[VISUAL:classification-metrics]

### Regression

[VISUAL:regression-metrics]

## Visualizations

The framework generates the following plots for trained models:

[VISUAL:generated-plots]

## Train/Test Protocol

Data is split before preprocessing to prevent leakage. All transformations (scaling, encoding, imputation) are fit on the training set and applied to the test set using transform-only. Classification splits use stratification to preserve class distribution.

## Comparing Models

All models trained in a session are displayed with their train score, CV score, test score, overfit gap, and generalization score. Models rejected for excessive overfitting are marked. The best model is highlighted based on generalization score, not raw test accuracy.

## Data Quality Score

A quality score (0-100) is computed during preprocessing:

[VISUAL:data-quality-score]`
    },
    {
      id: 'text-handling',
      title: 'Text Data Handling',
      category: 'Pipeline',
      content: `# Text Data Handling

## Text Column Detection

A column is classified as text (rather than categorical) if:

[VISUAL:text-column-detection]

## TF-IDF Vectorization

Text columns are vectorized using scikit-learn's TfidfVectorizer with the following configuration:

[VISUAL:tfidf-vectorization]

## Pipeline Integration

Text features are combined with numerical and categorical features via a ColumnTransformer. When text columns are present, sparse_threshold is set to 0.3 to maintain memory efficiency with sparse matrices. The TextVectorizer class implements sklearn's BaseEstimator and TransformerMixin interfaces for pipeline compatibility.

[VISUAL:pipeline-integration-notes]`
    },
    {
      id: 'configuration',
      title: 'Configuration',
      category: 'Reference',
      content: `# Configuration

## Environment Variables

All settings are defined in backend/.env (see .env.example for defaults):

[VISUAL:environment-variables]

## Training Defaults

[VISUAL:training-defaults]

## Preprocessing Defaults

[VISUAL:preprocessing-defaults]

## File Structure

- data/ — JSON metadata for datasets, models, experiments
- uploads/ — raw uploaded CSV files
- trained_models/ — serialized model and preprocessing pipeline files (.joblib)

## API Base URL

http://localhost:8000/api

Key endpoints:

[VISUAL:api-endpoints]`
    }
  ];

  const categories = [...new Set(docs.map(doc => doc.category))];
  const activeDocContent = docs.find(doc => doc.id === activeDoc);

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-1">Documentation</h1>
          <p className="text-muted-foreground text-sm">
            Technical reference for the AutoML Framework
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1">
            <div className="bg-card border rounded-xl p-4 sticky top-4">
              {categories.map((category) => (
                <div key={category} className="mb-4">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 px-2">
                    {category}
                  </h3>
                  <div className="space-y-0.5">
                    {docs
                      .filter(doc => doc.category === category)
                      .map((doc) => (
                        <button
                          key={doc.id}
                          onClick={() => setActiveDoc(doc.id)}
                          className={`w-full text-left px-3 py-2 rounded-lg transition-colors text-sm ${
                            activeDoc === doc.id
                              ? 'bg-primary text-primary-foreground font-medium'
                              : 'hover:bg-muted text-foreground'
                          }`}
                        >
                          {doc.title}
                        </button>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Content Area */}
          <div className="lg:col-span-3">
            <motion.div
              key={activeDoc}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.15 }}
              className="bg-card border rounded-xl p-8"
            >
              <div className="prose prose-slate dark:prose-invert max-w-none">
                <h1 className="text-2xl font-bold mb-6">{activeDocContent?.title}</h1>
                
                <div className="markdown-content">
                  {activeDocContent?.content.split('\n').map((line, index) => {
                    if (line.startsWith('# ')) {
                      return null;
                    }
                    if (line.startsWith('[VISUAL:')) {
                      const key = line.slice(8, -1);
                      const visuals = {
                        'pipeline-diagram': <PipelineDiagram />,
                        'strategy-matrix': <StrategyMatrix />,
                        'generalization-visual': <GeneralizationVisual />,
                        'architecture-tree': <ArchitectureTree />,
                        'preprocessing-pipeline': <PreprocessingPipeline />,
                        'adaptive-strategy-rules': <AdaptiveStrategyRules />,
                        'penalty-tiers': <PenaltyTiers />,
                        'stability-detection': <StabilityDetection />,
                        'model-comparison': <ModelComparison />,
                        'environment-variables': <EnvironmentVariables />,
                        'training-defaults': <TrainingDefaults />,
                        'preprocessing-defaults': <PreprocessingDefaults />,
                        'api-endpoints': <APIEndpoints />,
                        'hyperparameter-config': <HyperparameterConfig />,
                        'search-spaces-examples': <SearchSpacesExamples />,
                        'training-flow-steps': <TrainingFlowSteps />,
                        'gpu-resource-config': <GPUResourceConfig />,
                        'classification-metrics': <ClassificationMetrics />,
                        'regression-metrics': <RegressionMetrics />,
                        'generated-plots': <GeneratedPlots />,
                        'data-quality-score': <DataQualityScore />,
                        'text-column-detection': <TextColumnDetection />,
                        'tfidf-vectorization': <TFIDFVectorization />,
                        'pipeline-integration-notes': <PipelineIntegrationNotes />,
                        'workflow-steps': <WorkflowSteps />,
                        'supported-tasks': <SupportedTasks />,
                        'key-design-decisions': <KeyDesignDecisions />,
                      };
                      return <React.Fragment key={index}>{visuals[key]}</React.Fragment>;
                    }
                    if (line.startsWith('## ')) {
                      return <h2 key={index} className="text-xl font-semibold mt-8 mb-3 pb-1 border-b">{line.slice(3)}</h2>;
                    }
                    if (line.startsWith('### ')) {
                      return <h3 key={index} className="text-lg font-semibold mt-5 mb-2">{line.slice(4)}</h3>;
                    }
                    if (line.startsWith('- ')) {
                      const text = line.slice(2);
                      if (text.includes('**')) {
                        const parts = text.split('**');
                        return (
                          <li key={index} className="ml-4 mb-1.5 leading-relaxed">
                            {parts.map((part, i) => 
                              i % 2 === 1 ? <strong key={i}>{part}</strong> : part
                            )}
                          </li>
                        );
                      }
                      return <li key={index} className="ml-4 mb-1.5 leading-relaxed">{text}</li>;
                    }
                    if (line.match(/^\d+\.\s/)) {
                      const text = line.replace(/^\d+\.\s/, '');
                      return <li key={index} className="ml-4 mb-1.5 list-decimal leading-relaxed">{text}</li>;
                    }
                    if (line.trim() === '') {
                      return <div key={index} className="h-3"></div>;
                    }
                    if (line.includes('**')) {
                      const parts = line.split('**');
                      return (
                        <p key={index} className="mb-2.5 leading-relaxed text-sm">
                          {parts.map((part, i) => 
                            i % 2 === 1 ? <strong key={i}>{part}</strong> : part
                          )}
                        </p>
                      );
                    }
                    return <p key={index} className="mb-2.5 leading-relaxed text-sm">{line}</p>;
                  })}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Documentation;
