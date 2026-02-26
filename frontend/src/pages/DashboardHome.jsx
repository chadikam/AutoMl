/**
 * Actionable Dashboard Home — operational insights and decision-focused overview
 */
import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  Database,
  Brain,
  Zap,
  TrendingUp,
  Upload,
  Sparkles,
  ArrowRight,
  ArrowUpDown,
  Clock,
  CheckCircle,
  BarChart3,
  FileText,
  Activity,
  Layers,
  Target,
  AlertTriangle,
  AlertCircle,
  Info,
  Loader2,
  Trophy,
  Timer,
  XCircle,
  ChevronUp,
  ChevronDown,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '../components/ui/table';
import { automlAPI } from '../utils/api';

// ─── Helpers ────────────────────────────────────────────────────────────────

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

function formatDate(dateStr) {
  if (!dateStr) return '—';
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return '—';
  const now = new Date();
  const diff = now - d;
  if (diff < 60_000) return 'Just now';
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  if (diff < 604_800_000) return `${Math.floor(diff / 86_400_000)}d ago`;
  return d.toLocaleDateString();
}

function formatPercent(value) {
  if (value === null || value === undefined) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

function severityIcon(severity) {
  if (severity === 'error') return <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />;
  if (severity === 'warning') return <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />;
  return <Info className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />;
}

function severityBorder(severity) {
  if (severity === 'error') return 'border-red-200 dark:border-red-800 bg-red-50/50 dark:bg-red-950/30';
  if (severity === 'warning') return 'border-amber-200 dark:border-amber-800 bg-amber-50/50 dark:bg-amber-950/30';
  return 'border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-950/30';
}

// ─── Component ──────────────────────────────────────────────────────────────

const DashboardHome = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Table sorting
  const [sortField, setSortField] = useState('created_at');
  const [sortDir, setSortDir] = useState('desc');

  const fetchDashboard = useCallback(async () => {
    try {
      const result = await automlAPI.getDashboard();
      setData(result);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
      setError('Unable to load dashboard data.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboard();
    // Refresh every 30s to pick up training progress
    const interval = setInterval(fetchDashboard, 30_000);
    return () => clearInterval(interval);
  }, [fetchDashboard]);

  // ── Sorted model table ──────────────────────────────────────────────
  const sortedModels = useMemo(() => {
    if (!data?.model_table) return [];
    const rows = [...data.model_table];
    rows.sort((a, b) => {
      let va = a[sortField];
      let vb = b[sortField];
      if (va === null || va === undefined) va = '';
      if (vb === null || vb === undefined) vb = '';
      if (typeof va === 'number' && typeof vb === 'number') {
        return sortDir === 'asc' ? va - vb : vb - va;
      }
      const sa = String(va);
      const sb = String(vb);
      return sortDir === 'asc' ? sa.localeCompare(sb) : sb.localeCompare(sa);
    });
    return rows;
  }, [data?.model_table, sortField, sortDir]);

  const toggleSort = (field) => {
    if (sortField === field) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDir('desc');
    }
  };

  const SortIcon = ({ field }) => {
    if (sortField !== field) return <ArrowUpDown className="w-3 h-3 ml-1 opacity-40" />;
    return sortDir === 'asc'
      ? <ChevronUp className="w-3 h-3 ml-1" />
      : <ChevronDown className="w-3 h-3 ml-1" />;
  };

  // ── Loading / Error ──────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <div className="relative">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-primary"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <Sparkles className="w-6 h-6 text-primary" />
          </div>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4 text-muted-foreground">
        <AlertCircle className="w-12 h-12" />
        <p>{error}</p>
        <Button variant="outline" size="sm" onClick={() => { setLoading(true); fetchDashboard(); }}>
          Retry
        </Button>
      </div>
    );
  }

  const { training_jobs, best_model, last_dataset, attention_items, timeline, total_datasets, total_models } = data || {};

  // ── Empty state ──────────────────────────────────────────────────────
  const isEmpty = total_datasets === 0 && total_models === 0;
  if (isEmpty) {
    return (
      <div className="space-y-8 p-6">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-primary via-primary to-primary/60 bg-clip-text text-transparent">
            AutoML Framework
          </h1>
          <p className="mt-2 text-muted-foreground text-lg">
            Get started by uploading your first dataset
          </p>
        </div>
        <Card className="border-2 border-dashed">
          <CardContent className="p-12 text-center">
            <div className="mx-auto w-16 h-16 bg-gradient-to-br from-primary/10 to-primary/5 rounded-full flex items-center justify-center mb-6">
              <Sparkles className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Start Your ML Journey</h3>
            <p className="text-muted-foreground mb-6 max-w-md mx-auto">
              Upload your first dataset and train a machine learning model in just a few clicks
            </p>
            <div className="flex items-center justify-center gap-3">
              <Link to="/dashboard/datasets/upload">
                <Button size="lg" className="gap-2">
                  <Upload className="w-4 h-4" />
                  Upload Dataset
                </Button>
              </Link>
              <Link to="/dashboard/docs">
                <Button size="lg" variant="outline" className="gap-2">
                  <FileText className="w-4 h-4" />
                  Learn More
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ── Quick actions ────────────────────────────────────────────────────
  const quickActions = [
    { title: 'Upload Dataset', description: 'Add new data', icon: Upload, color: 'text-blue-600', bgColor: 'bg-blue-50 dark:bg-blue-950/50', link: '/dashboard/datasets/upload' },
    { title: 'Train Model', description: 'Start AutoML training', icon: Sparkles, color: 'text-purple-600', bgColor: 'bg-purple-50 dark:bg-purple-950/50', link: '/dashboard/models/automl/train' },
    { title: 'Test Model', description: 'Make predictions', icon: Zap, color: 'text-green-600', bgColor: 'bg-green-50 dark:bg-green-950/50', link: '/dashboard/models/test' },
  ];

  return (
    <div className="space-y-8 p-6">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div>
        <div className="flex items-center justify-between mt-2">
          <p className="text-muted-foreground text-lg">
            Here's what needs your attention
          </p>
          <div className="hidden md:flex items-center gap-3">
            {quickActions.map((qa, i) => {
              const Icon = qa.icon;
              return (
                <Link key={i} to={qa.link}>
                  <Button variant="outline" size="sm" className="gap-2">
                    <Icon className={`w-4 h-4 ${qa.color}`} />
                    {qa.title}
                  </Button>
                </Link>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── 1️⃣ Summary Cards ──────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {/* Active Training Jobs */}
        <Card className="border border-purple-200 dark:border-purple-800">
          <CardContent className="p-6">
            <div className="flex items-start justify-between">
              <div className="space-y-2">
                <p className="text-sm font-medium text-muted-foreground">Training Jobs</p>
                {training_jobs?.active_training ? (
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-purple-600" />
                      <span className="text-sm font-semibold text-purple-600">Running</span>
                    </div>
                    <p className="text-xs text-muted-foreground truncate max-w-[160px]">
                      {training_jobs.active_training.current_model || 'Starting…'}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {training_jobs.active_training.models_completed_count}/{training_jobs.active_training.total_models} models ·{' '}
                      {formatDuration(training_jobs.active_training.total_elapsed_seconds)}
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className="text-3xl font-bold">{training_jobs?.completed || 0}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {training_jobs?.completed || 0} completed · {training_jobs?.failed || 0} failed
                    </p>
                  </div>
                )}
              </div>
              <div className="p-3 rounded-xl bg-purple-50 dark:bg-purple-950/50">
                <Activity className={`w-6 h-6 text-purple-600 ${training_jobs?.active_training ? 'animate-pulse' : ''}`} />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Best Performing Model */}
        <Link to={best_model ? `/dashboard/models/${best_model.id}` : '/dashboard/models/automl'}>
          <Card className="border border-green-200 dark:border-green-800 h-full">
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="space-y-2 min-w-0">
                  <p className="text-sm font-medium text-muted-foreground">Best Performing Model</p>
                  {best_model ? (
                    <>
                      <p className="text-3xl font-bold text-green-600">{formatPercent(best_model.score)}</p>
                      <p className="text-xs text-muted-foreground truncate max-w-[160px]" title={best_model.name}>
                        {best_model.best_model_type || best_model.name}
                      </p>
                      <p className="text-xs text-muted-foreground truncate max-w-[160px]" title={best_model.dataset_name}>
                        on {best_model.dataset_name}
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="text-3xl font-bold">N/A</p>
                      <p className="text-xs text-muted-foreground">No models trained yet</p>
                    </>
                  )}
                </div>
                <div className="p-3 rounded-xl bg-green-50 dark:bg-green-950/50">
                  <Trophy className="w-6 h-6 text-green-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </Link>

        {/* Last Uploaded Dataset */}
        <Link to={last_dataset ? `/dashboard/datasets/${last_dataset.id}` : '/dashboard/datasets'}>
          <Card className="border border-blue-200 dark:border-blue-800 h-full">
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="space-y-2 min-w-0">
                  <p className="text-sm font-medium text-muted-foreground">Last Uploaded Dataset</p>
                  {last_dataset ? (
                    <>
                      <p className="text-lg font-bold truncate max-w-[160px]" title={last_dataset.name}>
                        {last_dataset.name}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {last_dataset.rows?.toLocaleString()} rows · {last_dataset.columns} cols
                      </p>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">
                          {last_dataset.missing_pct}% missing
                        </span>
                        <Badge variant={last_dataset.is_processed ? 'default' : 'secondary'} className="text-[10px] px-1.5 py-0">
                          {last_dataset.is_processed ? 'Processed' : 'Raw'}
                        </Badge>
                      </div>
                    </>
                  ) : (
                    <>
                      <p className="text-3xl font-bold">—</p>
                      <p className="text-xs text-muted-foreground">No datasets uploaded</p>
                    </>
                  )}
                </div>
                <div className="p-3 rounded-xl bg-blue-50 dark:bg-blue-950/50">
                  <Database className="w-6 h-6 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </Link>

        {/* Overview Counts */}
        <Card className="border border-orange-200 dark:border-orange-800">
          <CardContent className="p-6">
            <div className="flex items-start justify-between">
              <div className="space-y-2">
                <p className="text-sm font-medium text-muted-foreground">Workspace Overview</p>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1">
                  <div>
                    <p className="text-2xl font-bold">{total_datasets}</p>
                    <p className="text-xs text-muted-foreground">Datasets</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold">{total_models}</p>
                    <p className="text-xs text-muted-foreground">Models</p>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {(attention_items || []).length > 0
                    ? `${attention_items.length} item${attention_items.length !== 1 ? 's' : ''} need attention`
                    : 'All clear'}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-orange-50 dark:bg-orange-950/50">
                <Layers className="w-6 h-6 text-orange-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ── 2️⃣ Needs Attention ────────────────────────────────────────── */}
      {attention_items && attention_items.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <h2 className="text-2xl font-bold">Needs Attention</h2>
            <Badge variant="secondary" className="text-xs">{attention_items.length}</Badge>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {attention_items.map((item, idx) => (
              <Card key={idx} className={`border ${severityBorder(item.severity)}`}>
                <CardContent className="p-5">
                  <div className="flex items-start gap-3">
                    {severityIcon(item.severity)}
                    <div className="min-w-0 flex-1">
                      <p className="font-semibold text-sm">{item.title}</p>
                      <p className="text-xs text-muted-foreground mt-1">{item.description}</p>
                      {item.items && item.items.length > 0 && (
                        <div className="mt-3 space-y-1.5">
                          {item.items.map((sub, si) => (
                            <div key={si} className="flex items-center justify-between text-xs">
                              <Link
                                to={
                                  item.type === 'unprocessed_datasets'
                                    ? `/dashboard/datasets/${sub.id}`
                                    : `/dashboard/models/${sub.id}`
                                }
                                className="text-primary hover:underline truncate max-w-[180px]"
                                title={sub.name}
                              >
                                {sub.name}
                              </Link>
                              {sub.score !== undefined && (
                                <span className="text-red-500 font-medium ml-2">
                                  {formatPercent(sub.score)}
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      {item.type === 'unprocessed_datasets' && (
                        <Link to="/dashboard/datasets" className="inline-block mt-3">
                          <Button variant="outline" size="sm" className="h-7 text-xs gap-1">
                            View Datasets <ArrowRight className="w-3 h-3" />
                          </Button>
                        </Link>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* ── 3️⃣ Model Comparison Table ─────────────────────────────────── */}
      {sortedModels.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-primary" />
              <h2 className="text-2xl font-bold">Model Comparison</h2>
              <Badge variant="secondary" className="text-xs">{sortedModels.length}</Badge>
            </div>
            <Link to="/dashboard/models/automl">
              <Button variant="ghost" size="sm" className="gap-2">
                View all models
                <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
          </div>
          <Card>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="cursor-pointer select-none" onClick={() => toggleSort('dataset_name')}>
                      <span className="flex items-center">Dataset <SortIcon field="dataset_name" /></span>
                    </TableHead>
                    <TableHead className="cursor-pointer select-none" onClick={() => toggleSort('best_model_type')}>
                      <span className="flex items-center">Model Type <SortIcon field="best_model_type" /></span>
                    </TableHead>
                    <TableHead className="cursor-pointer select-none" onClick={() => toggleSort('task_type')}>
                      <span className="flex items-center">Task <SortIcon field="task_type" /></span>
                    </TableHead>
                    <TableHead className="cursor-pointer select-none text-right" onClick={() => toggleSort('accuracy')}>
                      <span className="flex items-center justify-end">Accuracy <SortIcon field="accuracy" /></span>
                    </TableHead>
                    <TableHead className="cursor-pointer select-none text-right" onClick={() => toggleSort('f1_score')}>
                      <span className="flex items-center justify-end">F1 Score <SortIcon field="f1_score" /></span>
                    </TableHead>
                    <TableHead className="cursor-pointer select-none text-right" onClick={() => toggleSort('test_score')}>
                      <span className="flex items-center justify-end">Test Score <SortIcon field="test_score" /></span>
                    </TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="cursor-pointer select-none" onClick={() => toggleSort('created_at')}>
                      <span className="flex items-center">Trained <SortIcon field="created_at" /></span>
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedModels.slice(0, 10).map((m) => (
                    <TableRow key={m.id} className="group cursor-pointer hover:bg-muted/50">
                      <TableCell>
                        <Link to={`/dashboard/models/${m.id}`} className="text-primary hover:underline font-medium text-sm truncate block max-w-[160px]" title={m.dataset_name}>
                          {m.dataset_name}
                        </Link>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="text-xs font-normal">
                          {m.best_model_type || '—'}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary" className="text-xs capitalize">
                          {m.task_type || '—'}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        {m.accuracy != null ? formatPercent(m.accuracy) : '—'}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        {m.f1_score != null ? formatPercent(m.f1_score) : '—'}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        {m.test_score != null ? formatPercent(m.test_score) : '—'}
                      </TableCell>
                      <TableCell>
                        <Badge variant="default" className="text-[10px] gap-1 bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400 border-0">
                          <CheckCircle className="w-3 h-3" />
                          {m.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
                        {formatDate(m.created_at)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            {sortedModels.length > 10 && (
              <div className="p-3 text-center border-t">
                <Link to="/dashboard/models/automl">
                  <Button variant="ghost" size="sm" className="gap-1 text-xs">
                    View all {sortedModels.length} models <ArrowRight className="w-3 h-3" />
                  </Button>
                </Link>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* ── 4️⃣ Activity Timeline ──────────────────────────────────────── */}
      {timeline && timeline.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Clock className="w-5 h-5 text-muted-foreground" />
            <h2 className="text-2xl font-bold">Recent Activity</h2>
          </div>
          <Card>
            <CardContent className="p-0">
              <div className="divide-y">
                {timeline.slice(0, 10).map((event, idx) => {
                  let Icon = Activity;
                  let iconColor = 'text-muted-foreground';
                  let iconBg = 'bg-muted';
                  if (event.type === 'dataset_upload') { Icon = Upload; iconColor = 'text-blue-600'; iconBg = 'bg-blue-50 dark:bg-blue-950/50'; }
                  if (event.type === 'dataset_processed') { Icon = CheckCircle; iconColor = 'text-emerald-600'; iconBg = 'bg-emerald-50 dark:bg-emerald-950/50'; }
                  if (event.type === 'model_trained') { Icon = Brain; iconColor = 'text-purple-600'; iconBg = 'bg-purple-50 dark:bg-purple-950/50'; }

                  return (
                    <div key={idx} className="flex items-center gap-4 px-5 py-3">
                      <div className={`p-2 rounded-lg ${iconBg} flex-shrink-0`}>
                        <Icon className={`w-4 h-4 ${iconColor}`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm truncate">{event.title}</p>
                        {event.meta && (
                          <p className="text-xs text-muted-foreground">
                            {event.meta.rows && `${event.meta.rows.toLocaleString()} rows`}
                            {event.meta.columns && ` · ${event.meta.columns} cols`}
                            {event.meta.best_model_type && `${event.meta.best_model_type}`}
                            {event.meta.score != null && ` · ${formatPercent(event.meta.score)}`}
                          </p>
                        )}
                      </div>
                      <span className="text-xs text-muted-foreground whitespace-nowrap flex-shrink-0">
                        {formatDate(event.timestamp)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ── Mobile Quick Actions (hidden on desktop where header has them) */}
      <div className="md:hidden">
        <h2 className="text-lg font-bold mb-3">Quick Actions</h2>
        <div className="grid grid-cols-3 gap-3">
          {quickActions.map((action, index) => {
            const Icon = action.icon;
            return (
              <Link key={index} to={action.link}>
                <Card className="hover:shadow-md transition-all duration-300 cursor-pointer group border-2">
                  <CardContent className="p-4 text-center">
                    <div className={`w-10 h-10 rounded-xl ${action.bgColor} flex items-center justify-center mb-2 mx-auto group-hover:scale-110 transition-transform`}>
                      <Icon className={`w-5 h-5 ${action.color}`} />
                    </div>
                    <h3 className="font-semibold text-xs">{action.title}</h3>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default DashboardHome;
