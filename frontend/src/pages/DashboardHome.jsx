/**
 * Dashboard home page with modern overview
 */
import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  Database, 
  Brain, 
  Zap, 
  TrendingUp, 
  Upload, 
  Sparkles,
  ArrowRight,
  Clock,
  CheckCircle,
  BarChart3,
  FileText,
  Activity,
  Layers,
  Target
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { datasetsAPI, modelsAPI } from '../utils/api';

const DashboardHome = () => {
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [datasetsData, modelsData] = await Promise.all([
        datasetsAPI.list(),
        modelsAPI.list(),
      ]);
      setDatasets(datasetsData);
      setModels(modelsData);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  };

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

  const totalModels = models.length;
  const processedDatasets = datasets.filter(d => d.is_preprocessed).length;
  const bestAccuracy = models.length > 0 ? Math.max(...models.map(m => m.accuracy)) * 100 : 0;

  const statCards = [
    {
      title: 'Total Datasets',
      value: datasets.length,
      change: `${processedDatasets} processed`,
      icon: Database,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-950/50',
      borderColor: 'border-blue-200 dark:border-blue-800',
      link: '/dashboard/datasets'
    },
    {
      title: 'Trained Models',
      value: totalModels,
      change: 'Ready to deploy',
      icon: Brain,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50 dark:bg-purple-950/50',
      borderColor: 'border-purple-200 dark:border-purple-800',
      link: '/dashboard/models/automl'
    },
    {
      title: 'Best Accuracy',
      value: totalModels > 0 ? `${bestAccuracy.toFixed(1)}%` : 'N/A',
      change: 'Top performer',
      icon: Target,
      color: 'text-green-600',
      bgColor: 'bg-green-50 dark:bg-green-950/50',
      borderColor: 'border-green-200 dark:border-green-800',
      link: '/dashboard/models/automl'
    },
    {
      title: 'Processed Data',
      value: processedDatasets,
      change: `${datasets.length - processedDatasets} raw`,
      icon: CheckCircle,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50 dark:bg-orange-950/50',
      borderColor: 'border-orange-200 dark:border-orange-800',
      link: '/dashboard/datasets/processed'
    }
  ];

  const quickActions = [
    {
      title: 'Upload Dataset',
      description: 'Add new data for training',
      icon: Upload,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-950/50',
      link: '/dashboard/datasets/upload'
    },
    {
      title: 'Train Model',
      description: 'Start AutoML training',
      icon: Sparkles,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50 dark:bg-purple-950/50',
      link: '/dashboard/models/automl/train'
    },
    {
      title: 'Test Model',
      description: 'Make predictions',
      icon: Zap,
      color: 'text-green-600',
      bgColor: 'bg-green-50 dark:bg-green-950/50',
      link: '/dashboard/models/test'
    },
    {
      title: 'View Reports',
      description: 'Analyze performance',
      icon: BarChart3,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50 dark:bg-orange-950/50',
      link: '/dashboard/models/automl'
    }
  ];

  return (
    <div className="space-y-8 p-6">
      {/* Welcome Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-primary via-primary to-primary/60 bg-clip-text text-transparent">
            AutoML Framework
          </h1>
          <p className="mt-2 text-muted-foreground text-lg">
            Here's what's happening in your ML workspace
          </p>
        </div>
        <div className="hidden md:flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/10 border border-primary/20">
          <Activity className="w-5 h-5 text-primary animate-pulse" />
          <span className="text-sm font-medium">All systems operational</span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {statCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Link key={index} to={stat.link}>
              <Card className={`border-2 ${stat.borderColor} hover:shadow-lg transition-all duration-300 hover:-translate-y-1 cursor-pointer group`}>
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="space-y-2">
                      <p className="text-sm font-medium text-muted-foreground">
                        {stat.title}
                      </p>
                      <p className="text-3xl font-bold">
                        {stat.value}
                      </p>
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        <TrendingUp className="w-3 h-3" />
                        {stat.change}
                      </div>
                    </div>
                    <div className={`p-3 rounded-xl ${stat.bgColor} group-hover:scale-110 transition-transform`}>
                      <Icon className={`w-6 h-6 ${stat.color}`} />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>

      {/* Quick Actions */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">Quick Actions</h2>
          <Button variant="ghost" size="sm" className="gap-2">
            <Layers className="w-4 h-4" />
            View All
          </Button>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {quickActions.map((action, index) => {
            const Icon = action.icon;
            return (
              <Link key={index} to={action.link}>
                <Card className="hover:shadow-md transition-all duration-300 hover:-translate-y-1 cursor-pointer group border-2">
                  <CardContent className="p-6">
                    <div className={`w-12 h-12 rounded-xl ${action.bgColor} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                      <Icon className={`w-6 h-6 ${action.color}`} />
                    </div>
                    <h3 className="font-semibold text-lg mb-1">{action.title}</h3>
                    <p className="text-sm text-muted-foreground">{action.description}</p>
                    <div className="mt-4 flex items-center gap-1 text-sm text-primary font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                      Get started
                      <ArrowRight className="w-4 h-4" />
                    </div>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Recent Activity */}
      {models.length > 0 ? (
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold">Recent Models</h2>
            <Link to="/dashboard/models/automl">
              <Button variant="ghost" size="sm" className="gap-2">
                View all models
                <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
          </div>
          <div className="grid gap-4">
            {models.slice(0, 3).map((model) => (
              <Link key={model.id} to={`/dashboard/models/${model.id}`}>
                <Card className="hover:shadow-md transition-all hover:border-primary/50 cursor-pointer">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="p-3 rounded-lg bg-gradient-to-br from-primary/10 to-primary/5">
                          <Brain className="w-6 h-6 text-primary" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-lg">{model.name}</h3>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge variant="secondary" className="text-xs">
                              {model.model_type.replace('_', ' ')}
                            </Badge>
                            <span className="text-xs text-muted-foreground flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {new Date(model.created_at).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold text-green-600">
                          {(model.accuracy * 100).toFixed(1)}%
                        </p>
                        <p className="text-xs text-muted-foreground">Accuracy</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      ) : (
        /* Empty State */
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
              <Link to="/dashboard/documentation">
                <Button size="lg" variant="outline" className="gap-2">
                  <FileText className="w-4 h-4" />
                  Learn More
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default DashboardHome;
