/**
 * Main Dashboard component with modern sidebar layout
 */
import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { AppSidebar } from '../components/AppSidebar';
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from '../components/ui/sidebar';
import { Separator } from '../components/ui/separator';

const Dashboard = () => {
  const location = useLocation();

  // Generate breadcrumb based on current path
  const getBreadcrumb = () => {
    const path = location.pathname;
    if (path === '/dashboard') return 'Dashboard';
    if (path.startsWith('/dashboard/datasets')) return 'Datasets';
    if (path.startsWith('/dashboard/processed')) return 'Processed Datasets';
    if (path.startsWith('/dashboard/models')) return 'Models';
    if (path.startsWith('/dashboard/docs')) return 'Documentation';
    return 'Dashboard';
  };

  return (
    <SidebarProvider defaultOpen={true}>
      <AppSidebar />
      <SidebarInset className="flex flex-col min-h-screen max-w-full overflow-x-hidden">
        <header className="flex h-16 shrink-0 items-center gap-2 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12 sticky top-0 z-10" style={{ zoom: '1' }}>
          <div className="flex items-center gap-2 px-4">
            <SidebarTrigger className="-ml-1 text-xs" />
            <Separator orientation="vertical" className="mr-2 h-4" />
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold">{getBreadcrumb()}</h1>
            </div>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4 pt-4 max-w-full overflow-x-hidden" style={{ zoom: '0.9' }}>
          <Outlet />
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
};

export default Dashboard;
