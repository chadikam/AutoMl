import React from 'react';
import { cn } from '../../lib/utils';
import { Separator } from './separator';

export function FieldGroup({ className, ...props }) {
  return (
    <div
      data-slot="field-group"
      className={cn(
        'group/field-group flex w-full flex-col gap-6',
        className
      )}
      {...props}
    />
  );
}

export function Field({ className, ...props }) {
  return (
    <div
      role="group"
      data-slot="field"
      className={cn('group/field flex w-full flex-col gap-1.5', className)}
      {...props}
    />
  );
}

export function FieldLabel({ className, htmlFor, ...props }) {
  return (
    <label
      htmlFor={htmlFor}
      data-slot="field-label"
      className={cn(
        'text-sm font-medium leading-none text-zinc-900 dark:text-zinc-100 peer-disabled:cursor-not-allowed peer-disabled:opacity-70',
        className
      )}
      {...props}
    />
  );
}

export function FieldDescription({ className, ...props }) {
  return (
    <p
      data-slot="field-description"
      className={cn(
        'text-sm text-zinc-600 dark:text-zinc-400',
        '[&>a]:text-zinc-900 dark:[&>a]:text-zinc-100 [&>a]:underline [&>a]:underline-offset-4 [&>a:hover]:text-blue-600 dark:[&>a:hover]:text-blue-400',
        className
      )}
      {...props}
    />
  );
}

export function FieldSeparator({ children, className, ...props }) {
  return (
    <div
      data-slot="field-separator"
      className={cn('relative -my-2 h-5 text-sm text-zinc-600 dark:text-zinc-400', className)}
      {...props}
    >
      <Separator className="absolute inset-0 top-1/2 bg-zinc-200 dark:bg-zinc-800" />
      {children && (
        <span
          className="relative mx-auto block w-fit bg-white dark:bg-zinc-950 px-2"
          data-slot="field-separator-content"
        >
          {children}
        </span>
      )}
    </div>
  );
}
