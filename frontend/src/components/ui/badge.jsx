import * as React from "react";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-zinc-900 dark:bg-zinc-50 text-zinc-50 dark:text-zinc-900 hover:bg-zinc-900/80 dark:hover:bg-zinc-50/80",
        secondary:
          "border-transparent bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 hover:bg-zinc-100/80 dark:hover:bg-zinc-800/80",
        destructive:
          "border-transparent bg-red-500 dark:bg-red-900 text-zinc-50 dark:text-zinc-50 hover:bg-red-500/80 dark:hover:bg-red-900/80",
        success:
          "border-transparent bg-green-500 dark:bg-green-900 text-zinc-50 dark:text-zinc-50 hover:bg-green-500/80 dark:hover:bg-green-900/80",
        warning:
          "border-transparent bg-yellow-500 dark:bg-yellow-900 text-zinc-900 dark:text-zinc-50 hover:bg-yellow-500/80 dark:hover:bg-yellow-900/80",
        outline: "text-zinc-900 dark:text-zinc-100 border-zinc-200 dark:border-zinc-800",
        admin:
          "border-transparent bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300 hover:bg-purple-100/80 dark:hover:bg-purple-900/60",
        standard:
          "border-transparent bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 hover:bg-blue-100/80 dark:hover:bg-blue-900/60",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export function Badge({ className, variant, ...props }) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}
