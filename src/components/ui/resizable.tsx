"use client";

import * as React from "react";
import {
  PanelGroup,
  Panel,
  PanelResizeHandle,
  type PanelGroupProps,
  type PanelProps,
  type PanelResizeHandleProps,
} from "react-resizable-panels";
import { cn } from "@/lib/utils";

const ResizablePanelGroup = React.forwardRef<HTMLDivElement, PanelGroupProps>(({ className, ...props }, ref) => (
  <PanelGroup
    ref={ref}
    className={cn("flex h-full w-full data-[panel-group-direction=vertical]:flex-col", className)}
    {...props}
  />
));
ResizablePanelGroup.displayName = "ResizablePanelGroup";

const ResizablePanel = React.forwardRef<HTMLDivElement, PanelProps>(({ className, ...props }, ref) => (
  <Panel ref={ref} className={cn("flex flex-col", className)} {...props} />
));
ResizablePanel.displayName = "ResizablePanel";

const ResizableHandle = React.forwardRef<HTMLDivElement, PanelResizeHandleProps>(
  ({ className, ...props }, ref) => (
    <PanelResizeHandle
      ref={ref}
      className={cn(
        "relative flex items-center justify-center bg-transparent transition-colors",
        "data-[panel-group-direction=horizontal]:w-1 data-[panel-group-direction=vertical]:h-1",
        "data-[panel-group-direction=horizontal]:cursor-col-resize data-[panel-group-direction=vertical]:cursor-row-resize",
        "data-[resize-handle-active=true]:bg-zinc-300 dark:data-[resize-handle-active=true]:bg-zinc-700",
        className,
      )}
      {...props}
    >
      <span
        className="block rounded-full bg-zinc-300 dark:bg-zinc-700 data-[panel-group-direction=horizontal]:h-12 data-[panel-group-direction=horizontal]:w-0.5 data-[panel-group-direction=vertical]:h-0.5 data-[panel-group-direction=vertical]:w-12"
        aria-hidden="true"
      />
    </PanelResizeHandle>
  ),
);
ResizableHandle.displayName = "ResizableHandle";

export { ResizablePanelGroup, ResizablePanel, ResizableHandle };
