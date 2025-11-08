
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface SidePanelProps {
  output: string;
  title: string;
  className?: string;
  contentClassName?: string;
}

export function SidePanel({ output, title, className, contentClassName }: SidePanelProps) {
  return (
    <Card className={cn("flex h-full flex-col", className)}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent
        className={cn(
          "flex-1 overflow-auto rounded-lg bg-zinc-50 p-4 font-mono text-xs text-black dark:bg-zinc-900 dark:text-zinc-100",
          contentClassName,
        )}
      >
        <pre className="whitespace-pre-wrap">{output}</pre>
      </CardContent>
    </Card>
  );
}
