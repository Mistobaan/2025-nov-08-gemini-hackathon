
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface SidePanelProps {
  output: string;
  title: string;
}

export function SidePanel({ output, title }: SidePanelProps) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="prose dark:prose-invert">
        <pre>{output}</pre>
      </CardContent>
    </Card>
  );
}
