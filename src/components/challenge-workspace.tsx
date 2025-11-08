"use client";

import { useState } from "react";
import Link from "next/link";
import Editor from "@monaco-editor/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Challenge } from "@/lib/challenges";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SidePanel } from "@/components/ui/side-panel";
import { Badge } from "@/components/ui/badge";

type ChallengeWorkspaceProps = {
  challenge: Challenge;
};

const fallbackTemplate = `import jax
import jax.numpy as jnp

def main():
    print("TPU Playground ready for JAX!")

if __name__ == "__main__":
    main()
`;

export default function ChallengeWorkspace({ challenge }: ChallengeWorkspaceProps) {
  const [editorContent, setEditorContent] = useState(challenge.code || fallbackTemplate);
  const [geminiOutput, setGeminiOutput] = useState("");
  const [geminiOutputTitle, setGeminiOutputTitle] = useState("");

  const handleRun = () => {
    setGeminiOutputTitle("Run Output");
    setGeminiOutput("Running code...");
    setTimeout(() => {
      const traceback = 'Traceback (most recent call last):\n  File "main.py", line 1, in <module>\n    print(hello)';
      setGeminiOutput(traceback);
      handleGeminiCall(editorContent, traceback);
    }, 1000);
  };

  const handleReview = () => {
    setGeminiOutputTitle("Code Review");
    setGeminiOutput("Requesting code review...");
    handleGeminiCall(editorContent);
  };

  const handleSave = () => {
    setGeminiOutputTitle("Save");
    setGeminiOutput("File saved!");
  };

  const handleGeminiCall = async (code: string, traceback?: string) => {
    setGeminiOutput("Calling Gemini API...");

    const prompt = `
Please review the following Python code.
${traceback ? `The code produced the following traceback:\n${traceback}` : ""}

Code:
\`\`\`python
${code}
\`\`\`

Provide feedback on the code, and if there is a traceback, explain the error and suggest a fix.
    `;

    try {
      const response = await fetch("/api/gemini", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        const error = await response.json();
        setGeminiOutput(`Error from Gemini API: ${error.error}`);
        return;
      }

      const { result } = await response.json();
      setGeminiOutput(result);
    } catch (error) {
      if (error instanceof Error) {
        setGeminiOutput(`Error calling Gemini API: ${error.message}`);
      } else {
        setGeminiOutput("An unknown error occurred while calling the Gemini API.");
      }
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 font-sans dark:bg-black">
      <header className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-800 sm:px-6">
        <div className="flex flex-col gap-1">
          <h1 className="text-lg font-semibold text-black dark:text-zinc-50">
            <Link href="/" className="text-zinc-500 hover:underline">
              TPU Playground
            </Link>{" "}
            / {challenge.title}
          </h1>
          <Badge variant="secondary" className="w-fit">
            Difficulty {challenge.difficulty}/5
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleSave}>
            Save
          </Button>
          <Button variant="outline" size="sm" onClick={handleRun}>
            Run
          </Button>
          <Button size="sm" onClick={handleReview}>
            Review
          </Button>
        </div>
      </header>
      <main className="grid flex-1 gap-4 p-4 sm:p-6 lg:grid-cols-2">
        <div className="col-span-1">
          <Tabs defaultValue="readme">
            <TabsList>
              <TabsTrigger value="readme">README.md</TabsTrigger>
              <TabsTrigger value="main.py">main.py</TabsTrigger>
            </TabsList>
            <TabsContent value="readme">
              <Card>
                <CardContent className="prose dark:prose-invert p-6">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{challenge.description}</ReactMarkdown>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="main.py">
              <Card className="h-[600px] overflow-hidden">
                <Editor
                  height="100%"
                  language="python"
                  theme="vs-dark"
                  value={editorContent}
                  onChange={(value) => setEditorContent(value || "")}
                  options={{
                    minimap: { enabled: false },
                    fontSize: 14,
                    wordWrap: "on",
                    scrollBeyondLastLine: false,
                  }}
                />
              </Card>
            </TabsContent>
          </Tabs>
        </div>
        <div className="col-span-1">
          <SidePanel output={geminiOutput} title={geminiOutputTitle} />
        </div>
      </main>
    </div>
  );
}
