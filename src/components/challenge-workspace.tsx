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
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Textarea } from "@/components/ui/textarea";

type ChallengeWorkspaceProps = {
  challenge: Challenge;
};

const fallbackTemplate = `def solution(*args, **kwargs):
    """Implement your TPU solution here."""
    pass
`;

export default function ChallengeWorkspace({ challenge }: ChallengeWorkspaceProps) {
  const starter = challenge.studentTemplate ?? challenge.student_template ?? fallbackTemplate;
  const [editorContent, setEditorContent] = useState(starter);
  const [tracebackInput, setTracebackInput] = useState("");
  const [geminiOutput, setGeminiOutput] = useState("");
  const [geminiOutputTitle, setGeminiOutputTitle] = useState("");
  const difficultyLabel = challenge.difficulty
    ? `${challenge.difficulty.charAt(0).toUpperCase()}${challenge.difficulty.slice(1)}`
    : "Challenge";

  const submitPromptToGemini = async (prompt: string) => {
    setGeminiOutput("Calling Gemini API...");

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

  const buildRunPrompt = (code: string, traceback?: string) => `
Please review the following Python code.
${traceback ? `The code produced the following traceback:\n${traceback}` : ""}

Code:
\`\`\`python
${code}
\`\`\`

Provide feedback on the code, and if there is a traceback, explain the error and suggest a fix.
  `;

  const buildReviewPrompt = (code: string, description: string, traceback?: string) => {
    const trimmedTraceback = traceback?.trim();
    const tracebackSection = trimmedTraceback ? ` and the traceback ${trimmedTraceback}` : "";
    return `please given the problem description ${description} the code of the user ${code}${tracebackSection} you are a TPU expert programmer please give recommendations to the user tips and tricks in case he is stuck`;
  };

  const handleRun = () => {
    setGeminiOutputTitle("Run Output");
    setGeminiOutput("Running code...");
    setTimeout(() => {
      const traceback = 'Traceback (most recent call last):\n  File "main.py", line 1, in <module>\n    print(hello)';
      setGeminiOutput(traceback);
      void submitPromptToGemini(buildRunPrompt(editorContent, traceback));
    }, 1000);
  };

  const handleReview = async () => {
    setGeminiOutputTitle("Code Review");
    setGeminiOutput("Requesting code review...");
    const prompt = buildReviewPrompt(editorContent, challenge.description, tracebackInput);
    await submitPromptToGemini(prompt);
  };

  const handleSave = () => {
    setGeminiOutputTitle("Save");
    setGeminiOutput("File saved!");
  };

  return (
    <div className="flex h-full min-h-screen flex-col bg-zinc-50 font-sans dark:bg-black">
      <header className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-800 sm:px-6">
        <div className="flex flex-col gap-1">
          <h1 className="text-lg font-semibold text-black dark:text-zinc-50">
            <Link href="/" className="text-zinc-500 hover:underline">
              TPU Playground
            </Link>{" "}
            / {challenge.title}
          </h1>
          <Badge variant="secondary" className="w-fit">
            Difficulty · {difficultyLabel}
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleSave}>
            Save
          </Button>
          <Button variant="outline" size="sm" onClick={handleRun}>
            Run
          </Button>
        </div>
      </header>
      <main className="flex-1 min-h-0 p-4 sm:p-6">
        <ResizablePanelGroup direction="horizontal" className="h-full min-h-[520px] gap-4">
          <ResizablePanel defaultSize={60} minSize={35} className="min-h-[400px] overflow-hidden">
            <div className="flex h-full min-h-0 flex-col rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
              <Tabs defaultValue="readme" className="flex h-full min-h-[480px] flex-col">
                <TabsList className="w-fit shrink-0">
                  <TabsTrigger value="readme">README.md</TabsTrigger>
                  <TabsTrigger value="main.py">main.py</TabsTrigger>
                </TabsList>
                <TabsContent value="readme" className="mt-4 flex-1 overflow-hidden">
                  <Card className="flex h-full flex-col overflow-auto">
                    <CardContent className="prose h-full flex-1 overflow-auto p-6 dark:prose-invert">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{challenge.description}</ReactMarkdown>
                    </CardContent>
                  </Card>
                </TabsContent>
                <TabsContent value="main.py" className="mt-4 flex flex-1 min-h-[360px]">
                  <div className="flex flex-1 flex-col gap-4">
                    <Card className="flex h-full flex-1 overflow-hidden">
                      <Editor
                        height="100%"
                        language="python"
                        theme="vs-light"
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
                    <div>
                      <label className="text-sm font-medium text-zinc-600 dark:text-zinc-300">
                        Traceback (optional)
                      </label>
                      <Textarea
                        placeholder="Paste the traceback here if you have one"
                        value={tracebackInput}
                        onChange={(event) => setTracebackInput(event.target.value)}
                        className="mt-2"
                      />
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          </ResizablePanel>
          <ResizableHandle className="mx-1 rounded-full bg-zinc-200 dark:bg-zinc-800" />
          <ResizablePanel defaultSize={40} minSize={20} className="min-h-[320px]">
            <div className="flex h-full flex-col gap-3">
              <SidePanel
                output={geminiOutput}
                title={geminiOutputTitle || "Code Review"}
                className="flex-1 rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950"
              />
              <Button size="sm" onClick={handleReview}>
                Review
              </Button>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </main>
    </div>
  );
}
