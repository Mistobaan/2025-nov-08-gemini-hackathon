"use client";

import { useParams } from "next/navigation";
import Editor from "@monaco-editor/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

// Mock challenges data - in a real app, this would be fetched from a database
const mockChallenges = [
  {
    id: "challenge-1",
    title: "Matrix Multiplication in JAX",
    description: `
### Task
Implement a performant matrix multiplication function using JAX.

### Requirements
- Your function should be named \`matrix_multiply\`.
- It should accept two \`jax.numpy.ndarray\` objects.
- Use \`@jax.jit\` to compile your function for performance.

### Example
\\\`\`\`python
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (100, 200))
b = jax.random.normal(key, (200, 300))

# Your function should work like this:
# result = matrix_multiply(a, b)
# assert result.shape == (100, 300)
\\\`\`\`
    `,
  },
  {
    id: "challenge-2",
    title: "Image Classification with Flax",
    description: `
### Task
Build and train a simple CNN for image classification on a dummy dataset using Flax.

### Requirements
- Define a simple CNN model using \`flax.linen\`.
- Create a training step function.
- Your code should be runnable and demonstrate a basic training loop.
    `,
  },
  {
    id: "challenge-3",
    title: "Optimize a Function with JIT",
    description: `
### Task
Use \`@jax.jit\` compilation to speed up a numerical computation.

### Description
You are given a slow function. Your task is to apply JAX's Just-In-Time (JIT) compilation to make it run faster and verify the speed-up.
    `,
  },
];

const jaxTemplate = `
import jax
import jax.numpy as jnp

# This is a basic JAX template.
# Start by defining your functions.

def main():
  """Main function to run your JAX code."""
  print("Hello from JAX!")
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(key, (10,))
  print("Here is a random JAX array:")
  print(x)

if __name__ == "__main__":
  main()
`;

export default function ChallengePage() {
  const params = useParams();
  const challengeId = params.id as string;

  const challenge = mockChallenges.find((c) => c.id === challengeId);

  if (!challenge) {
    return <div className="p-6">Challenge not found.</div>;
  }

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 font-sans dark:bg-black">
       <header className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-800 sm:px-6">
        <h1 className="text-lg font-semibold text-black dark:text-zinc-50">
          <a href="/">JAX TPU Lab</a> / {challenge.title}
        </h1>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">Run</Button>
          <Button size="sm">Submit</Button>
        </div>
      </header>
      <main className="flex-1 p-4 sm:p-6">
        <Tabs defaultValue="readme">
          <TabsList>
            <TabsTrigger value="readme">README.md</TabsTrigger>
            <TabsTrigger value="main.py">main.py</TabsTrigger>
          </TabsList>
          <TabsContent value="readme">
            <Card>
              <CardContent className="prose dark:prose-invert p-6">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {challenge.description}
                </ReactMarkdown>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="main.py">
            <Card className="h-[600px] overflow-hidden">
              <Editor
                height="100%"
                language="python"
                theme="vs-dark"
                defaultValue={jaxTemplate}
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
      </main>
    </div>
  );
}
