"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import type { Challenge } from "@/lib/challenges";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type ChallengesDashboardProps = {
  challenges: Challenge[];
};

const mockUser = {
  email: "user@example.com",
};

const difficultyLabels = ["", "Intro", "Easy", "Intermediate", "Advanced", "Expert"];

function pickSummary(markdown: string) {
  const firstUsableLine = markdown
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line && !line.startsWith("#") && !line.startsWith("```"));

  if (!firstUsableLine) {
    return "Ready to run on TPU hardware.";
  }

  return firstUsableLine.replace(/[*`_]/g, "");
}

export default function ChallengesDashboard({ challenges }: ChallengesDashboardProps) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const summaries = useMemo(() => {
    return Object.fromEntries(challenges.map((challenge) => [challenge.id, pickSummary(challenge.description)]));
  }, [challenges]);

  const handleSignIn = () => setIsAuthenticated(true);
  const handleSignOut = () => setIsAuthenticated(false);

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 font-sans dark:bg-black">
      <header className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-800 sm:px-6">
        <h1 className="text-lg font-semibold text-black dark:text-zinc-50">TPU Playground</h1>
        {isAuthenticated ? (
          <div className="flex items-center gap-4">
            <span className="text-sm text-zinc-600 dark:text-zinc-400">Welcome, {mockUser.email}</span>
            <Button variant="outline" size="sm" onClick={handleSignOut}>
              Sign Out
            </Button>
          </div>
        ) : (
          <Button onClick={handleSignIn}>Sign In with Google</Button>
        )}
      </header>

      <main className="flex-1 p-4 sm:p-6">
        {isAuthenticated ? (
          <div className="mx-auto max-w-4xl">
            <h2 className="mb-4 text-2xl font-bold tracking-tight text-black dark:text-zinc-50">Your Challenges</h2>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {challenges.map((challenge) => (
                <Link key={challenge.id} href={`/challenge/${challenge.id}`} className="no-underline">
                  <Card className="h-full transition-colors hover:border-zinc-400 dark:hover:border-zinc-600">
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between gap-2">
                        <span className="truncate">{challenge.title}</span>
                        <Badge variant="secondary">
                          {difficultyLabels[challenge.difficulty] || "Challenge"} · {challenge.difficulty}/5
                        </Badge>
                      </CardTitle>
                      <CardDescription>{summaries[challenge.id]}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Button className="w-full">Start Challenge</Button>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl rounded-lg bg-white p-8 text-center shadow-sm dark:bg-zinc-900">
            <h2 className="text-3xl font-bold tracking-tight text-black dark:text-zinc-50">Welcome to the TPU Playground</h2>
            <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
              A seamless, web-based environment to write, review, and execute code on Google&rsquo;s TPUs efficiently and securely.
            </p>
            <p className="mt-2 text-zinc-500 dark:text-zinc-500">Sign in to access your challenges and start coding.</p>
            <div className="mt-6">
              <Button size="lg" onClick={handleSignIn}>
                Sign In with Google
              </Button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
