"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import type { Challenge } from "@/lib/challenges";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";

type ChallengesDashboardProps = {
  challenges: Challenge[];
};

const mockUser = {
  email: "user@example.com",
};

const difficultyLabels = ["", "Intro", "Easy", "Intermediate", "Advanced", "Expert"];

const difficultyFilters = [
  { value: "all", label: "All difficulties" },
  ...difficultyLabels.slice(1).map((label, index) => ({
    value: String(index + 1),
    label: `${label} (${index + 1})`,
  })),
];

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
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>("all");
  const summaries = useMemo(() => {
    return Object.fromEntries(challenges.map((challenge) => [challenge.id, pickSummary(challenge.description)]));
  }, [challenges]);
  const filteredChallenges = useMemo(() => {
    if (selectedDifficulty === "all") {
      return challenges;
    }

    const difficultyNumber = Number(selectedDifficulty);

    return challenges.filter((challenge) => challenge.difficulty === difficultyNumber);
  }, [challenges, selectedDifficulty]);

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
            <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-2xl font-bold tracking-tight text-black dark:text-zinc-50">Your Challenges</h2>
              <div className="flex flex-col gap-1 text-left text-sm">
                <Label htmlFor="difficulty-filter" className="text-zinc-600 dark:text-zinc-300">
                  Difficulty
                </Label>
                <select
                  id="difficulty-filter"
                  className="w-48 rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-700 shadow-sm focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
                  value={selectedDifficulty}
                  onChange={(event) => setSelectedDifficulty(event.target.value)}
                >
                  {difficultyFilters.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="space-y-4">
              {filteredChallenges.length === 0 ? (
                <div className="rounded-lg border border-dashed border-zinc-200 bg-white px-4 py-6 text-center text-sm text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300">
                  No challenges match this difficulty yet.
                </div>
              ) : (
                filteredChallenges.map((challenge) => (
                  <Link key={challenge.id} href={`/challenge/${challenge.id}`} className="no-underline">
                    <Card className="transition-colors hover:border-zinc-400 dark:hover:border-zinc-600">
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
                ))
              )}
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
