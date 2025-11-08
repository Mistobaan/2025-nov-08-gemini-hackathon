"use client";

import { useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// Mock user data for demonstration
const mockUser = {
  email: "user@example.com",
};

// Mock challenges data
const mockChallenges = [
  {
    id: "challenge-1",
    title: "Matrix Multiplication in JAX",
    description: "Implement a performant matrix multiplication function using JAX.",
    status: "Not Submitted",
  },
  {
    id: "challenge-2",
    title: "Image Classification with Flax",
    description: "Build and train a simple CNN for image classification on a dummy dataset.",
    status: "Not Submitted",
  },
  {
    id: "challenge-3",
    title: "Optimize a Function with JIT",
    description: "Use jit compilation to speed up a numerical computation.",
    status: "Benchmarked",
  },
];

export default function Home() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const handleSignIn = () => setIsAuthenticated(true);
  const handleSignOut = () => setIsAuthenticated(false);

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 font-sans dark:bg-black">
      <header className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-800 sm:px-6">
        <h1 className="text-lg font-semibold text-black dark:text-zinc-50">JAX TPU Lab</h1>
        {isAuthenticated ? (
          <div className="flex items-center gap-4">
            <span className="text-sm text-zinc-600 dark:text-zinc-400">
              Welcome, {mockUser.email}
            </span>
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
            <h2 className="mb-4 text-2xl font-bold tracking-tight text-black dark:text-zinc-50">
              Your Challenges
            </h2>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {mockChallenges.map((challenge) => (
                <Link key={challenge.id} href={`/challenge/${challenge.id}`} className="no-underline">
                  <Card className="h-full hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors">
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        {challenge.title}
                        <Badge 
                          variant={challenge.status === 'Benchmarked' ? 'default' : 'secondary'}
                        >
                          {challenge.status}
                        </Badge>
                      </CardTitle>
                      <CardDescription>{challenge.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Button className="w-full">
                        {challenge.status === 'Benchmarked' ? 'View Submission' : 'Start Challenge'}
                      </Button>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl rounded-lg bg-white p-8 text-center shadow-sm dark:bg-zinc-900">
            <h2 className="text-3xl font-bold tracking-tight text-black dark:text-zinc-50">
              Welcome to the JAX TPU Lab
            </h2>
            <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
              A seamless, web-based environment to write, review, and execute JAX code on Google's TPUs efficiently and securely.
            </p>
            <p className="mt-2 text-zinc-500 dark:text-zinc-500">
              Sign in to access your challenges and start coding.
            </p>
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