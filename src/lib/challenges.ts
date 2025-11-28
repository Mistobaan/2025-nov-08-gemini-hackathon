import { promises as fs } from "fs";
import path from "path";

export type Challenge = {
  id: string;
  title: string;
  description: string;
  code: string;
  difficulty: "easy" | "medium" | "hard";
  student_template?: string;
  studentTemplate?: string;
  subtasks?: Array<{
    title: string;
    description: string;
  }>;
};

const CHALLENGES_PATH = path.join(process.cwd(), "data", "challenges.jsonl");

async function readChallengeFile() {
  const file = await fs.readFile(CHALLENGES_PATH, "utf-8");

  return file
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

export async function loadChallenges(): Promise<Challenge[]> {
  const lines = await readChallengeFile();

  return lines.map((line) => JSON.parse(line) as Challenge);
}

export async function loadChallengeById(id: string): Promise<Challenge | null> {
  const challenges = await loadChallenges();

  return challenges.find((challenge) => challenge.id === id) ?? null;
}
