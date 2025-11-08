import ChallengesDashboard from "@/components/challenges-dashboard";
import { loadChallenges } from "@/lib/challenges";

export default async function Home() {
  const challenges = await loadChallenges();

  return <ChallengesDashboard challenges={challenges} />;
}
