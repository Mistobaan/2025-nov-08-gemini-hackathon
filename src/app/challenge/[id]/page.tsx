import { notFound } from "next/navigation";
import ChallengeWorkspace from "@/components/challenge-workspace";
import { loadChallengeById } from "@/lib/challenges";

type ChallengePageProps = {
  params: { id: string };
};

export default async function ChallengePage({ params }: ChallengePageProps) {
  const challenge = await loadChallengeById(params.id);

  if (!challenge) {
    notFound();
  }

  return <ChallengeWorkspace challenge={challenge} />;
}
