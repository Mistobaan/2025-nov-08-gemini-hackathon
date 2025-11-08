import { notFound } from "next/navigation";
import ChallengeWorkspace from "@/components/challenge-workspace";
import { loadChallengeById } from "@/lib/challenges";

type ChallengePageProps = {
  params: Promise<{ id: string }>;
};

export default async function ChallengePage({ params }: ChallengePageProps) {
  const { id } = await params;
  const challenge = await loadChallengeById(id);

  if (!challenge) {
    notFound();
  }

  return (
    <div className="min-h-screen">
      <ChallengeWorkspace challenge={challenge} />
    </div>
  );
}
