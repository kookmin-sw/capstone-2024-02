import { ProfilePage } from '@/app/pages';

export default function Page({
  params: { memberId },
}: {
  params: { memberId: string };
}) {
  return <ProfilePage memberId={memberId} />;
}
