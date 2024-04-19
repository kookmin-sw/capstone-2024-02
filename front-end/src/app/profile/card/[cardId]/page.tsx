import { SettingPage } from '@/app/pages';

export default function Page({
  params: { cardId },
}: {
  params: { cardId: number };
}) {
  return <SettingPage cardId={cardId} />;
}
