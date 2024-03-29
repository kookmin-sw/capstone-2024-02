import { SettingPage } from '@/app/pages';

export default function Page({
  params: { userId },
}: {
  params: { userId: string };
}) {
  return <SettingPage type="메이트" name="메이트" />;
}
