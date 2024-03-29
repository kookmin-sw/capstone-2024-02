import { SettingPage } from '@/app/pages';

export default function Page({
  params: { userId },
}: {
  params: { userId: string };
}) {
  return <SettingPage type="내" name="김마루" />;
}
