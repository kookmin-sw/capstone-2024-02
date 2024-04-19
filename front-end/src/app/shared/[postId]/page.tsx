import { SharedPostPage } from '@/app/pages';

export default function Page({
  params: { postId },
}: {
  params: { postId: string };
}) {
  return <SharedPostPage postId={+postId} />;
}
