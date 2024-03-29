import { SharedPostPage } from '@/app/pages';

export default function Page({
  params: { postId },
}: {
  params: { postId: string };
}) {
  return (
    <SharedPostPage
      post={{
        title: '혼자 살긴 너무 큰 방 같이 살 룸메이트 구해요!',
        content: `안녕하세요! 저는 현재 룸메이트를 찾고 있는 정연수입니다. 
      서울시 정릉동에서 함께 살아갈 룸메이트를 구하고 있습니다.
      주로 밤에 작업을 하며 새벽 2시~3시쯤에 취침합니다.
      관심있으신 분들 연락 주세요!`,
      }}
    />
  );
}
