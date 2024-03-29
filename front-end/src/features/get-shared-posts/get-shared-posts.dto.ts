export interface GetSharedPostsDTO {
  data: [
    {
      post: { postId: number; title: string; img: string };
      writer: { userId: string; profileURL: string; name: string };
    },
  ];
}
