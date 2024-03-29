'use client';

import { useState } from 'react';
import styled from 'styled-components';

import { UserCard } from '@/components/main-page';
import {
  SharedPostsMenu,
  SharedPostsFilter,
  PostCard,
} from '@/components/shared-posts';
import { type SharedPostsType } from '@/entities/shared-posts-filter';

const styles = {
  container: styled.div`
    padding-top: 4.12rem;
    width: 100%;

    display: flex;
    flex-direction: column;
  `,
  SharedPostsMenu: styled(SharedPostsMenu)`
    margin-bottom: 2rem;
  `,
  SharedPostsFilter: styled(SharedPostsFilter)`
    margin-bottom: 5.19rem;
  `,
  posts: styled.div`
    display: flex;
    flex-direction: column;
    gap: 2rem;
  `,
  cards: styled.div`
    padding-left: 2.62rem;
    display: flex;
    flex-wrap: wrap;
    gap: 2rem 2.62rem;
  `,
};

export function SharedPostsPage() {
  const [selected, setSelected] = useState<SharedPostsType>('hasRoom');

  return (
    <styles.container>
      <styles.SharedPostsMenu selected={selected} handleSelect={setSelected} />
      <styles.SharedPostsFilter selected={selected} />
      {selected === 'hasRoom' ? (
        <styles.posts>
          <PostCard />
          <PostCard />
          <PostCard />
          <PostCard />
          <PostCard />
        </styles.posts>
      ) : (
        <styles.cards>
          <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
          <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
          <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
          <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
          <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
        </styles.cards>
      )}
    </styles.container>
  );
}
