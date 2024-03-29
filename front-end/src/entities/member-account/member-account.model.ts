import { type Follow } from '../follow';
import { type MemberCard } from '../member-card';

export interface MemberAccount {
  memberId: string;
  email: string;
  nickname: string;
  birthYear: string;
  gender: string;
  phoneNumber: string;
  initialized: boolean;
  myCard: MemberCard;
  mateCard: MemberCard;
  followers: Follow[];
  followings: Follow[];
}
