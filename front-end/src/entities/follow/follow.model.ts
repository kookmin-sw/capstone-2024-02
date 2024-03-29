import { type MemberAccount } from '../member-account';

export interface Follow {
  follower: MemberAccount;
  following: MemberAccount;
}
