package org.capstone.maru.service;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import lombok.RequiredArgsConstructor;
import org.capstone.maru.domain.Follow;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.dto.FollowingDto;
import org.capstone.maru.repository.FollowRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@RequiredArgsConstructor
@Service
public class FollowService {

    private final MemberAccountService memberAccountService;

    private final FollowRepository followRepository;

    /*
     * 팔로우 기능
     * follower가 following을 팔로우합니다.
     */
    @Transactional
    public void followUser(String follower, String following) {
        MemberAccount followerAccount = memberAccountService.searchMemberAccount(follower);
        MemberAccount followingAccount = memberAccountService.searchMemberAccount(following);

        Follow follow = new Follow(followerAccount, followingAccount);
        followRepository.save(follow);
    }

    @Transactional(readOnly = true)
    public FollowingDto getFollowings(String follower) {
        MemberAccount followerAccount = memberAccountService.searchMemberAccount(follower);

        Map<String, String> followingList = followerAccount
            .getFollowings().stream().collect(Collectors.toMap(
                follow -> follow.getFollowing().getMemberId(),
                follow -> follow.getFollowing().getNickname()
            ));

        return FollowingDto.from(followingList);
    }
}
