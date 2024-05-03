package org.capstone.maru.service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.Follow;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.dto.FollowingDto;
import org.capstone.maru.repository.postgre.FollowRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@RequiredArgsConstructor
@Service
public class FollowService {

    private final MemberAccountService memberAccountService;

    private final S3FileService s3FileService;

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

        log.info("followerAccount: {}", followerAccount.getFollowings());

        Map<String, List<String>> followingList = followerAccount
            .getFollowings().stream().collect(
                Collectors.toMap(
                    follow -> follow.getFollowing().getMemberId(),
                    follow -> List.of(
                        follow.getFollowing().getNickname(),
                        s3FileService.getPreSignedUrlForLoad(
                            follow.getFollowing().getProfileImage().getFileName()))));

        log.info("followingList: {}", followingList);

        return FollowingDto.from(followingList);
    }

    public void unfollowUser(String memberId, String followingMemberId) {
        MemberAccount followerAccount = memberAccountService.searchMemberAccount(memberId);
        MemberAccount followingAccount = memberAccountService.searchMemberAccount(
            followingMemberId);

        Follow follow = followRepository.findByFollowerAndFollowing(followerAccount,
            followingAccount).orElseThrow(() -> new IllegalArgumentException("팔로우 관계가 존재하지 않습니다."));

        followRepository.delete(follow);
    }
}
