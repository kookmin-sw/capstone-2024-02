package org.capstone.maru.service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
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

        List<Follow> followingList = followerAccount.getFollowings().stream().toList();
        followingList.forEach(follow -> follow
            .getFollowing()
            .getProfileImage()
            .updateFileName(
                s3FileService.getMemberPreSignedUrlForLoad(
                    followerAccount.getGender(),
                    follow.getFollowing().getProfileImage().getFileName())
            )
        );

        return FollowingDto.fromFollow(followingList);
    }

    @Transactional(readOnly = true)
    public FollowingDto getMutualFollower(String memberId) {
        List<MemberAccount> followingList = followRepository.findAllMutualFollower(memberId)
            .stream().toList();
        followingList.forEach(memberAccount -> memberAccount
            .getProfileImage()
            .updateFileName(
                s3FileService.getMemberPreSignedUrlForLoad(
                    memberAccount.getGender(),
                    memberAccount.getProfileImage().getFileName()
                )
            )
        );

        return FollowingDto.fromMemberAccount(
            followingList
        );
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
