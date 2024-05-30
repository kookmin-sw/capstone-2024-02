package org.capstone.maru.dto;

import java.util.List;
import java.util.Map;
import lombok.Builder;
import org.capstone.maru.domain.Follow;
import org.capstone.maru.domain.MemberAccount;

@Builder
public record FollowingDto(
    List<SimpleMemberInfoDto> followingList
) {

    record SimpleMemberInfoDto(
        String memberId,
        String nickname,
        String profileImage
    ) {

    }

    public static FollowingDto fromFollow(List<Follow> followingList) {
        return FollowingDto
            .builder()
            .followingList(
                followingList
                    .stream()
                    .map(
                        follow ->
                            new SimpleMemberInfoDto(
                                follow.getFollowing().getMemberId(),
                                follow.getFollowing().getNickname(),
                                follow.getFollowing().getProfileImage().getFileName()
                            )
                    )
                    .toList()
            )
            .build();
    }

    public static FollowingDto fromMemberAccount(List<MemberAccount> followingList) {
        return FollowingDto
            .builder()
            .followingList(
                followingList
                    .stream()
                    .map(
                        memberAccount ->
                            new SimpleMemberInfoDto(
                                memberAccount.getMemberId(),
                                memberAccount.getNickname(),
                                memberAccount.getProfileImage().getFileName()
                            )
                    )
                    .toList()
            )
            .build();
    }
}
