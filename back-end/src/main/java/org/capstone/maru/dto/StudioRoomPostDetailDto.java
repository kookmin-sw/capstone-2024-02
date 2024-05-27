package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.Builder;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.Participation;
import org.capstone.maru.domain.StudioRoomPost;

@Builder
public record StudioRoomPostDetailDto(
    Long id,
    String title,
    String content,
    String publisherGender,
    FeatureCardDto roomMateCard,
    List<SimpleMemberProfileDto> participants,
    Set<RoomImageDto> roomImages,
    MemberAccountDto publisherAccount,
    Address address,
    RoomInfoDto roomInfo,
    Short recruitmentCapacity,
    Boolean isScrapped,
    Long scrapCount,
    Long viewCount,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static StudioRoomPostDetailDto from(
        StudioRoomPost entity,
        Boolean isPostScrapped,
        List<String> followingIds,
        Long scrapCount,
        Long viewCount
    ) {
        return StudioRoomPostDetailDto
            .builder()
            .id(entity.getId())
            .title(entity.getTitle())
            .content(entity.getContent())
            .roomMateCard(FeatureCardDto.from(entity.getRoomMateCard()))
            .participants(
                entity.getSharedRoomPostRecruits()
                      .stream()
                      .map(Participation::getRecruitedMemberAccount)
                      .map(memberAccount ->
                          SimpleMemberProfileDto
                              .from(
                                  memberAccount,
                                  followingIds
                                      .stream()
                                      .anyMatch(memberId -> Objects.equals(memberId,
                                          memberAccount.getMemberId()))
                              )
                      )
                      .toList()
            )
            .roomImages(
                entity.getRoomImages()
                      .stream()
                      .map(RoomImageDto::from)
                      .collect(Collectors.toSet())
            )
            .publisherGender(entity.getPublisherGender())
            .publisherAccount(MemberAccountDto.from(entity.getPublisherAccount()))
            .address(entity.getAddress())
            .roomInfo(RoomInfoDto.from(entity.getRoomInfo()))
            .recruitmentCapacity(entity.getRecruitmentCapacity())
            .isScrapped(isPostScrapped)
            .scrapCount(scrapCount)
            .viewCount(viewCount)
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .build();
    }
}