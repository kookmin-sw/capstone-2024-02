package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.Builder;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.Participation;
import org.capstone.maru.domain.StudioRoomPost;

@Builder
public record StudioRoomPostDetailDto(
    Long id,
    String title,
    String content,
    String publisherGender,
    MemberCardDto roomMateCard,
    List<MemberAccountDto> participants,
    Set<RoomImageDto> roomImages,
    MemberAccountDto publisherAccount,
    RoomInfoDto roomInfo,
    Boolean isScrapped,
    Long scrapCount,
    Long viewCount,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static StudioRoomPostDetailDto from(StudioRoomPost entity,
        Boolean isScrapped,
        Long scrapCount, Long viewCount) {
        return StudioRoomPostDetailDto
            .builder()
            .id(entity.getId())
            .title(entity.getTitle())
            .content(entity.getContent())
            .roomMateCard(MemberCardDto.from(entity.getRoomMateCard()))
            .participants(
                entity.getSharedRoomPostRecruits()
                      .stream()
                      .map(Participation::getRecruitedMemberAccount)
                      .map(MemberAccountDto::from)
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
            .roomInfo(RoomInfoDto.from(entity.getRoomInfo()))
            .isScrapped(isScrapped)
            .scrapCount(scrapCount)
            .viewCount(viewCount)
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .build();
    }
}