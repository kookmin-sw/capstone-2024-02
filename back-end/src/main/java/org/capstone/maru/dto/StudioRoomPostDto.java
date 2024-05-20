package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.RoomInfo;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.repository.postgre.projection.ScrapPostView;

@Slf4j
@Builder
public record StudioRoomPostDto(
    Long id,
    String title,
    String content,
    String publisherGender,
    List<RoomImageDto> roomImages,
    MemberAccountDto publisherAccount,
    Address address,
    RoomInfoDto roomInfo,
    Short recruitmentCapacity,
    Boolean isScrapped,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static StudioRoomPostDto from(StudioRoomPost entity,
        List<ScrapPostView> scrapViewEntity) {
        return StudioRoomPostDto
            .builder()
            .id(entity.getId())
            .title(entity.getTitle())
            .content(entity.getContent())
            .roomImages(
                entity.getRoomImages()
                    .stream()
                    .map(RoomImageDto::from)
                    .toList()
            )
            .publisherGender(entity.getPublisherGender())
            .publisherAccount(MemberAccountDto.from(entity.getPublisherAccount()))
            .address(entity.getAddress())
            .roomInfo(RoomInfoDto.from(entity.getRoomInfo()))
            .recruitmentCapacity(entity.getRecruitmentCapacity())
            .isScrapped(
                scrapViewEntity
                    .stream()
                    .filter(scrapPostView ->
                        Objects.equals(scrapPostView.getScrappedId(), entity.getId()))
                    .map(ScrapPostView::getIsScrapped)
                    .findAny()
                    .orElse(false)
            )
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .build();
    }

    public StudioRoomPost toEntity(
        FeatureCard roomMateCardEntity,
        MemberAccount publisherAccountEntity,
        RoomInfo roomInfoEntity
    ) {
        return StudioRoomPost.of(
            title,
            content,
            publisherGender,
            recruitmentCapacity,
            publisherAccountEntity,
            roomMateCardEntity,
            address,
            roomInfoEntity
        );
    }

    public RoomImageDto thumbnail() {
        return roomImages
            .stream()
            .filter(RoomImageDto::isThumbnail)
            .findAny()
            .orElseThrow(IllegalArgumentException::new);
    }
}