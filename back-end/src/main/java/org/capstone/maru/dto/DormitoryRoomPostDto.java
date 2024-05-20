package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;
import lombok.Builder;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.DormitoryRoomPost;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.repository.postgre.projection.ScrapPostView;

@Builder
public record DormitoryRoomPostDto(
    Long id,
    String title,
    String content,
    String publisherGender,
    List<RoomImageDto> roomImages,
    MemberAccountDto publisherAccount,
    Address address,
    Short recruitmentCapacity,
    Boolean isScrapped,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static DormitoryRoomPostDto from(DormitoryRoomPost entity,
        List<ScrapPostView> scrapViewEntity) {
        return DormitoryRoomPostDto
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

    public DormitoryRoomPost toEntity(
        FeatureCard roomMateCardEntity,
        MemberAccount publisherAccountEntity
    ) {
        return DormitoryRoomPost.of(
            title,
            content,
            publisherGender,
            recruitmentCapacity,
            publisherAccountEntity,
            roomMateCardEntity,
            address
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
