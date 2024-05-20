package org.capstone.maru.domain;

import jakarta.persistence.DiscriminatorValue;
import jakarta.persistence.Entity;
import java.util.List;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.capstone.maru.dto.DormitoryRoomPostDto;
import org.capstone.maru.dto.FeatureCardDto;
import org.capstone.maru.dto.RoomImageDto;
import org.hibernate.annotations.DynamicUpdate;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@DiscriminatorValue("D")
@DynamicUpdate
@Entity
public class DormitoryRoomPost extends SharedRoomPost {

    // -- 생성자 메서드 -- //
    private DormitoryRoomPost(String title, String content, String publisherGender,
        Short recruitmentCapacity, MemberAccount publisherAccount, FeatureCard roomMateCard,
        Address address) {
        super(title, content, publisherGender, recruitmentCapacity, publisherAccount, roomMateCard,
            address);
    }

    public static DormitoryRoomPost of(
        String title,
        String content,
        String publisherGender,
        Short recruitmentCapacity,
        MemberAccount publisherAccount,
        FeatureCard roomMateCard,
        Address address
    ) {
        return new DormitoryRoomPost(
            title, content, publisherGender, recruitmentCapacity, publisherAccount, roomMateCard,
            address
        );
    }

    // -- 비지니스 로직 -- //
    public void updateDormitoryRoomPost(
        DormitoryRoomPostDto dormitoryRoomPostDto,
        FeatureCardDto roomMateCardDto,
        List<RoomImageDto> roomImagesDto
    ) {
        super.updateSharedRoomPost(
            dormitoryRoomPostDto.title(),
            dormitoryRoomPostDto.content(),
            roomMateCardDto.toEntity(),
            dormitoryRoomPostDto.address(),
            dormitoryRoomPostDto.recruitmentCapacity()
        );
        this.updateRoomImages(roomImagesDto);
    }
}
