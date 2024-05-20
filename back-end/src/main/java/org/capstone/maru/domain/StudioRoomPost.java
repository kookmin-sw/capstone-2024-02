package org.capstone.maru.domain;

import jakarta.persistence.CascadeType;
import jakarta.persistence.DiscriminatorValue;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToOne;

import java.util.List;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.capstone.maru.dto.FeatureCardDto;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.RoomInfoDto;
import org.capstone.maru.dto.StudioRoomPostDto;
import org.hibernate.annotations.DynamicUpdate;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@DiscriminatorValue("S")
@DynamicUpdate
@Entity
public class StudioRoomPost extends SharedRoomPost {

    @OneToOne(fetch = FetchType.LAZY, cascade = CascadeType.ALL)
    @JoinColumn(name = "room_info_id")
    private RoomInfo roomInfo;

    // -- 생성자 메서드 -- //
    private StudioRoomPost(String title, String content, String publisherGender,
        Short recruitmentCapacity,
        MemberAccount publisherAccount, FeatureCard roomMateCard, Address address,
        RoomInfo roomInfo) {
        super(title, content, publisherGender, recruitmentCapacity, publisherAccount, roomMateCard,
            address);
        this.roomInfo = roomInfo;
    }

    public static StudioRoomPost of(
        String title,
        String content,
        String publisherGender,
        Short recruitmentCapacity,
        MemberAccount publisherAccount,
        FeatureCard roomMateCard,
        Address address,
        RoomInfo roomInfo
    ) {
        return new StudioRoomPost(
            title, content, publisherGender, recruitmentCapacity, publisherAccount, roomMateCard,
            address, roomInfo
        );
    }

    // -- 비지니스 로직 -- //
    public void updateStudioRoomPost(
        StudioRoomPostDto studioRoomPostDto,
        FeatureCardDto roomMateCardDto,
        RoomInfoDto roomInfoDto,
        List<RoomImageDto> roomImagesDto
    ) {
        super.updateSharedRoomPost(
            studioRoomPostDto.title(),
            studioRoomPostDto.content(),
            roomMateCardDto.toEntity(),
            studioRoomPostDto.address(),
            studioRoomPostDto.recruitmentCapacity()
        );
        this.roomInfo.updateRoomInfo(roomInfoDto);
        this.updateRoomImages(roomImagesDto);
    }


    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        return super.equals(o);
    }

    @Override
    public int hashCode() {
        return super.hashCode();
    }
}
