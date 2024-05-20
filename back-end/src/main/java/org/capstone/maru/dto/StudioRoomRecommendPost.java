package org.capstone.maru.dto;

import com.querydsl.core.annotations.QueryProjection;
import java.time.LocalDateTime;
import java.util.List;
import lombok.Getter;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.Recommend;
import org.capstone.maru.domain.RoomImage;
import org.capstone.maru.domain.RoomInfo;
import org.capstone.maru.domain.StudioRoomPost;

@Getter
public class StudioRoomRecommendPost {

    private final Long id;
    private final String title;
    private final String content;
    private final String publisherGender;
    private final List<RoomImage> roomImages;
    private final MemberAccount publisherAccount;
    private final Address address;
    private final RoomInfo roomInfo;
    private final Short recruitmentCapacity;
    private final LocalDateTime createdAt;
    private final String createdBy;
    private final LocalDateTime modifiedAt;
    private final String modifiedBy;
    private final Integer score;

    @QueryProjection
    public StudioRoomRecommendPost(
        StudioRoomPost studioRoomPost,
        Recommend recommend
    ) {
        this.id = studioRoomPost.getId();
        this.title = studioRoomPost.getTitle();
        this.content = studioRoomPost.getContent();
        this.publisherGender = studioRoomPost.getPublisherGender();
        this.roomImages = studioRoomPost.getRoomImages();
        this.publisherAccount = studioRoomPost.getPublisherAccount();
        this.address = studioRoomPost.getAddress();
        this.recruitmentCapacity = studioRoomPost.getRecruitmentCapacity();
        this.roomInfo = studioRoomPost.getRoomInfo();
        this.createdAt = studioRoomPost.getCreatedAt();
        this.createdBy = studioRoomPost.getCreatedBy();
        this.modifiedAt = studioRoomPost.getModifiedAt();
        this.modifiedBy = studioRoomPost.getModifiedBy();
        this.score = recommend.getScore();
    }


}
