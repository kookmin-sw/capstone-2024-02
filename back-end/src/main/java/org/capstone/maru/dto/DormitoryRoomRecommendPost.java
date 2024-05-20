package org.capstone.maru.dto;

import com.querydsl.core.annotations.QueryProjection;
import java.time.LocalDateTime;
import java.util.List;
import lombok.Getter;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.DormitoryRoomPost;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.Recommend;
import org.capstone.maru.domain.RoomImage;

@Getter
public class DormitoryRoomRecommendPost {

    private final Long id;
    private final String title;
    private final String content;
    private final String publisherGender;
    private final List<RoomImage> roomImages;
    private final MemberAccount publisherAccount;
    private final Address address;
    private final Short recruitmentCapacity;
    private final LocalDateTime createdAt;
    private final String createdBy;
    private final LocalDateTime modifiedAt;
    private final String modifiedBy;
    private final Integer score;

    @QueryProjection
    public DormitoryRoomRecommendPost(
        DormitoryRoomPost dormitoryRoomPost,
        Recommend recommend
    ) {
        this.id = dormitoryRoomPost.getId();
        this.title = dormitoryRoomPost.getTitle();
        this.content = dormitoryRoomPost.getContent();
        this.publisherGender = dormitoryRoomPost.getPublisherGender();
        this.roomImages = dormitoryRoomPost.getRoomImages();
        this.publisherAccount = dormitoryRoomPost.getPublisherAccount();
        this.address = dormitoryRoomPost.getAddress();
        this.recruitmentCapacity = dormitoryRoomPost.getRecruitmentCapacity();
        this.createdAt = dormitoryRoomPost.getCreatedAt();
        this.createdBy = dormitoryRoomPost.getCreatedBy();
        this.modifiedAt = dormitoryRoomPost.getModifiedAt();
        this.modifiedBy = dormitoryRoomPost.getModifiedBy();
        this.score = recommend.getScore();
    }
}
