package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.DiscriminatorValue;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.springframework.core.annotation.AnnotationUtils;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString(callSuper = true, exclude = {"studioRoomPost"})
@Entity
public class RoomImage extends Image {

    @Column
    private Boolean isThumbnail;

    @Column
    private Short orderNumber;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "studio_room_post_id", nullable = false)
    private StudioRoomPost studioRoomPost;

    // -- 생성자 메서드 -- //
    private RoomImage(String fileName, Boolean isThumbnail, Short orderNumber,
        StudioRoomPost studioRoomPost) {
        super(fileName);
        this.isThumbnail = isThumbnail;
        this.orderNumber = orderNumber;
        this.studioRoomPost = studioRoomPost;
    }

    public static RoomImage of(
        String fileName,
        Boolean isThumbnail,
        Short orderNumber,
        StudioRoomPost studioRoomPost
    ) {
        RoomImage result = new RoomImage(
            fileName, isThumbnail, orderNumber, studioRoomPost
        );

        studioRoomPost.addRoomImage(result);
        return result;
    }

    // -- 비지니스 로직 -- //

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
