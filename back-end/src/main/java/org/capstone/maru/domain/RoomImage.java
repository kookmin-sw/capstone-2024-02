package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.hibernate.annotations.DynamicUpdate;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString(callSuper = true, exclude = {"sharedRoomPost"})
@DynamicUpdate
@Entity
public class RoomImage extends Image {

    @Column
    private Boolean isThumbnail;

    @Column
    private Short orderNumber;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "shared_room_post_id", nullable = false)
    private SharedRoomPost sharedRoomPost;

    // -- 생성자 메서드 -- //
    private RoomImage(String fileName, Boolean isThumbnail, Short orderNumber,
        SharedRoomPost sharedRoomPost) {
        super(fileName);
        this.isThumbnail = isThumbnail;
        this.orderNumber = orderNumber;
        this.sharedRoomPost = sharedRoomPost;
    }

    public static RoomImage of(
        String fileName,
        Boolean isThumbnail,
        Short orderNumber,
        SharedRoomPost roomPost
    ) {
        return new RoomImage(
            fileName,
            isThumbnail,
            orderNumber,
            roomPost
        );
    }

    // -- 비지니스 로직 -- //
    public void updateIsThumbNail(Boolean isThumbnail) {
        if (Objects.equals(this.isThumbnail, isThumbnail)) {
            return;
        }
        this.isThumbnail = isThumbnail;
    }

    public void updateOrderNumber(Short orderNumber) {
        if (Objects.equals(this.orderNumber, orderNumber)) {
            return;
        }
        this.orderNumber = orderNumber;
    }

    public void updateSharedRoomPost(SharedRoomPost roomPost) {
        this.sharedRoomPost = roomPost;
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
