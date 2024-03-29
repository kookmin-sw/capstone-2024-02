package org.capstone.maru.domain;

import jakarta.persistence.Column;
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

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString(exclude = {"studioRoomPost"})
@Entity
public class RoomImage extends AuditingFields {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column
    private String fileName;

    @Column
    private String storeImagePath;

    @Column
    private Boolean isThumbnail;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "studio_room_post_id")
    private StudioRoomPost studioRoomPost;

    // -- 생성자 메서드 -- //
    private RoomImage(String fileName, String storeImagePath,
        Boolean isThumbnail, StudioRoomPost studioRoomPost) {
        this.fileName = fileName;
        this.storeImagePath = storeImagePath;
        this.isThumbnail = isThumbnail;
        this.studioRoomPost = studioRoomPost;
    }

    public static RoomImage of(
        String fileName,
        String storeImagePath,
        Boolean isThumbnail,
        StudioRoomPost studioRoomPost
    ) {
        return new RoomImage(
            fileName, storeImagePath, isThumbnail, studioRoomPost
        );
    }

    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof RoomImage roomImage)) {
            return false;
        }
        return id != null && id.equals(roomImage.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
