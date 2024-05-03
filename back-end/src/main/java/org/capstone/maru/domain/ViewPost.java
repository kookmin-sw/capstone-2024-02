package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Table;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(indexes = {
    @Index(columnList = "shared_room_post_id", unique = true)
})
@Entity
public class ViewPost {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column
    private Long sharedRoomPostId;

    @Column
    private Long viewCount;

    // -- 생성자 메서드 -- //
    private ViewPost(Long sharedRoomPostId, Long viewCount) {
        this.sharedRoomPostId = sharedRoomPostId;
        this.viewCount = viewCount;
    }

    public static ViewPost of(Long sharedRoomPostId, Long viewCount) {
        return new ViewPost(sharedRoomPostId, viewCount);
    }

    // -- 비지니스 로직 -- //
    public void updateViewCount(Long viewCount) {
        if (viewCount < 0) {
            return;
        }

        this.viewCount = viewCount;
    }

    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ViewPost viewPost)) {
            return false;
        }
        return id != null && id.equals(viewPost.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
