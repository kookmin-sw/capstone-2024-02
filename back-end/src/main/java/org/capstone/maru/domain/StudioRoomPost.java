package org.capstone.maru.domain;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.CascadeType;
import jakarta.persistence.DiscriminatorValue;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OneToOne;
import jakarta.persistence.OrderBy;

import java.util.ArrayList;
import java.util.List;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.hibernate.annotations.BatchSize;
import org.hibernate.annotations.OnDelete;
import org.hibernate.annotations.OnDeleteAction;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@DiscriminatorValue("S")
@Entity
public class StudioRoomPost extends SharedRoomPost {

    @OneToMany(mappedBy = "studioRoomPost", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    @OrderBy("orderNumber ASC")
    private final List<RoomImage> roomImages = new ArrayList<>();

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "publisher_id", nullable = false)
    @OnDelete(action = OnDeleteAction.CASCADE)
    @JsonIgnore
    private MemberAccount publisherAccount;

    @OneToOne(fetch = FetchType.LAZY, cascade = CascadeType.ALL)
    @JoinColumn(name = "room_info_id", nullable = false)
    private RoomInfo roomInfo;

    // -- 생성자 메서드 -- //
    private StudioRoomPost(
        String title, String content, String publisherGender, FeatureCard roomMateCard,
        MemberAccount publisherAccount, RoomInfo roomInfo
    ) {
        super(title, content, publisherGender, roomMateCard);
        this.publisherAccount = publisherAccount;
        this.roomInfo = roomInfo;
    }

    public static StudioRoomPost of(
        String title,
        String content,
        String publisherGender,
        FeatureCard roomMateCard,
        MemberAccount publisherAccount,
        RoomInfo roomInfo
    ) {
        return new StudioRoomPost(
            title, content, publisherGender, roomMateCard, publisherAccount, roomInfo
        );
    }

    // -- 비지니스 로직 -- //
    public void addRoomImage(RoomImage roomImage) {
        this.roomImages.add(roomImage);
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
