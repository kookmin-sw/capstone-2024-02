package org.capstone.maru.domain;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.DiscriminatorColumn;
import jakarta.persistence.Embedded;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Inheritance;
import jakarta.persistence.InheritanceType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OneToOne;
import jakarta.persistence.OrderBy;
import jakarta.persistence.Table;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import java.util.stream.Collectors;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import org.capstone.maru.dto.RoomImageDto;
import org.hibernate.annotations.DynamicUpdate;
import org.hibernate.annotations.OnDelete;
import org.hibernate.annotations.OnDeleteAction;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString(callSuper = true, exclude = {"roomMateCard", "sharedRoomPostRecruits", "roomImages",
    "publisherAccount"})
@Table(indexes = {
    @Index(columnList = "id", unique = true),
    @Index(columnList = "title"),
    @Index(columnList = "publisherGender"),
    @Index(columnList = "createdAt"),
    @Index(columnList = "createdBy")
})
@Inheritance(strategy = InheritanceType.SINGLE_TABLE)
@DiscriminatorColumn
@DynamicUpdate
@Entity
public abstract class SharedRoomPost extends AuditingFields {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Setter
    @Column(nullable = false, length = 50)
    private String title;

    @Setter
    @Column(columnDefinition = "TEXT")
    private String content;

    @Column(updatable = false)
    private String publisherGender;

    @Column
    private Short recruitmentCapacity;

    @OneToMany(
        mappedBy = "sharedRoomPost",
        cascade = CascadeType.ALL,
        fetch = FetchType.LAZY,
        orphanRemoval = true
    )
    @OrderBy("orderNumber ASC")
    private final List<RoomImage> roomImages = new ArrayList<>();

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "publisher_id", nullable = false, updatable = false)
    @OnDelete(action = OnDeleteAction.CASCADE)
    @JsonIgnore
    private MemberAccount publisherAccount;

    @OneToOne(fetch = FetchType.LAZY, cascade = CascadeType.ALL)
    @JoinColumn(name = "room_mate_card_id", nullable = false)
    private FeatureCard roomMateCard;

    @Embedded
    private Address address;

    @OneToMany(
        mappedBy = "recruitsSharedRoomPost",
        cascade = CascadeType.ALL,
        fetch = FetchType.LAZY,
        orphanRemoval = true
    )
    private final List<Participation> sharedRoomPostRecruits = new ArrayList<>();


    // -- 생성자 메서드 -- //
    protected SharedRoomPost(String title, String content, String publisherGender,
        Short recruitmentCapacity,
        MemberAccount publisherAccount, FeatureCard roomMateCard, Address address) {
        this.title = title;
        this.content = content;
        this.publisherGender = publisherGender;
        this.recruitmentCapacity = recruitmentCapacity;
        this.publisherAccount = publisherAccount;
        this.roomMateCard = roomMateCard;
        this.address = address;
    }


    // -- 연관관계 편의 메서드 -- //
    private void addRoomImage(RoomImage roomImage) {
        this.roomImages.add(roomImage);
    }

    private void removeRoomImage(RoomImage roomImage) {
        this.roomImages.remove(roomImage);
        roomImage.updateSharedRoomPost(null);
    }

    public void addRoomImages(List<RoomImageDto> roomImages) {
        for (RoomImageDto roomImage : roomImages) {
            addRoomImage(roomImage.toEntity(this));
        }
    }

    public void addSharedRoomPostRecruits(Participation participation) {
        this.sharedRoomPostRecruits.add(participation);
    }


    // -- 비지니스 로직 -- //
    public void updateRoomImages(List<RoomImageDto> roomImages) {
        Map<String, RoomImage> existingRoomImages = this.roomImages
            .stream()
            .collect(Collectors.toMap(RoomImage::getFileName, roomImage -> roomImage));

        for (RoomImageDto dto : roomImages) {
            if (existingRoomImages.containsKey(dto.fileName())) {
                RoomImage roomImage = existingRoomImages.get(dto.fileName());
                roomImage.updateOrderNumber(dto.order());
                roomImage.updateIsThumbNail(dto.isThumbnail());
                existingRoomImages.remove(dto.fileName());
            } else {
                addRoomImage(dto.toEntity(this));
            }
        }

        existingRoomImages.values().forEach(this::removeRoomImage);
    }

    public void updateSharedRoomPost(String title, String content, FeatureCard roomMateCard,
        Address address, Short recruitmentCapacity) {
        updateTitle(title);
        updateContent(content);
        updateRoomMateCard(roomMateCard);
        updateAddress(address);
        updateRecruitmentCapacity(recruitmentCapacity);
    }

    private void updateTitle(String title) {
        if (title.equals(this.title)) {
            return;
        }
        this.title = title;
    }

    private void updateContent(String content) {
        if (content.equals(this.content)) {
            return;
        }
        this.content = content;
    }

    private void updateRoomMateCard(FeatureCard roomMateCard) {
        this.roomMateCard.updateFeatureCard(roomMateCard);
    }

    private void updateAddress(Address address) {
        if (Objects.equals(address, this.address)) {
            return;
        }
        this.address = address;
    }

    private void updateRecruitmentCapacity(Short recruitmentCapacity) {
        if (Objects.equals(this.recruitmentCapacity, recruitmentCapacity)) {
            return;
        }
        this.recruitmentCapacity = recruitmentCapacity;
    }

    public void initParticipants() {
        if (this.sharedRoomPostRecruits.isEmpty()) {
            return;
        }

        for (Participation participation : this.sharedRoomPostRecruits) {
            participation.updateSharedRoomPost(null);
        }

        this.sharedRoomPostRecruits.clear();
    }

    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof SharedRoomPost sharedRoomPost)) {
            return false;
        }
        return id != null && id.equals(sharedRoomPost.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
