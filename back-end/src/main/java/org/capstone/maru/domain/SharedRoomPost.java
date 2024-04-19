package org.capstone.maru.domain;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.DiscriminatorColumn;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Inheritance;
import jakarta.persistence.InheritanceType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString(callSuper = true, exclude = {"roomMateCard", "sharedRoomPostRecruits"})
@Table(indexes = {
    @Index(columnList = "id", unique = true),
    @Index(columnList = "title"),
    @Index(columnList = "publisherGender"),
    @Index(columnList = "createdAt"),
    @Index(columnList = "createdBy")
})
@Inheritance(strategy = InheritanceType.SINGLE_TABLE)
@DiscriminatorColumn
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

    @Column
    private String publisherGender;

    @OneToOne(fetch = FetchType.LAZY, cascade = CascadeType.ALL)
    @JoinColumn(name = "room_mate_card_id", nullable = false)
    private FeatureCard roomMateCard;

    @OneToMany(mappedBy = "recruitsSharedRoomPost", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private final List<Participation> sharedRoomPostRecruits = new ArrayList<>();


    // -- 생성자 메서드 -- //
    protected SharedRoomPost(String title, String content, String publisherGender,
        FeatureCard roomMateCard) {
        this.title = title;
        this.content = content;
        this.publisherGender = publisherGender;
        this.roomMateCard = roomMateCard;
    }

    // -- 연관관계 편의 메서드 -- //
    public void addSharedRoomPostRecruits(Participation participation) {
        this.sharedRoomPostRecruits.add(participation);
    }

    // -- 비지니스 로직 -- //

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
