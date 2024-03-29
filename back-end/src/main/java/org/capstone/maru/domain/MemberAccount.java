package org.capstone.maru.domain;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.springframework.data.domain.Persistable;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString(callSuper = true, exclude = {"myCard", "mateCard", "followers", "followings"})
@Table(indexes = {
    @Index(columnList = "memberId", unique = true),
    @Index(columnList = "email", unique = true),
    @Index(columnList = "createdAt"),
    @Index(columnList = "createdBy")
})
@Entity
public class MemberAccount extends AuditingFields implements Persistable<String> {

    @Id
    @Column(nullable = false, length = 100)
    private String memberId;

    @Column(nullable = false, length = 50)
    private String email;

    @Column(nullable = false, length = 50)
    private String nickname;

    @Column(nullable = false)
    private String birthYear;

    @Column(nullable = false)
    private String gender;

    @Column
    private String phoneNumber;

    @Column(nullable = false)
    private Boolean initialized;

    @OneToOne(fetch = FetchType.LAZY, cascade = CascadeType.PERSIST)
    @JoinColumn(
        name = "myCardId",
        referencedColumnName = "member_card_id",
        nullable = false
    )
    private MemberCard myCard;

    @OneToOne(fetch = FetchType.LAZY, cascade = CascadeType.PERSIST)
    @JoinColumn(
        name = "mateCardId",
        referencedColumnName = "member_card_id",
        nullable = false
    )
    private MemberCard mateCard;

    @OneToMany(mappedBy = "following", cascade = CascadeType.PERSIST)
    private Set<Follow> followers;

    @OneToMany(mappedBy = "follower", cascade = CascadeType.PERSIST)
    private Set<Follow> followings;

    private MemberAccount(
        String memberId,
        String email,
        String nickname,
        String birthYear,
        String gender,
        String phoneNumber,
        String createdBy
    ) {
        this.memberId = memberId;
        this.email = email;
        this.nickname = nickname;
        this.birthYear = birthYear;
        this.gender = gender;
        this.phoneNumber = phoneNumber;
        this.createdBy = createdBy;
        this.modifiedBy = createdBy;
        this.initialized = true;

        this.myCard = new MemberCard(List.of());
        this.mateCard = new MemberCard(List.of());

        this.followers = new HashSet<>();
        this.followings = new HashSet<>();
    }

    public static MemberAccount of(
        String memberId,
        String email,
        String nickname,
        String birthYear,
        String gender,
        String phoneNumber
    ) {
        return new MemberAccount(
            memberId,
            email,
            nickname,
            birthYear,
            gender,
            phoneNumber,
            null
        );
    }

    public static MemberAccount of(
        String memberId,
        String email,
        String nickname,
        String birthYear,
        String gender,
        String phoneNumber,
        String createdBy
    ) {
        return new MemberAccount(
            memberId,
            email,
            nickname,
            birthYear,
            gender,
            phoneNumber,
            createdBy
        );
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof MemberAccount that)) {
            return false;
        }
        return this.getMemberId() != null && this.getMemberId().equals(that.getMemberId());
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.getMemberId());
    }

    @Override
    public String getId() {
        return memberId;
    }

    @Override
    public boolean isNew() {
        return getCreatedAt() == null;
    }

    // -- 비즈니스 로직 -- //

    /*
        특성이 없는 경우는 initialized를 false로 변경
        따라서 user를 특성을 입력하는 곳으로 이동
     */
    public void updateInitialized(List<String> myFeatures) {
        if (myFeatures == null || myFeatures.isEmpty()) {
            this.initialized = false;
            return;
        }
        this.initialized = true;
    }
}
