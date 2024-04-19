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
import org.hibernate.annotations.ColumnDefault;
import org.springframework.data.domain.Persistable;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString(callSuper = true, exclude = {
    "myCard",
    "mateCard",
    "followers",
    "followings",
    "profileImage"
})
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

    @Column(nullable = false)
    @ColumnDefault("'TRUE'")
    private Boolean recommendOn;

    @OneToOne(fetch = FetchType.LAZY, cascade = CascadeType.ALL)
    @JoinColumn(
        name = "myCardId",
        referencedColumnName = "feature_card_id",
        nullable = false
    )
    private FeatureCard myCard;

    @OneToOne(fetch = FetchType.LAZY, cascade = CascadeType.ALL)
    @JoinColumn(
        name = "mateCardId",
        referencedColumnName = "feature_card_id",
        nullable = false
    )
    private FeatureCard mateCard;

    @OneToMany(mappedBy = "following", cascade = CascadeType.PERSIST)
    private Set<Follow> followers;

    @OneToMany(mappedBy = "follower", cascade = CascadeType.PERSIST)
    private Set<Follow> followings;

    @OneToOne(fetch = FetchType.LAZY, cascade = CascadeType.PERSIST)
    @JoinColumn(
        name = "profile_image_id",
        referencedColumnName = "file_name"
    )
    private ProfileImage profileImage;

    private MemberAccount(
        String memberId,
        String email,
        String nickname,
        String birthYear,
        String gender,
        String phoneNumber,
        String createdBy,
        Boolean initialized,
        Boolean recommendOn,
        FeatureCard myCard,
        FeatureCard mateCard,
        Set<Follow> followers,
        Set<Follow> followings,
        ProfileImage profileImage
    ) {
        this.memberId = memberId;
        this.email = email;
        this.nickname = nickname;
        this.birthYear = birthYear;
        this.gender = gender;
        this.phoneNumber = phoneNumber;
        this.createdBy = createdBy;
        this.modifiedBy = createdBy;

        this.initialized = initialized;

        this.recommendOn = recommendOn;

        this.myCard = myCard;
        this.mateCard = mateCard;

        this.followers = followers;
        this.followings = followings;
        this.profileImage = profileImage;
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
            null,
            true,
            true,
            FeatureCard.of(null, List.of()),
            FeatureCard.of(null, List.of()),
            new HashSet<>(),
            new HashSet<>(),
            ProfileImage.defaultImage(memberId)
        );
    }

    public static MemberAccount of(
        String memberId,
        String email,
        String nickname,
        String birthYear,
        String gender,
        String phoneNumber,
        String createdBy,
        Boolean initialized,
        Boolean recommendOn,
        FeatureCard myCard,
        FeatureCard mateCard,
        Set<Follow> followers,
        Set<Follow> followings,
        ProfileImage profileImage
    ) {
        return new MemberAccount(
            memberId,
            email,
            nickname,
            birthYear,
            gender,
            phoneNumber,
            createdBy,
            initialized,
            recommendOn,
            myCard,
            mateCard,
            followers,
            followings,
            profileImage
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
       false면 user의 특성이 있는 것으로 판단
       true면 user의 특성이 없는 것으로 판단
     */
    public void updateInitialized(List<String> myFeatures) {
        /*
        feature 가 없는 경우
        */
        if (myFeatures == null || myFeatures.isEmpty()) {
            this.initialized = true;
            return;
        }
        this.initialized = false;
    }

    public void updateProfileImage(ProfileImage profileImage) {
        this.profileImage = profileImage;
    }

    public Boolean updateRecommendOn(Boolean recommendOn) {
        if (recommendOn == null) {
            return this.recommendOn;
        }
        this.recommendOn = recommendOn;
        return this.recommendOn;
    }
}
