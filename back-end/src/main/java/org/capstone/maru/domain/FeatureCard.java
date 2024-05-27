package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Convert;
import jakarta.persistence.Embedded;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import java.util.List;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.capstone.maru.domain.converter.MemberFeaturesConverter;
import org.capstone.maru.domain.jsonb.MemberFeatures;
import org.hibernate.annotations.ColumnDefault;
import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

@DynamicInsert
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity
public class FeatureCard {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "feature_card_id", nullable = false)
    private Long id;

    /*
      회원 희망 지역
    */
    @Column(name = "location", length = 50)
    @ColumnDefault("'default'")
    private String location;

    @JdbcTypeCode(SqlTypes.JSON)
    @Embedded
    private MemberFeatures memberFeatures;

    private FeatureCard(Long memberCardId, MemberFeatures memberFeatures) {
        this.id = memberCardId;
        this.memberFeatures = memberFeatures;
    }

    private FeatureCard(String location, MemberFeatures memberFeatures) {
        this.location = location;
        this.memberFeatures = memberFeatures;
    }

    public void updateFeatureCard(FeatureCard featureCard) {
        updateMemberFeatures(featureCard.memberFeatures);
        updateLocation(featureCard.location);
    }

    public void updateMemberFeatures(MemberFeatures memberFeatures) {
        if (Objects.equals(this.memberFeatures, memberFeatures)) {
            return;
        }
        this.memberFeatures = memberFeatures;
    }

    public void updateLocation(String location) {
        if (location == null || location.equals(this.location)) {
            return;
        }
        this.location = location;
    }

    public static FeatureCard of(String location, MemberFeatures memberFeatures) {
        return new FeatureCard(location, memberFeatures);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof FeatureCard that)) {
            return false;
        }
        return this.getId() != null && this.getId().equals(that.getId());
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.getId());
    }

}
