package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Convert;
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
import org.hibernate.annotations.ColumnDefault;
import org.hibernate.annotations.DynamicInsert;

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

    @Convert(converter = MemberFeaturesConverter.class)
    @Column(name = "features", length = 100, nullable = false)
    private List<String> memberFeatures;

    private FeatureCard(Long memberCardId, List<String> memberFeatures) {
        this.id = memberCardId;
        this.memberFeatures = memberFeatures;
    }

    private FeatureCard(List<String> memberFeatures) {
        this.memberFeatures = memberFeatures;
    }

    public void updateMemberFeatures(List<String> memberFeatures) {
        if (memberFeatures == null) {
            return;
        }
        this.memberFeatures = memberFeatures;
    }

    public void updateLocation(String location) {
        if (location == null) {
            return;
        }
        this.location = location;
    }

    public static FeatureCard of(Long memberCardId, List<String> memberFeatures) {
        return new FeatureCard(memberCardId, memberFeatures);
    }

    public static FeatureCard of(List<String> memberFeatures) {
        return new FeatureCard(memberFeatures);
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
