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
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.capstone.maru.domain.converter.MemberFeaturesConverter;
import org.hibernate.annotations.ColumnDefault;
import org.hibernate.annotations.DynamicInsert;

@DynamicInsert
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity
public class MemberCard {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "member_card_id", nullable = false)
    private Long id;

    /*
        회원 희망 지역
     */
    @Column(name = "location", length = 50)
    @ColumnDefault("'default'")
    private String location;

    @Convert(converter = MemberFeaturesConverter.class)
    private List<String> memberFeatures;

    public MemberCard(Long memberCardId, List<String> memberFeatures) {
        this.id = memberCardId;
        this.memberFeatures = memberFeatures;
    }

    public void updateMemberFeatures(List<String> memberFeatures) {
        this.memberFeatures = memberFeatures;
    }

    public void updateLocation(String location) {
        this.location = location;
    }

    public static MemberCard of(Long memberCardId, List<String> memberFeatures) {
        return new MemberCard(memberCardId, memberFeatures);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof MemberCard that)) {
            return false;
        }
        return this.getId() != null && this.getId().equals(that.getId());
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.getId());
    }

}
