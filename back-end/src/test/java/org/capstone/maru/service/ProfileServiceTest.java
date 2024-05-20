package org.capstone.maru.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.ProfileImage;
import org.capstone.maru.dto.MemberProfileDto;
import org.capstone.maru.repository.postgre.MemberAccountRepository;
import org.capstone.maru.repository.postgre.MemberCardRepository;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.util.EntityCreator;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@DisplayName("Service - 프로필")
@ExtendWith(MockitoExtension.class)
class ProfileServiceTest {

    @InjectMocks
    ProfileService sut;
    @Mock
    MemberAccountService memberAccountService;
    @Mock
    S3FileService s3FileService;

    @Mock
    MemberCardRepository memberCardRepository;

    @Mock
    MemberAccountRepository memberAccountRepository;

    @DisplayName("내 프로필 조회 할때, 내 프로필, 내 카드, 메이트 카드를 반환한다.")
    @Test
    void givenMemberId_whenSearching_thenReturnsMemberProfile() {
        // given
        String memberId = "test_1";
        MemberAccount memberAccount = EntityCreator.createMemberAccount(1);
        MemberPrincipal memberPrincipal = createMemberPrincipal(memberId);

        // when
        when(memberAccountService.searchMemberAccount(memberId)).thenReturn(memberAccount);

        // then
        MemberProfileDto memberProfile = sut.getMemberProfile(memberId, "MALE");

        assertThat(memberProfile).isNotNull();
        assertThat(memberProfile.myCard().id()).isEqualTo(1L);
        assertThat(memberProfile.mateCard().id()).isEqualTo(2L);
    }

    private MemberPrincipal createMemberPrincipal(String memberId) {
        return MemberPrincipal.of(
            memberId,
            "test@mail.com",
            "nickname",
            "2024",
            "MALE",
            "010-1234-5678",
            new HashMap<>()
        );
    }
}
