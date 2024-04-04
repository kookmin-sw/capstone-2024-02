package org.capstone.maru.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.MemberCard;
import org.capstone.maru.domain.ProfileImage;
import org.capstone.maru.dto.MemberAccountDto;
import org.capstone.maru.dto.MemberProfileDto;
import org.capstone.maru.repository.MemberAccountRepository;
import org.capstone.maru.repository.MemberCardRepository;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.annotation.Autowired;

@DisplayName("Service - 프로필")
@ExtendWith(MockitoExtension.class)
class ProfileServiceTest {

    @InjectMocks
    ProfileService sut;
    @Mock
    MemberAccountService memberAccountService;

    @Mock
    MemberCardRepository memberCardRepository;

    @Mock
    MemberAccountRepository memberAccountRepository;

    @DisplayName("내 프로필 조회 할때, 내 프로필, 내 카드, 메이트 카드를 반환한다.")
    @Test
    void givenMemberId_whenSearching_thenReturnsMemberProfile() {
        // given
        String memberId = "testId";
        MemberAccount memberAccount = createMemberAccount(memberId);
        MemberPrincipal memberPrincipal = createMemberPrincipal(memberId);

        // when
        when(memberAccountService.searchMemberAccount(memberId)).thenReturn(memberAccount);

        // then
        MemberProfileDto memberProfile = sut.getMemberProfile(memberId);

        assertThat(memberProfile).isNotNull();
        assertThat(memberProfile.myCard().id()).isEqualTo(1L);
        assertThat(memberProfile.mateCard().id()).isEqualTo(2L);
    }

    private MemberCard createMyCard() {
        return MemberCard.of(
            1L,
            List.of("feature1", "feature2", "feature3")
        );
    }

    private MemberCard createMateCard() {
        return MemberCard.of(
            2L,
            List.of("feature1", "feature2", "feature3")
        );
    }

    private MemberAccount createMemberAccount(String memberId) {
        return MemberAccount.of(
            memberId,
            "test@mail.com",
            "nickname",
            "2024",
            "MALE",
            "010-1234-5678",
            "nickname",
            true,
            createMyCard(),
            createMateCard(),
            new HashSet<>(),
            new HashSet<>(),
            ProfileImage.defaultImage("test@mail.com")
        );
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
