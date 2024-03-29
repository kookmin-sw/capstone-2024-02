package org.capstone.maru.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.BDDMockito.given;
import static org.mockito.BDDMockito.then;

import java.util.Optional;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.dto.MemberAccountDto;
import org.capstone.maru.repository.MemberAccountRepository;
import org.capstone.maru.security.exception.MemberAccountNotFoundException;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@DisplayName("Service - 회원")
@ExtendWith(MockitoExtension.class)
class MemberAccountServiceTest {

    @InjectMocks
    private MemberAccountService sut;

    @Mock
    private MemberAccountRepository memberAccountRepository;

    @DisplayName("존재하는 회원 ID를 검색하면, 회원 데이터를 Optional 로 반환한다.")
    @Test
    void givenExistentMemberId_whenSearching_thenReturnsOptionalMemberData() throws Exception {
        // given
        String memberId = "testId";
        given(memberAccountRepository.findById(memberId))
            .willReturn(Optional.of(createMemberAccount(memberId)));

        // when
        Optional<MemberAccountDto> result = Optional.of(sut.searchMember(memberId));
        // then
        assertThat(result).isPresent();
        then(memberAccountRepository).should().findById(memberId);
    }

    @DisplayName("존재하지 않는 회원 ID를 검색하면, 비어있는 Optional 을 반환한다.")
    @Test
    void givenNonexistentMemberId_whenSearching_thenReturnsOptionalMemberData() throws Exception {
        // given
        String wrongMemberId = "wrongId";
        given(memberAccountRepository.findById(wrongMemberId))
            .willReturn(Optional.empty());

        // when
        Exception exception = assertThrows(MemberAccountNotFoundException.class, () -> {
            sut.searchMember(wrongMemberId);
        });

        // then
        assertThat(exception).isInstanceOf(MemberAccountNotFoundException.class);
        then(memberAccountRepository).should().findById(wrongMemberId);
    }

    @DisplayName("회원 정보를 입력하면, 새로운 회원 정보를 저장하여 가입시키고 해당 회원 데이터를 반환한다.")
    @Test
    void givenMemberParams_whenSaving_thenReturnsMemberAccount() throws Exception {
        // given
        MemberAccount memberAccount = createMemberAccount("testId");
        MemberAccount savedMemberAccount = createSigningUpMemberAccount("testId");
        given(memberAccountRepository.save(memberAccount)).willReturn(savedMemberAccount);

        MemberAccount member = MemberAccount.of(
            memberAccount.getMemberId(),
            memberAccount.getEmail(),
            memberAccount.getNickname(),
            memberAccount.getBirthYear(),
            memberAccount.getGender(),
            memberAccount.getPhoneNumber(),
            memberAccount.getMemberId()
        );

        // when
        MemberAccountDto result = sut.login(
            member.getMemberId(),
            member.getEmail(),
            member.getNickname(),
            member.getBirthYear(),
            member.getGender(),
            member.getPhoneNumber()
        );

        // then
        assertThat(result)
            .hasFieldOrPropertyWithValue("memberId", memberAccount.getMemberId())
            .hasFieldOrPropertyWithValue("email", memberAccount.getEmail())
            .hasFieldOrPropertyWithValue("nickname", memberAccount.getNickname())
            .hasFieldOrPropertyWithValue("createdBy", memberAccount.getMemberId())
            .hasFieldOrPropertyWithValue("modifiedBy", memberAccount.getMemberId());
        then(memberAccountRepository).should().save(memberAccount);
    }

    private MemberAccount createMemberAccount(String memberId) {
        return createMemberAccount(memberId, null);
    }

    private MemberAccount createSigningUpMemberAccount(String memberId) {
        return createMemberAccount(memberId, memberId);
    }

    private MemberAccount createMemberAccount(String memberId, String createdBy) {
        return MemberAccount.of(
            memberId,
            "test@mail.com",
            "nickname",
            "2024",
            "MALE",
            "010-1234-5678",
            createdBy
        );
    }
}