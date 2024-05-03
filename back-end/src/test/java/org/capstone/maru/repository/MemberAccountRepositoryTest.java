package org.capstone.maru.repository;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.capstone.maru.config.TestJpaConfig;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.repository.postgre.MemberAccountRepository;
import org.capstone.maru.repository.postgre.MemberCardRepository;
import org.capstone.maru.util.EntityCreator;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.context.annotation.Import;
import org.springframework.test.context.TestPropertySource;

@DisplayName("JPA 연결 테스트")
@Import(TestJpaConfig.class)
@DataJpaTest
@TestPropertySource(locations = "classpath:application-test.yaml")
class MemberAccountRepositoryTest {

    private final MemberAccountRepository memberAccountRepository;
    private final MemberCardRepository memberCardRepository;

    public MemberAccountRepositoryTest(
        @Autowired MemberAccountRepository memberAccountRepository,
        @Autowired MemberCardRepository memberCardRepository
    ) {
        this.memberAccountRepository = memberAccountRepository;
        this.memberCardRepository = memberCardRepository;
    }

    @DisplayName("[MemberAccount] select 테스트")
    @Test
    void givenNothing_whenQueryingSelect_thenReturnMembers() throws Exception {
        // given
        int expected = 100;
        for (int i = 1; i <= expected; i++) {
            memberAccountRepository.save(EntityCreator.createMemberAccount(i));
        }

        // when
        List<MemberAccount> members = memberAccountRepository.findAll();

        // then
        assertThat(members)
            .isNotNull()
            .hasSize(expected);
    }

    @DisplayName("[MemberAccount] insert 테스트")
    @Test
    void givenNothing_whenQueryingInsert_thenReturnNothing() throws Exception {
        // given
        long previousCount = memberAccountRepository.count();
        long previousCardCount = memberCardRepository.count();

        // when

        MemberAccount memberAccount = EntityCreator.createMemberAccount(1);

        var memberAccountTest = memberAccountRepository.save(memberAccount);

        // then
        assertThat(memberAccountTest.getMemberId())
            .isEqualTo("test_1");
        assertThat(memberAccountRepository.count())
            .isEqualTo(previousCount + 1);
        assertThat(memberCardRepository.count())
            .isEqualTo(previousCardCount + 2);

        assertThat(memberAccountTest.getMyCard().getMemberFeatures())
            .isEmpty();
    }

    @DisplayName("[MemberCard] update 테스트")
    @Test
    void givenNothing_whenQueryingUpdate_thenReturnNothing() throws Exception {
        // given

        // when
        MemberAccount memberAccount = EntityCreator.createMemberAccount(1);

        var memberAccountTest = memberAccountRepository.save(memberAccount);

        FeatureCard featureCard = memberAccountTest.getMyCard();
        featureCard.updateMemberFeatures(List.of("아침형", "흡연", "음주"));

        long previousCardCount = memberCardRepository.count();

        var memberCardTest = memberCardRepository.save(featureCard);

        // then
        assertThat(memberCardTest.getMemberFeatures())
            .isEqualTo(List.of("아침형", "흡연", "음주"));
        assertThat(memberCardRepository.count())
            .isEqualTo(previousCardCount);
    }

}
