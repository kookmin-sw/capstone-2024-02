package org.capstone.maru.repository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.context.jdbc.Sql.ExecutionPhase.AFTER_TEST_METHOD;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.context.jdbc.Sql;
import org.springframework.test.context.junit.jupiter.SpringExtension;

@DataJpaTest
@ExtendWith(SpringExtension.class)
class JpaProjectionTest {

    @Autowired
    private MemberAccountRepository memberAccountRepository;

    @Autowired
    private MemberCardRepository memberCardRepository;

    @Test
    public void whenUsingClosedProjection() {
        // when
        System.out.println(memberAccountRepository.getInitializedById("kakao_12345678987654321"));
//        assertThat(memberAccountRepository.getInitializedById("kakao_12345678987654321"))
//            .hasSize(1);
    }

}
