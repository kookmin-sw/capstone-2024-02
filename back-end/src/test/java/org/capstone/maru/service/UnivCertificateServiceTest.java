package org.capstone.maru.service;

import static org.assertj.core.api.Assertions.assertThat;

import com.univcert.api.UnivCert;
import java.util.Map;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

//@Disabled("써드 파티 테스트")
@DisplayName("Service - 학교 인증")
class UnivCertificateServiceTest {

    @DisplayName("인증 가능한 대학교명을 넣었을 때 성공 반환")
    @Test
    void givenNothing_whenUnivCert_thenResponseSuccess() throws Exception {
        // given
        String univName = "국민대학교";
        String key = "success";

        // when
        Map<String, Object> result = UnivCert.check(univName);

        // then
        assertThat(result).containsEntry(key, true);
    }

    @DisplayName("인증 할 이메일로 인증 번호 발송")
    @Test
    void givenEmail_whenUnivCert_thenReturnCertNumber() throws Exception {
        // given
        String apiKey = "91927da3-bc95-4574-8d43-47e10a9272a7";
        String email = "777joonho@naver.com";
        String univName = "없는대학교";

        String key = "success";

        // when
        Map<String, Object> result = UnivCert.certify(apiKey, email, univName, true);

        // then
        assertThat(result).containsEntry(key, true);
    }

    @DisplayName("인증코드로 이메일 인증")
    @Test
    void givenCertNumber_whenUnivCert_thenCertificate() throws Exception {
        // given
        String apiKey = "91927da3-bc95-4574-8d43-47e10a9272a7";
        String email = "777joonho@gmail.com";
        String univName = "가나대학교";
        int code = 7314;

        String key = "success";

        // when
        Map<String, Object> result = UnivCert.certifyCode(apiKey, email, univName, code);

        // then
        assertThat(result).containsEntry(key, true);
    }

    @DisplayName("인증된 이메일인지 확인")
    @Test
    void givenEmail_whenUnivCert_thenReturnStatus() throws Exception {
        // given
        String apiKey = "91927da3-bc95-4574-8d43-47e10a9272a7";
        String email = "777joonho@kookmin.ac.kr";

        String key = "success";

        // when
        Map<String, Object> result = UnivCert.status(apiKey, email);

        // then
        assertThat(result).containsEntry(key, true);
    }
}
