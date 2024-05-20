package org.capstone.maru.service;

import com.univcert.api.UnivCert;
import java.io.IOException;
import java.util.Map;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.exception.CertificateException;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.repository.postgre.MemberAccountRepository;
import org.capstone.maru.security.exception.MemberAccountNotFoundException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class CertificateService {

    @Value("${univcert.api_key}")
    private String apiKey;

    private static final String KEY = "success";
    private final MemberAccountRepository memberAccountRepository;

    public CertificateService(MemberAccountRepository memberAccountRepository) {
        this.memberAccountRepository = memberAccountRepository;
    }

    public void certifyUniv(String univName) throws IOException {
        boolean success = (boolean) UnivCert
            .check(univName)
            .getOrDefault(KEY, false);

        if (!success) {
            throw new CertificateException(RestErrorCode.INVALID_CERTIFICATE_VALUE);
        }
    }

    public void certifyEmail(String univName, String email) throws IOException {
        boolean success = (boolean) UnivCert
            .certify(apiKey, email, univName, true)
            .getOrDefault(KEY, false);

        if (!success) {
            throw new CertificateException(RestErrorCode.INVALID_CERTIFICATE_VALUE);
        }
    }

    public void certifyUnivAndEmail(String email, String univName) {
        try {
            certifyUniv(univName);
            certifyEmail(univName, email);
        } catch (Exception e) {
            throw new CertificateException(RestErrorCode.INVALID_CERTIFICATE_VALUE);
        }
    }

    @Transactional
    public void certifyCode(String memberId, String email, String univName, Integer code) {
        boolean success;
        try {
            success = (boolean) UnivCert
                .certifyCode(apiKey, email, univName, code)
                .getOrDefault(KEY, false);
        } catch (IOException e) {
            throw new CertificateException(RestErrorCode.INVALID_CERTIFICATE_VALUE);
        }

        if (!success) {
            throw new CertificateException(RestErrorCode.INVALID_CERTIFICATE_VALUE);
        }
        
        MemberAccount memberAccount = memberAccountRepository
            .findByMemberId(memberId)
            .orElseThrow(() -> new MemberAccountNotFoundException(RestErrorCode.MEMBER_NOT_FOUND));

        memberAccount.updateUnivName(univName);
    }
}
