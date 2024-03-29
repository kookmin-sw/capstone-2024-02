package org.capstone.maru.security.token;

import lombok.RequiredArgsConstructor;
import org.capstone.maru.security.exception.RefreshTokenNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@RequiredArgsConstructor
@Transactional
@Service
public class RefreshTokenService {

    private final RefreshTokenRepository repository;

    public void saveRefreshToken(String refreshToken) {
        repository.save(new RefreshToken(refreshToken));
    }

    public void removeRefreshToken(String refreshToken) {
        RefreshToken token = repository
            .findByRefreshToken(refreshToken)
            .orElseThrow(RefreshTokenNotFoundException::new);

        repository.delete(token);
    }

    public boolean hasRefreshToken(String refreshToken) {
        return repository.existsByRefreshToken(refreshToken);
    }
}
