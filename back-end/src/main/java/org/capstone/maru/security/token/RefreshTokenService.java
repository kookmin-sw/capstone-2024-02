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

    public void saveRefreshToken(String token, Long ttl) {
        repository.save(new RefreshToken(token, ttl));
    }

    public void removeRefreshToken(String token) {
        RefreshToken refreshToken = repository
            .findByToken(token)
            .orElseThrow(RefreshTokenNotFoundException::new);

        repository.delete(refreshToken);
    }

    public boolean hasRefreshToken(String token) {
        return repository.existsByToken(token);
    }
}
