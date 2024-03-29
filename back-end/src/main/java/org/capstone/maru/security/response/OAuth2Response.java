package org.capstone.maru.security.response;

import lombok.Getter;
import org.capstone.maru.domain.constant.Gender;

@Getter
public abstract class OAuth2Response {

    public abstract String id();

    public abstract String email();

    public abstract String nickname();

    public abstract String birthYear();

    public abstract Gender gender();

    public abstract String phoneNumber();
}
