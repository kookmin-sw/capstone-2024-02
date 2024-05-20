package org.capstone.maru.dto.request;

import org.capstone.maru.domain.jsonb.MemberFeatures;

public record MemberFeatureRequest(
    String location,
    MemberFeatures features
) {

}
