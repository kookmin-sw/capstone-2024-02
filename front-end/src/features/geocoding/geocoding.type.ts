export interface NaverAddress {
  roadAddress: string;
  jibunAddress: string;
  englishAddress: string;
  x: string;
  y: string;
  distance: number;
  addressElements: Array<{
    types: string[];
    longName: string;
    shortName: string;
    code: string;
  }>;
}

export interface FromAddrToCoordDTO {
  status: string;
  errorMessage: string;
  meta: {
    totalCount: number;
    page: number;
    count: number;
  };
  addresses: NaverAddress[];
}
