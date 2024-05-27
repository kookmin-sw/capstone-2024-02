export interface SharedPostListItem {
  id: number;
  title: string;
  content: string;
  score: number;
  thumbnail: {
    fileName: string;
    isThumbnail: boolean;
    order: number;
  };
  publisherAccount: {
    memberId: string;
    email: string;
    nickname: string;
    birthYear: string;
    gender: string;
    phoneNumber: string;
    profileImageFileName: string;
    createdAt: Date;
    createdBy: string;
    modifiedAt: Date;
    modifiedBy: string;
  };
  address: {
    oldAddress: string;
    roadAddress: string;
  };
  roomInfo: {
    id: number;
    roomType: string;
    floorType: string;
    size: number;
    numberOfRoom: number;
    numberOfBathRoom: number;
    rentalType: string;
    expectedPayment: number;
    recruitmentCapacity: number;
  };
  isScrapped: boolean;
  createdAt: Date;
  createdBy: string;
  modifiedAt: Date;
  modifiedBy: string;
}

export interface SharedPost {
  id: number;
  title: string;
  content: string;
  roomMateFeatures: {
    smoking?: string;
    roomSharingOption?: string;
    mateAge?: string;
    options: string; // 프런트에서 파싱 필요.
  };
  participants: Array<{
    memberId: string;
    nickname: string;
    profileImageFileName: string;
    birthYear: string;
    isScrapped: boolean;
  }>;
  roomImages: Array<{
    fileName: string;
    isThumbnail: boolean;
    order: number;
  }>;
  publisherAccount: {
    memberId: string;
    email: string;
    nickname: string;
    birthYear: string;
    gender: string;
    phoneNumber: string;
    profileImageFileName: string;
    createdAt: Date;
    createdBy: string;
    modifiedAt: Date;
    modifiedBy: string;
  };
  address: {
    oldAddress: string;
    roadAddress: string;
  };
  roomInfo: {
    id: number;
    roomType: string;
    floorType: string;
    size: number;
    numberOfRoom: number;
    numberOfBathRoom: number;
    hasLivingRoom: boolean;
    recruitmentCapacity: number;
    rentalType: string;
    expectedPayment: number;
    extraOption: {
      canPark: boolean;
      hasAirConditioner: boolean;
      hasRefrigerator: boolean;
      hasWasher: boolean;
      hasTerrace: boolean;
    };
  };
  isScrapped: boolean;
  scrapCount: number;
  viewCount: number;
  createdAt: Date;
  createdBy: string;
  modifiedAt: Date;
  modifiedBy: string;
}

export interface DormitorySharedPostListItem {
  id: number;
  title: string;
  content: string;
  score: number;
  thumbnail: {
    fileName: string;
    isThumbNail: boolean;
    order: number;
  };
  publisherAccount: {
    memberId: string;
    email: string;
    nickname: string;
    birthYear: string;
    gender: string;
    phoneNumber: string;
    profileImageFileName: string;
    createdAt: Date;
    createdBy: string;
    modifiedAt: Date;
    modifiedBy: string;
  };
  address: {
    oldAddress: string;
    roadAddress: string;
  };
  recruitmentCapacity: number;
  isScrapped: boolean;
  createdAt: Date;
  createdBy: string;
  modifiedAt: Date;
  modifiedBy: string;
}

export interface DormitorySharedPost {
  id: number;
  title: string;
  content: string;
  roomMateFeatures: {
    smoking?: string;
    roomSharingOption?: string;
    mateAge?: string;
    options: string;
  };
  participants: Array<{
    memberId: string;
    nickname: string;
    profileImageFileName: string;
    birthYear: string;
    isScrapped: boolean;
  }>;
  roomImages: Array<{
    fileName: string;
    isThumbNail: boolean;
    order: number;
  }>;
  publisherAccount: {
    memberId: string;
    email: string;
    nickname: string;
    birthYear: string;
    gender: string;
    phoneNumber: string;
    profileImageFileName: string;
    createdAt: Date;
    createdBy: string;
    modifiedAt: Date;
    modifiedBy: string;
  };
  address: {
    oldAddress: string;
    roadAddress: string;
  };
  recruitmentCapacity: number;
  isScrapped: boolean;
  scrapCount: number;
  viewCount: number;
  createdAt: Date;
  createdBy: string;
  modifiedAt: Date;
  modifiedBy: string;
}
