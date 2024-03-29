import { ProfilePage } from '@/app/pages';

export default function Page({
  params: { userId },
}: {
  params: { userId: string };
}) {
  return (
    <ProfilePage
      src="https://s3-alpha-sig.figma.com/img/59a5/3c6f/ae49249b51c7d5d81ab89eeb0bf610f1?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=bS1lS~7-WQsD9x5vHJBOiMnhhFjI~VgCwJH6Bzz~IxFWob-PV-XZweWFIhU6yJC3XHv5qZKZxnP9RWT~0ciIbQfofuhbODEUxnMHe6oq8Dl45khsD30dnXOK~FLBPpWhMumJO-zPpuWjiRwsZ35mfWLbgyT7dND41I9yXCyRASQx9v2iAGzDoVzTfvtkjRyGw6es6fSXRsFGMqthnzYmv7DZT~FCz2avi3-NqXruXQpkijQHNEQUM61ThFiNYEIv8vb1wZWf-USbbJpE8bdbUneblY2T0cWwMRBtKbCrJ0Y~P9lvFbzqBv7h9WOzNyJW~~KeG9vVrBmLRRo1BsNdng__"
      name="김마루"
      age="24세"
      addr="성북 길음동"
    />
  );
}
