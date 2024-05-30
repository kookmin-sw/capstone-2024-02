export const convertPhoneNumber = (input: string) => {
  const phonePattern1 = /^010-\d{4}-\d{4}$/;
  const phonePattern2 = /^\+82 10-\d{4}-\d{4}$/;

  if (phonePattern1.test(input)) return input;
  if (phonePattern2.test(input))
    return input.replace(/^\+82 10-(\d{4})-(\d{4})$/, '010-$1-$2');
  return input;
};
