export function getAge(birth: Date) {
  return new Date().getFullYear() - birth.getFullYear() + 1;
}
