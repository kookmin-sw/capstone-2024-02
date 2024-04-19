export function getAge(birthYear: number) {
  return new Date().getFullYear() - birthYear + 1;
}
