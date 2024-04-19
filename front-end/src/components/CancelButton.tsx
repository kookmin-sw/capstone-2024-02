export function CancelButton({
  onClick,
  fill,
}: {
  onClick: () => void;
  fill: string;
}) {
  return (
    <svg
      onClick={onClick}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
    >
      <path
        d="M4.70703 3.29297L3.29297 4.70703L10.5859 12L3.29297 19.293L4.70703 20.707L12 13.4141L19.293 20.707L20.707 19.293L13.4141 12L20.707 4.70703L19.293 3.29297L12 10.5859L4.70703 3.29297Z"
        fill={fill}
      />
    </svg>
  );
}
