function Circle() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="50"
      height="50"
      viewBox="0 0 50 50"
      fill="none"
    >
      <circle cx="25" cy="25" r="25" fill="#D9D9D9" />
    </svg>
  );
}

export function AddButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      aria-label="Add-Button"
      type="button"
      onClick={() => {
        onClick();
      }}
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        all: 'unset',
        cursor: 'pointer',
        position: 'relative',
      }}
    >
      <Circle />
      <span
        style={{
          userSelect: 'none',
          WebkitUserSelect: 'none',
          MozUserSelect: 'none',
          msUserSelect: 'none',
          height: '100%',
          position: 'absolute',
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          color: '#000',
          textAlign: 'center',
          verticalAlign: 'middle',
          fontFamily: 'Noto Sans KR',
          fontSize: '1.875rem',
          fontStyle: 'normal',
          fontWeight: '400',
          lineHeight: 'normal',
        }}
      >
        +
      </span>
    </button>
  );
}
