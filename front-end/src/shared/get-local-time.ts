export function getLocalTime(isoString: string, type: string) {
  if (type === 'server') {
    const utcTime = new Date(isoString);
    const localTime = utcTime.setHours(utcTime.getHours() + 9);

    return new Date(localTime).toLocaleString('en-US', {
      hour12: true,
      hour: 'numeric',
      minute: '2-digit',
    });
  }

  return new Date(isoString).toLocaleString('en-US', {
    hour12: true,
    hour: 'numeric',
    minute: '2-digit',
  });
}
