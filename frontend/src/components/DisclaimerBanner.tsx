export function DisclaimerBanner({ compact = false }: { compact?: boolean }) {
  return <div className={`disclaimer ${compact ? 'compact' : ''}`} role="note"><span aria-hidden="true">ⓘ</span><span>{compact ? 'Educational project only. Not for diagnosis or medical decisions.' : 'Educational project only. This is not a medical device and must not be used for diagnosis or medical decisions.'}</span></div>;
}
