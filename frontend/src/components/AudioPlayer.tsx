import { useEffect, useState } from 'react';

export function AudioPlayer({ file }: { file: File }) {
  const [url, setUrl] = useState('');

  useEffect(() => {
    const next = typeof URL.createObjectURL === 'function' ? URL.createObjectURL(file) : '';
    setUrl(next);
    return () => {
      if (next && typeof URL.revokeObjectURL === 'function') URL.revokeObjectURL(next);
    };
  }, [file]);

  return (
    <div className="audio-player">
      <audio controls src={url || undefined} aria-label={`Listen to ${file.name}`}>
        Audio playback is unavailable in this browser.
      </audio>
    </div>
  );
}
